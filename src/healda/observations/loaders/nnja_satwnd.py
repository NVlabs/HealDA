# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NNJA satellite-derived atmospheric motion vectors as conventional wind rows.

APPLIED here: report type must be in 240-260 (read from the archive, which
resolves it at write time); SWCM 1-7; finite lat/lon/pressure/wind within the
channel table's bounds; thinning (see NNJASatwndLoader).

NOT applied: zenith limb, qifn, expected error, type-specific pressure limits,
convinfo iuse, or anything needing a background. See satwnd_qc for what is.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from dataclasses import dataclass
from typing import Sequence

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from healda.config.environment import nnja_archive
from healda.observations.loaders.nnja_base import (
    CycleTableSource,
    GLOBAL_CHANNEL_ID,
    LOCAL_CHANNEL_ID,
    SENSOR_ID,
    NNJAArchiveLoader,
    row_generator,
)

from healda.observations.preprocessing import satwnd_kernels, satwnd_qc
from healda.observations.loaders.archive_row_groups import window_row_groups
from healda.observations.loaders.threads import configure_arrow_pools, thread_pool
from healda.observations.sensors import (
    CONV_CHANNELS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)

logger = logging.getLogger(__name__)

# Read back as DictionaryArray rather than plain strings. Parquet already
# stores these RLE_DICTIONARY with a handful of distinct values, so decoding
# to strings only to re-encode downstream costs ~38 ms per window.
DICTIONARY_COLUMNS = ["ncep_dump_subtype", "gsi_type_mapping_tier"]

SATWND_ARCHIVE = nnja_archive("parquet", "satwnd_v2")

SATWND_REPORT_TYPES = frozenset(range(240, 261))

SATWND_TYPE_MIN = min(SATWND_REPORT_TYPES)
SATWND_TYPE_MAX = max(SATWND_REPORT_TYPES)
ARCHIVE_HPX_LEVEL = 12
U_LOCAL_CHANNEL_ID = 6
V_LOCAL_CHANNEL_ID = 7
# Sat_Zenith_Angle is kept, and SIGNED: negatives are a real off-nadir angle
# for cross-track scanners, and GSI screens on it.
NULL_COLUMNS = ("Sol_Zenith_Angle", "Scan_Angle")
REQUIRED_COLUMNS = (
    "time_utc",
    "da_window",
    "latitude",
    "longitude",
    "hpx4096_nest",
    "wind_u_derived",
    "wind_v_derived",
    "assigned_pressure",
    "gsi_observation_type",
    "gsi_type_mapping_tier",
    # satwnd_qc screens on SWCM and uses it to identify layer winds.
    "wind_computation_method",
    # The quarantine keys on the source subset.
    "ncep_dump_subtype",
    # Source file identity, which the quarantine keys on. da_window cannot
    # substitute: it splits one cycle across two windows.
    "cycle",
)
OPTIONAL_COLUMNS = (
    "sdmedit_wind_quality_mark",
    "satellite_zenith_angle",
    "gsi_internal_subtype",
    # GNAP-resolved QI. The promoted per_cent_confidence scalar is not read:
    # it is the lowest GNAP, which is qify for EUMETSAT and JMA.
    "quality_diagnostics",
)

# One type can carry two provenances -- 241 is INSAT via prepdata (NC005024/25)
# and GOES-R via the GSI sattab (NC005099) -- so the tier rides in QC_Flag,
# which Observation_Type alone cannot separate. These are the values the ETL
# writes into gsi_type_mapping_tier; order is persisted as the QC_Flag code, so
# append only.
TIER_CODES = (
    "1-direct-gsi-sattab",
    "1-direct-gsi-sattab-unusable-istype-1",
    "2-pre-refactor-gsi",
    "3-prepdata",
    "0-stored-as-built",
)
TIER_QC_CODES = {name: index + 1 for index, name in enumerate(TIER_CODES)}
TIER_QC_UNRESOLVED = 0


@numba.njit(cache=True, nogil=True)
def _winner_kernel(cell, score, best, owner):
    for row in range(cell.shape[0]):
        group = cell[row]
        if score[row] < best[group]:
            best[group] = score[row]
            owner[group] = row
    return owner


def _winners_by_cell(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Index of the lowest-`score` row within each distinct `key`, ties to the
    earliest row.

    `key` is sparse over a huge domain, so it is compacted to dense group ids
    first (hash-based, O(n)) rather than indexing an array over the raw key
    space -- at hpx7 that space is ~2e8 cells and allocating it per window
    would cost more than the sort this replaces.
    """
    codes, uniques = pd.factorize(key, sort=False)
    n_cells = len(uniques)
    best = np.full(n_cells, np.iinfo(np.int64).max, dtype=np.int64)
    owner = np.zeros(n_cells, dtype=np.int64)
    codes = codes.astype(np.int64, copy=False)

    return _winner_kernel(codes, score, best, owner)


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    stddev: float
    min_valid: float
    max_valid: float


# U.S. Standard Atmosphere 1976 geopotential layers: base height (m), lapse
# rate (K/m), base temperature (K), base pressure (Pa) at full precision. One
# table, one inversion. The hand-written per-layer form this replaces had
# 32-47 km isothermal where the layer lapses at +0.0028 K/m (a 658 m step at
# 1.109 hPa) and a sign error above 51 km that made height fall as pressure
# fell.
_USSA_H_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0])
_USSA_LAPSE_K_M = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
_USSA_T_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_USSA_P_PA = np.array(
    [
        101_325.0,
        22_632.06397346295,
        5_474.888669677785,
        868.0186847552303,
        110.9063055549665,
        66.93887311868764,
        3.9564204280407553,
    ]
)
# Top of the last layer, 84.852 km. Below this the table does not define an
# atmosphere, so extrapolating it would invent one.
_USSA_TOP_P_PA = 0.37338358997621796
# R* / (g0 * M0), in m/K
_USSA_K = 8.31432 / (9.80665 * 0.0289644)


def pressure_to_height_m(pressure_hpa: np.ndarray) -> np.ndarray:
    """USSA-1976 geopotential height for a pressure, in metres.

    A climatological pressure altitude, not an observed geometric height.
    Non-finite, non-positive, or out-of-domain pressure gives NaN rather than a
    number that would pass downstream as a real height.

    Floored at 0. Pressure above standard sea level has a genuinely negative
    pressure altitude (1100 hPa is about -698 m), but Height feeds a
    HEIGHT_MIN=0 filter, so returning it unclamped would silently drop those
    rows -- 1,218 of 15.2M kept. The floor is a downstream contract, not
    physics.

    TODO: allow negative heights for all conv obs, screened for validity rather
    than floored. Sub-sea-level pressure altitude is real and the 0 floor is a
    repo-wide limitation, not one of this loader: QCLimits.HEIGHT_MIN and the
    height_ok screen in obs_filtering_utils would have to move together.
    """
    pressure_pa = np.asarray(pressure_hpa, dtype=np.float64) * 100.0
    height = np.full(pressure_pa.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(pressure_pa) & (pressure_pa >= _USSA_TOP_P_PA)

    # A mask per layer, not a searchsorted gather: measured 2.1x faster at the
    # ~136k winners a window carries, because it skips the empty upper layers
    # and avoids a per-element binary search plus indexed gathers.
    for index in range(_USSA_P_PA.size):
        top_p = _USSA_P_PA[index + 1] if index + 1 < _USSA_P_PA.size else _USSA_TOP_P_PA
        # Layer 0 also takes pressure above its base, i.e. below sea level.
        below = (
            pressure_pa <= _USSA_P_PA[index]
            if index
            else np.ones(pressure_pa.shape, dtype=bool)
        )
        # The last layer owns its own top boundary; the others stop short of
        # the next layer's base, which that layer claims.
        above_top = (
            pressure_pa >= top_p
            if index + 1 == _USSA_P_PA.size
            else pressure_pa > top_p
        )
        layer = usable & below & above_top
        if not layer.any():
            continue
        ratio = pressure_pa[layer] / _USSA_P_PA[index]
        lapse = _USSA_LAPSE_K_M[index]
        if lapse:
            height[layer] = _USSA_H_M[index] + _USSA_T_K[index] / lapse * (
                np.power(ratio, -lapse * _USSA_K) - 1.0
            )
        else:
            height[layer] = _USSA_H_M[index] - _USSA_K * _USSA_T_K[index] * np.log(
                ratio
            )
    return np.maximum(height, 0.0).astype(np.float32)


class NNJASatwndLoader(NNJAArchiveLoader):
    """Read daily AMV Parquet files and emit thinned conv ``u``/``v`` rows.

    THINNING. Each 3h DA window is thinned on its own. The archive assigns every
    row to the window ``W`` whose ``(W - 3h, W]`` holds its time; a request's
    ``obs_context_hours`` selects which windows are read. Within a window one row
    is kept per cell of

    - space: NESTED HEALPix pixel at ``thin_hpx_level``. Set from
      ``ObsConfig.nnja_satwnd_thin_hpx_level``. The default 5 (~200 km) follows
      GSI's 200 km AMV thinning mesh.
    - pressure: the nearest of the model's 13 levels (``PRESSURE_LEVELS_HPA``).
      ``pressure_bin_hpa`` replaces them with fixed-width bins.
    - time: ``floor((t - W + 3h) / time_bin_hours)``. The default 2h follows
      GSI's AMV ``ptime`` and splits a window into ``(W - 3h, W - 1h)`` and
      ``[W - 1h, W]``.
    - report type.

    The kept row is the one nearest ``W``. ``pressure_bin_hpa`` and
    ``time_bin_hours`` are constructor arguments only.

    VERTICAL COORDINATE. The data arrives as a PRESSURE. The producing centre's
    height-assignment algorithm (BUFR HAMD) outputs a pressure (BUFR PRLC); raw
    SATWND may carry up to 11 of them -- final, window-channel, histogram,
    H2O-intercept, CO2-slicing -- with method codes and estimated errors. PREPDATA
    hardwires JTP=1, keeping only the final assignment as POB, and this loader
    likewise reads only `assigned_pressure`. The alternatives, and their
    disagreement, are the genuinely unused information here.

    `Height` in the output is FILLED IN HERE, by standard-atmosphere conversion of
    that pressure. It is a unit change, not an observation, and the archive's HGHT
    column is null on every row checked (55.7M, 2000-2025).

    NCEP fills ZOB the same way, but conditionally: iw3unpbf.f takes a provider
    HGHT when present and otherwise applies the fixed relation
    ZOB = 44330.77 * [1 - (POB/1013.25)**0.19026]. For the audited JMA/EUMETSAT
    population no raw HGHT existed, so every ZOB was derived; that is not a rule for
    every template. Our conversion equals theirs to <=0.2 m in the troposphere, then
    diverges above ~226 hPa because ours is piecewise and theirs extrapolates one
    layer: 9 m at 200 hPa, 205 m at 125 hPa, 383 m at 100 hPa.

    None of this reaches assimilation. GSI positions AMVs by POB in log pressure and
    ignores ZOB in the observation operator; the height-coordinate branch is for
    PIBAL 221-229, not 240-260. Beware the label: a GSI diagnostic field named
    Height is ZOB on the PrepBUFR route and the expected-error `ee` on the direct
    raw SATWND route.
    """

    def __init__(
        self,
        archive_root: str = SATWND_ARCHIVE,
        obs_context_hours: tuple[int, int] = (-3, 3),
        data_spacing: int = 3,
        normalize: bool = True,
        thin_hpx_level: int = 5,
        pressure_bin_hpa: float | None = None,
        time_bin_hours: float = 2.0,
        qc_config: satwnd_qc.SatwndQCConfig | None = None,
        wind_obs_dropout: float = 0.0,
        dropout_scope: str = "row",
        table_source: CycleTableSource | None = None,
    ) -> None:
        if data_spacing != 3:
            raise ValueError("NNJA satwnd windows require data_spacing=3")
        start, end = obs_context_hours
        if start % 3 or end % 3:
            raise ValueError(
                "obs_context_hours endpoints must be 3-hour aligned, "
                f"got {obs_context_hours}"
            )
        if not 0 <= thin_hpx_level <= ARCHIVE_HPX_LEVEL:
            raise ValueError(
                f"thin_hpx_level must be in [0, {ARCHIVE_HPX_LEVEL}], "
                f"got {thin_hpx_level}"
            )
        if pressure_bin_hpa is not None and (
            not np.isfinite(pressure_bin_hpa) or pressure_bin_hpa <= 0
        ):
            raise ValueError(
                "pressure_bin_hpa must be positive or None (thin on the model's "
                f"13 conv-plevel levels), got {pressure_bin_hpa}"
            )
        if not np.isfinite(time_bin_hours) or time_bin_hours <= 0:
            raise ValueError(f"time_bin_hours must be positive, got {time_bin_hours}")
        if not 0.0 <= wind_obs_dropout <= 1.0:
            raise ValueError(
                f"wind_obs_dropout must be in [0, 1], got {wind_obs_dropout}"
            )
        self.archive_root = archive_root
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        self.normalize = normalize
        self.thin_hpx_level = thin_hpx_level
        self.pressure_bin_hpa = pressure_bin_hpa
        self.time_bin_hours = time_bin_hours
        self.wind_obs_dropout = wind_obs_dropout
        self.dropout_scope = dropout_scope
        # GSI's main AMV value screen is `qifn >= 85`. Off by default so this
        # port changes volume for one reason only (type resolution); turn it on
        # deliberately rather than discovering it in a row count.
        # None means satwnd_qc's own defaults: the GSI screens that are global
        # and forecast-independent, with the QI screen OFF.
        self.qc_config = qc_config or satwnd_qc.SatwndQCConfig()
        # Read instead of archive_root when given.
        self.table_source = table_source
        configure_arrow_pools()

    @functools.cached_property
    def channel_stats(self) -> dict[int, ChannelStats]:
        """Packaged u/v conv stats; see NNJAConvLoader.channel_stats."""
        conv = SENSOR_CONFIGS["conv"]
        return {
            local_id: ChannelStats(
                mean=float(conv.means[local_id]),
                stddev=float(conv.stds[local_id]),
                min_valid=float(CONV_CHANNELS[local_id].min_valid),
                max_valid=float(CONV_CHANNELS[local_id].max_valid),
            )
            for local_id in (U_LOCAL_CHANNEL_ID, V_LOCAL_CHANNEL_ID)
        }

    def _path(self, day: pd.Timestamp) -> str:
        return os.path.join(self.archive_root, f"{day:%Y%m%d}.parquet")

    def _winner_rows(
        self,
        table: pa.Table,
        valid: np.ndarray,
        window: pd.Timestamp,
        pressure_hpa: np.ndarray,
        report_type: np.ndarray,
        pixel_f64: np.ndarray,
    ) -> np.ndarray:
        rows = np.flatnonzero(valid)
        if not rows.size:
            return rows

        time_ns = (
            self._numpy(table, "time_utc")[rows]
            .astype("datetime64[ns]", copy=False)
            .view(np.int64)
        )
        window_ns = int(np.datetime64(window.to_datetime64(), "ns").astype(np.int64))
        thinning_keys, distance_ns = satwnd_kernels.build_thinning_keys(
            rows=rows,
            pixel_f64=pixel_f64,
            pressure_hpa=pressure_hpa,
            time_ns=time_ns,
            report_type=report_type,
            window_ns=window_ns,
            thin_hpx_level=self.thin_hpx_level,
            archive_hpx_level=ARCHIVE_HPX_LEVEL,
            pressure_bin_hpa=self.pressure_bin_hpa,
            time_bin_hours=self.time_bin_hours,
            report_type_min=SATWND_TYPE_MIN,
        )
        return np.sort(rows[_winners_by_cell(thinning_keys, distance_ns)])

    def _to_long(
        self,
        table: pa.Table,
        window: pd.Timestamp,
        rng: np.random.Generator | None = None,
        wind_dropout: float = 0.0,
    ) -> pa.Table:
        if not table.num_rows:
            return self._empty()

        u = self._numpy(table, "wind_u_derived", np.float64)
        v = self._numpy(table, "wind_v_derived", np.float64)
        pressure_hpa = self._numpy(table, "assigned_pressure", np.float64) / 100.0
        # The archive resolves the GSI type at write time. It is stored uint16,
        # so the null mask comes from Arrow rather than a float64 round-trip
        # just to call isfinite. -1 marks a row the archive left untyped.
        column = table["gsi_observation_type"]
        raw = column.to_numpy(zero_copy_only=False)
        if column.null_count:
            # Nulls widen the column to float64 with NaN, which cannot be cast
            # to an integer type directly. -1 marks a row the archive left
            # untyped; the report-type range screen then drops it.
            untyped = ~np.isfinite(raw)
            report_type = np.where(untyped, -1.0, raw).astype(np.int32)
        else:
            report_type = raw.astype(np.int32, copy=False)
        window64 = np.datetime64(window.to_datetime64(), "ns")

        u_stats = self.channel_stats[U_LOCAL_CHANNEL_ID]
        v_stats = self.channel_stats[V_LOCAL_CHANNEL_ID]

        # GSI-derived screening lives in satwnd_qc, which owns the policy; the
        # loader owns only what is specific to ITS output contract -- the
        # channel table's validity bounds, which exist for normalisation, not
        # for quality.
        result = satwnd_qc.evaluate(
            table,
            window64,
            report_type,
            self.qc_config,
            # Already materialised above; handing them over avoids a second
            # full-length float64 conversion of each inside the QC.
            columns={
                "wind_u_derived": u,
                "wind_v_derived": v,
                "pressure_hpa": pressure_hpa,
            },
        )
        # NaN -> uint64 is undefined, so a null pixel must be dropped before
        # thinning rather than cast -- rare (1-2 rows/cycle) but real.
        pixel_f64 = np.asarray(
            table["hpx4096_nest"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        has_pixel = np.isfinite(pixel_f64)

        valid = (
            result.keep_mask
            & has_pixel
            & (report_type >= SATWND_TYPE_MIN)
            & (report_type <= SATWND_TYPE_MAX)
            & np.isfinite(u)
            & (u >= u_stats.min_valid)
            & (u <= u_stats.max_valid)
            & np.isfinite(v)
            & (v >= v_stats.min_valid)
            & (v <= v_stats.max_valid)
        )
        rows = self._winner_rows(
            table, valid, window, pressure_hpa, report_type, pixel_f64
        )
        # Serves both scopes: `wind_dropout` is the configured p under "row", and 0 or 1
        # under "sample", where this keeps every AMV or none of them.
        if wind_dropout:
            rng = rng or np.random.default_rng()
            rows = rows[rng.random(rows.size) >= wind_dropout]
        if not rows.size:
            return self._empty()

        # Everything below is per-output-row, so it is computed on the thinned
        # winners only. At hpx5 that is ~1 in 8 of the rows read.
        lat = self._numpy(table, "latitude", np.float64)[rows]
        lon = self._numpy(table, "longitude", np.float64)[rows]
        time = self._numpy(table, "time_utc")[rows].astype("datetime64[ns]", copy=False)
        # NOTE ~always 1: the column it derives from is always 2 where present.
        quality = (
            self._numpy(table, "sdmedit_wind_quality_mark", np.float64)[rows]
            if "sdmedit_wind_quality_mark" in table.column_names
            else np.full(rows.size, np.nan)
        )
        assimilable = ~np.isfinite(quality) | (quality <= 2)
        height = pressure_to_height_m(pressure_hpa[rows])

        if "satellite_zenith_angle" in table.column_names:
            zenith = self._numpy(table, "satellite_zenith_angle", np.float64)[rows]
        else:
            zenith = np.full(rows.size, np.nan, dtype=np.float64)
        # QC_Flag carries the tier as a 1-based TIER_CODES index, 0 = untyped.
        tier_code = self._tier_codes(table, rows)

        # Only Observation, Global_Channel_ID and local_channel_id differ between U and
        # V; the rest is shared rather than rebuilt per channel.
        count = rows.size
        shared_arrays = {
            "Latitude": pa.array(lat, type=pa.float32()),
            "Longitude": pa.array(np.mod(lon, 360.0), type=pa.float32()),
            "Absolute_Obs_Time": pa.array(time),
            "DA_window": pa.array(np.full(count, window64, dtype="datetime64[ns]")),
            "Platform_ID": pa.array(
                np.full(count, PLATFORM_NAME_TO_ID["uv"], dtype=np.uint16)
            ),
            "Pressure": pa.array(pressure_hpa[rows], type=pa.float32()),
            "Height": pa.array(height, type=pa.float32()),
            "Observation_Type": pa.array(report_type[rows], type=pa.uint16()),
            "Analysis_Use_Flag": pa.array(assimilable.astype(np.int8), type=pa.int8()),
            "Sat_Zenith_Angle": pa.array(zenith, type=pa.float32()),
            "QC_Flag": pa.array(tier_code, type=pa.int32()),
            SENSOR_ID.name: pa.array(
                np.full(count, SENSOR_NAME_TO_ID["conv"], dtype=np.uint16)
            ),
        }
        for name in NULL_COLUMNS:
            shared_arrays[name] = pa.nulls(
                count, type=self.output_schema.field(name).type
            )

        parts = []
        for local_id, values, stats in (
            (U_LOCAL_CHANNEL_ID, u, u_stats),
            (V_LOCAL_CHANNEL_ID, v, v_stats),
        ):
            observation = values[rows].astype(np.float32)
            if self.normalize:
                if not np.isfinite(stats.stddev) or stats.stddev <= 0:
                    raise ValueError(
                        f"invalid stddev for conv channel {local_id}: {stats.stddev}"
                    )
                observation = (
                    (observation - np.float32(stats.mean)) / np.float32(stats.stddev)
                ).astype(np.float32)
            arrays = shared_arrays | {
                "Observation": pa.array(observation, type=pa.float32()),
                GLOBAL_CHANNEL_ID.name: pa.array(
                    np.full(count, SENSOR_OFFSET["conv"] + local_id, dtype=np.uint16),
                    type=GLOBAL_CHANNEL_ID.type,
                ),
                LOCAL_CHANNEL_ID.name: pa.array(
                    np.full(count, local_id, dtype=np.uint16)
                ),
            }
            parts.append(
                pa.table(
                    [arrays[field.name] for field in self.output_schema],
                    schema=self.output_schema,
                )
            )
        return pa.concat_tables(parts)

    @staticmethod
    def _tier_codes(table: pa.Table, rows: np.ndarray) -> np.ndarray:
        """QC_Flag codes from the stored tier names.

        Decoded through the dictionary rather than to_pylist: a handful of
        distinct names over millions of rows, so the mapping is done once per
        name and gathered, not once per row.
        """
        if "gsi_type_mapping_tier" not in table.column_names:
            return np.full(rows.size, TIER_QC_UNRESOLVED, dtype=np.int32)
        column = table.column("gsi_type_mapping_tier").combine_chunks()
        if isinstance(column, pa.ChunkedArray):
            column = column.chunk(0) if column.num_chunks else pa.array([], pa.string())
        if not pa.types.is_dictionary(column.type):
            column = column.dictionary_encode()
        lookup = np.array(
            [
                TIER_QC_CODES.get(name, TIER_QC_UNRESOLVED)
                for name in column.dictionary.to_pylist()
            ],
            dtype=np.int32,
        )
        # A null index comes back as NaN in a float array, which cannot index.
        indices = column.indices.to_numpy(zero_copy_only=False)
        null = ~np.isfinite(indices) if indices.dtype.kind == "f" else None
        indices = np.nan_to_num(indices, nan=0.0).astype(np.intp, copy=False)[rows]
        codes = lookup[indices]
        if null is not None:
            codes = np.where(null[rows], TIER_QC_UNRESOLVED, codes)
        return codes.astype(np.int32)

    def _load_day(
        self,
        day: pd.Timestamp,
        windows: Sequence[pd.Timestamp],
        dropout_seed: int | None = None,
        wind_dropout: float = 0.0,
    ) -> list[tuple[pd.Timestamp, pa.Table]]:
        if self.table_source is not None:
            return self._load_windows_from_source(
                windows, row_generator(dropout_seed), wind_dropout
            )
        path = self._path(day)
        if not os.path.exists(path):
            return []
        parquet = pq.ParquetFile(path, pre_buffer=True)
        available = set(parquet.schema_arrow.names)
        missing = set(REQUIRED_COLUMNS) - available
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        # Reopen only to set read_dictionary; it raises KeyError on an absent
        # column, so it can only be passed once the schema is known.
        dictionary = [c for c in DICTIONARY_COLUMNS if c in available]
        if dictionary:
            parquet = pq.ParquetFile(path, pre_buffer=True, read_dictionary=dictionary)
        # da_window is REQUIRED to exist -- its row-group statistics are how a
        # window is located -- but it is never read: the archive contract gives
        # one window per row group, and _to_long takes `window` as an argument.
        # quality_diagnostics is a nested list column and by far the largest
        # optional read, and it is untouched unless the QI screen is on, so it
        # is requested only when it will actually be used.
        skip = {"da_window"}
        if not self.qc_config.apply_qifn:
            # quality_diagnostics is the GNAP-resolved QI source and is only
            # read when the screen is on; the promoted scalar is never used.
            skip.add("quality_diagnostics")
        columns = [name for name in REQUIRED_COLUMNS if name not in skip]
        columns += [
            name for name in OPTIONAL_COLUMNS if name in available and name not in skip
        ]

        wanted = {pd.Timestamp(window).tz_localize(None) for window in windows}
        by_window: dict[pd.Timestamp, list[pa.Table]] = {
            window: [] for window in wanted
        }
        rng = row_generator(dropout_seed)
        # One da_window per row group is an archive contract, not a hope: the
        # day writer emits eight single-window groups. The shared helper
        # asserts it, and derives the statistics column index from the file's
        # is a different numbering once a file has nested columns (a SATWND day
        # has 119 Arrow fields and 145 leaves).
        for window, group in window_row_groups(parquet, sorted(wanted), path=path):
            long = self._to_long(
                parquet.read_row_group(group, columns=columns),
                window,
                rng,
                wind_dropout,
            )
            if long.num_rows:
                by_window[window].append(long)
        return [
            (window, pa.concat_tables(parts))
            for window, parts in sorted(by_window.items())
            if parts
        ]

    def _load_windows_from_source(
        self,
        windows: Sequence[pd.Timestamp],
        rng: np.random.Generator | None,
        wind_dropout: float,
    ) -> list[tuple[pd.Timestamp, pa.Table]]:
        result = []
        for window in sorted({pd.Timestamp(w).tz_localize(None) for w in windows}):
            table = self.table_source.get(self._cycle_and_half(window)[0])
            if table is None or not table.num_rows:
                continue
            missing = set(REQUIRED_COLUMNS) - set(table.column_names)
            if missing:
                raise ValueError(
                    f"table_source[{window}]: missing columns {sorted(missing)}"
                )
            da_window = self._numpy(table, "da_window").astype("datetime64[ns]")
            rows = da_window == np.datetime64(window.to_datetime64(), "ns")
            long = self._to_long(
                table.filter(pa.array(rows)), window, rng, wind_dropout
            )
            if long.num_rows:
                result.append((window, long))
        return result

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        needed: dict[pd.Timestamp, set[pd.Timestamp]] = {}
        for target in times:
            for window in self._interval_times(target):
                day = pd.Timestamp(window).normalize()
                needed.setdefault(day, set()).add(pd.Timestamp(window))

        loop = asyncio.get_running_loop()
        dropout = self._dropout()
        wind_dropout = dropout.rate(self.wind_obs_dropout)
        dropout_seeds = dropout.seeds(needed)
        loaded = await asyncio.gather(
            *(
                loop.run_in_executor(
                    thread_pool(),
                    self._load_day,
                    day,
                    tuple(sorted(windows)),
                    dropout_seeds[day],
                    wind_dropout,
                )
                for day, windows in sorted(needed.items())
            )
        )
        by_window = {
            window: table for day_parts in loaded for window, table in day_parts
        }
        empty = self._empty()
        return {
            "obs_v2": [
                pa.concat_tables(
                    [
                        by_window[window]
                        for window in self._interval_times(target)
                        if window in by_window
                    ]
                )
                if any(window in by_window for window in self._interval_times(target))
                else empty
                for target in times
            ]
        }

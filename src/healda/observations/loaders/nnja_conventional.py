# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NNJA conventional Parquet loader for the unified long observation schema.

The archive under ``parquet/cycles`` is PrepBUFR flattened to Parquet: one row
per report level and one column per physical variable. This loader projects
those columns, applies each variable's quality mark, converts to Healda units,
and unpivots the surviving values into the model-facing long schema.

When ``include_gpsro`` is set, GPS-RO bending angles from ``parquet/gpsro`` are
appended as the ``gps_angle`` channel (conv local id 0).
"""

from __future__ import annotations

import asyncio
import functools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from healda.config.environment import nnja_archive
from healda.observations.loaders.nnja_base import (
    GLOBAL_CHANNEL_ID,
    CycleTableSource,
    LOCAL_CHANNEL_ID,
    SENSOR_ID,
    NNJAArchiveLoader,
    row_generator,
)

from healda.observations.loaders.threads import configure_arrow_pools, thread_pool
from healda.observations.loaders.nnja_gpsro import GPS_LEGACY_SAIDS, NNJAGpsroLoader
from healda.observations.loaders.nnja_satwnd import (
    SATWND_ARCHIVE,
    SATWND_REPORT_TYPES,
    NNJASatwndLoader,
)
from healda.observations.system import FAMILY_REPORT_TYPES, ObsFamily
from healda.observations.sensors import (
    CONV_CHANNELS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)

# Report-type allowlists follow global_convinfo and the existing GSI comparison
# filters.  In particular, POB is a level coordinate on most rows and must only
# become a station-pressure observation on the pressure path.
# fmt: off
T_REPORT_TYPES = frozenset({
    118, 119, 120, 126, 130, 131, 132, 133, 134, 135, 136, 180, 181, 182, 183, 187,
    301, 302,
})
Q_REPORT_TYPES = frozenset({
    118, 119, 120, 130, 131, 132, 133, 134, 135, 136, 180, 181, 182, 183, 187,
    301, 302,
})
UV_REPORT_TYPES = frozenset({
    210, 216, 217, 218, 219, 220, 221, 223, 224, 228, 229, 230, 231, 232, 233, 234,
    235, 236, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254,
    256, 257, 258, 259, 260, 270, 280, 281, 282, 284, 285, 286, 287, 289, 290, 291,
    296, 401, 402,
})
# fmt: on
PRESSURE_REPORT_TYPES = frozenset({120, 180, 181, 187})
# NCEP PrepBUFR types marked R (restricted commercial aircraft). Public NOMADS
# GDAS prepbufr.nr omits them; the NNJA reanalysis dump includes them.
# 131/231 AMDAR, 133/233 MDCRS ACARS, 134/234 TAMDAR, 135/235 Canadian AMDAR.
RESTRICTED_AIRCRAFT_TYPES = frozenset({131, 133, 134, 135, 231, 233, 234, 235})
AIRCRAFT_WIND_REPORT_TYPES = frozenset({230, 231, 233, 234, 235})
# Every scatterometer, not just ASCAT: QuikSCAT/ERS/WindSat carry the 2000-2010 record
# and none of them reports a height, so a {290} set drops them at the height screen.
SCATTEROMETER_REPORT_TYPES = FAMILY_REPORT_TYPES[ObsFamily.SCATTEROMETER]
# Arms started before the pre-ASCAT fix saw only 290 in both the height fill and the wind
# dropout set, so ascat_only_scatterometer needs this in both places.
ASCAT_REPORT_TYPES = frozenset({290})


@dataclass(frozen=True)
class VariableSpec:
    name: str
    value_column: str
    quality_column: str
    local_channel_id: int
    platform: str
    report_types: frozenset[int]
    scale: float = 1.0
    offset: float = 0.0
    surface_pressure: bool = False


VARIABLES = (
    VariableSpec(
        "ps", "POB", "PQM", 3, "ps", PRESSURE_REPORT_TYPES, surface_pressure=True
    ),
    # PrepBUFR QOB is mg/kg; Healda's q channel is kg/kg.
    VariableSpec("q", "QOB", "QQM", 4, "q", Q_REPORT_TYPES, scale=1e-6),
    # PrepBUFR TOB is Celsius; Healda's t channel is Kelvin.
    VariableSpec("t", "TOB", "TQM", 5, "t", T_REPORT_TYPES, offset=273.15),
    VariableSpec("u", "UOB", "WQM", 6, "uv", UV_REPORT_TYPES),
    VariableSpec("v", "VOB", "WQM", 7, "uv", UV_REPORT_TYPES),
)

# GSI places 280-299 surface winds at ELV+10 m, except 282 at ELV+20
# (read_prepbufr.F90:2143-2167 @3c1f5fe). 282 and 284 are out because their POB is not a
# measurement: 282 is a nominal 1013 hPa constant, 284 is reduced from MSLP. 282 is also
# 17 buoys in one basin, 21 obs a cycle against the ~105,000 recovered here.
SURFACE_WIND_TYPES = (280, 281, 287)

CORE_COLUMNS = ("XOB", "YOB", "DHR", "POB", "TYP")
OPTIONAL_METADATA_COLUMNS = ("ZOB", "CAT", "ELV")
NULL_COLUMNS = ("Sat_Zenith_Angle", "Sol_Zenith_Angle", "Scan_Angle", "QC_Flag")


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    stddev: float
    min_valid: float
    max_valid: float


class NNJAConvLoader(NNJAArchiveLoader):
    """Read NNJA PrepBUFR, GPS-RO, and optional SATWND as normalized long obs."""

    def __init__(
        self,
        archive_root: str = nnja_archive("parquet", "cycles"),
        obs_context_hours: tuple[int, int] = (-3, 3),
        data_spacing: int = 3,
        max_quality_mark: int | None = 2,
        normalize: bool = True,
        include_gpsro: bool = True,
        surface_winds: bool = False,
        gpsro_saids: str = "legacy",
        gpsro_archive_root: str = nnja_archive("parquet", "gpsro_v3"),
        drop_restricted_aircraft: bool = False,
        include_satwnd: bool = False,
        satwnd_archive_root: str = SATWND_ARCHIVE,
        gpsro_table_source: CycleTableSource | None = None,
        satwnd_table_source: CycleTableSource | None = None,
        wind_obs_dropout: float = 0.0,
        surface_pressure_dropout: float = 0.0,
        dropout_scope: str = "row",
        ascat_only_scatterometer: bool = False,
    ) -> None:
        if data_spacing != 3:
            raise ValueError("NNJA conventional cycle halves require data_spacing=3")
        start, end = obs_context_hours
        if start % 3 or end % 3:
            raise ValueError(
                "obs_context_hours endpoints must be 3-hour aligned, "
                f"got {obs_context_hours}"
            )
        if not 0.0 <= wind_obs_dropout <= 1.0:
            raise ValueError(
                f"wind_obs_dropout must be in [0, 1], got {wind_obs_dropout}"
            )
        if gpsro_saids not in ("legacy", "full"):
            raise ValueError(
                f'gpsro_saids must be "legacy" or "full", got {gpsro_saids!r}'
            )
        self.archive_root = archive_root
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        self.max_quality_mark = max_quality_mark
        self.normalize = normalize
        self.include_gpsro = include_gpsro
        self.surface_winds = surface_winds
        self.drop_restricted_aircraft = drop_restricted_aircraft
        self.include_satwnd = include_satwnd
        self.wind_obs_dropout = wind_obs_dropout
        self.surface_pressure_dropout = surface_pressure_dropout
        self.dropout_scope = dropout_scope
        self.scatterometer_types = (
            ASCAT_REPORT_TYPES
            if ascat_only_scatterometer
            else SCATTEROMETER_REPORT_TYPES
        )
        # Each family draws its own verdict under sample scope. No AMV entry: PrepBUFR
        # AMVs are filtered out unconditionally, so AMV dropout is the SATWND loader's.
        self.dropout_families = {
            ObsFamily.AIRCRAFT: AIRCRAFT_WIND_REPORT_TYPES,
            ObsFamily.SCATTEROMETER: self.scatterometer_types,
        }
        self.gpsro = (
            NNJAGpsroLoader(
                said_allowlist=None if gpsro_saids == "full" else GPS_LEGACY_SAIDS,
                archive_root=gpsro_archive_root,
                obs_context_hours=obs_context_hours,
                data_spacing=data_spacing,
                normalize=normalize,
                table_source=gpsro_table_source,
            )
            if include_gpsro
            else None
        )
        self.satwnd = (
            NNJASatwndLoader(
                archive_root=satwnd_archive_root,
                obs_context_hours=obs_context_hours,
                data_spacing=data_spacing,
                normalize=normalize,
                wind_obs_dropout=wind_obs_dropout,
                dropout_scope=dropout_scope,
                table_source=satwnd_table_source,
            )
            if include_satwnd
            else None
        )
        configure_arrow_pools()

    @functools.cached_property
    def channel_stats(self) -> dict[int, ChannelStats]:
        """Packaged conv vocabulary: bounds from CONV_CHANNELS, moments from the CSV.

        The QC bounds gate which observations survive, so they are part of the
        recipe rather than of whichever archive happens to be mounted.
        """
        conv = SENSOR_CONFIGS["conv"]
        return {
            spec.local_channel_id: ChannelStats(
                mean=float(conv.means[spec.local_channel_id]),
                stddev=float(conv.stds[spec.local_channel_id]),
                min_valid=float(CONV_CHANNELS[spec.local_channel_id].min_valid),
                max_valid=float(CONV_CHANNELS[spec.local_channel_id].max_valid),
            )
            for spec in VARIABLES
        }

    def _path(self, cycle: pd.Timestamp) -> str:
        return os.path.join(
            self.archive_root,
            cycle.strftime("%Y"),
            f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.prepbufr.nr.parquet",
        )

    def _to_long(
        self,
        table: pa.Table,
        cycle: pd.Timestamp,
        positive_half: bool,
        rng: np.random.Generator | None = None,
        wind_dropout: dict[ObsFamily, float] | None = None,
        surface_pressure_dropout: float = 0.0,
    ) -> pa.Table:
        if not table.num_rows:
            return self._empty()

        lon = self._numpy(table, "XOB", np.float32)
        lat = self._numpy(table, "YOB", np.float32)
        dhr = self._numpy(table, "DHR", np.float64)
        pressure = self._numpy(table, "POB", np.float32)
        report_type_float = self._numpy(table, "TYP", np.float64)
        report_type = np.where(
            np.isnan(report_type_float), -1, report_type_float
        ).astype(np.int64)

        half = dhr > 0 if positive_half else dhr <= 0
        base_valid = (
            half
            & np.isfinite(dhr)
            & (np.abs(dhr) <= 3)
            & np.isfinite(lat)
            & (lat >= -90)
            & (lat <= 90)
            & np.isfinite(lon)
            & np.isfinite(pressure)
            & (pressure > 0)
        )
        if self.drop_restricted_aircraft:
            base_valid = base_valid & ~np.isin(
                report_type, tuple(RESTRICTED_AIRCRAFT_TYPES)
            )
        if not base_valid.any():
            return self._empty()

        cycle_ns = np.datetime64(cycle.to_datetime64(), "ns").astype(np.int64)
        time_ns = cycle_ns + np.rint(dhr * 3_600_000_000_000).astype(np.int64)
        window = cycle + pd.Timedelta(hours=3 if positive_half else 0)
        height = (
            self._numpy(table, "ZOB", np.float32).copy()
            if "ZOB" in table.column_names
            else np.full(table.num_rows, np.nan, dtype=np.float32)
        )
        # Scatterometer winds are a 10 m reference wind and report no ZOB
        missing_height = ~np.isfinite(height) & np.isin(
            report_type, tuple(self.scatterometer_types)
        )
        height[missing_height] = 10.0
        if self.surface_winds:
            # No ZOB on any row, so the height clause above drops every one of them.
            elv = (
                self._numpy(table, "ELV", np.float32)
                if "ELV" in table.column_names
                else np.full(table.num_rows, np.nan, dtype=np.float32)
            )
            fill = (
                ~np.isfinite(height)
                & np.isin(report_type, tuple(SURFACE_WIND_TYPES))
                & np.isfinite(elv)
            )
            height[fill] = elv[fill] + 10.0
        category = (
            self._numpy(table, "CAT", np.float64)
            if "CAT" in table.column_names
            else None
        )
        wind_dropout_keep = np.ones(table.num_rows, dtype=bool)
        # Per family, not over one merged set: each carries its own rate, and under
        # sample scope its own verdict. The families are disjoint, so one draw per row.
        for family, types in self.dropout_families.items():
            rate = (wind_dropout or {}).get(family, 0.0)
            if not rate:
                continue
            targeted = np.isin(report_type, tuple(types))
            # Serves both scopes: `rate` is the configured p under "row", and 0 or 1
            # under "sample", where this drops all of the family or none of it.
            if targeted.any():
                rng = rng or np.random.default_rng()
                wind_dropout_keep[targeted] = rng.random(int(targeted.sum())) >= rate

        parts = []
        for spec in VARIABLES:
            missing = [
                name
                for name in (spec.value_column, spec.quality_column)
                if name not in table.column_names
            ]
            if missing:
                raise ValueError(
                    f"PrepBUFR table missing required columns for {spec.name!r}: "
                    f"{missing}"
                )
            raw = self._numpy(table, spec.value_column, np.float32)
            quality = self._numpy(table, spec.quality_column, np.float32)
            values = raw * np.float32(spec.scale) + np.float32(spec.offset)
            stats = self.channel_stats[spec.local_channel_id]

            keep = (
                base_valid
                & np.isin(report_type, tuple(spec.report_types))
                & np.isfinite(values)
                & np.isfinite(quality)
                & (values >= stats.min_valid)
                & (values <= stats.max_valid)
            )
            if self.max_quality_mark is not None:
                keep &= quality <= self.max_quality_mark
            if spec.local_channel_id in (6, 7):
                keep &= wind_dropout_keep
            if spec.surface_pressure:
                # Type 120 is a profile report. Only CAT=0 is its surface pressure;
                # without CAT, exclude it rather than turning every level into ps.
                type_120 = report_type == 120
                at_surface = np.zeros(table.num_rows, dtype=bool)
                if category is not None:
                    at_surface = np.isfinite(category) & (category == 0)
                keep &= ~type_120 | at_surface
                if surface_pressure_dropout:
                    # Serves both scopes: the configured p under "row", 0 or 1 under
                    # "sample". Only ps goes; the row's other variables survive.
                    rng = rng or np.random.default_rng()
                    keep &= rng.random(table.num_rows) >= surface_pressure_dropout

            rows = np.flatnonzero(keep)
            if not rows.size:
                continue
            observation = values[rows]
            if self.normalize:
                if not np.isfinite(stats.stddev) or stats.stddev <= 0:
                    raise ValueError(
                        f"invalid stddev for conv channel {spec.name!r}: {stats.stddev}"
                    )
                observation = (observation - stats.mean) / stats.stddev

            count = rows.size
            arrays = {
                "Latitude": pa.array(lat[rows], type=pa.float32()),
                # PrepBUFR XOB is 0..360; match UFS / nnja_wide_loader convention.
                "Longitude": pa.array(np.mod(lon[rows], 360.0), type=pa.float32()),
                "Absolute_Obs_Time": pa.array(time_ns[rows].astype("datetime64[ns]")),
                "DA_window": pa.array(
                    np.full(count, window.to_datetime64(), dtype="datetime64[ns]")
                ),
                "Platform_ID": pa.array(
                    np.full(count, PLATFORM_NAME_TO_ID[spec.platform], dtype=np.uint16)
                ),
                "Observation": pa.array(observation, type=pa.float32()),
                GLOBAL_CHANNEL_ID.name: pa.array(
                    np.full(
                        count,
                        SENSOR_OFFSET["conv"] + spec.local_channel_id,
                        dtype=np.uint16,
                    ),
                    type=GLOBAL_CHANNEL_ID.type,
                ),
                "Pressure": pa.array(pressure[rows], type=pa.float32()),
                "Height": pa.array(height[rows], type=pa.float32(), from_pandas=True),
                "Observation_Type": pa.array(report_type[rows], type=pa.uint16()),
                "Analysis_Use_Flag": pa.array(
                    (quality[rows] <= 2).astype(np.int8), type=pa.int8()
                ),
                LOCAL_CHANNEL_ID.name: pa.array(
                    np.full(count, spec.local_channel_id, dtype=np.uint16)
                ),
                SENSOR_ID.name: pa.array(
                    np.full(count, SENSOR_NAME_TO_ID["conv"], dtype=np.uint16)
                ),
            }
            for name in NULL_COLUMNS:
                arrays[name] = pa.nulls(count, type=self.output_schema.field(name).type)
            parts.append(
                pa.table(
                    [arrays[field.name] for field in self.output_schema],
                    schema=self.output_schema,
                )
            )
        return pa.concat_tables(parts) if parts else self._empty()

    def _load_cycle(
        self,
        cycle: pd.Timestamp,
        halves: Sequence[bool],
        dropout_seed: int | None = None,
        wind_dropout: dict[ObsFamily, float] | None = None,
        surface_pressure_dropout: float = 0.0,
    ) -> list[tuple[pd.Timestamp, pa.Table]]:
        path = self._path(cycle)
        if not os.path.exists(path):
            return []
        parquet = pq.ParquetFile(path, pre_buffer=True)
        available = set(parquet.schema_arrow.names)
        missing = set(CORE_COLUMNS) - available
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        columns = list(CORE_COLUMNS)
        columns += [name for name in OPTIONAL_METADATA_COLUMNS if name in available]
        for spec in VARIABLES:
            columns += [
                name
                for name in (spec.value_column, spec.quality_column)
                if name in available
            ]
        columns = list(dict.fromkeys(columns))
        rng = row_generator(dropout_seed)

        by_half: dict[bool, list[pa.Table]] = {half: [] for half in halves}
        dhr_index = parquet.schema_arrow.get_field_index("DHR")
        for group in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(group).column(dhr_index).statistics
            candidates = []
            if False in by_half and (stats is None or stats.min <= 0):
                candidates.append(False)
            if True in by_half and (stats is None or stats.max > 0):
                candidates.append(True)
            if not candidates:
                continue
            source = parquet.read_row_group(group, columns=columns)
            for half in candidates:
                long = self._to_long(
                    source, cycle, half, rng, wind_dropout, surface_pressure_dropout
                )
                if long.num_rows:
                    by_half[half].append(long)

        result = []
        for half in halves:
            parts = by_half[half]
            if parts:
                window = cycle + pd.Timedelta(hours=3 if half else 0)
                result.append((window, pa.concat_tables(parts)))
        return result

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        """Return one unified observation table per requested target time."""
        needed: dict[pd.Timestamp, set[bool]] = {}
        for target in times:
            for window in self._interval_times(target):
                cycle, half = self._cycle_and_half(window)
                needed.setdefault(cycle, set()).add(half)

        loop = asyncio.get_running_loop()
        dropout = self._dropout()
        # Per family: same probability, independent verdicts. ps is a single target.
        wind_dropout = {
            family: dropout.rate(self.wind_obs_dropout)
            for family in self.dropout_families
        }
        surface_pressure_dropout = dropout.rate(self.surface_pressure_dropout)
        dropout_seeds = dropout.seeds(needed)
        prepbufr_future = asyncio.gather(
            *(
                loop.run_in_executor(
                    thread_pool(),
                    self._load_cycle,
                    cycle,
                    tuple(sorted(halves)),
                    dropout_seeds[cycle],
                    wind_dropout,
                    surface_pressure_dropout,
                )
                for cycle, halves in sorted(needed.items())
            )
        )
        empty_result = {"obs_v2": [self._empty() for _ in times]}
        gpsro_future = (
            self.gpsro.sel_time(times)
            if self.gpsro is not None
            else asyncio.sleep(0, result=empty_result)
        )
        satwnd_future = (
            self.satwnd.sel_time(times)
            if self.satwnd is not None
            else asyncio.sleep(0, result=empty_result)
        )
        loaded, gpsro, satwnd = await asyncio.gather(
            prepbufr_future, gpsro_future, satwnd_future
        )
        by_window = {
            window: table for cycle_parts in loaded for window, table in cycle_parts
        }
        empty = self._empty()

        def combine(
            target: datetime,
            gpsro_table: pa.Table,
            satwnd_table: pa.Table,
        ) -> pa.Table:
            parts = []
            for window in self._interval_times(target):
                if window not in by_window:
                    continue
                prepbufr = by_window[window]
                # Unconditional: AMVs come from the dedicated archive or not at all.
                report_type = np.asarray(prepbufr["Observation_Type"])
                prepbufr = prepbufr.filter(
                    pa.array(~np.isin(report_type, tuple(SATWND_REPORT_TYPES)))
                )
                if prepbufr.num_rows:
                    parts.append(prepbufr)
            if gpsro_table.num_rows:
                parts.append(gpsro_table)
            if satwnd_table.num_rows:
                parts.append(satwnd_table)
            return pa.concat_tables(parts) if parts else empty

        return {
            "obs_v2": [
                combine(target, gpsro_table, satwnd_table)
                for target, gpsro_table, satwnd_table in zip(
                    times, gpsro["obs_v2"], satwnd["obs_v2"]
                )
            ]
        }

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NNJA GPS-RO Parquet loader for the unified long observation schema.

The GPS-RO archive under ``parquet/gpsro_v3`` stores one row per (occultation,
level) with L1b bending angle plus source-only pressure/height columns derived
from BUFR refractivity: ``dry_pressure_hpa``, ``standard_atmosphere_pressure_hpa``
(now keyed off refraction-corrected ``heit_at_impact_m``, not the ray-geometric
``impact_height_m``), and ``blended_pressure_5km_hpa`` (0.8*standard-atmosphere +
0.2*dry below 5km, pure dry above). Healda's ``gps_angle`` channel (conv local
id 0) is the matching product. The archive does not retain 1D-Var ``gps_t`` /
``gps_q``.

``pressure_source`` selects which column feeds ``Pressure``:
  - "v3" (default): blended_pressure_5km_hpa, falling back to dry_pressure_hpa,
    then standard_atmosphere_pressure_hpa, then on-the-fly ISA at impact height.
  - "v2": dry_pressure_hpa, falling back to standard_atmosphere_pressure_hpa,
    then on-the-fly ISA at impact height. Same fallback chain as the gpsro_v2
    archive layout, without ``blended_pressure_5km_hpa``; gpsro_v3 also derives
    ``standard_atmosphere_pressure_hpa`` from refraction-corrected impact height.
"""

from __future__ import annotations

import asyncio
import functools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

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
)

from healda.observations.loaders.threads import configure_arrow_pools, thread_pool
from healda.observations.sensors import (
    CONV_CHANNELS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)

GPS_ANGLE_LOCAL_CHANNEL = 0

ARCHIVE_ROOT = nnja_archive("parquet", "gpsro_v3")

PressureSource = Literal["v3", "v2"]

# A subset: the receivers flying in 2024, so it drops every mission retired before then.
GPS_LEGACY_SAIDS = frozenset(
    {3, 5, 42, 43, 44, 267, 269, 750, 751, 752, 753, 754, 755, 803, 804, 825}
)

CORE_COLUMNS = (
    "bending_angle",
    "qfro",
    "obs_time",
    "impact_parameter",
    "earth_radius_curvature",
    "satellite_id",
)
OPTIONAL_COLUMNS = (
    "tangent_latitude",
    "tangent_longitude",
    "occ_latitude",
    "occ_longitude",
    "impact_height_m",
    "heit_at_impact_m",
    "dry_pressure_hpa",
    "standard_atmosphere_pressure_hpa",
    "blended_pressure_5km_hpa",
)
NULL_COLUMNS = ("Sat_Zenith_Angle", "Sol_Zenith_Angle", "Scan_Angle", "QC_Flag")


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    stddev: float
    min_valid: float
    max_valid: float


def qfro_bit_set(qfro: np.ndarray, bit: int) -> np.ndarray:
    """True where WMO QFRO bit *bit* is set (bit 1 = MSB of a 16-bit word)."""
    shift = 16 - int(bit)
    values = np.asarray(qfro, dtype=np.float64)
    values = np.where(np.isnan(values), 0.0, values).astype(np.int64)
    return ((values >> shift) & 1) == 1


def height_to_pressure_hpa(height_m: np.ndarray) -> np.ndarray:
    """Approximate pressure (hPa) from geometric height (US Standard Atmosphere).

    GPS-RO levels carry impact height, not pressure. Healda's conv filters and
    ``conv-plevel`` binning both expect a finite Pressure column, so this fills
    that coordinate without claiming a retrieved thermodynamic profile.

    Layers follow the 1976 US Standard Atmosphere through 71 km so heights kept
    by the loader (up to 60 km) retain distinct pressures instead of collapsing
    at a 20 km troposphere ceiling.
    """
    h = np.clip(np.asarray(height_m, dtype=np.float64), 0.0, 60_000.0)
    p = np.empty(h.shape, dtype=np.float64)

    trop = h <= 11_000.0
    p[trop] = 1013.25 * np.power(np.maximum(1.0 - 2.25577e-5 * h[trop], 1e-6), 5.25588)

    strat1 = (h > 11_000.0) & (h <= 20_000.0)
    p[strat1] = 226.321 * np.exp(-1.57686e-4 * (h[strat1] - 11_000.0))

    strat2 = (h > 20_000.0) & (h <= 32_000.0)
    t2 = 216.65 + 0.001 * (h[strat2] - 20_000.0)
    p[strat2] = 54.7489 * np.power(t2 / 216.65, -34.1632)

    strat3 = (h > 32_000.0) & (h <= 47_000.0)
    t3 = 228.65 + 0.0028 * (h[strat3] - 32_000.0)
    p[strat3] = 8.68019 * np.power(t3 / 228.65, -12.2011)

    strat4 = (h > 47_000.0) & (h <= 51_000.0)
    p[strat4] = 1.10906 * np.exp(-1.26227e-4 * (h[strat4] - 47_000.0))

    meso = h > 51_000.0
    t5 = 270.65 - 0.0028 * (h[meso] - 51_000.0)
    p[meso] = 0.669387 * np.power(np.maximum(t5 / 270.65, 1e-6), 12.2011)

    return p.astype(np.float32)


class NNJAGpsroLoader(NNJAArchiveLoader):
    """Read cycle-based NNJA GPS-RO Parquet as ``gps_angle`` long observations."""

    def __init__(
        self,
        archive_root: str = ARCHIVE_ROOT,
        obs_context_hours: tuple[int, int] = (-3, 3),
        data_spacing: int = 3,
        normalize: bool = True,
        said_allowlist: frozenset[int] | None = GPS_LEGACY_SAIDS,
        pressure_source: PressureSource = "v3",
        table_source: CycleTableSource | None = None,
    ) -> None:
        if data_spacing != 3:
            raise ValueError("NNJA GPS-RO cycle halves require data_spacing=3")
        if pressure_source not in ("v3", "v2"):
            raise ValueError(f"unknown pressure_source: {pressure_source!r}")
        self.archive_root = archive_root
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        self.normalize = normalize
        self.said_allowlist = said_allowlist
        self.pressure_source = pressure_source
        # Read instead of archive_root when given.
        self.table_source = table_source
        configure_arrow_pools()

    @functools.cached_property
    def channel_stats(self) -> ChannelStats:
        """Packaged gps_angle stats; see NNJAConvLoader.channel_stats."""
        conv = SENSOR_CONFIGS["conv"]
        return ChannelStats(
            mean=float(conv.means[GPS_ANGLE_LOCAL_CHANNEL]),
            stddev=float(conv.stds[GPS_ANGLE_LOCAL_CHANNEL]),
            min_valid=float(CONV_CHANNELS[GPS_ANGLE_LOCAL_CHANNEL].min_valid),
            max_valid=float(CONV_CHANNELS[GPS_ANGLE_LOCAL_CHANNEL].max_valid),
        )

    def _path(self, cycle: pd.Timestamp) -> str:
        return os.path.join(
            self.archive_root,
            cycle.strftime("%Y"),
            f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.gpsro.tm00.bufr_d.parquet",
        )

    def _select_pressure(
        self, table: pa.Table, rows: np.ndarray, height: np.ndarray
    ) -> np.ndarray:
        """Pick the Pressure column per ``self.pressure_source``.

        Every candidate falls back to on-the-fly ISA at impact height (which is
        always finite for finite height), so this never introduces new NaNs.
        """

        def col(name: str) -> np.ndarray:
            if name not in table.column_names:
                return np.full(rows.size, np.nan, dtype=np.float64)
            return self._numpy(table, name, np.float64)[rows]

        height_isa = height_to_pressure_hpa(height[rows]).astype(np.float64)

        def with_fallback(*candidates: np.ndarray) -> np.ndarray:
            out = height_isa
            for candidate in reversed(candidates):
                out = np.where(
                    np.isfinite(candidate) & (candidate > 0.0), candidate, out
                )
            return out

        if self.pressure_source == "v3":
            pressure = with_fallback(
                col("blended_pressure_5km_hpa"),
                col("dry_pressure_hpa"),
                col("standard_atmosphere_pressure_hpa"),
            )
        else:
            pressure = with_fallback(
                col("dry_pressure_hpa"),
                col("standard_atmosphere_pressure_hpa"),
            )
        return pressure.astype(np.float32)

    def _to_long(
        self,
        table: pa.Table,
        cycle: pd.Timestamp,
        positive_half: bool,
    ) -> pa.Table:
        if not table.num_rows:
            return self._empty()

        obs_time = self._numpy(table, "obs_time")
        # Parquet stores UTC timestamps; drop tz so comparisons match the cycle.
        if getattr(obs_time.dtype, "tz", None) is not None:
            obs_time = obs_time.astype("datetime64[ns]")
        cycle_ns = np.datetime64(cycle.to_datetime64(), "ns")
        half = obs_time > cycle_ns if positive_half else obs_time <= cycle_ns

        def latlon(primary: str, fallback: str) -> np.ndarray:
            if primary in table.column_names:
                values = self._numpy(table, primary, np.float32)
                if fallback in table.column_names:
                    other = self._numpy(table, fallback, np.float32)
                    return np.where(np.isfinite(values), values, other).astype(
                        np.float32
                    )
                return values
            return self._numpy(table, fallback, np.float32)

        lat = latlon("tangent_latitude", "occ_latitude")
        lon = latlon("tangent_longitude", "occ_longitude")
        bending = self._numpy(table, "bending_angle", np.float32)
        qfro = self._numpy(table, "qfro", np.float64)
        impact = self._numpy(table, "impact_parameter", np.float64)
        curvature = self._numpy(table, "earth_radius_curvature", np.float64)
        said_f = self._numpy(table, "satellite_id", np.float64)
        said = np.where(np.isnan(said_f), -1, said_f).astype(np.int64)
        height = (impact - curvature).astype(np.float32)
        # gpsro_v3 carries the refraction-corrected BUFR height used to derive
        # its standard-atmosphere and blended pressure columns. Prefer it for
        # the model-facing Height feature, with the archived/computed impact
        # height as a per-row fallback.
        for name in ("impact_height_m", "heit_at_impact_m"):
            if name in table.column_names:
                candidate = self._numpy(table, name, np.float32)
                usable = np.isfinite(candidate) & (candidate > 0.0)
                height = np.where(usable, candidate, height).astype(np.float32)
        stats = self.channel_stats

        window_end = cycle + pd.Timedelta(hours=3 if positive_half else 0)
        window_start = window_end - pd.Timedelta(hours=self.data_spacing)
        in_window = (obs_time > np.datetime64(window_start.to_datetime64(), "ns")) & (
            obs_time <= np.datetime64(window_end.to_datetime64(), "ns")
        )

        keep = (
            half
            & in_window
            & np.isfinite(obs_time.astype("datetime64[ns]").astype(np.int64))
            & np.isfinite(lat)
            & (lat >= -90)
            & (lat <= 90)
            & np.isfinite(lon)
            & np.isfinite(bending)
            & (bending > 0)
            & (bending >= stats.min_valid)
            & (bending <= stats.max_valid)
            & np.isfinite(height)
            & (height > 0)
            & (height <= 60_000)
            & ~qfro_bit_set(qfro, 5)
        )
        if self.said_allowlist is not None:
            keep &= np.isin(said, tuple(self.said_allowlist))

        rows = np.flatnonzero(keep)
        if not rows.size:
            return self._empty()

        observation = bending[rows]
        if self.normalize:
            if not np.isfinite(stats.stddev) or stats.stddev <= 0:
                raise ValueError(f"invalid stddev for gps_angle: {stats.stddev}")
            observation = (observation - stats.mean) / stats.stddev

        pressure = self._select_pressure(table, rows, height)
        # Drop levels that still lack a usable pressure coordinate (should be rare:
        # the on-the-fly ISA fallback below is always finite for finite height).
        pressure_ok = np.isfinite(pressure) & (pressure > 0.0)
        if not pressure_ok.all():
            rows = rows[pressure_ok]
            observation = observation[pressure_ok]
            pressure = pressure[pressure_ok]
            if not rows.size:
                return self._empty()
        count = rows.size
        arrays = {
            "Latitude": pa.array(lat[rows], type=pa.float32()),
            "Longitude": pa.array(np.mod(lon[rows], 360.0), type=pa.float32()),
            "Absolute_Obs_Time": pa.array(obs_time[rows].astype("datetime64[ns]")),
            "DA_window": pa.array(
                np.full(count, window_end.to_datetime64(), dtype="datetime64[ns]")
            ),
            "Platform_ID": pa.array(
                np.full(count, PLATFORM_NAME_TO_ID["gps"], dtype=np.uint16)
            ),
            "Observation": pa.array(observation, type=pa.float32()),
            GLOBAL_CHANNEL_ID.name: pa.array(
                np.full(
                    count,
                    SENSOR_OFFSET["conv"] + GPS_ANGLE_LOCAL_CHANNEL,
                    dtype=np.uint16,
                ),
                type=GLOBAL_CHANNEL_ID.type,
            ),
            "Pressure": pa.array(pressure, type=pa.float32()),
            "Height": pa.array(height[rows], type=pa.float32()),
            "Observation_Type": pa.array(said[rows], type=pa.uint16()),
            "Analysis_Use_Flag": pa.array(
                np.ones(count, dtype=np.int8), type=pa.int8()
            ),
            LOCAL_CHANNEL_ID.name: pa.array(
                np.full(count, GPS_ANGLE_LOCAL_CHANNEL, dtype=np.uint16)
            ),
            SENSOR_ID.name: pa.array(
                np.full(count, SENSOR_NAME_TO_ID["conv"], dtype=np.uint16)
            ),
        }
        for name in NULL_COLUMNS:
            arrays[name] = pa.nulls(count, type=self.output_schema.field(name).type)
        return pa.table(
            [arrays[field.name] for field in self.output_schema],
            schema=self.output_schema,
        )

    def _load_cycle(
        self,
        cycle: pd.Timestamp,
        halves: Sequence[bool],
    ) -> list[tuple[pd.Timestamp, pa.Table]]:
        by_half: dict[bool, list[pa.Table]] = {half: [] for half in halves}
        for source, candidates in self._cycle_sources(cycle, list(by_half)):
            for half in candidates:
                long = self._to_long(source, cycle, half)
                if long.num_rows:
                    by_half[half].append(long)

        result = []
        for half in halves:
            parts = by_half[half]
            if parts:
                window = cycle + pd.Timedelta(hours=3 if half else 0)
                result.append((window, pa.concat_tables(parts)))
        return result

    @staticmethod
    def _check_columns(available: set[str], origin: str) -> None:
        missing = set(CORE_COLUMNS) - available
        if missing:
            raise ValueError(f"{origin}: missing required columns {sorted(missing)}")
        if "occ_latitude" not in available and "tangent_latitude" not in available:
            raise ValueError(f"{origin}: need occ_latitude or tangent_latitude")
        if "occ_longitude" not in available and "tangent_longitude" not in available:
            raise ValueError(f"{origin}: need occ_longitude or tangent_longitude")

    def _cycle_sources(self, cycle: pd.Timestamp, halves: list[bool]):
        """Yield ``(table, halves it may serve)`` for one cycle."""
        if self.table_source is not None:
            table = self.table_source.get(cycle)
            if table is not None and table.num_rows:
                self._check_columns(set(table.column_names), f"table_source[{cycle}]")
                yield table, halves
            return
        path = self._path(cycle)
        if not os.path.exists(path):
            return
        parquet = pq.ParquetFile(path, pre_buffer=True)
        available = set(parquet.schema_arrow.names)
        self._check_columns(available, path)

        columns = list(CORE_COLUMNS)
        columns += [name for name in OPTIONAL_COLUMNS if name in available]
        columns = list(dict.fromkeys(columns))

        time_index = parquet.schema_arrow.get_field_index("obs_time")
        cycle_py = cycle.to_pydatetime()
        for group in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(group).column(time_index).statistics
            candidates = []
            if stats is None:
                candidates = list(halves)
            else:
                lo = stats.min.replace(tzinfo=None) if stats.min.tzinfo else stats.min
                hi = stats.max.replace(tzinfo=None) if stats.max.tzinfo else stats.max
                if False in halves and lo <= cycle_py:
                    candidates.append(False)
                if True in halves and hi > cycle_py:
                    candidates.append(True)
            if not candidates:
                continue
            yield parquet.read_row_group(group, columns=columns), candidates

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        """Return one unified observation table per requested target time."""
        needed: dict[pd.Timestamp, set[bool]] = {}
        for target in times:
            for window in self._interval_times(target):
                cycle, half = self._cycle_and_half(window)
                needed.setdefault(cycle, set()).add(half)

        loop = asyncio.get_running_loop()
        loaded = await asyncio.gather(
            *(
                loop.run_in_executor(
                    thread_pool(), self._load_cycle, cycle, tuple(sorted(halves))
                )
                for cycle, halves in sorted(needed.items())
            )
        )
        by_window = {
            window: table for cycle_parts in loaded for window, table in cycle_parts
        }
        empty = self._empty()

        def combine(target: datetime) -> pa.Table:
            parts = [
                by_window[window]
                for window in self._interval_times(target)
                if window in by_window
            ]
            return pa.concat_tables(parts) if parts else empty

        return {"obs_v2": [combine(target) for target in times]}

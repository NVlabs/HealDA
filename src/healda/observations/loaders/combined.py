# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NNJA satellite observations paired with conventional ones.

By default conventional rows still come from the UFS replay through
``UFSUnifiedLoader``. With ``ObsConfig.use_nnja_conv``, they come from the NNJA
PrepBUFR + GPS-RO parquet archives via ``NNJAConvLoader``, then are expanded to
the same ``conv-plevel`` vocabulary the combined training path expects.

Each loader emits ``local_channel_id``, a sensor's own 0-based channel index.
A global id is therefore always ``SENSOR_OFFSET[sensor] + local_channel_id``, and
rebasing conv into this vocabulary is that one expression.
"""

from __future__ import annotations

import asyncio
import functools
import pathlib
from typing import Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.config.environment import nnja_archive
from healda.observations import sensors, sensors_nnja
from healda.observations.preprocessing.conv_plevel import (
    build_conv_plevel_channel_table,
    expanded_local_channel,
)
from healda.observations.loaders.nnja_base import CycleTableSource
from healda.observations.loaders.nnja_conventional import NNJAConvLoader
from healda.observations.loaders.nnja_satwnd import SATWND_ARCHIVE
from healda.observations.loaders.nnja_wide import NNJAWideLoader
from healda.observations.preprocessing.filtering import filter_observations
from healda.observations.loaders.ufs import LOCAL_CHANNEL_ID
from healda.observations.schema import GLOBAL_CHANNEL_ID, SENSOR_ID

CONV_SENSOR = "conv-plevel"

SENSOR_ORDER = (*sensors_nnja.SENSOR_ORDER, CONV_SENSOR)
SENSOR_NAME_TO_ID = {name: index for index, name in enumerate(SENSOR_ORDER)}
SENSOR_OFFSET = {**sensors_nnja.SENSOR_OFFSET, CONV_SENSOR: sensors_nnja.NCHANNEL}

CONV_CHANNELS = sensors.SENSOR_CONFIGS[CONV_SENSOR].channels
NCHANNEL = sensors_nnja.NCHANNEL + CONV_CHANNELS

NORMALIZATION_DIR = sensors.NORMALIZATION_DIR
NNJA_CONV_NORMALIZATION_DIR = sensors_nnja.NORMALIZATION_DIR


def _conv_norm_paths(*, nnja_conv: bool) -> tuple[pathlib.Path, pathlib.Path]:
    """Base + by-level CSV paths for UFS or NNJA-computed conventional stats."""
    if nnja_conv:
        base = NNJA_CONV_NORMALIZATION_DIR / "conv_normalizations.csv"
        by_level = NNJA_CONV_NORMALIZATION_DIR / "conv_normalizations_by_level.csv"
        missing = [str(path) for path in (base, by_level) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "NNJA conventional normalization CSVs are required when "
                f"nnja_conv=True; missing: {', '.join(missing)}"
            )
        return base, by_level
    base = NORMALIZATION_DIR / "conv_normalizations.csv"
    by_level = NORMALIZATION_DIR / "conv_normalizations_by_level.csv"
    return base, by_level


@functools.lru_cache(maxsize=2)
def _conv_plevel_table(*, nnja_conv: bool = False) -> pa.Table:
    """The 92 conv-plevel rows, with per-level stats, rebased into this vocabulary."""
    base_path, level_path = _conv_norm_paths(nnja_conv=nnja_conv)
    base = pd.read_csv(base_path)
    base_stats = pa.table(
        {
            # Raw_Channel_ID is 1-based over the 8-channel conv vocabulary.
            "Global_Channel_ID": (
                sensors.SENSOR_OFFSET["conv"] + base["Raw_Channel_ID"] - 1
            ).to_numpy(),
            "mean": base["obs_mean"].to_numpy(),
            "stddev": base["obs_std"].to_numpy(),
        }
    )
    level_stats = pa.Table.from_pandas(
        pd.read_csv(level_path),
        preserve_index=False,
    )
    table = build_conv_plevel_channel_table(
        level_stats=level_stats,
        base_channel_table=base_stats,
        include_local_channel_id=True,
    )
    local = table[LOCAL_CHANNEL_ID.name].to_numpy(zero_copy_only=False)
    return _rebase(table, local).drop([LOCAL_CHANNEL_ID.name])


def _rebase(table: pa.Table, local_channel_id: np.ndarray) -> pa.Table:
    """Restate conv's channel and sensor identity in this vocabulary."""
    global_id = (SENSOR_OFFSET[CONV_SENSOR] + local_channel_id).astype(np.uint16)
    sensor = np.full(table.num_rows, SENSOR_NAME_TO_ID[CONV_SENSOR], dtype=np.uint16)
    table = table.set_column(
        table.schema.get_field_index(GLOBAL_CHANNEL_ID.name),
        GLOBAL_CHANNEL_ID,
        pa.array(global_id, type=GLOBAL_CHANNEL_ID.type),
    )
    return table.set_column(
        table.schema.get_field_index(SENSOR_ID.name),
        SENSOR_ID,
        pa.array(sensor, type=SENSOR_ID.type),
    )


def channel_table(*, nnja_conv: bool = False) -> pa.Table:
    return pa.concat_tables(
        [sensors_nnja.channel_table(), _conv_plevel_table(nnja_conv=nnja_conv)]
    )


class ConventionalLoader(Protocol):
    """Anything CombinedObsLoader can rebase into the NNJA vocabulary."""

    sensors: list[str]

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]: ...


def channel_count(sensor: str) -> int:
    if sensor == CONV_SENSOR:
        return CONV_CHANNELS
    return len(sensors_nnja.SENSOR_CONFIGS[sensor].channels)


def platform_ids(sensor: str) -> tuple[int, ...]:
    if sensor == CONV_SENSOR:
        return tuple(
            sensors.PLATFORM_NAME_TO_ID[name]
            for name in sensors.SENSOR_CONFIGS[CONV_SENSOR].platforms
        )
    return sensors_nnja.SENSOR_CONFIGS[sensor].platform_ids


def _set_typed_column(table: pa.Table, field: pa.Field, values) -> pa.Table:
    index = table.schema.get_field_index(field.name)
    return table.set_column(
        index,
        table.schema.field(index).with_type(field.type),
        pa.array(values, type=field.type),
    )


class NNJAConventionalLoader:
    """NNJA PrepBUFR + GPS-RO as GSI ``conv-plevel`` rows for CombinedObsLoader.

    Reads physical values from ``NNJAConvLoader``, applies the shared conventional
    filters, expands to pressure-level channels, and normalizes with NNJA
    by-level stats (``normalizations/nnja/conv_normalizations*.csv``). Emitted
    identity columns use the GSI ``conv-plevel`` vocabulary so
    ``CombinedObsLoader`` can rebase them.
    """

    def __init__(
        self,
        obs_context_hours: tuple[int, int] = (-3, 3),
        archive_root: str = nnja_archive("parquet", "cycles"),
        gpsro_archive_root: str = nnja_archive("parquet", "gpsro_v3"),
        include_gpsro: bool = True,
        surface_winds: bool = False,
        max_quality_mark: int | None = 2,
        gpsro_saids: str = "legacy",
        include_satwnd: bool = False,
        satwnd_archive_root: str = SATWND_ARCHIVE,
        satwnd_thin_hpx_level: int = 5,
        gpsro_table_source: CycleTableSource | None = None,
        satwnd_table_source: CycleTableSource | None = None,
        prepbufr_table_source: CycleTableSource | None = None,
        conv_uv_in_situ_only: bool = False,
        conv_gps_level1_only: bool = False,
        drop_restricted_aircraft: bool = False,
        balloon_drift: bool = False,
        drop_report_types: tuple[int, ...] = (),
        conv_min_pressure_hpa: float | None = None,
        drop_obs_channel_ids: list[int] | None = None,
        use_conv_level_stats: bool = True,
        wind_obs_dropout: float = 0.0,
        surface_pressure_dropout: float = 0.0,
        dropout_scope: str = "row",
        ascat_only_scatterometer: bool = False,
        pressure_height_fill: str | None = None,
    ) -> None:
        self.sensors = [CONV_SENSOR]
        self.obs_context_hours = obs_context_hours
        self.conv_uv_in_situ_only = conv_uv_in_situ_only
        self.conv_gps_level1_only = conv_gps_level1_only
        self.drop_restricted_aircraft = drop_restricted_aircraft
        self.drop_report_types = tuple(drop_report_types)
        self.conv_min_pressure_hpa = conv_min_pressure_hpa
        self.use_conv_level_stats = use_conv_level_stats
        self.drop_obs_channel_ids = (
            list(drop_obs_channel_ids) if drop_obs_channel_ids is not None else []
        )
        self._loader = NNJAConvLoader(
            archive_root=archive_root,
            gpsro_archive_root=gpsro_archive_root,
            obs_context_hours=obs_context_hours,
            # Physical values here; z-score happens after conv-plevel expansion when
            # use_conv_level_stats is set (matching UFSUnifiedLoader).
            normalize=False,
            include_gpsro=include_gpsro,
            surface_winds=surface_winds,
            max_quality_mark=max_quality_mark,
            gpsro_saids=gpsro_saids,
            include_satwnd=include_satwnd,
            satwnd_archive_root=satwnd_archive_root,
            satwnd_thin_hpx_level=satwnd_thin_hpx_level,
            gpsro_table_source=gpsro_table_source,
            satwnd_table_source=satwnd_table_source,
            prepbufr_table_source=prepbufr_table_source,
            drop_restricted_aircraft=drop_restricted_aircraft,
            balloon_drift=balloon_drift,
            wind_obs_dropout=wind_obs_dropout,
            surface_pressure_dropout=surface_pressure_dropout,
            dropout_scope=dropout_scope,
            ascat_only_scatterometer=ascat_only_scatterometer,
            pressure_height_fill=pressure_height_fill,
        )

    @functools.cached_property
    def _base_channel_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """min_valid / max_valid / is_conv indexed by base conv Global_Channel_ID.

        Only the 8 conv rows reach this loader, so the packaged vocabulary is the
        whole table; the sat branch of the shared filter is unused here.
        """
        offset = sensors.SENSOR_OFFSET["conv"]
        base = sensors.CONV_CHANNELS
        size = offset + len(base)
        min_valid = np.full(size, np.nan, dtype=np.float32)
        max_valid = np.full(size, np.nan, dtype=np.float32)
        is_conv = np.zeros(size, dtype=bool)
        for i, channel in enumerate(base):
            min_valid[offset + i] = channel.min_valid
            max_valid[offset + i] = channel.max_valid
            is_conv[offset + i] = True
        return min_valid, max_valid, is_conv

    @functools.cached_property
    def _plevel_norm_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        conv = channel_table(nnja_conv=True).slice(SENSOR_OFFSET[CONV_SENSOR])
        mean = np.asarray(conv["mean"], dtype=np.float32)
        stddev = np.asarray(conv["stddev"], dtype=np.float32)
        if mean.size != CONV_CHANNELS:
            raise ValueError(
                f"expected {CONV_CHANNELS} conv-plevel stats, got {mean.size}"
            )
        return mean, stddev

    def _attach_base_bounds(self, table: pa.Table) -> pa.Table:
        min_lut, max_lut, is_conv_lut = self._base_channel_bounds
        gid = np.asarray(table[GLOBAL_CHANNEL_ID.name], dtype=np.int64)
        table = table.append_column(
            "min_valid", pa.array(min_lut[gid], type=pa.float32())
        )
        table = table.append_column(
            "max_valid", pa.array(max_lut[gid], type=pa.float32())
        )
        return table.append_column("is_conv", pa.array(is_conv_lut[gid]))

    def _to_conv_plevel(self, table: pa.Table) -> pa.Table:
        if table.num_rows == 0:
            return table
        if self.drop_report_types:
            # Downstream of combine(), so this reaches AMVs from the dedicated SATWND
            # archive as well as from PrepBUFR. GPS-RO rows carry a satellite id in this
            # column, not a report type.
            deny = pa.array(self.drop_report_types, type=table["Observation_Type"].type)
            table = table.filter(
                pc.invert(pc.is_in(table["Observation_Type"], deny)).fill_null(True)
            )
            if table.num_rows == 0:
                return table
        if self.drop_obs_channel_ids:
            # Ids named in the base conv vocabulary (conv_var_global_ids) match here,
            # before levels are expanded; level ids match after.
            table = self._drop_channels(table)
            if table.num_rows == 0:
                return table
        table = self._attach_base_bounds(table)
        table = filter_observations(
            table,
            qc_filter=False,
            conv_uv_in_situ_only=self.conv_uv_in_situ_only,
            conv_gps_level1_only=self.conv_gps_level1_only,
            conv_min_pressure_hpa=self.conv_min_pressure_hpa,
        )
        table = table.drop(["min_valid", "max_valid", "is_conv"])
        if table.num_rows == 0:
            return table

        base_local = np.asarray(table[LOCAL_CHANNEL_ID.name], dtype=np.int16)
        pressure = np.asarray(table["Pressure"], dtype=np.float32)
        expanded = expanded_local_channel(base_local, pressure).astype(np.uint16)
        if self.use_conv_level_stats:
            mean_lut, std_lut = self._plevel_norm_lookup
            normalized = (
                np.asarray(table["Observation"], dtype=np.float32) - mean_lut[expanded]
            ) / std_lut[expanded]
            table = table.set_column(
                table.schema.get_field_index("Observation"),
                table.schema.field("Observation"),
                pa.array(normalized, type=pa.float32()),
            )

        # Identity in the GSI conv-plevel vocabulary; CombinedObsLoader rebases next.
        global_channel = (sensors.SENSOR_OFFSET["conv-plevel"] + expanded).astype(
            np.uint16
        )
        sensor_id = np.full(
            table.num_rows, sensors.SENSOR_NAME_TO_ID["conv-plevel"], dtype=np.uint16
        )
        table = _set_typed_column(table, LOCAL_CHANNEL_ID, expanded)
        table = _set_typed_column(table, GLOBAL_CHANNEL_ID, global_channel)
        table = _set_typed_column(table, SENSOR_ID, sensor_id)

        if self.drop_obs_channel_ids:
            table = self._drop_channels(table)
        return table

    def _drop_channels(self, table: pa.Table) -> pa.Table:
        drop = pa.array(self.drop_obs_channel_ids).cast(
            table[GLOBAL_CHANNEL_ID.name].type
        )
        return table.filter(pc.invert(pc.is_in(table[GLOBAL_CHANNEL_ID.name], drop)))

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        loaded = await self._loader.sel_time(times)
        return {"obs_v2": [self._to_conv_plevel(table) for table in loaded["obs_v2"]]}


class CombinedObsLoader:
    """NNJA satellite and conventional observations as one unified long table."""

    def __init__(self, satellite: NNJAWideLoader, conventional: ConventionalLoader):
        if list(conventional.sensors) != [CONV_SENSOR]:
            raise ValueError(
                f"the conventional loader must carry exactly [{CONV_SENSOR!r}], got "
                f"{conventional.sensors}"
            )
        self.satellite = satellite
        self.conventional = conventional
        self.sensors = (*satellite.sensors, CONV_SENSOR)

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        # order matters, slowest first
        satellite, conventional = await asyncio.gather(
            self.satellite.sel_time(times), self.conventional.sel_time(times)
        )
        return {
            "obs_v2": [
                pa.concat_tables([sat, _rebase(conv, _local(conv))])
                for sat, conv in zip(satellite["obs_v2"], conventional["obs_v2"])
            ]
        }


def _local(table: pa.Table) -> np.ndarray:
    return table[LOCAL_CHANNEL_ID.name].to_numpy(zero_copy_only=False)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
UFS Unified Loader for the new combined schema data format.

This loader handles both satellite and conventional observations using the
unified schema in ``healda.observations.schema``. It provides an async interface
compatible with TimeMergedDataset and includes quality control filtering,
normalization, and innovation filtering.
"""

import asyncio
import dataclasses
import io
import math
import os
from datetime import datetime
from typing import List, Literal

import functools
import fsspec
import healda.utils.profiling
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import healda.utils.storage
from healda.observations.loaders.threads import (
    WORKERS,
    configure_arrow_pools,
    thread_pool,
)
from healda.utils.profiling import cpu_timing_range
from healda.observations.sensors import (
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)
from healda.observations.preprocessing.conv_plevel import (
    CONV_PLEVEL_BASE_LOCAL_CHANNELS,
    CONV_PLEVEL_N_LEVELS,
    CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL,
    CONV_PLEVEL_SURFACE_LOCAL_CHANNEL,
    PACKAGED_LEVEL_STATS,
    base_local_channel_to_expanded_group,
    build_conv_plevel_channel_table,
    nearest_pressure_level_index,
)
from healda.observations.schema import (
    get_combined_observation_schema,
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
)
from healda.datasets.base import DatasetMetadata, TimeUnit
from healda.observations.preprocessing.filtering import filter_observations

LOCAL_CHANNEL_ID = pa.field("local_channel_id", pa.uint16())


def _split(items: list, count: int) -> list[list]:
    """`items` as at most `count` contiguous, near-equal runs."""
    count = min(count, len(items))
    if count <= 1:
        return [items] if items else []
    size = math.ceil(len(items) / count)
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclasses.dataclass(frozen=True, slots=True)
class _RowGroupJob:
    """One row group to read, and the window its DA_window statistics place it in."""

    sensor: str
    path: str
    # Whole-file contents for remote reads; local reads reopen from the path.
    remote_bytes: bytes | None
    window: pd.Timestamp
    row_group: int


# Time coverage of the processed UFS obs tree (date-partitioned parquet).
# conv/amsua go back to 2000-01-01; atms starts 2012-02, mhs 2005-11, but all
# sensors currently end 2025-04-30 (end date of data on the NOAA's public
# UFS Replay bucket)

UFS_OBS_COVERAGE = DatasetMetadata(
    name="ufs_obs",
    start="2000-01-01 00:00:00",
    end="2025-04-30 18:00:00",
    time_step=6,  # 6h DA windows (3h available)
    time_unit=TimeUnit.HOUR,
)


def get_channel_table():
    import healda.config.environment as config

    return UFSUnifiedLoader(
        config.UFS_OBS_PATH,
        sensors=[],
        obs_context_hours=(-3, 3),
        filesystem_type="s3" if config.UFS_OBS_PATH.startswith("s3://") else "local",
        remote_name=config.UFS_OBS_PROFILE,
    ).channel_table


class UFSUnifiedLoader:
    """
    Unified loader for UFS observation data using the new combined schema.

    This loader handles both satellite and conventional observations in a
    unified format, providing async interface compatibility with TimeMergedDataset.
    """

    def __init__(
        self,
        data_path: str,
        sensors: List[str],
        filesystem_type: Literal["s3", "local"] = "local",
        remote_name: str = "",
        innovation_type: Literal["none", "adjusted", "unadjusted"] = "none",
        qc_filter: bool = False,
        filter_innovation: bool = False,
        check_corrected: bool = True,
        obs_context_hours: tuple[int, int] = (-24, 0),
        data_spacing: int = 3,  # hours
        drop_obs_channel_ids: list[int] | None = None,
        conv_uv_in_situ_only: bool = False,
        conv_gps_level1_only: bool = False,
        conv_min_pressure_hpa: float | None = None,
        use_conv_level_stats: bool = False,
        conv_level_channels: bool = False,
    ):
        """
        Initialize the UFS Unified Loader.

        Args:
            data_path: Path to the processed observation data
            sensors: List of sensors to load (e.g., ['atms', 'mhs', 'conv'])
            filesystem_type: Type of filesystem ('s3' or 'local')
            remote_name: Remote storage name for S3
            innovation_type: Innovation type to use ('none', 'adjusted', 'unadjusted')
            qc_filter: Whether to apply quality control filtering
            filter_innovation: Whether to filter based on innovation values
            check_corrected: Whether to validate corrected observation values
            obs_context_hours: Hours relative to target time for observation context
            data_spacing: Hours between data points
            drop_obs_channel_ids: Global channel IDs to drop
            conv_uv_in_situ_only: Exclude satellite UV (keep in-situ only)
            conv_gps_level1_only: Exclude GPS T/Q (keep bending angle)
            conv_min_pressure_hpa: Minimum pressure for non-GPS conv obs in hPa.
                GPS keeps its 0.5 hPa floor.
            use_conv_level_stats: Use pressure-level normalization stats for
                vertically structured conv channels. Station pressure remains
                single-norm.
            conv_level_channels: Experimental expanded conv channel vocabulary. Not
                supported by the current processed parquet/channel table.
        """
        self.data_path = data_path
        self.sensors = sensors
        self.filesystem_type = filesystem_type
        self.remote_name = remote_name
        self.innovation_type = innovation_type
        self.qc_filter = qc_filter
        self.filter_innovation = filter_innovation
        self.check_corrected = check_corrected
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        # Optional list of global observation channel IDs (GLOBAL_CHANNEL_ID)
        # to drop before normalization and further processing.
        self.drop_obs_channel_ids = (
            list(drop_obs_channel_ids) if drop_obs_channel_ids is not None else []
        )
        self.conv_uv_in_situ_only = conv_uv_in_situ_only
        self.conv_gps_level1_only = conv_gps_level1_only
        self.conv_min_pressure_hpa = conv_min_pressure_hpa
        self.use_conv_level_stats = use_conv_level_stats or conv_level_channels
        self.conv_level_channels = conv_level_channels
        configure_arrow_pools()

        # Validate sensors
        for sensor in self.sensors:
            if sensor not in SENSOR_CONFIGS:
                raise ValueError(
                    f"Unconfigured sensor: {sensor}. Available: {list(SENSOR_CONFIGS.keys())}"
                )

        # Setup filesystem
        if self.filesystem_type == "s3":
            self.fs = fsspec.filesystem(
                "s3", **healda.utils.storage.get_storage_options(remote_name)
            )
        elif self.filesystem_type == "local":
            self.fs = None
        else:
            raise ValueError(
                f"Unsupported filesystem_type: {filesystem_type}. Use 's3' or 'local'"
            )

        # Load channel table for normalization
        self._channel_table = None

    @functools.cached_property
    def _base_schema(self) -> pa.Schema:
        return get_combined_observation_schema()

    @functools.cached_property
    def _read_columns(self) -> list[str]:
        return self._base_schema.names

    @property
    def output_schema(self) -> pa.Schema:
        """Get the output schema including the sensor and platform columns."""
        return self._base_schema.append(LOCAL_CHANNEL_ID).append(SENSOR_ID)

    @functools.cached_property
    def channel_table(self) -> pa.Table:
        """Load the channel table for normalization."""
        channel_table_path = os.path.join(self.data_path, "channel_table.parquet")
        if self.fs is not None:
            file = io.BytesIO(self.fs.cat_file(channel_table_path))
        else:
            file = channel_table_path

        table = pq.read_table(file)
        sensor_id = np.asarray(table["sensor_id"])
        local_channel_ids = []
        offset = 0
        for i in range(len(sensor_id)):
            if sensor_id[i] != sensor_id[i - 1]:
                offset = i
            local_channel_ids.append(i - offset)
        array = pa.array(local_channel_ids).cast(LOCAL_CHANNEL_ID.type)
        table = table.append_column(LOCAL_CHANNEL_ID, array)
        if not self.use_conv_level_stats:
            return table
        conv_plevel_sensor_id = SENSOR_NAME_TO_ID["conv-plevel"]
        table = table.filter(pc.not_equal(table[SENSOR_ID.name], conv_plevel_sensor_id))
        return pa.concat_tables(
            [
                table,
                build_conv_plevel_channel_table(
                    level_stats=self.conv_level_stats,
                    base_channel_table=table,
                    include_local_channel_id=True,
                ),
            ]
        )

    @functools.cached_property
    def conv_level_stats(self) -> pa.Table | None:
        if not self.use_conv_level_stats:
            return None

        stats_path = os.path.join(self.data_path, "conv_normalizations_by_level.csv")

        if self.fs is not None and stats_path.startswith(self.data_path):
            stats_csv = io.BytesIO(self.fs.cat_file(stats_path))
        else:
            if not os.path.exists(stats_path):
                stats_path = PACKAGED_LEVEL_STATS
            stats_csv = stats_path
        table = pa.Table.from_pandas(pd.read_csv(stats_csv), preserve_index=False)

        # The CSV is keyed by the original conv Global_Channel_ID plus nearest
        # ERA5 Level_hPa. Cast those keys to the same types used by channel_table
        # so later lookups do not silently miss.
        table = table.set_column(
            table.schema.get_field_index("Global_Channel_ID"),
            "Global_Channel_ID",
            table["Global_Channel_ID"].cast(GLOBAL_CHANNEL_ID.type),
        )
        table = table.set_column(
            table.schema.get_field_index("Level_hPa"),
            "Level_hPa",
            table["Level_hPa"].cast(pa.int16()),
        )

        # CONV_PLEVEL_BASE_LOCAL_CHANNELS are local conv channel ids
        # (gps_angle/gps_t/gps_q/q/t/u/v). Add the base conv sensor offset to get
        # the matching original Global_Channel_IDs in the stats CSV; ps is excluded
        # because it remains one channel with one normalization.
        base_conv_offset = SENSOR_OFFSET["conv"]
        vertical_gids = pa.array(
            base_conv_offset + CONV_PLEVEL_BASE_LOCAL_CHANNELS,
            type=GLOBAL_CHANNEL_ID.type,
        )
        table = table.filter(pc.is_in(table["Global_Channel_ID"], vertical_gids))
        return table.select(
            [
                "Global_Channel_ID",
                "Level_hPa",
                "obs_mean",
                "obs_std",
            ]
        ).rename_columns(
            ["Global_Channel_ID", "Level_hPa", "level_mean", "level_stddev"]
        )

    @functools.cached_property
    def _conv_plevel_local_channel_lut(self) -> np.ndarray:
        """Map (base conv local channel, pressure-level index) to expanded local id.

        Shape is (8, 13): the original conv vocabulary by the level index returned
        from ``nearest_pressure_level_index``. A vertical channel's row holds its
        13 consecutive expanded ids, so the ``q`` row is 39..51. ``ps`` is not
        expanded, so all 13 entries of its row hold the same id, 91, which lets
        callers gather every observation in one indexing op with no surface
        branch. See ``conv_plevel`` for the full channel layout.
        """
        conv_channels = SENSOR_CONFIGS["conv"].channels
        base_to_group = base_local_channel_to_expanded_group()

        lut = np.empty((conv_channels, CONV_PLEVEL_N_LEVELS), dtype=np.uint16)
        for local_channel in range(conv_channels):
            if local_channel == CONV_PLEVEL_SURFACE_LOCAL_CHANNEL:
                lut[local_channel, :] = CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL
                continue
            group = base_to_group[local_channel]
            if group < 0:
                raise ValueError(f"cannot map conv local channel {local_channel}")
            lut[local_channel, :] = group * CONV_PLEVEL_N_LEVELS + np.arange(
                CONV_PLEVEL_N_LEVELS, dtype=np.uint16
            )
        return lut

    @functools.cached_property
    def _conv_plevel_norm_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mean/std arrays indexed by conv-plevel local channel id."""
        conv_plevel_sensor_id = SENSOR_NAME_TO_ID["conv-plevel"]
        channel_table = self.channel_table.filter(
            pc.equal(self.channel_table[SENSOR_ID.name], conv_plevel_sensor_id)
        )
        local_channel = np.asarray(
            channel_table[LOCAL_CHANNEL_ID.name], dtype=np.uint16
        )
        size = SENSOR_CONFIGS["conv-plevel"].channels
        mean = np.zeros(size, dtype=np.float32)
        stddev = np.ones(size, dtype=np.float32)
        mean[local_channel] = np.asarray(channel_table["mean"], dtype=np.float32)
        stddev[local_channel] = np.asarray(channel_table["stddev"], dtype=np.float32)
        return mean, stddev

    def _get_interval_times(self, dt: datetime) -> pd.DatetimeIndex:
        """Get times in the observation context interval."""
        start, end = self.obs_context_hours
        # DA windows are end-aligned: the "03:00" window covers obs from 00:00–03:00.
        start += self.data_spacing

        return pd.date_range(
            dt + pd.Timedelta(hours=start),
            dt + pd.Timedelta(hours=end),
            freq=f"{self.data_spacing}h",
        )

    def _get_parquet_files_to_read(self, interval_times: pd.DatetimeIndex):
        """Get parquet files to read for given time interval."""
        required_dates = {t.strftime("%Y%m%d") for t in interval_times}

        for sensor in self.sensors:
            for date in required_dates:
                parquet_sensor = "conv" if sensor == "conv-plevel" else sensor
                file_path = os.path.join(
                    self.data_path, parquet_sensor, f"{date}", "0.parquet"
                )
                yield (sensor, file_path)

    def _open_parquet(self, path: str, remote_bytes: bytes | None) -> pq.ParquetFile:
        """A handle of the caller's own, since pyarrow serializes reads through one."""
        source = io.BytesIO(remote_bytes) if remote_bytes is not None else path
        with cpu_timing_range("ufs.open"):
            return pq.ParquetFile(source, pre_buffer=self.parquet_pre_buffer)

    def _plan_row_groups(
        self,
        sensor: str,
        path: str,
        target_windows: pd.DatetimeIndex,
    ) -> list[_RowGroupJob]:
        """Row groups whose DA_window statistics name one of the target windows.

        Footer only. A missing file contributes nothing, so readers past here take it as good.
        """
        try:
            remote_bytes = self.fs.cat_file(path) if self.fs is not None else None
            parquet = self._open_parquet(path, remote_bytes)
            da_window = parquet.schema_arrow.get_field_index("DA_window")
        except (FileNotFoundError, OSError):
            return []

        wanted = set(target_windows)
        jobs = []
        for row_group in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(row_group).column(da_window).statistics
            # A row group holds exactly one DA_window by ETL contract, so its min is that
            # singular window.
            if stats.min != stats.max:
                raise ValueError(
                    f"{path} row group {row_group} spans {stats.min}..{stats.max}"
                )
            if stats.min not in wanted:
                continue
            jobs.append(
                _RowGroupJob(
                    sensor=sensor,
                    path=path,
                    remote_bytes=remote_bytes,
                    window=stats.min,
                    row_group=row_group,
                )
            )
        return jobs

    def _read_row_group(
        self, parquet: pq.ParquetFile, job: _RowGroupJob
    ) -> pa.Table | None:
        """The job's rows, all of them in job.window, or None if the row group is empty."""
        with cpu_timing_range("ufs.read"):
            table = parquet.read_row_group(job.row_group, columns=self._read_columns)

        # A zero-row table still adds a chunk to every column downstream.
        return table if table.num_rows else None

    def _filter_observations(self, table: pa.Table) -> pa.Table:
        return filter_observations(
            table,
            self.qc_filter,
            conv_uv_in_situ_only=self.conv_uv_in_situ_only,
            conv_gps_level1_only=self.conv_gps_level1_only,
            conv_min_pressure_hpa=self.conv_min_pressure_hpa,
        )

    @staticmethod
    def _set_typed_column(table: pa.Table, field: pa.Field, values) -> pa.Table:
        """Replace an existing column while preserving the declared Arrow type."""
        index = table.schema.get_field_index(field.name)
        # Passing a name rather than a field declares the replacement nullable, which would
        # leave conv-plevel's window disagreeing with the rest and fail the concatenation.
        return table.set_column(
            index,
            table.schema.field(index).with_type(field.type),
            pa.array(values, type=field.type),
        )

    def _normalize_conv_with_plevel_stats(
        self, table: pa.Table, output_plevel_channels: bool
    ) -> pa.Table:
        """Normalize conv obs with pressure-level stats, optionally emitting conv-plevel ids.

        Input rows have already been joined to base conv metadata. This computes
        the expanded local channel from base local channel + pressure, gathers
        level-specific mean/std, and then either keeps base conv ids or replaces
        them with the expanded ``conv-plevel`` channel identity.
        """
        if table.num_rows == 0:
            return table

        # Bin each observation pressure to one of the 13 ERA5 pressure levels.
        level_idx = nearest_pressure_level_index(
            np.asarray(table["Pressure"], dtype=np.float32)
        )

        # Convert base conv local channel + pressure bin into the expanded
        # conv-plevel local channel used for per-level normalization.
        base_local = np.asarray(table[LOCAL_CHANNEL_ID.name], dtype=np.uint16)
        expanded_local = self._conv_plevel_local_channel_lut[base_local, level_idx]

        # Gather mean/std by expanded local channel and apply z-score normalization.
        mean_lut, std_lut = self._conv_plevel_norm_lookup
        normalized = (
            np.asarray(table["Observation"], dtype=np.float32)
            - mean_lut[expanded_local]
        ) / std_lut[expanded_local]
        index = table.schema.get_field_index("Observation")
        table = table.set_column(
            index,
            table.schema.field(index),
            pa.array(normalized, type=table["Observation"].type),
        )

        # Normalization can use expanded pressure-level stats while keeping the
        # original base conv channel ids. Only the conv-plevel sensor path emits
        # the expanded channel identity to the model.
        if not output_plevel_channels:
            return table

        # Arrow tables are immutable; replace the three identity columns with the
        # expanded conv-plevel ids after normalization.
        global_channel = (SENSOR_OFFSET["conv-plevel"] + expanded_local).astype(
            np.uint16
        )
        sensor_id = np.full(
            table.num_rows, SENSOR_NAME_TO_ID["conv-plevel"], dtype=np.uint16
        )
        table = self._set_typed_column(table, LOCAL_CHANNEL_ID, expanded_local)
        table = self._set_typed_column(table, GLOBAL_CHANNEL_ID, global_channel)
        return self._set_typed_column(table, SENSOR_ID, sensor_id)

    def _normalize_observations(
        self,
        table: pa.Table,
    ) -> pa.Table:
        """Z-score normalize observations using joined channel-table mean/std."""
        normalized = pc.divide(
            pc.subtract(table["Observation"], table["mean"]), table["stddev"]
        )
        return table.set_column(
            table.schema.get_field_index("Observation"),
            "Observation",
            normalized,
        )

    _extra_channel_fields = ["min_valid", "max_valid", "is_conv", "mean", "stddev"]

    # Coalesces column ranges into fewer reads; marginal here, where files are a few MiB.
    parquet_pre_buffer = True

    # Each (sensor, day) file is independent, so reads and conversion overlap; Arrow
    # releases the GIL for both. 0 or False reads serially.
    concurrent_file_reads = WORKERS

    @functools.cached_property
    def _channel_lookup(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Channel-table columns as arrays indexed by Global_Channel_ID.

        The id is a small dense integer, so the metadata attaches by gather rather than
        by key. Built once per loader.
        """
        global_id = np.asarray(
            self.channel_table[GLOBAL_CHANNEL_ID.name], dtype=np.int64
        )
        size = int(global_id.max()) + 1
        columns = {}
        for name in (
            LOCAL_CHANNEL_ID.name,
            SENSOR_ID.name,
            *self._extra_channel_fields,
        ):
            values = np.asarray(self.channel_table[name])
            table = np.zeros(size, dtype=values.dtype)
            table[global_id] = values
            columns[name] = table
        present = np.zeros(size, dtype=bool)
        present[global_id] = True
        return columns, present

    def _add_channel_metadata(self, table):
        """Attach per-channel metadata by Global_Channel_ID.

        The hash join this replaces rebuilt a table over the channel table every window,
        cost more than the read did, and lost row order. It only overtakes the gather
        past 8 Arrow threads, twice what `loader_threads` sets, so there is no flag.
        """
        columns, present = self._channel_lookup
        global_id = np.asarray(table[GLOBAL_CHANNEL_ID.name], dtype=np.int64)
        # An unknown id would gather row zero and silently attach another channel's
        # stats; one past the end of `present` raises IndexError, so range-check first.
        known = np.zeros(global_id.size, dtype=bool)
        in_range = (global_id >= 0) & (global_id < present.size)
        known[in_range] = present[global_id[in_range]]
        if not known.all():
            raise KeyError(
                f"global channel ids absent from the channel table: "
                f"{np.unique(global_id[~known]).tolist()}"
            )
        for name, lookup in columns.items():
            table = table.append_column(name, pa.array(lookup[global_id]))
        return table

    async def sel_time(self, times: pd.DatetimeIndex) -> dict[str, list[pa.Table]]:
        """
        Load observation data for specified times.

        Args:
            times: Target times to load data for

        Returns:
            One table per target time under `obs_v2`, each grouped by observation window
            and then by the order the files were read.
        """
        # The store's DA_window is naive; an aware index would match no row group and come
        # back empty rather than fail.
        if times.tz is not None:
            raise ValueError(f"times must be tz-naive UTC, got {times.tz}")

        # Get all times needed for the context window
        all_times = set()
        for t in times:
            interval_times = self._get_interval_times(t)
            all_times.update(interval_times)

        interval_times = pd.DatetimeIndex(sorted(all_times))

        async def on_pool(work, items):
            """`work` over `items`, spread across the shared pool, results left in order."""
            if not self.concurrent_file_reads or len(items) < 2:
                return [work(item) for item in items]
            loop = asyncio.get_running_loop()
            return await asyncio.gather(
                *(loop.run_in_executor(thread_pool(), work, item) for item in items)
            )

        def plan(sensor_and_path) -> list[_RowGroupJob]:
            sensor, path = sensor_and_path
            return self._plan_row_groups(sensor, path, interval_times)

        def load(job: _RowGroupJob, parquet: pq.ParquetFile) -> pa.Table | None:
            table = self._read_row_group(parquet, job)
            if table is None:
                return None
            with cpu_timing_range("ufs.metadata"):
                table = self._add_channel_metadata(table)
            with cpu_timing_range("ufs.filter"):
                table = self._filter_observations(table)
            # Drop specified global channels, if any
            if self.drop_obs_channel_ids:
                mask = pc.is_in(
                    table[GLOBAL_CHANNEL_ID.name],
                    pa.array(self.drop_obs_channel_ids).cast(
                        table[GLOBAL_CHANNEL_ID.name].type
                    ),
                )
                # Keep rows whose GLOBAL_CHANNEL_ID is NOT in drop list
                table = table.filter(pc.invert(mask))
            with cpu_timing_range("ufs.normalize"):
                if job.sensor == "conv-plevel":
                    table = self._normalize_conv_with_plevel_stats(
                        table, output_plevel_channels=True
                    )
                elif job.sensor == "conv" and self.use_conv_level_stats:
                    table = self._normalize_conv_with_plevel_stats(
                        table, output_plevel_channels=False
                    )
                else:
                    table = self._normalize_observations(table)
                table = table.drop(self._extra_channel_fields)
            return table

        # Conv is the slowest, so submit it first; matters when jobs outnumber the pool.
        def conv_first(sensor_and_path):
            sensor, _path = sensor_and_path
            return 0 if sensor in ("conv", "conv-plevel") else 1

        files = sorted(self._get_parquet_files_to_read(interval_times), key=conv_first)
        per_file = await on_pool(plan, files)

        # Files come first for width, each on its own OST. Splitting one into row groups only
        # pays when there are too few files to fill the pool, which is conv at one per sample.
        readers_per_file = max(1, WORKERS // max(len(per_file), 1))
        runs = [
            run for file_jobs in per_file for run in _split(file_jobs, readers_per_file)
        ]

        def load_run(jobs: list[_RowGroupJob]) -> list[pa.Table | None]:
            # One handle per run keeps its row groups a sequential scan.
            parquet = self._open_parquet(jobs[0].path, jobs[0].remote_bytes)
            return [load(job, parquet) for job in jobs]

        # Runs are contiguous and `gather` resolves in order, so tables come back in
        # file-then-row-group order, as a sequential read would leave them.
        loaded = await on_pool(load_run, runs)

        tables = {}
        for run, results in zip(runs, loaded, strict=True):
            for job, table in zip(run, results, strict=True):
                if table is not None:
                    tables.setdefault(job.window, []).append(table)

        # Combine all observations
        def process(t):
            all_tables = []
            for interval_time in self._get_interval_times(t):
                for table in tables.get(interval_time, []):
                    all_tables.append(table)

            if not all_tables:
                return empty

            table = pa.concat_tables(all_tables)
            # Cast to ensure proper nullability and types
            return table.cast(self.output_schema)

        empty = self._get_empty_table()
        return {"obs_v2": [process(t) for t in times]}

    def _get_empty_table(self):
        # Return empty table with proper schema
        # Create empty arrays for each field in the schema
        empty_arrays = []
        for field in self.output_schema:
            empty_arrays.append(pa.array([], type=field.type))
        template = pa.table(empty_arrays, schema=self.output_schema)
        return template


if __name__ == "__main__":
    import time

    import healda.config.environment as config

    DATA_PATH = config.UFS_OBS_PATH
    sensors = ["atms", "mhs", "amsua", "amsub", "iasi-pca", "cris-fsr-pca", "conv"]
    sample_time = pd.Timestamp("2022-01-18T21:00:00")

    loader = UFSUnifiedLoader(
        data_path=DATA_PATH,
        sensors=sensors,
        obs_context_hours=(-3, +3),
    )

    files = [
        (s, p)
        for s, p in loader._get_parquet_files_to_read(
            loader._get_interval_times(sample_time)
        )
        if os.path.exists(p)
    ]
    disk_bytes = sum(os.path.getsize(p) for _, p in files)
    print(f"Read files:   {len(files)} parquet files")
    for s, p in files:
        print(f"  {s:20s}  {os.path.getsize(p) / 1e6:6.1f} MB  {p}")

    start = time.perf_counter()
    result = asyncio.run(loader.sel_time(pd.DatetimeIndex([sample_time])))
    elapsed = time.perf_counter() - start
    table = result["obs_v2"][0]

    da_col = table.column("DA_window")
    print(f"Rows:    {table.num_rows:,}")
    print(f"DA range: {pc.min(da_col).as_py()} → {pc.max(da_col).as_py()}")
    print(f"On disk: {disk_bytes / 1e6:.1f} MB")
    print(f"In mem:  {table.nbytes / 1e6:.1f} MB")
    print(f"Load:    {elapsed * 1e3:.0f} ms")

    # By id, not by zip against `sensors`: the ids present are neither all of the
    # configured sensors nor in that order.
    id_to_name = {SENSOR_NAME_TO_ID[sensor]: sensor for sensor in sensors}
    ids, counts = np.unique(table["sensor_id"].to_numpy(), return_counts=True)
    for sensor_id, count in zip(ids, counts):
        print(f"Sensor {id_to_name[int(sensor_id)]}: {count:,} obs")

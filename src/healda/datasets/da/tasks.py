# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DA training task configs plus the UFS observation plumbing they share.

A ``TrainingTaskConfig`` names the supervised gridded-state problem: the target
state, optional gridded state inputs, and residual metadata. Observations are not
part of the task identity; the train loop toggles them with ``use_obs`` /
``ObsConfig``. ``build_training_dataset`` turns a task into the right dataset.
"""

import dataclasses
from typing import Any, Mapping

import torch

import healda.config.environment as config
from healda.config.models import ObsConfig
from healda.datasets import catalog
from healda.datasets.catalog import _Zarr
from healda.observations.loaders import combined
from healda.observations.loaders.nnja_base import CycleTableSource
from healda.observations.loaders.nnja_wide import NNJAWideLoader
from healda.observations.loaders.ufs import UFSUnifiedLoader
from healda.observations.system import ObsPipeline
from healda.observations.sensors_nnja import DEFAULT_SENSORS
from healda.observations.preprocessing import scan_geometry

__all__ = [
    "TASK_CONFIGS",
    "StateConfig",
    "TrainingTaskConfig",
    "build_obs_loader",
    "build_training_dataset",
    "collate",
    "get_sensors_for_config",
]


@dataclasses.dataclass(frozen=True)
class StateConfig:
    name: str
    entry: _Zarr
    variable_config: str
    source: str
    stats_file: str
    time_offset_hours: int = 0


@dataclasses.dataclass(frozen=True)
class TrainingTaskConfig:
    name: str
    target: StateConfig
    inputs: tuple[StateConfig, ...] = ()
    residual_against: str | None = None
    residual_stats_file: str | None = None
    # Optional override of the input the skill metric is scored against. None
    # leaves the baseline to fall back (where skill is computed) to the residual
    # input, then the first input.
    skill_against: str | None = None
    default_prediction_mode: str = "full"
    dataset_backend: str = "packed"

    @property
    def has_background(self) -> bool:
        return bool(self.inputs)

    def input_by_name(self, name: str | None) -> StateConfig | None:
        if name is None:
            return None
        for state in self.inputs:
            if state.name == name:
                return state
        raise ValueError(f"{self.name}: {name!r} is not an input")

    @property
    def residual_input(self) -> StateConfig | None:
        return self.input_by_name(self.residual_against)

    @property
    def skill_input(self) -> StateConfig | None:
        return self.input_by_name(self.skill_against)

    def validate_prediction_mode(self, prediction_mode: str) -> None:
        if prediction_mode == "full":
            return
        if prediction_mode != "residual":
            raise ValueError(
                f"{self.name}: unsupported prediction_mode={prediction_mode!r}"
            )
        residual_input = self.residual_input
        if residual_input is None or self.residual_stats_file is None:
            raise ValueError(f"{self.name}: residual mode requires residual metadata")
        if self.target.variable_config != residual_input.variable_config:
            raise ValueError(
                f"{self.name}: residual mode requires matching target/residual-input "
                f"channels, got {self.target.variable_config!r} and "
                f"{residual_input.variable_config!r}"
            )


# All tasks use YearHoldoutSplit (val=2022, test=2025); each store's time range sets
# the first year.
TASK_CONFIGS: dict[str, TrainingTaskConfig] = {
    "era5_74ch": TrainingTaskConfig(
        name="era5_74ch",
        target=StateConfig(
            name="era5_analysis",
            entry=catalog.era5_104var,
            variable_config="era5_74ch",
            source="era5",
            stats_file="era5_13_levels_stats.csv",
        ),
    ),
    # adding land/cloud variables
    "era5_86ch": TrainingTaskConfig(
        name="era5_86ch",
        target=StateConfig(
            name="era5_analysis",
            entry=catalog.era5_104var,
            variable_config="era5_86ch",
            source="era5",
            stats_file="era5_13_levels_stats.csv",
        ),
    ),
    # Adding vertical velocity.
    "era5_99ch": TrainingTaskConfig(
        name="era5_99ch",
        target=StateConfig(
            name="era5_analysis",
            entry=catalog.era5_104var,
            variable_config="era5_99ch",
            source="era5",
            stats_file="era5_13_levels_stats.csv",
        ),
    ),
    # era5_99ch plus (u10/v10/z10/t10) and tcw.
    #
    # KNOWN WRONG, left alone so arms in flight keep their normalisation: both grids predict
    # the tcw-tcwv residual (state_transforms), but this file's tcw row is fitted to raw tcw,
    # so the residual z-scores to about -1.44 with almost no variance. era5_104ch_hpx_stats.csv
    # is the corrected file; switching to it changes what every 104ch HPX arm sees.
    "era5_104ch": TrainingTaskConfig(
        name="era5_104ch",
        target=StateConfig(
            name="era5_analysis",
            entry=catalog.era5_104var,
            variable_config="era5_104ch",
            source="era5",
            stats_file="era5_13_levels_stats.csv",
        ),
    ),
    # The 0.25 degree counterpart. Stats are cos(lat)-weighted on this grid, which keeps
    # variance HPX64 averages away (W most), so the two grids cannot share a file.
    "era5_104ch_latlon": TrainingTaskConfig(
        name="era5_104ch_latlon",
        target=StateConfig(
            name="era5_analysis",
            entry=catalog.era5_104var,
            variable_config="era5_104ch",
            source="era5",
            stats_file="era5_104ch_latlon_stats.csv",
        ),
        dataset_backend="latlon",
    ),
}


def validate_task_configs() -> None:
    """Check every registered task. Call again after registering more."""
    from healda.config.variables import VARIABLE_CONFIGS
    from healda.datasets.da.state_stats import resolve_stats_file

    for key, task in TASK_CONFIGS.items():
        if key != task.name:
            raise ValueError(f"{key}: task registry key does not match {task.name!r}")
        states = (task.target, *task.inputs)
        state_names = [state.name for state in states]
        if len(state_names) != len(set(state_names)):
            raise ValueError(f"{key}: duplicate state names: {state_names}")
        for state in states:
            if state.variable_config not in VARIABLE_CONFIGS:
                raise ValueError(
                    f"{key}: unknown variable config {state.variable_config!r}"
                )
            resolve_stats_file(state.stats_file)
        if task.residual_stats_file:
            resolve_stats_file(task.residual_stats_file)
        task.validate_prediction_mode(task.default_prediction_mode)


validate_task_configs()


def build_training_dataset(
    name: str,
    *,
    transform_options,
    **kwargs,
) -> torch.utils.data.Dataset:
    """Build a dataset for a registered training task."""
    config_ = TASK_CONFIGS[name]
    if config_.dataset_backend == "latlon":
        from healda.datasets.da.hourly_latlon_dataset import HourlyLatlonDataset

        return HourlyLatlonDataset(
            variable_config=config_.target.variable_config,
            stats_file=config_.target.stats_file,
            source=config_.target.source,
            transform_options=transform_options,
            **kwargs,
        )

    if config_.dataset_backend != "packed":
        raise ValueError(f"{name}: unknown dataset_backend={config_.dataset_backend!r}")
    from healda.datasets.da.packed_da_dataset import PackedDADataset

    return PackedDADataset(config_, transform_options=transform_options, **kwargs)


def collate(batch: list[dict[str, Any]]):
    """Collate a batch of samples for masked training.

    Uses default_collate for tensor fields.
    """
    data_tensors = [
        {k: v for k, v in sample.items() if torch.is_tensor(v)} for sample in batch
    ]
    return torch.utils.data.default_collate(data_tensors)


def satellite_sensors(config: ObsConfig) -> tuple[str, ...]:
    dropped = set(config.drop_sensors)
    if {"conv", combined.CONV_SENSOR} & dropped:
        # CombinedObsLoader requires a conventional loader carrying exactly CONV_SENSOR, so
        # dropping conv here would leave the model without it while the rows kept arriving.
        raise ValueError(f"use_nnja_sat cannot drop {combined.CONV_SENSOR}")
    return tuple(
        s for s in (config.nnja_sensors or DEFAULT_SENSORS) if s not in dropped
    )


def get_sensors_for_config(config: ObsConfig):
    if config.use_nnja_sat:
        return [*satellite_sensors(config), combined.CONV_SENSOR]
    sensors = ["atms", "mhs", "amsua", "amsub"]
    if config.use_infrared:
        sensors.append("iasi")
    if config.use_conv:
        sensors.append("conv-plevel" if config.conv_level_channels else "conv")
    if config.use_infrared_pca:
        sensors.extend(["iasi-pca", "cris-fsr-pca"])
    if config.use_airs_pca:
        sensors.append("airs-pca")
    if config.drop_sensors:
        # "conv" is an alias for conv/conv-plevel
        drop = set(config.drop_sensors)
        if "conv" in drop:
            drop.add("conv-plevel")
        sensors = [s for s in sensors if s not in drop]
    return sensors


def _get_sat_loader(
    obs_config: ObsConfig,
    pipeline: ObsPipeline,
    *,
    table_source: Mapping[str, CycleTableSource] | None = None,
) -> NNJAWideLoader:
    sensors = satellite_sensors(obs_config)
    return NNJAWideLoader(
        sensors=sensors,
        obs_context_hours=(obs_config.context_start, obs_config.context_end),
        thin_nside=obs_config.nnja_thin_nside,
        ir_channels=obs_config.nnja_ir_channels,
        fov_keep_range={
            sensor: scan_geometry.SCAN_GEOMETRY[sensor].keep for sensor in sensors
        },
        platform_channel_dropout=pipeline.random_drop.platform_channel,
        dropout_scope=pipeline.random_drop.scope,
        table_source=table_source,
    )


def _get_conv_loader(
    obs_config: ObsConfig,
    pipeline: ObsPipeline,
    *,
    gpsro_table_source: CycleTableSource | None = None,
    satwnd_table_source: CycleTableSource | None = None,
    prepbufr_table_source: CycleTableSource | None = None,
):
    """Conventional side of CombinedObsLoader: UFS replay or NNJA PrepBUFR+GPS-RO."""
    table_sources = (gpsro_table_source, satwnd_table_source, prepbufr_table_source)
    if any(source is not None for source in table_sources) and not pipeline.nnja_conv:
        raise ValueError(
            "observation table sources are supported only for NNJA conventional "
            "observations (ObsConfig.use_nnja_conv)"
        )
    filters = pipeline.filters
    if pipeline.nnja_conv:
        # No UFS archive here: the conv vocabulary, its QC bounds and the NNJA
        # by-level normalization all ship with the package.
        return combined.NNJAConventionalLoader(
            gpsro_table_source=gpsro_table_source,
            satwnd_table_source=satwnd_table_source,
            prepbufr_table_source=prepbufr_table_source,
            include_satwnd=pipeline.satwnd,
            satwnd_thin_hpx_level=pipeline.satwnd_thin_hpx_level,
            gpsro_saids=pipeline.gpsro_saids,
            surface_winds=pipeline.surface_winds,
            max_quality_mark=pipeline.max_quality_mark,
            obs_context_hours=(obs_config.context_start, obs_config.context_end),
            # Applied against the GSI conv-plevel vocabulary, before nnja_combined
            # rebases onto the NNJA one, matching the UFS path.
            drop_obs_channel_ids=list(filters.channel_ids),
            conv_uv_in_situ_only=filters.uv_in_situ_only,
            conv_gps_level1_only=filters.gps_level1_only,
            drop_restricted_aircraft=filters.restricted_aircraft,
            balloon_drift=filters.balloon_drift,
            drop_report_types=filters.report_types,
            conv_min_pressure_hpa=filters.non_gps_min_pressure_hpa,
            use_conv_level_stats=obs_config.use_conv_level_stats
            or obs_config.conv_level_channels,
            wind_obs_dropout=pipeline.random_drop.wind,
            surface_pressure_dropout=pipeline.random_drop.surface_pressure,
            dropout_scope=pipeline.random_drop.scope,
            # Not a filter or a random drop: a compatibility switch that changes both.
            ascat_only_scatterometer=obs_config.ascat_only_scatterometer,
        )
    return UFSUnifiedLoader(
        config.UFS_OBS_PATH,
        sensors=[combined.CONV_SENSOR],
        obs_context_hours=(obs_config.context_start, obs_config.context_end),
        filesystem_type="s3" if config.UFS_OBS_PATH.startswith("s3://") else "local",
        remote_name=config.UFS_OBS_PROFILE,
        # Applied against the UFS vocabulary, before nnja_combined rebases conv onto the
        # NNJA one, so the ids are the same ones the non-nnja loops drop.
        drop_obs_channel_ids=list(filters.channel_ids),
        conv_uv_in_situ_only=filters.uv_in_situ_only,
        conv_gps_level1_only=filters.gps_level1_only,
        conv_min_pressure_hpa=filters.non_gps_min_pressure_hpa,
        conv_level_channels=obs_config.conv_level_channels,
        use_conv_level_stats=obs_config.use_conv_level_stats,
    )


def build_obs_loader(
    obs_config: ObsConfig,
    *,
    training: bool,
    satellite_table_source: Mapping[str, CycleTableSource] | None = None,
    gpsro_table_source: CycleTableSource | None = None,
    satwnd_table_source: CycleTableSource | None = None,
    prepbufr_table_source: CycleTableSource | None = None,
):
    """The observation loader for one phase. `training` has no default on purpose:
    a default is what let the latlon backend train with its dropout silently off.

    Dropout takes no seed: it draws from the worker's ambient numpy stream, which
    PyTorch seeds per (rank, worker) from the run seed. Each loader draws exactly once
    per sel_time, which is what keeps time-parallel ranks in lockstep.

    The table sources replace the NNJA archives with in-memory cycle tables
    (``healda.observations.adapters.e2s_nnja``); ``satellite_table_source`` is keyed by
    sensor. A ``None`` source reads the archive on disk.
    """
    assert obs_config.innovation_type == "none"
    pipeline = ObsPipeline(obs_config, training=training)

    if pipeline.nnja_sat:
        return combined.CombinedObsLoader(
            satellite=_get_sat_loader(
                obs_config, pipeline, table_source=satellite_table_source
            ),
            conventional=_get_conv_loader(
                obs_config,
                pipeline,
                gpsro_table_source=gpsro_table_source,
                satwnd_table_source=satwnd_table_source,
                prepbufr_table_source=prepbufr_table_source,
            ),
        )

    table_sources = (
        satellite_table_source,
        gpsro_table_source,
        satwnd_table_source,
        prepbufr_table_source,
    )
    if any(source is not None for source in table_sources):
        raise ValueError(
            "observation table sources are supported only for NNJA observations "
            "(ObsConfig.use_nnja_sat)"
        )
    return UFSUnifiedLoader(
        config.UFS_OBS_PATH,
        sensors=get_sensors_for_config(obs_config),
        obs_context_hours=(obs_config.context_start, obs_config.context_end),
        filesystem_type="s3" if config.UFS_OBS_PATH.startswith("s3://") else "local",
        remote_name=config.UFS_OBS_PROFILE,
        drop_obs_channel_ids=list(pipeline.filters.channel_ids),
        conv_uv_in_situ_only=pipeline.filters.uv_in_situ_only,
        conv_gps_level1_only=pipeline.filters.gps_level1_only,
        conv_min_pressure_hpa=pipeline.filters.non_gps_min_pressure_hpa,
        use_conv_level_stats=obs_config.use_conv_level_stats,
        conv_level_channels=obs_config.conv_level_channels,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyses from a trained HealDA checkpoint and observation tables.

``load_da_model`` rebuilds the network the way the run trained it, from the
``loop.json`` the checkpoint carries; ``DAModel.run_analysis`` runs the recipe's own
observation loaders and transform on in-memory cycle tables, and
``DAModel.run_analysis_unified`` the transform alone on a unified table; both return
physical fields. Downstream wrappers (Earth2Studio) build on this and add nothing to the
numerics.
"""

import asyncio
import dataclasses
import gc
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch._dynamo.utils import counters as dynamo_counters

import healda.models
import healda.utils.datetime
from healda.cli.train import TrainingLoop
from healda.datasets.da import state_masks, state_transforms
from healda.datasets.da.tasks import (
    build_obs_loader,
    get_sensors_for_config,
    satellite_sensors,
)
from healda.datasets.da.transform import TransformV2
from healda.datasets.da.hourly_latlon_dataset import NLAT as LATLON_NLAT
from healda.datasets.da.hourly_latlon_dataset import NLON as LATLON_NLON
from healda.observations import schema
from healda.observations.system import ObsFilters, inference_filters
from healda.observations.loaders import threads as obs_threads
from healda.observations.loaders.nnja_base import CycleTableSource, da_windows
from healda.observations.preprocessing.ir_spectral import ir_channel_preset
from healda.training.checkpoint import Checkpoint

# What run_analysis and run_analysis_unified take as analysis times.
AnalysisTimes = pd.DatetimeIndex | Sequence[pd.Timestamp | str]

# Obs pool threads while an analysis runs; training keeps the pool's default size
INFERENCE_OBS_WORKERS = 32


@dataclasses.dataclass
class DAModel:
    net: torch.nn.Module
    loop: TrainingLoop
    transform: TransformV2
    device: torch.device
    extra_filters: ObsFilters = ObsFilters()

    @property
    def channels(self) -> list[str]:
        return list(self.loop.batch_info.channels)

    def to(self, device) -> "DAModel":
        """Move the network and the device observations are prepared on."""
        self.device = torch.device(device)
        self.net.to(self.device)
        self.transform.compile_pixel_ops(self.device)
        return self

    @property
    def satellite_sensors(self) -> tuple[str, ...]:
        return satellite_sensors(self.loop.obs_config)

    @property
    def ir_channels(self) -> dict[str, list[int]]:
        """Channel numbers the recipe reads per IR sounder; it reads no others."""
        preset = self.loop.obs_config.nnja_ir_channels
        if preset is None:
            return {}
        return {
            sensor: sorted(channels)
            for sensor, channels in ir_channel_preset(preset).items()
            if sensor in self.satellite_sensors
        }

    def frame_times(self, analysis_time) -> pd.DatetimeIndex:
        """The ``time_length`` frames ending at ``analysis_time``."""
        return pd.date_range(
            end=pd.Timestamp(analysis_time),
            periods=self.loop.time_length,
            freq=pd.Timedelta(hours=self.loop.time_step),
        )

    def observation_window(self, analysis_time) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Span of observation times one analysis reads, ``[start, end)``."""
        frames = self.frame_times(analysis_time)
        obs = self.loop.obs_config
        return (
            frames[0] + pd.Timedelta(hours=obs.context_start),
            frames[-1] + pd.Timedelta(hours=obs.context_end),
        )

    def run_analysis(
        self,
        analysis_times: AnalysisTimes,
        *,
        satellite_tables: Mapping[str, CycleTableSource] | None = None,
        gpsro_tables: CycleTableSource | None = None,
        satwnd_tables: CycleTableSource | None = None,
        prepbufr_tables: CycleTableSource | None = None,
    ) -> torch.Tensor:
        """Analyses from per-family observation tables, run through the recipe's loaders.

        Each analysis is the last frame of a ``time_length`` window fed with the
        observations of every frame. The loaders quality control, thin and normalize the
        tables as in training. Analyses run one at a time.

        Args:
            analysis_times: Analysis times, UTC, anything ``pd.DatetimeIndex`` accepts.
            satellite_tables: Per sensor name, ``{cycle: table}`` in
                ``schema.table_schema(sensor)``. A sensor not passed contributes no
                observations.
            gpsro_tables: ``{cycle: table}`` in ``schema.table_schema("gpsro")``.
            satwnd_tables: ``{cycle: table}`` in ``schema.table_schema("satwnd")``.
            prepbufr_tables: ``{cycle: table}`` in ``schema.table_schema("prepbufr")``.
                A family not passed contributes no observations.

        Returns:
            ``(len(analysis_times), len(self.channels), 721, 1440)`` float32 analyses in
            physical units on the 0.25 degree lat/lon grid, on ``self.device``. Bounded
            channels (humidity, water columns, snow depth, soil moisture, sea ice and
            cloud fractions) are clipped to ``state_transforms.PHYSICAL_BOUNDS``.
        """
        # Training runs several dataloader worker processes that share the CPUs; an
        # analysis runs in one process with every CPU to itself.
        obs_threads.set_workers(
            min(INFERENCE_OBS_WORKERS, len(os.sched_getaffinity(0)))
        )
        # A None source reads the on-disk archive; a stream not passed is empty instead.
        # `is None`, not `or`: len() of an analysis_tables source waits for every table.
        loader = build_obs_loader(
            self.loop.obs_config,
            training=False,
            extra_filters=self.extra_filters,
            satellite_table_source={} if satellite_tables is None else satellite_tables,
            gpsro_table_source={} if gpsro_tables is None else gpsro_tables,
            satwnd_table_source={} if satwnd_tables is None else satwnd_tables,
            prepbufr_table_source={} if prepbufr_tables is None else prepbufr_tables,
        )
        # One analysis per time, run alone (batch size 1); they stack on the batch axis.
        analyses = []
        for time in pd.DatetimeIndex(analysis_times):
            frame_times = self.frame_times(time)
            loaded = _run_coroutine(loader.sel_time(frame_times))
            frames = [{"obs_v2": obs} for obs in loaded["obs_v2"]]
            analyses.append(self._analysis(frame_times, frames))
        return torch.cat(analyses)

    def run_analysis_unified(
        self, analysis_times: AnalysisTimes, observations: pa.Table
    ) -> torch.Tensor:
        """Analyses from a unified observation table, skipping the loaders.

        Each frame reads the rows whose ``DA_window`` is one of its windows and passes
        them to featurization unchanged. Analyses run one at a time.

        Args:
            analysis_times: Analysis times, UTC, anything ``pd.DatetimeIndex`` accepts.
            observations: One table in ``schema.table_schema("unified")``, rows already
                thinned, quality controlled and normalized; rows outside the analyses'
                windows are not read.

        Returns:
            ``(len(analysis_times), len(self.channels), 721, 1440)`` float32 analyses in
            physical units on the 0.25 degree lat/lon grid, on ``self.device``. Bounded
            channels (humidity, water columns, snow depth, soil moisture, sea ice and
            cloud fractions) are clipped to ``state_transforms.PHYSICAL_BOUNDS``.

        Raises:
            ValueError: ``observations`` does not match the unified schema, or a read row
                holds a ``sensor_id`` the transform does not group or an id outside the
                network's embedding tables.
        """
        times = pd.DatetimeIndex(analysis_times)
        obs = self.loop.obs_config
        context = (obs.context_start, obs.context_end)
        windows = {
            t: da_windows(t, context) for time in times for t in self.frame_times(time)
        }
        table = self._unified_table(observations)
        needed = pd.DatetimeIndex(sorted(set().union(*windows.values())))
        table = table.filter(pc.is_in(table["DA_window"], pa.array(needed.values)))
        self._check_ids(table)
        # One analysis per time, run alone (batch size 1); they stack on the batch axis.
        analyses = []
        for time in times:
            frame_times = self.frame_times(time)
            frames = [_frame_rows(table, windows[t]) for t in frame_times]
            analyses.append(self._analysis(frame_times, frames))
        return torch.cat(analyses)

    @staticmethod
    def _unified_table(table: pa.Table) -> pa.Table:
        schema.validate(table, "unified")
        unified = schema.table_schema("unified")
        return schema.long_table(
            {
                field.name: table[field.name].cast(field.type)
                for field in unified
                if field.name in table.column_names
            },
            unified,
        )

    def _check_ids(self, table: pa.Table) -> None:
        sensors = set(pc.unique(table[schema.SENSOR_ID.name]).to_pylist())
        unknown = sorted(sensors - set(self.transform.sensor_ids))
        if unknown:
            raise ValueError(
                f"sensor_id {unknown} are not among the transform's sensors "
                f"{list(self.transform.sensor_ids)}"
            )
        for column, rows in self.net.observation_id_bounds().items():
            largest = pc.max(table[column]).as_py()
            if largest is not None and largest >= rows:
                raise ValueError(
                    f"{column} {largest} is outside the network's embedding table of "
                    f"{rows} rows"
                )

    def _analysis(
        self, frame_times: pd.DatetimeIndex, frames: list[dict]
    ) -> torch.Tensor:
        """``(1, C, lat, lon)`` physical-space analysis from one window's frames,
        ``frames[t]["obs_v2"]``, the frame layout the transform shares with training."""
        times = [[healda.utils.datetime.as_cftime(t) for t in frame_times]]
        return self.to_physical(self._forward(times, [frames]))

    def _forward(self, times, frames) -> torch.Tensor:
        """Model-space analysis ``(b, C, lat, lon)``, the last frame, for one batch."""
        graphs_before = dynamo_counters["stats"]["unique_graphs"]
        # The packed tensors never leave transform_observations, and `unified_obs` is
        # rebound to its tokens, so neither is held into the network.
        unified_obs, encodings = self.transform.transform_observations(
            times, frames, self.device
        )
        with torch.inference_mode():
            second_of_day = encodings["second_of_day"].to(self.device)
            b, t = second_of_day.shape
            # The lat/lon model rebuilds its condition; this is the placeholder training
            # feeds.
            condition = torch.zeros(b, 1, t, 1, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda" and self.loop.bf16,
            ):
                unified_obs = self.net.tokenize_observations(unified_obs)
                out = self.net(
                    condition,
                    noise_labels=torch.zeros(b, device=self.device),
                    class_labels=torch.empty(b, 0, device=self.device),
                    second_of_day=second_of_day,
                    day_of_year=encodings["day_of_year"].to(self.device),
                    unified_obs=unified_obs,
                    timestamp=encodings["timestamp"].to(self.device),
                    is_causal=self.loop.dit_temporal_attention_causal,
                )
            if dynamo_counters["stats"]["unique_graphs"] != graphs_before:
                # A compile leaves its tracing state as cyclic garbage that holds this
                # forward's inputs to the compiled blocks (the obs tokens among them).
                gc.collect()
            return out.out[:, :, -1].float()

    def to_physical(self, prediction: torch.Tensor) -> torch.Tensor:
        """Model space ``(b, C, lat, lon)`` -> physical units, bounded channels clipped
        to their physical range, fills re-imposed."""
        return to_physical(
            prediction,
            self.loop,
            restore_fill=self.loop.use_masked_domain_loss,
            clamp=True,
        )


def _frame_rows(table: pa.Table, windows: pd.DatetimeIndex) -> dict[str, pa.Table]:
    """The frame of `table`'s rows in `windows`, one sensor per record batch as the
    transform takes them."""
    rows = table.filter(pc.is_in(table["DA_window"], pa.array(windows.values)))
    sensors = rows[schema.SENSOR_ID.name]
    parts = [rows.filter(pc.equal(sensors, s)) for s in pc.unique(sensors)]
    return {"obs_v2": pa.concat_tables(parts) if parts else rows}


def to_physical(
    prediction: torch.Tensor, loop, *, restore_fill: bool, clamp: bool = False
) -> torch.Tensor:
    """Model space -> physical units, channel axis 1.

    Spatial dims are ``(lat, lon)`` or one flat axis of the run's grid. With
    ``restore_fill``, masked channels get the loader's fill outside their domain.
    With ``clamp``, bounded channels are clipped to ``state_transforms.PHYSICAL_BOUNDS``.
    """
    channels = list(loop.batch_info.channels)
    physical = state_transforms.to_physical_space(
        loop.batch_info.denormalize(prediction),
        channels,
        loop.variable_config.name,
        channel_axis=1,
    )
    if clamp:
        physical = state_transforms.clamp_physical(physical, channels, channel_axis=1)
    if not restore_fill or not any(c in state_masks.CHANNEL_DOMAIN for c in channels):
        return physical
    if not loop.latlon_decode:
        # The run's own order, not a constant: a disk-nest arm emits nest.
        return state_transforms.restore_fill(
            physical,
            channels,
            state_masks.HPX64,
            loop._pipeline_pixel_order,
            channel_axis=1,
        )
    shape = physical.shape
    grid = (
        physical.reshape(*shape[:-1], LATLON_NLAT, LATLON_NLON)
        if shape[-1] == LATLON_NLAT * LATLON_NLON
        else physical
    )
    return state_transforms.restore_fill(
        grid, channels, state_masks.LATLON_025, None, channel_axis=1
    ).reshape(shape)


def _run_coroutine(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Called from a running event loop (a notebook): run on a thread with its own loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def read_training_loop(checkpoint) -> TrainingLoop:
    """The ``TrainingLoop`` a checkpoint was written by.

    Read from the checkpoint's ``loop.json``, which ``healda-train`` writes with every
    checkpoint.
    """
    with Checkpoint(checkpoint) as ckpt:
        loop_json = ckpt.read_loop_json()
    if loop_json is None:
        raise ValueError(
            "checkpoint has no loop.json; use a checkpoint written by healda-train"
        )
    return TrainingLoop.loads(loop_json)


def load_da_model(
    checkpoint: str | os.PathLike,
    device: torch.device | str,
    *,
    extra_filters: ObsFilters | None = None,
) -> DAModel:
    """Rebuild the trained network and its observation pipeline from a checkpoint.

    The recipe is the ``loop.json`` stored in the checkpoint; it must be an NNJA lat/lon
    recipe. The first call downloads the ERA5 static fields into the healda
    cache.

    Args:
        checkpoint: A ``.checkpoint`` file written by ``healda-train``.
        device: Device the network runs on.
        extra_filters: Observations removed beyond the recipe's own filters. Defaults to
            ``inference_filters()``: the dated channel denials, and the provider quality
            flags of tables that carry them.

    Returns:
        The model, ready for ``run_analysis`` and ``run_analysis_unified``.

    Raises:
        ValueError: The checkpoint has no ``loop.json`` or is not an NNJA lat/lon
            recipe.
    """
    device = torch.device(device)
    loop = read_training_loop(checkpoint)
    obs = loop.obs_config
    if not (loop.latlon_decode and obs.use_nnja_sat and obs.use_nnja_conv):
        raise ValueError(
            "load_da_model supports only an NNJA lat/lon recipe (latlon_decode with "
            "use_nnja_sat and use_nnja_conv)"
        )
    net = loop.get_network()
    with Checkpoint(checkpoint) as ckpt:
        net = ckpt.read_model(net=net, map_location="cpu")
    net.eval().requires_grad_(False)
    net.to(device)
    net.last_frame_only = True

    options = healda.models.transform_options_for(loop.model_config)
    transform = TransformV2(
        loop.variable_config,
        hpx_level=options.observation_hpx_level,
        sensors=get_sensors_for_config(loop.obs_config),
        use_nnja_sat=loop.obs_config.use_nnja_sat,
        target_normalization=loop.batch_info.normalization,
        features_position=options.features_position,
        attention_prepack=options.attention_prepack,
        build_attention_group_map=options.build_attention_group_map,
        pixel_order=options.pixel_order,
    )
    return DAModel(
        net=net,
        loop=loop,
        transform=transform,
        device=device,
        extra_filters=inference_filters() if extra_filters is None else extra_filters,
    )

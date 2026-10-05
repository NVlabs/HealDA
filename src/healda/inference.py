# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyses from a trained HealDA checkpoint and observation tables.

``load_da_model`` rebuilds the network the way the run trained it, from the
``loop.json`` the checkpoint carries; ``DAModel.run_analysis`` runs the recipe's own
observation loaders and transform on in-memory cycle tables and returns physical
fields. Downstream wrappers (Earth2Studio) build on this and add nothing to the
numerics.
"""

import asyncio
import dataclasses
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch

import healda.models
import healda.utils.datetime
from healda.cli.train import LOOPS, TrainingLoop
from healda.datasets.da import state_masks, state_transforms
from healda.datasets.da.tasks import (
    build_obs_loader,
    get_sensors_for_config,
    satellite_sensors,
)
from healda.datasets.da.transform import TransformV2
from healda.datasets.da.hourly_latlon_dataset import NLAT as LATLON_NLAT
from healda.datasets.da.hourly_latlon_dataset import NLON as LATLON_NLON
from healda.observations.system import ObsFilters, inference_filters
from healda.observations.loaders.nnja_base import CycleTableSource
from healda.observations.preprocessing.ir_spectral import ir_channel_preset
from healda.training.checkpoint import Checkpoint

# Inference defaults shared by load_da_model and the inference CLI.
PRESSURE_HEIGHT_FILL = "sonde"


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
        analysis_times,
        *,
        satellite_tables: Mapping[str, CycleTableSource] | None = None,
        gpsro_tables: CycleTableSource | None = None,
        satwnd_tables: CycleTableSource | None = None,
        prepbufr_tables: CycleTableSource | None = None,
    ) -> torch.Tensor:
        """Physical-space analyses ``(b, C, lat, lon)``, one per analysis time.

        Each analysis is the last frame of a ``time_length`` window fed with the
        observations of every frame. ``satellite_tables`` is keyed by sensor; a stream
        or sensor not passed contributes no observations.
        """
        times = pd.DatetimeIndex(analysis_times)
        # A None source reads the on-disk archive; a stream not passed is empty instead.
        loader = build_obs_loader(
            self.loop.obs_config,
            training=False,
            extra_filters=self.extra_filters,
            satellite_table_source=satellite_tables or {},
            gpsro_table_source=gpsro_tables or {},
            satwnd_table_source=satwnd_tables or {},
            prepbufr_table_source=prepbufr_tables or {},
        )
        frame_times = [self.frame_times(t) for t in times]
        loaded = _run_coroutine(self._sel_all(loader, frame_times))
        cftimes = [
            [healda.utils.datetime.as_cftime(t) for t in window]
            for window in frame_times
        ]
        frames = [
            [{"obs_v2": obs["obs_v2"][i]} for i in range(self.loop.time_length)]
            for obs in loaded
        ]
        batch = self.transform.transform_observations(cftimes, frames)
        return self.to_physical(self._forward(batch))

    @staticmethod
    async def _sel_all(loader, frame_times):
        return await asyncio.gather(*(loader.sel_time(w) for w in frame_times))

    @torch.inference_mode()
    def _forward(self, batch) -> torch.Tensor:
        """Model-space analysis ``(b, C, lat, lon)``, the last frame, for one batch."""
        second_of_day = batch["second_of_day"].to(self.device)
        b, t = second_of_day.shape
        obs_tensors, lengths = batch["unified_obs"]
        unified_obs = self.transform._device_transform_unified_obs(
            obs_tensors, lengths, self.device
        )
        # The lat/lon model rebuilds its condition; this is the placeholder training feeds.
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
                day_of_year=batch["day_of_year"].to(self.device),
                unified_obs=unified_obs,
                timestamp=batch["timestamp"].to(self.device),
                is_causal=self.loop.dit_temporal_attention_causal,
            )
        return out.out[:, :, -1].float()

    def to_physical(self, prediction: torch.Tensor) -> torch.Tensor:
        """Model space ``(b, C, lat, lon)`` -> physical units, fills re-imposed."""
        return to_physical(
            prediction, self.loop, restore_fill=self.loop.use_masked_domain_loss
        )


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


def read_training_loop(checkpoint, loop_name: str | None = None) -> TrainingLoop:
    """The ``TrainingLoop`` a checkpoint was written by.

    Read from the checkpoint's ``loop.json``; ``loop_name`` names a ``LOOPS`` preset
    to fall back to for checkpoints written without one.
    """
    with Checkpoint(checkpoint) as ckpt:
        loop_json = ckpt.read_loop_json()
    if loop_json is None:
        if loop_name is None:
            raise ValueError(
                "checkpoint has no loop.json; pass loop_name (a healda-train preset) "
                "or use a checkpoint written by healda-train"
            )
        return LOOPS[loop_name]
    return TrainingLoop.loads(loop_json)


def load_da_model(
    checkpoint,
    device,
    *,
    loop_name: str | None = None,
    extra_filters: ObsFilters | None = None,
    pressure_height_fill: str | None = PRESSURE_HEIGHT_FILL,
    fp32_output_head: bool = True,
) -> DAModel:
    """Rebuild the trained network and its observation pipeline from a checkpoint.

    Only the NNJA lat/lon recipe is supported: a 0.25 degree decode fed by NNJA
    satellite and NNJA conventional observations, the streams ``run_analysis`` takes as
    tables. ``checkpoint`` is a ``.checkpoint`` zip written by ``healda-train``;
    ``loop_name`` is the preset to fall back to when it carries no ``loop.json``.
    ``extra_filters`` defaults to ``inference_filters()``, as the inference CLI does. Provider
    flags act only where a table carries them; ``e2s_nnja`` tables carry none.
    ``pressure_height_fill`` replaces the recipe's (None: no fill), and
    ``fp32_output_head`` keeps the decoder output in fp32, both as the CLI defaults.
    Building the recipe fetches ERA5 statics into the healda cache on first use.
    """
    device = torch.device(device)
    loop = read_training_loop(checkpoint, loop_name)
    loop.obs_config = dataclasses.replace(
        loop.obs_config, nnja_pressure_height_fill=pressure_height_fill
    )
    obs = loop.obs_config
    if not (loop.latlon_decode and obs.use_nnja_sat and obs.use_nnja_conv):
        raise ValueError(
            "load_da_model supports only the NNJA lat/lon recipe (latlon_decode with "
            "use_nnja_sat and use_nnja_conv)"
        )
    loop.latlon_decode = dataclasses.replace(
        loop.latlon_decode, fp32_output=fp32_output_head
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
        features_v2=options.features_v2,
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

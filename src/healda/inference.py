# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyses from a trained HealDA checkpoint and observation tables.

``load_analysis_model`` rebuilds the network the way the run trained it, from the
``loop.json`` the checkpoint carries; ``AnalysisModel.analyze`` runs the recipe's own
observation loaders and transform on in-memory cycle tables and returns physical
fields. Downstream wrappers (Earth2Studio) build on this and add nothing to the
numerics.
"""

from __future__ import annotations

import asyncio
import dataclasses
import zipfile
from typing import TYPE_CHECKING

import pandas as pd
import torch

import healda.utils.datetime
from healda.datasets.da import state_masks, state_transforms
from healda.datasets.da.tasks import build_conventional_loader, get_sensors_for_config
from healda.observations.loaders import combined
from healda.observations.loaders.nnja_base import CycleTableSource

# The network and the transform pull in Triton; keep them out of the module import so
# the loader and denormalization paths stay usable (and testable) without a GPU stack.
if TYPE_CHECKING:
    from healda.cli.train import TrainingLoop
    from healda.datasets.da.transform import TransformV2


class RebasedConventionalLoader:
    """The conventional loader alone, emitting the NNJA combined vocabulary.

    ``CombinedObsLoader`` rebases conv rows while merging satellites in; with no
    satellite loader the rebase still has to happen or the rows sort under a foreign
    sensor id and land in empty buckets.
    """

    def __init__(self, conventional):
        if list(conventional.sensors) != [combined.CONV_SENSOR]:
            raise ValueError(
                f"expected a loader carrying exactly [{combined.CONV_SENSOR!r}], got "
                f"{conventional.sensors}"
            )
        self.conventional = conventional
        self.sensors = [combined.CONV_SENSOR]

    async def sel_time(self, times: pd.DatetimeIndex):
        loaded = await self.conventional.sel_time(times)
        return {"obs_v2": [combined.rebase_conventional(t) for t in loaded["obs_v2"]]}


@dataclasses.dataclass
class AnalysisModel:
    net: torch.nn.Module
    loop: TrainingLoop
    transform: TransformV2
    device: torch.device

    @property
    def channels(self) -> list[str]:
        return list(self.loop.batch_info.channels)

    @property
    def time_length(self) -> int:
        return self.loop.time_length

    @property
    def time_step(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.loop.time_step)

    def frame_times(self, analysis_time) -> pd.DatetimeIndex:
        """The ``time_length`` frames ending at ``analysis_time``."""
        return pd.date_range(
            end=pd.Timestamp(analysis_time),
            periods=self.time_length,
            freq=self.time_step,
        )

    def observation_window(self, analysis_time) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Span of observation times one analysis reads, ``[start, end]``."""
        frames = self.frame_times(analysis_time)
        obs = self.loop.obs_config
        return (
            frames[0] + pd.Timedelta(hours=obs.context_start),
            frames[-1] + pd.Timedelta(hours=obs.context_end),
        )

    def build_loader(
        self,
        *,
        gpsro_tables: CycleTableSource | None = None,
        satwnd_tables: CycleTableSource | None = None,
    ) -> RebasedConventionalLoader:
        conventional = build_conventional_loader(
            self.loop.obs_config,
            training=False,
            gpsro_table_source=gpsro_tables,
            satwnd_table_source=satwnd_tables,
        )
        return RebasedConventionalLoader(conventional)

    def analyze(
        self,
        analysis_times,
        *,
        gpsro_tables: CycleTableSource | None = None,
        satwnd_tables: CycleTableSource | None = None,
    ) -> torch.Tensor:
        """Physical-space analyses ``(b, C, *spatial)``, one per analysis time.

        Each analysis is the last frame of a ``time_length`` window fed with the
        observations of every frame; sensors with no rows are legal and empty.
        """
        times = pd.DatetimeIndex(analysis_times)
        if len(times) == 0:
            raise ValueError("analyze needs at least one analysis time")
        loader = self.build_loader(
            gpsro_tables=gpsro_tables, satwnd_tables=satwnd_tables
        )
        frame_times = [self.frame_times(t) for t in times]
        loaded = asyncio.run(self._sel_all(loader, frame_times))
        cftimes = [
            [healda.utils.datetime.as_cftime(t) for t in window]
            for window in frame_times
        ]
        frames = [
            [{"obs_v2": obs["obs_v2"][i]} for i in range(self.time_length)]
            for obs in loaded
        ]
        batch = self.transform.transform_observations(cftimes, frames)
        prediction = self._forward(batch)
        return self.to_physical(prediction[:, :, -1])

    @staticmethod
    async def _sel_all(loader, frame_times):
        return await asyncio.gather(*(loader.sel_time(w) for w in frame_times))

    @torch.inference_mode()
    def _forward(self, batch) -> torch.Tensor:
        """Model-space output ``(b, C, T, *spatial)`` for one transformed batch."""
        second_of_day = batch["second_of_day"].to(self.device)
        b, t = second_of_day.shape
        obs_tensors, lengths = batch["unified_obs"]
        unified_obs = self.transform._device_transform_unified_obs(
            obs_tensors, lengths, self.device
        )
        # The latlon recipe ignores the condition and the HPX recipes here are obs-only;
        # this is the placeholder the training dataset feeds.
        condition = torch.zeros(b, 1, t, 1, device=self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and self.loop.bf16,
        ):
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
        return out.out.float()

    def to_physical(self, prediction: torch.Tensor) -> torch.Tensor:
        """Model space ``(b, C, *spatial)`` -> physical units, fills re-imposed."""
        stats = self.loop.batch_info.normalization
        shape = (1, -1) + (1,) * (prediction.ndim - 2)
        center = torch.as_tensor(
            stats.center, dtype=prediction.dtype, device=prediction.device
        ).view(shape)
        scales = torch.as_tensor(
            stats.scales, dtype=prediction.dtype, device=prediction.device
        ).view(shape)
        physical = prediction * scales + center
        physical = state_transforms.to_physical_space(
            physical, self.channels, self.loop.variable_config.name, channel_axis=1
        )
        if self.loop.use_masked_domain_loss:
            if self.loop.latlon_decode:
                mask_file, pixel_order = state_masks.LATLON_025, None
            else:
                mask_file = state_masks.HPX64
                pixel_order = self.loop._pipeline_pixel_order
            physical = state_transforms.restore_fill(
                physical, self.channels, mask_file, pixel_order, channel_axis=1
            )
        return physical


def read_training_loop(checkpoint, loop_name: str | None = None) -> TrainingLoop:
    """The ``TrainingLoop`` a checkpoint was written by.

    Read from the checkpoint's ``loop.json``; ``loop_name`` names a ``LOOPS`` preset
    to fall back to for checkpoints written without one.
    """
    with zipfile.ZipFile(checkpoint, "r") as archive:
        has_loop = "loop.json" in archive.namelist()
        loop_json = archive.read("loop.json") if has_loop else None
    if loop_json is None:
        if loop_name is None:
            raise ValueError(
                "checkpoint has no loop.json; pass loop_name (a healda-train preset) "
                "or use a checkpoint written by healda-train"
            )
        from healda.cli.train import LOOPS

        return LOOPS[loop_name]
    from healda.cli.train import TrainingLoop

    return TrainingLoop.loads(loop_json.decode())


def single_process_loop(
    loop: TrainingLoop, *, compile_dit: bool = True
) -> TrainingLoop:
    """The recipe as one unsharded process runs it.

    ``time_parallel`` collapses to 1 and FSDP is off. The fused FiLM tokenizer kernel
    does not survive the whole observation window on one rank, so the weight-identical
    pure-torch tokenizer is selected. ``compile_dit`` stays on by default: eager
    execution of the trained kernels was measured to bias the geopotential column.
    """
    embedder = loop.sensor_embedder_config
    if embedder is not None:
        embedder = dataclasses.replace(embedder, use_fused_mlp=False)
    return dataclasses.replace(
        loop,
        time_parallel=1,
        fsdp=False,
        compile_dit=compile_dit,
        sensor_embedder_config=embedder,
    )


def load_analysis_model(
    checkpoint,
    device="cuda",
    *,
    loop_name: str | None = None,
    compile_dit: bool = True,
) -> AnalysisModel:
    """Rebuild the trained network and its observation pipeline from a checkpoint.

    ``checkpoint`` is a ``.checkpoint`` zip written by ``healda-train``; ``loop_name``
    is the preset to fall back to when it carries no ``loop.json``. Building the
    lat/lon recipe fetches ERA5 statics into the healda cache on first use.
    """
    import healda.models
    from healda.datasets.da.transform import TransformV2
    from healda.training.checkpoint import Checkpoint

    device = torch.device(device)
    loop = single_process_loop(
        read_training_loop(checkpoint, loop_name), compile_dit=compile_dit
    )
    if not loop.obs_config.use_obs:
        raise ValueError("the checkpoint's recipe does not ingest observations")
    net = loop.get_network()
    with Checkpoint(checkpoint) as ckpt:
        net = ckpt.read_model(net=net, map_location="cpu")
    net.eval().requires_grad_(False)
    net.to(device)

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
    return AnalysisModel(net=net, loop=loop, transform=transform, device=device)

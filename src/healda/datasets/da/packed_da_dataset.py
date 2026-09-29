# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed-store DA dataset: target (+ optional background, + optional obs) at t.

``PackedDADataset`` is the general target(+optional gridded inputs, +optional
obs) dataset; its static condition (orog + land fraction) is read from the target
store. The task configs (``TrainingTaskConfig`` / ``TASK_CONFIGS``) and the
``build_training_dataset`` factory live in ``tasks``.
"""

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import xarray as xr

import healda.utils.datetime
from healda.datasets.base import NormalizationStats
from healda.datasets.da.state_stats import (
    load_channel_stats,
    make_batch_info,
)
from healda.datasets.da.tasks import (
    TrainingTaskConfig,
    build_obs_loader,
    get_sensors_for_config,
)
from healda.observations.coverage import obs_coverage
from healda.observations.loaders.ufs import get_channel_table
from healda.datasets.da.packed_state import PackedStateReader
from healda.datasets.da.transform import (
    TransformOptions,
    TransformV2,
    reorder_from_nest,
    zscore_static,
)
from healda.config.variables import VARIABLE_CONFIGS, encode_channels
from healda.datasets.merged_dataset import get_flat_indexer
from healda.datasets.splits import Split, YearHoldoutSplit


def _shift_times(times, offset_hours: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times) + pd.Timedelta(hours=offset_hours)


@dataclass
class _AlignedSource:
    """A store reader read at ``query_time + offset_hours`` (target is offset 0)."""

    reader: PackedStateReader
    offset_hours: int = 0
    _store_idx: np.ndarray | None = field(default=None, init=False)

    def valid_mask(self, query_times) -> np.ndarray:
        return self.reader.valid_store_mask(
            _shift_times(query_times, self.offset_hours)
        )

    def bind(self, query_times) -> None:
        self._store_idx = self.reader.index_of(
            _shift_times(query_times, self.offset_hours)
        )

    def read_at(self, coords) -> np.ndarray:
        return self.reader.read(self._store_idx[np.asarray(coords["time"])])


def _concat_normalizations(stats: list[NormalizationStats]) -> NormalizationStats:
    return NormalizationStats(
        center=np.concatenate([np.asarray(s.center) for s in stats]),
        scales=np.concatenate([np.asarray(s.scales) for s in stats]),
    )


class PackedDADataset(torch.utils.data.Dataset):
    """Target (+ optional gridded inputs, + optional obs) from packed stores.

    Gridded inputs are read at target time plus each input's
    ``time_offset_hours`` and concatenated into the ``background`` tensor.

    Inputs can live at different lags, so readers are aligned onto a common query
    timeline. Stores also have dropout/invalid frames, so the usable timeline is the
    intersection of every source's validity at its own offset.
    """

    def __init__(
        self,
        config: TrainingTaskConfig,
        *,
        split: Split = "",
        time_length: int = 1,
        frame_step: int = 1,
        model_rank: int = 0,
        model_world_size: int = 1,
        obs_config=None,
        transform_options: TransformOptions,
        training: bool = False,
    ):
        self.config = config
        self.variable_config = VARIABLE_CONFIGS[config.target.variable_config]
        self.channels = encode_channels(self.variable_config)
        self.input_variable_configs = [
            VARIABLE_CONFIGS[state.variable_config] for state in config.inputs
        ]
        self.input_channels = [
            encode_channels(variable_config)
            for variable_config in self.input_variable_configs
        ]
        self.background_channels = [
            channel for channels in self.input_channels for channel in channels
        ]

        # Share one reader across sources of the same store at different offsets:
        # reads never mutate the reader.
        reader_cache: dict = {}

        def make_reader(entry, channels, source) -> PackedStateReader:
            key = (id(entry), tuple(channels), source)
            if key not in reader_cache:
                reader_cache[key] = PackedStateReader(entry, channels, source=source)
            return reader_cache[key]

        self._target = _AlignedSource(
            make_reader(config.target.entry, self.channels, config.target.source)
        )
        self._inputs = tuple(
            _AlignedSource(
                make_reader(state.entry, channels, state.source),
                state.time_offset_hours,
            )
            for state, channels in zip(config.inputs, self.input_channels, strict=True)
        )
        self._sources = (self._target, *self._inputs)

        use_obs = obs_config is not None and obs_config.use_obs
        self._use_obs = use_obs
        self._obs_config = obs_config
        times = self._valid_times(split)
        self._query_times = times
        for source in self._sources:
            source.bind(times)

        self.batch_info = make_batch_info(
            self.variable_config,
            config.target.stats_file,
            config.target.source,
        )
        target_norm = self.batch_info.normalization
        input_norms = [
            load_channel_stats(
                state.stats_file,
                channels,
                state.source,
            )
            for state, channels in zip(config.inputs, self.input_channels, strict=True)
        ]
        background_norm = (
            _concat_normalizations(input_norms) if input_norms else target_norm
        )

        self._obs_loader = (
            build_obs_loader(obs_config, training=training) if use_obs else None
        )
        self.channel_table = get_channel_table()
        self.npix = 49152
        self.time_length = time_length
        # Indexer positions address the query timeline; each source maps them to
        # its own store rows (see _AlignedSource.bind/read_at).
        self._indexer = get_flat_indexer(
            xr.Dataset(coords={"time": self._query_times}),
            [],
            "time",
            time_length=time_length,
            frame_step=frame_step,
            model_rank=model_rank,
            model_world_size=model_world_size,
        )
        self.transform = TransformV2(
            self.variable_config,
            hpx_level=transform_options.observation_hpx_level,
            sensors=get_sensors_for_config(obs_config) if use_obs else [],
            use_nnja_sat=use_obs and obs_config.use_nnja_sat,
            target_normalization=target_norm,
            background_normalization=background_norm,
            static_condition=self._build_static_condition(
                transform_options.pixel_order
            ),
            features_v2=transform_options.features_v2,
            attention_prepack=transform_options.attention_prepack,
            build_attention_group_map=(transform_options.build_attention_group_map),
            pixel_order=transform_options.pixel_order,
        )

    def _valid_times(self, split: Split) -> pd.DatetimeIndex:
        # Target clock is the query timeline.
        times = self._target.reader.times
        times = times[YearHoldoutSplit().mask(times, split)]
        # Obs out of range silently return empty, so clamp to their coverage.
        if self._use_obs:
            coverage = obs_coverage(self._obs_config)
            times = times[
                (times >= pd.Timestamp(coverage.start))
                & (times <= pd.Timestamp(coverage.end))
            ]
        # Drop each source's invalid frames at its own offset.
        for source in self._sources:
            times = times[source.valid_mask(times)]
        if len(times) == 0:
            raise RuntimeError(f"{self.config.name}: no valid times for {split=}")
        return times

    def _build_static_condition(self, pixel_order) -> torch.Tensor:
        # orog + land fraction from the target store, each z-scored on the spot.
        names = list(self.variable_config.variables_static)
        raw = self._target.reader.static(names)  # (C, X)
        arrays = [zscore_static(raw[i]) for i in range(len(names))]
        condition = (
            torch.stack(arrays).float().unsqueeze(1).unsqueeze(0)
        )  # (1, C, 1, X)
        return reorder_from_nest(condition, pixel_order)

    def __len__(self) -> int:
        return len(self._indexer)

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._query_times

    def _get_times(self, i):
        coords = self._indexer[i]
        return pd.to_datetime(self._query_times[np.asarray(coords["time"])])

    def _get_obs(self, i):
        if self._obs_loader is None:
            return None
        return asyncio.run(self._obs_loader.sel_time(self._get_times(i)))["obs_v2"]

    def get(self, i):
        frame_times = self._get_times(i)
        times = [healda.utils.datetime.as_cftime(t) for t in frame_times]
        coords = self._indexer[i]
        state = self._target.read_at(coords)
        input_blocks = [source.read_at(coords) for source in self._inputs]
        background = np.concatenate(input_blocks, axis=1) if input_blocks else None
        obs = self._get_obs(i)
        objs = []
        for t in range(state.shape[0]):
            frame = {"state": state[t]}
            if background is not None:
                frame["background"] = background[t]
            if obs is not None:
                frame["obs_v2"] = obs[t]
            objs.append(frame)
        return times, objs

    def __getitems__(self, indexes):
        times, objs = zip(*[self.get(i) for i in indexes])
        return self.transform.transform(times, objs)

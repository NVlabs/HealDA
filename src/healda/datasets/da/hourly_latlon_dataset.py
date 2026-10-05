# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load 0.25° ERA5 from hourly-file packs. CPU just reads. GPU z-scores the lat/lon target.

Pack files are hourly on disk. This dataset samples 6-hourly synoptic hours
(00/06/12/18). Hourly sampling is a separate axis and is not enabled here.

The DiT condition is statics, not coarsened ERA5: this transform does not regrid
weather to HPX. ``condition`` is a dummy (B, 1, T, 1) so the step loop has a
tensor; ``Hpx256LatlonModel`` ignores it and concatenates IFS statics + calendar.
"""

from __future__ import annotations

import re
import asyncio
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

import healda.config.environment as config
import healda.utils.datetime
from healda.datasets.base import VariableConfig
from healda.datasets.da.tasks import (
    build_obs_loader,
    get_sensors_for_config,
)
from healda.observations.coverage import obs_coverage
from healda.observations.system import ObsFilters
from healda.datasets.da.state_stats import make_batch_info
from healda.datasets.da import state_transforms
from healda.datasets.da.state_masks import CHANNEL_DOMAIN_FILL
from healda.datasets.da.transform import (
    TransformOptions,
    TransformV2,
    _compute_day_of_year,
    _compute_second_of_day,
    _compute_timestamp,
)
from healda.config.variables import VARIABLE_CONFIGS, encode_channels
from healda.observations.loaders.threads import thread_pool
from healda.datasets.era5_hourly_h5 import Era5HourlyH5
from healda.datasets.merged_dataset import get_flat_indexer
from healda.datasets.splits import Split, YearHoldoutSplit
from healda.config.channels import LEVELS
from healda.config.channels import STANDARD_TO_QUERY

NLAT, NLON = 721, 1440
SYNOPTIC_HOURS = (0, 6, 12, 18)

HOURLY_PACK_NAME = {
    **STANDARD_TO_QUERY,
    **{f"W{level}": f"w{level}" for level in LEVELS},
    **{f"{variable}10": f"{variable.lower()}10" for variable in ("U", "V", "T", "Z")},
    "tcw": "tcw",
    "d2m": "d2m",
    "skt": "skt",
    "tcc": "tcc",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "stl1": "stl1",
    "stl2": "stl2",
    "swvl1": "swvl1",
    "swvl2": "swvl2",
    "sd": "sd",
}


_HOURLY_PACK_ENV = ("ERA5_HOURLY_73CH",)

# One directory per entry under ERA5_HOURLY_EXT_ROOT/<variable>/<year>.h5.
_EXT_VARIABLES = (
    "sst",
    "sic",
    "d2m",
    "skt",
    "sp",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
    "sd",
    "tcw",
    "t10",
    "u10",
    "v10",
    "z10",
    "w",
)

_W_CHANNEL = re.compile(r"w\d+$")


def _ext_directory(pack_channel: str) -> str:
    return "w" if _W_CHANNEL.fullmatch(pack_channel) else pack_channel


def hourly_pack_roots(pack_names: Sequence[str] | None = None) -> list[str]:
    """Roots needed for ``pack_names`` (default: all ext variables).

    Both settings take a ``:``-separated list, and every root holding a pack is
    returned, not just the first: a variable's years may be split across them.
    """
    wanted_ext = set(
        _EXT_VARIABLES
        if pack_names is None
        else (_ext_directory(name) for name in pack_names)
    ) & set(_EXT_VARIABLES)
    missing = []
    present = []
    for name in _HOURLY_PACK_ENV:
        setting = getattr(config, name)
        roots = [
            root
            for root in config.path_list(setting)
            if Path(root, "metadata", "data.json").is_file()
        ]
        if roots:
            present.extend(roots)
        else:
            missing.append(f"{name}={setting or '<unset>'}")

    if wanted_ext:
        ext_roots = config.path_list(config.ERA5_HOURLY_EXT_ROOT)
        if not ext_roots:
            missing.append("ERA5_HOURLY_EXT_ROOT=<unset>")
        else:
            for variable in sorted(wanted_ext):
                found = [
                    str(Path(ext_root, variable))
                    for ext_root in ext_roots
                    if Path(ext_root, variable, "metadata", "data.json").is_file()
                ]
                if found:
                    present.extend(found)
                else:
                    missing.append(
                        f"{variable} under ERA5_HOURLY_EXT_ROOT="
                        f"{config.ERA5_HOURLY_EXT_ROOT}"
                    )

    if missing:
        raise FileNotFoundError(
            "ERA5 packs missing sidecar or unset: " + "; ".join(missing)
        )
    return present


def _pack_times(steps_by_year: dict[int, int], step: pd.Timedelta) -> pd.DatetimeIndex:
    """Every timestamp held by the selected packs."""
    parts = [
        pd.date_range(
            pd.Timestamp(year=year, month=1, day=1),
            periods=nsteps,
            freq=step,
        )
        for year, nsteps in sorted(steps_by_year.items())
    ]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def _synoptic(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return times[times.hour.isin(SYNOPTIC_HOURS)]


class HourlyLatlonTransform(torch.nn.Module):
    """CPU: identity read. GPU: z-scores 0.25° ERA5 into the loss target.

    This only builds the target; the DiT's own condition input is statics/calendar
    only, not a regridded copy of this data (see module docstring).
    """

    def __init__(
        self,
        channels: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        obs_transform: TransformV2 | None = None,
        config_name: str | None = None,
    ):
        super().__init__()
        self._obs = obs_transform
        self._config_name = config_name
        self.channels = list(channels)
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        )
        fills = torch.zeros(len(channels))
        for name, value in CHANNEL_DOMAIN_FILL.items():
            if name in channels:
                fills[channels.index(name)] = value
        self.register_buffer("_domain_fill", fills.view(1, -1, 1, 1))

    def device_transform(self, batch, device):
        self.to(device)
        latlon = batch["latlon"].to(device, non_blocking=True)
        b, t, c, h, w = latlon.shape
        flat = latlon.reshape(b * t, c, h, w)
        fill = self._domain_fill.to(dtype=latlon.dtype)
        flat = torch.where(torch.isfinite(flat), flat, fill.expand_as(flat))
        # Before the z-score: the statistics rows are fitted to the transformed field.
        if self._config_name is not None:
            flat = state_transforms.to_model_space(
                flat, self.channels, self._config_name, channel_axis=1
            )
        z = (flat - self.mean) / self.std
        target = z.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        condition = torch.zeros(b, 1, t, 1, device=device, dtype=z.dtype)
        out = {
            "target": target,
            "condition": condition,
            "second_of_day": batch["second_of_day"].to(device, non_blocking=True),
            "day_of_year": batch["day_of_year"].to(device, non_blocking=True),
            "timestamp": batch["timestamp"].to(device, non_blocking=True),
            "labels": batch["labels"].to(device, non_blocking=True),
        }
        if self._obs is not None and "unified_obs" in batch:
            obs_tensors, lengths = batch["unified_obs"]
            out["unified_obs"] = self._obs._device_transform_unified_obs(
                obs_tensors, lengths, device
            )
        return out


class HourlyLatlonDataset(torch.utils.data.Dataset):
    """6-hourly synoptic 0.25° ERA5. CPU transform is a load."""

    def __init__(
        self,
        *,
        variable_config: VariableConfig | str = "era5_99ch",
        stats_file: str = "era5_13_levels_stats.csv",
        source: str = "era5",
        split: Split = "train",
        time_length: int = 1,
        frame_step: int = 1,
        model_rank: int = 0,
        model_world_size: int = 1,
        roots: Sequence[str] | None = None,
        synoptic_only: bool = True,
        obs_config=None,
        transform_options: TransformOptions | None = None,
        training: bool = False,
        extra_obs_filters: ObsFilters = ObsFilters(),
    ):
        if isinstance(variable_config, str):
            variable_config = VARIABLE_CONFIGS[variable_config]
        self.variable_config = variable_config
        self.channels = encode_channels(variable_config)
        self.batch_info = make_batch_info(variable_config, stats_file, source)
        pack_names = [HOURLY_PACK_NAME[name] for name in self.channels]
        roots = list(roots) if roots is not None else hourly_pack_roots(pack_names)
        self._reader = Era5HourlyH5(roots, pack_names)
        times = _pack_times(self._reader.steps_by_year, self._reader.step)
        if synoptic_only:
            times = _synoptic(times)
        times = times[YearHoldoutSplit().mask(times, split)]
        use_obs = obs_config is not None and obs_config.use_obs
        if use_obs:
            coverage = obs_coverage(obs_config)
            times = times[
                (times >= pd.Timestamp(coverage.start))
                & (times <= pd.Timestamp(coverage.end))
            ]
        if len(times) == 0:
            raise RuntimeError(f"hourly latlon: no times for {split=}")
        self._query_times = times
        # Frames per window; maps a window index to its final timestamp.
        self.time_length = time_length
        self._indexer = get_flat_indexer(
            xr.Dataset(coords={"time": times}),
            [],
            "time",
            time_length=time_length,
            frame_step=frame_step,
            model_rank=model_rank,
            model_world_size=model_world_size,
        )
        self._obs_loader = (
            build_obs_loader(
                obs_config, training=training, extra_filters=extra_obs_filters
            )
            if use_obs
            else None
        )
        self._obs_transform = None
        if use_obs:
            if transform_options is None:
                raise ValueError("obs require transform_options")
            self._obs_transform = TransformV2(
                self.variable_config,
                hpx_level=transform_options.observation_hpx_level,
                sensors=get_sensors_for_config(obs_config),
                use_nnja_sat=obs_config.use_nnja_sat,
                target_normalization=self.batch_info.normalization,
                features_v2=transform_options.features_v2,
                attention_prepack=transform_options.attention_prepack,
                build_attention_group_map=(transform_options.build_attention_group_map),
                pixel_order=transform_options.pixel_order,
            )
        stats = self.batch_info.normalization
        self.transform = HourlyLatlonTransform(
            self.channels,
            np.asarray(stats.center, dtype=np.float32),
            np.asarray(stats.scales, dtype=np.float32),
            obs_transform=self._obs_transform,
            config_name=self.variable_config.name,
        )

    def __len__(self):
        return len(self._indexer)

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._query_times

    def _frame_times(self, i) -> pd.DatetimeIndex:
        coords = self._indexer[i]
        return pd.to_datetime(self._query_times[np.asarray(coords["time"])])

    async def _load_state_and_obs(self, all_times, frame_times):
        loop = asyncio.get_running_loop()
        loaded, *obs_results = await asyncio.gather(
            loop.run_in_executor(thread_pool(), self._reader.read, all_times),
            *(self._obs_loader.sel_time(window) for window in frame_times),
        )
        return loaded, obs_results

    def __getitems__(self, indexes):
        frame_times = [self._frame_times(i) for i in indexes]
        all_times = pd.DatetimeIndex([t for window in frame_times for t in window])
        n_t = len(frame_times[0])
        if self._obs_loader is None:
            loaded = self._reader.read(all_times)
            obs_results = None
        else:
            loaded, obs_results = asyncio.run(
                self._load_state_and_obs(all_times, frame_times)
            )
        latlon = torch.from_numpy(loaded.reshape(len(indexes), n_t, *loaded.shape[1:]))
        cftimes = [
            [healda.utils.datetime.as_cftime(t) for t in window]
            for window in frame_times
        ]

        def _apply(func):
            return torch.from_numpy(np.vectorize(func)(cftimes)).float()

        out = {
            "latlon": latlon,
            "second_of_day": _apply(_compute_second_of_day),
            "day_of_year": _apply(_compute_day_of_year),
            "timestamp": torch.from_numpy(
                np.vectorize(_compute_timestamp)(cftimes)
            ).long(),
            "labels": torch.empty(len(indexes), 0),
        }
        if obs_results is not None:
            frames = [
                [{"obs_v2": obs["obs_v2"][t]} for t in range(n_t)]
                for obs in obs_results
            ]
            out["unified_obs"] = self._obs_transform._process_obs(cftimes, frames)
        return out

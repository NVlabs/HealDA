# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import numpy as np
import pandas as pd
import pathlib

from healda.datasets.base import BatchInfo, NormalizationStats, TimeUnit, VariableConfig


from healda.config.variables import (
    NO_LEVEL,
    VARIABLE_CONFIGS,
    _encode_channel,
    channel_index,
    encode_channels,
    resolve_store_channels,
)

__all__ = ["PredictionNormalizations", "get_batch_info", "load_channel_stats"]


@dataclasses.dataclass(frozen=True)
class PredictionNormalizations:
    """Per-channel stats the diffusion step uses for the target, the background,
    and (for residual prediction) the target-minus-background increment."""

    target: NormalizationStats
    background: NormalizationStats
    residual: NormalizationStats | None


def get_batch_info(
    config: VariableConfig,
    time_step: int = 1,
    time_unit: TimeUnit = TimeUnit.HOUR,
) -> BatchInfo:
    return BatchInfo(
        channels=encode_channels(config),
        scales=_get_std(config),
        center=_get_mean(config),
        time_step=time_step,
        time_unit=time_unit,
    )


def _get_mean(config: VariableConfig) -> np.ndarray:
    mean = _get_nearest_stats(config)["mean"].values
    return mean


def _get_std(config: VariableConfig) -> np.ndarray:
    std = _get_nearest_stats(config)["std"].values
    return std


# Per-config state normalization CSVs, indexed by (variable, level). Live in the
# normalizations/ dir alongside the obs stats convention.
NORMALIZATIONS_DIR = pathlib.Path(__file__).parent / "state_normalizations"
# Extra directories searched after the packaged one, for stats that ship elsewhere.
EXTRA_STATS_DIRS: list[pathlib.Path] = []


def resolve_stats_file(file_name: str) -> pathlib.Path:
    """First match on the search path; the error lists where we looked."""
    for base in [NORMALIZATIONS_DIR, *EXTRA_STATS_DIRS]:
        candidate = base / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{file_name} not found in "
        f"{[str(b) for b in [NORMALIZATIONS_DIR, *EXTRA_STATS_DIRS]]}"
    )


_ERA5_STATE_STATS = "era5_13_levels_stats.csv"
# Every registered config shares one stats file. A config registered after import
# must add its own entry, or _load_raw_stats raises naming it.
STATE_STATS_FILES = {name: _ERA5_STATE_STATS for name in VARIABLE_CONFIGS}


def _load_raw_stats(config: VariableConfig) -> pd.DataFrame:
    file_name = STATE_STATS_FILES.get(config.name)
    if file_name is None:
        raise ValueError(f"No state normalization stats registered for: {config.name}")
    return pd.read_csv(resolve_stats_file(file_name)).set_index(["variable", "level"])


# def get_sst_stats(config: VariableConfig = _default_config):
#     df = _load_raw_stats(config)
#     row = df.loc[("sst", NO_LEVEL)]
#     return row["mean"].item(), row["std"].item()


def _get_nearest_stats(config: VariableConfig):
    # To handle float levels, gets nearest level
    raw = _load_raw_stats(config)
    idx = channel_index(config)

    mapped_idx = []
    for var, level in idx:
        if level != NO_LEVEL:
            available = raw.loc[var].index.values
            nearest = available[np.abs(available - level).argmin()]
            mapped_idx.append((var, nearest))
        else:
            mapped_idx.append((var, level))

    mapped_idx = pd.MultiIndex.from_tuples(mapped_idx, names=["variable", "level"])
    return raw.loc[mapped_idx]


def _read_stats_csv(stats_file: str) -> pd.DataFrame:
    raw = pd.read_csv(resolve_stats_file(stats_file))
    raw["channel"] = [
        _encode_channel((variable, int(level)))
        for variable, level in zip(raw["variable"], raw["level"])
    ]
    return raw.set_index("channel")


def _match_index_case(
    index: pd.Index, channels: list[str], stats_file: str
) -> list[str]:
    """Resolve each channel against ``index``, falling back to a case-insensitive match.

    The CSVs disagree on capitalisation: some spell 10 hPa `U10`, others `u10`, the same
    way the packed HPX64 store does. Exact match wins, so a consistent file is unaffected.
    """
    folded: dict[str, list[str]] = {}
    for name in index:
        folded.setdefault(name.lower(), []).append(name)

    resolved = []
    for channel in channels:
        if channel in index:
            resolved.append(channel)
            continue
        candidates = folded.get(channel.lower(), [])
        if len(candidates) != 1:
            raise KeyError(
                f"channel {channel!r} matches {len(candidates)} rows in {stats_file}"
                f" case-insensitively{f' ({candidates})' if candidates else ''}"
            )
        resolved.append(candidates[0])
    return resolved


def load_channel_stats(
    stats_file: str,
    channels: list[str],
    source: str = "",
    overlay_file: str | None = None,
) -> NormalizationStats:
    """Load per-channel mean/std from a state-normalizations CSV.

    ``source`` maps canonical names to a store's names (e.g. ``q2m`` -> ``sh2``)
    before lookup. Stats are returned in the requested channel order.

    ``overlay_file`` replaces individual channels and must contain only those,
    so channels scored on a physical domain (sst over ocean) get domain stats
    while the rest stay identical to the base file.
    """
    raw = _read_stats_csv(stats_file)
    store_channels = _match_index_case(
        raw.index, resolve_store_channels(list(channels), source), stats_file
    )
    center = raw.loc[store_channels, "mean"].to_numpy().copy()
    scales = raw.loc[store_channels, "std"].to_numpy().copy()

    if overlay_file is not None:
        over = _read_stats_csv(overlay_file)
        unknown = sorted(set(over.index) - set(raw.index))
        if unknown:
            raise KeyError(
                f"{overlay_file} overlays channels absent from {stats_file}: "
                f"{unknown}. An overlay may only replace, never introduce."
            )
        position = {name: i for i, name in enumerate(store_channels)}
        for name, row in over.iterrows():
            i = position.get(name)
            if i is not None:
                center[i] = row["mean"]
                scales[i] = row["std"]

    return NormalizationStats(center=center, scales=scales)


def make_batch_info(
    variable_config: VariableConfig,
    stats_file: str,
    source: str = "",
    time_step: int = 6,
) -> BatchInfo:
    """BatchInfo for a variable config, normalized by a state-normalizations CSV.

    Single stats path shared by every dataset: encode the channels, read their
    per-channel mean/std from ``stats_file``, and carry them on the BatchInfo
    (``batch_info.normalization`` is what the transform applies).
    """
    channels = encode_channels(variable_config)
    stats = load_channel_stats(stats_file, channels, source)
    return BatchInfo(
        channels=channels,
        center=stats.center,
        scales=stats.scales,
        time_step=time_step,
        time_unit=TimeUnit.HOUR,
    )

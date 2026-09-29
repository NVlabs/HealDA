# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-channel (pressure-level / per-variable) loss weighting.

A ``ChannelWeightSpec`` weights each state channel in the training loss via an
upper-air pressure-level profile plus a surface weight. ``resolve`` expands it
into a mean-1 vector aligned to the channel list. To weight a run, set
``TrainingLoop.channel_weight_config`` to a name registered in ``CONFIGS``.

Every weight field has a scalar default and a ``*_by_var`` map of per-variable
overrides (e.g. ``base_weight=1.0, base_weight_by_var={"Z": 3.0}`` -> Z at 3x,
rest at 1x). Overrides naming a variable absent from the run's config are
ignored, so a spec may list weights for variables it does not train on.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from healda.datasets.base import VariableConfig
from healda.config.variables import (
    NO_LEVEL,
    channel_index,
    encode_channels,
)


@dataclasses.dataclass(frozen=True)
class LevelStep:
    """One pressure threshold in a ``StepProfile``: at or above ``hpa`` in altitude
    (pressure <= ``hpa``), multiply by ``scale`` (or its per-variable override)."""

    hpa: float
    scale: float = 1.0
    scale_by_var: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class StepProfile:
    """Per-variable base weight, attenuated by pressure-level steps.

    A variable's weight is its base weight, multiplied cumulatively by every
    step in ``steps`` whose ``hpa`` threshold the level is at or above in
    altitude. ``steps`` is applied in the given order, so listing a coarser
    threshold before a finer one (e.g. 200 hPa then 20 hPa) compounds them for
    a level above both -- adding another threshold in between is just another
    ``LevelStep``, never a new field.
    """

    base_weight: float = 1.0
    base_weight_by_var: Mapping[str, float] = dataclasses.field(default_factory=dict)
    steps: tuple[LevelStep, ...] = ()

    def weight(self, var: str, level_hpa: int) -> float:
        w = self.base_weight_by_var.get(var, self.base_weight)
        for step in self.steps:
            if level_hpa <= step.hpa:
                w *= step.scale_by_var.get(var, step.scale)
        return w


@dataclasses.dataclass(frozen=True)
class LogTaperProfile:
    """FuXi-style log decay from a per-variable peak down to a shared floor.

    A variable's weight is its peak weight at ``peak_hpa`` and decays
    log-linearly (a straight line on a log-y plot) to ``floor_weight`` at
    ``floor_hpa``, staying at ``floor_weight`` for every higher level (pressure
    <= ``floor_hpa``).
    """

    peak_weight_by_var: Mapping[str, float] = dataclasses.field(default_factory=dict)
    floor_weight: float = 0.02
    peak_hpa: float = 1000.0
    floor_hpa: float = 500.0

    def weight(self, var: str, level_hpa: int) -> float:
        if level_hpa <= self.floor_hpa:
            return self.floor_weight
        peak = self.peak_weight_by_var.get(var, self.floor_weight)
        frac = (level_hpa - self.floor_hpa) / (self.peak_hpa - self.floor_hpa)
        return self.floor_weight * (peak / self.floor_weight) ** frac


Profile = StepProfile | LogTaperProfile


@dataclasses.dataclass(frozen=True)
class ChannelWeightSpec:
    """Upper-air profile + surface weight. ``resolve`` -> mean-1 vector."""

    upper_air: Profile
    # Weight for surface channels (per-variable override via *_by_var).
    surface_weight: float = 1.0
    surface_weight_by_var: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def _channel_weight(self, name: str, var: str, level: int) -> float:
        if level == NO_LEVEL:
            return self.surface_weight_by_var.get(name, self.surface_weight)
        return self.upper_air.weight(var, level)

    def resolve(self, channels: list[str], config: VariableConfig) -> np.ndarray:
        """Per-channel weights aligned to ``channels``, normalized to mean 1.

        ``config`` provides the canonical name -> (variable, level) decode; a
        channel name absent from it is treated as a surface channel.
        """
        decode = dict(zip(encode_channels(config), channel_index(config), strict=True))
        w = np.array(
            [
                self._channel_weight(ch, *decode.get(ch, (ch, NO_LEVEL)))
                for ch in channels
            ],
            dtype=np.float64,
        )
        # A zero or non-finite weight silently removes a channel from the loss.
        if not len(w):
            raise ValueError("at least one channel is required")
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("channel weights must be finite and positive")
        return (w / w.mean()).astype(np.float32)


CONFIGS: dict[str, ChannelWeightSpec] = {
    "step": ChannelWeightSpec(
        upper_air=StepProfile(
            base_weight_by_var={"Z": 3.0, "T": 1.5},
            steps=(LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25}),),
        ),
    ),
    "step-cloudLand": ChannelWeightSpec(
        upper_air=StepProfile(
            base_weight_by_var={"Z": 3.0, "T": 1.5},
            steps=(LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25}),),
        ),
        surface_weight_by_var={
            "sst": 0.5,
            "sic": 0.5,
            "tcc": 0.5,
            "lcc": 0.25,
            "mcc": 0.25,
            "hcc": 0.25,
            "stl1": 1.0,
            "stl2": 0.5,
            "swvl1": 0.5,
            "swvl2": 0.5,
            "sd": 0.5,
        },
    ),
    "step-cloudLandW": ChannelWeightSpec(
        upper_air=StepProfile(
            base_weight_by_var={"Z": 3.0, "T": 1.5, "W": 0.1},
            steps=(
                LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25, "W": 0.25}),
            ),
        ),
        surface_weight_by_var={
            "sst": 0.5,
            "sic": 0.5,
            "tcc": 0.14,
            "lcc": 0.07,
            "mcc": 0.07,
            "hcc": 0.07,
            "stl1": 1.0,
            "stl2": 0.5,
            "swvl1": 0.5,
            "swvl2": 0.5,
            "sd": 0.5,
        },
    ),
    # 104ch: step-cloudLandW plus a 10 hPa step and tcw
    "step-cloudLandW-10hpa": ChannelWeightSpec(
        upper_air=StepProfile(
            base_weight_by_var={"Z": 3.0, "T": 1.5, "W": 0.1},
            steps=(
                LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25, "W": 0.25}),
                LevelStep(hpa=20.0, scale=0.1),
            ),
        ),
        surface_weight_by_var={
            "sst": 0.5,
            "sic": 0.5,
            "tcc": 0.14,
            "lcc": 0.07,
            "mcc": 0.07,
            "hcc": 0.07,
            "stl1": 1.0,
            "stl2": 0.5,
            "swvl1": 0.25,
            "swvl2": 0.25,
            "sd": 0.5,
            "tcw": 0.1,
        },
    ),
    # step-cloudLandW-10hpa with W x1.2 and sd x1.5. w and sd have significantly higher std
    # under the 0.25 deg stats/ domain only stats, meaning their share of loss is decreased.
    "step-cloudLandW-10hpa-wsdUp": ChannelWeightSpec(
        upper_air=StepProfile(
            base_weight_by_var={"Z": 3.0, "T": 1.5, "W": 0.12},
            steps=(
                LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25, "W": 0.25}),
                LevelStep(hpa=20.0, scale=0.1),
            ),
        ),
        surface_weight_by_var={
            "sst": 0.5,
            "sic": 0.5,
            "tcc": 0.14,
            "lcc": 0.07,
            "mcc": 0.07,
            "hcc": 0.07,
            "stl1": 1.0,
            "stl2": 0.5,
            "swvl1": 0.25,
            "swvl2": 0.25,
            "sd": 0.75,
            "tcw": 0.1,
        },
    ),
    "fuxi": ChannelWeightSpec(
        surface_weight=0.2,
        upper_air=LogTaperProfile(
            peak_weight_by_var={"T": 0.2, "Q": 0.1, "U": 0.1, "V": 0.1, "Z": 0.1},
        ),
    ),
}


def load_channel_weights(
    name: str, channels: list[str], config: VariableConfig
) -> np.ndarray:
    try:
        spec = CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"unknown channel weight config {name!r}; available: {sorted(CONFIGS)}"
        ) from None
    return spec.resolve(channels, config)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-channel loss-weight resolution: alignment, normalization, profile math."""

import numpy as np
import pytest

from healda.config.variables import VARIABLE_CONFIGS, encode_channels
from healda.training.channel_weights import (
    CONFIGS,
    ChannelWeightSpec,
    LevelStep,
    LogTaperProfile,
    StepProfile,
    load_channel_weights,
)

VAR_CONFIG = VARIABLE_CONFIGS["era5_74ch"]
CHANNELS = encode_channels(VAR_CONFIG)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_resolve_is_mean_one(name):
    w = load_channel_weights(name, CHANNELS, VAR_CONFIG)
    assert len(w) == len(CHANNELS)
    assert np.isfinite(w).all()
    assert float(np.mean(w)) == pytest.approx(1.0)


def test_aligns_by_name_not_position():
    # The weight a channel gets must not depend on its position in the list.
    spec = CONFIGS["step"]
    forward = dict(zip(CHANNELS, spec.resolve(CHANNELS, VAR_CONFIG), strict=True))
    rev = list(reversed(CHANNELS))
    reordered = dict(zip(rev, spec.resolve(rev, VAR_CONFIG), strict=True))
    for ch in CHANNELS:
        assert reordered[ch] == pytest.approx(forward[ch])


def test_dropping_a_variable_renormalizes_over_subset():
    kept = [c for c in CHANNELS if not c.startswith("Z")]
    w = load_channel_weights("step", kept, VAR_CONFIG)
    assert len(w) == len(CHANNELS) - len(VAR_CONFIG.levels)
    assert float(np.mean(w)) == pytest.approx(1.0)


def test_step_profile_math():
    p = StepProfile(
        base_weight_by_var={"Z": 3.0, "T": 1.5},
        steps=(LevelStep(hpa=200.0, scale=0.5, scale_by_var={"Q": 0.25}),),
    )
    assert p.weight("Z", 500) == pytest.approx(3.0)  # below upper level: base only
    assert p.weight("Z", 100) == pytest.approx(1.5)  # <= 200 hPa: base x 0.5
    assert p.weight("Q", 100) == pytest.approx(0.25)  # Q uses its own scale
    assert p.weight("U", 850) == pytest.approx(1.0)  # default base, no attenuation


def test_step_profile_steps_compound_in_order():
    # A level above two thresholds gets multiplied by both, in the order listed --
    # this is what lets a caller add a threshold in between without a new field.
    p = StepProfile(
        base_weight_by_var={"Z": 3.0},
        steps=(
            LevelStep(hpa=200.0, scale=0.5),
            LevelStep(hpa=20.0, scale=0.2),
        ),
    )
    assert p.weight("Z", 500) == pytest.approx(3.0)  # above both thresholds
    assert p.weight("Z", 100) == pytest.approx(1.5)  # <= 200 only: x0.5
    assert p.weight("Z", 10) == pytest.approx(0.3)  # <= both: x0.5 x0.2


def test_log_taper_endpoints():
    p = LogTaperProfile(peak_weight_by_var={"T": 0.2})
    assert p.weight("T", 1000) == pytest.approx(0.2)  # peak at peak_hpa
    assert p.weight("T", 500) == pytest.approx(0.02)  # floor at floor_hpa
    assert p.weight("T", 100) == pytest.approx(0.02)  # pinned above floor_hpa


def test_surface_override_applied_and_missing_ignored():
    # An override on a real surface channel (sst) changes its weight; an override
    # naming a channel absent from the config (bogus) is silently ignored.
    base = ChannelWeightSpec(upper_air=StepProfile())
    with_sst = ChannelWeightSpec(
        upper_air=StepProfile(), surface_weight_by_var={"sst": 0.1}
    )
    with_bogus = ChannelWeightSpec(
        upper_air=StepProfile(), surface_weight_by_var={"nope": 0.1}
    )
    i = CHANNELS.index("sst")
    assert (
        with_sst.resolve(CHANNELS, VAR_CONFIG)[i]
        < base.resolve(CHANNELS, VAR_CONFIG)[i]
    )
    np.testing.assert_allclose(
        base.resolve(CHANNELS, VAR_CONFIG), with_bogus.resolve(CHANNELS, VAR_CONFIG)
    )

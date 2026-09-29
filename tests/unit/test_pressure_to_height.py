# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""U.S. Standard Atmosphere 1976 inversion.

The hand-written version this replaces passed nothing below 32 km: 32-47 km was
coded isothermal though the layer lapses at +0.0028 K/m, the branch above 51 km
had a reversed exponent, and NaN fell through every mask into uninitialised
memory. Each test here is one of those failures.
"""

import numpy as np
import pytest

from healda.observations.loaders.nnja_satwnd import pressure_to_height_m

# USSA-1976 layer bases: pressure (hPa) and geopotential height (m).
BOUNDARIES = [
    (1013.25, 0.0),
    (226.3204, 11000.0),
    (54.7488, 20000.0),
    (8.68016, 32000.0),
    (1.10906, 47000.0),
    (0.669385, 51000.0),
    (0.0395642, 71000.0),
]


@pytest.mark.parametrize("pressure,height", BOUNDARIES)
def test_layer_boundaries_match_the_published_table(pressure, height):
    got = float(pressure_to_height_m(np.array([pressure]))[0])
    assert abs(got - height) < 1.0, f"{pressure} hPa -> {got} m, expected {height} m"


def test_continuous_across_every_layer_boundary():
    """A step here means two layers disagree -- the 658 m bug at 1.109 hPa."""
    for pressure, _ in BOUNDARIES[1:]:
        above = float(pressure_to_height_m(np.array([pressure * (1 + 1e-9)]))[0])
        below = float(pressure_to_height_m(np.array([pressure * (1 - 1e-9)]))[0])
        assert abs(above - below) < 1.0, f"{pressure} hPa: {above} vs {below}"


def test_height_rises_monotonically_as_pressure_falls():
    """The reversed exponent made height fall with pressure above 51 km."""
    # Up to sea-level pressure; above it the result is floored at 0.
    pressure = np.logspace(np.log10(1013.25), np.log10(0.01), 20000)
    height = pressure_to_height_m(pressure).astype(np.float64)
    assert np.all(np.diff(height) > 0.0)


def test_sub_sea_level_pressure_is_floored_not_negative():
    """AMVs reach 1032 hPa; a negative height would lose them downstream."""
    height = pressure_to_height_m(np.array([1013.25, 1032.1, 1100.0]))
    assert np.all(height >= 0.0)


def test_the_exact_top_of_domain_resolves():
    """The last layer owns its own top boundary; it used to fall through."""
    from healda.observations.loaders.nnja_satwnd import _USSA_TOP_P_PA

    got = float(pressure_to_height_m(np.array([_USSA_TOP_P_PA / 100.0]))[0])
    assert abs(got - 84852.0) < 1.0


def test_below_the_table_domain_is_nan():
    """Under 84.852 km the layers end; extrapolating would invent an atmosphere."""
    assert np.isnan(pressure_to_height_m(np.array([0.003, 1e-6]))).all()


def test_invalid_pressure_is_nan_not_a_plausible_height():
    """Clipping turned 0 and -10 hPa into 47,725 m, which reads as real."""
    out = pressure_to_height_m(np.array([np.nan, 0.0, -10.0, np.inf]))
    assert np.isnan(out).all()


def test_amv_range_is_sane():
    """125-1008 hPa is what the QC pressure floor admits."""
    height = pressure_to_height_m(np.linspace(125.0, 1008.0, 5000)).astype(np.float64)
    assert np.isfinite(height).all()
    assert height.min() > 0.0
    assert height.max() < 16000.0

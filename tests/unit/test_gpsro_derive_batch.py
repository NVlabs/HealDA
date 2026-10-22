# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from healda.observations.preprocessing.gpsro_pressure import (
    derive_level_coordinates,
    derive_level_coordinates_batch,
)


def _profile(rng, n):
    height = np.sort(rng.uniform(0.0, 60_000.0, n))
    refractivity = 300.0 * np.exp(-height / 7_000.0)
    return height, refractivity


def test_batch_matches_one_occultation_at_a_time():
    rng = np.random.default_rng(0)
    occultations = []
    for case in range(12):
        n_levels = int(rng.integers(5, 40))
        height, refractivity = _profile(rng, n_levels)
        radius = 6_371_000.0
        impact = radius + rng.uniform(0.0, 50_000.0, int(rng.integers(1, 30)))
        lat, elrc, geoid = (
            float(rng.uniform(-80, 80)),
            radius,
            float(rng.uniform(-50, 50)),
        )
        if case == 1:
            height, refractivity = height[:0], refractivity[:0]  # empty profile
        if case == 2:
            height, refractivity = height[:1], refractivity[:1]  # single level
        if case == 3:
            elrc = None  # no radius of curvature
        if case == 4:
            lat = None  # no latitude
        if case == 5:
            height[::3] = np.nan  # non-finite levels
            refractivity[1::4] = -1.0
            impact[0] = np.nan
        if case == 6:
            impact = impact[:0]  # no bending-angle levels
        if case == 7:
            geoid = None
        occultations.append((impact, height, refractivity, lat, elrc, geoid))

    expected = [
        derive_level_coordinates(
            i,
            h,
            n,
            latitude_degrees=lat,
            earth_radius_of_curvature_m=r,
            geoid_undulation_m=g,
        )
        for i, h, n, lat, r, g in occultations
    ]
    batch = derive_level_coordinates_batch(
        np.concatenate([o[0] for o in occultations]),
        np.cumsum([0, *(o[0].size for o in occultations)]),
        np.concatenate([o[1] for o in occultations]),
        np.concatenate([o[2] for o in occultations]),
        np.cumsum([0, *(o[1].size for o in occultations)]),
        latitude_degrees=np.array(
            [np.nan if o[3] is None else o[3] for o in occultations]
        ),
        earth_radius_of_curvature_m=np.array(
            [np.nan if o[4] is None else o[4] for o in occultations]
        ),
        geoid_undulation_m=np.array(
            [0.0 if o[5] is None else o[5] for o in occultations]
        ),
    )
    for name, values in batch.items():
        np.testing.assert_array_equal(
            values, np.concatenate([e[name] for e in expected])
        )
    # Cases that cannot be derived stay NaN; a reported radius still gives the impact height.
    assert (
        np.isnan(expected[1]["dry_pressure_hpa"]).all()
        and np.isnan(expected[2]["heit_at_impact_m"]).all()
    )
    assert (
        np.isnan(expected[3]["impact_height_m"]).all()
        and np.isnan(expected[4]["dry_pressure_hpa"]).all()
    )
    assert np.isfinite(expected[4]["impact_height_m"]).all()
    assert (
        np.isfinite(expected[0]["dry_pressure_hpa"]).any()
        and np.isfinite(expected[0]["heit_at_impact_m"]).any()
    )


def test_batch_matches_reference_values():
    radius = 6_371_000.0
    occultations = [
        (
            [1_000.0, 3_000.0, 8_000.0, 18_000.0],
            [0.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0],
            [300.0, 240.0, 165.0, 90.0, 25.0],
            45.0,
            10.0,
        ),
        (
            [3_000.0, np.nan, 12_000.0],
            [0.0, np.nan, 4_000.0, 8_000.0, 16_000.0],
            [310.0, 200.0, -1.0, 110.0, 40.0],
            -30.0,
            0.0,
        ),
        (
            [2_500.0, 6_000.0],
            [1_000.0, 3_000.0, 6_000.0, 12_000.0],
            [270.0, 210.0, 150.0, 70.0],
            np.nan,
            0.0,
        ),
    ]
    batch = derive_level_coordinates_batch(
        radius + np.concatenate([o[0] for o in occultations]),
        np.cumsum([0, *(len(o[0]) for o in occultations)]),
        np.concatenate([o[1] for o in occultations]),
        np.concatenate([o[2] for o in occultations]),
        np.cumsum([0, *(len(o[1]) for o in occultations)]),
        latitude_degrees=np.array([o[3] for o in occultations]),
        earth_radius_of_curvature_m=np.full(len(occultations), radius),
        geoid_undulation_m=np.array([o[4] for o in occultations]),
    )
    nan = np.nan
    expected = {
        "heit_at_impact_m": [nan, 1333.190048272, 7142.692986239, 17736.39477070,
                             1219.014432083, nan, 11493.14450097, nan, nan],
        "dry_pressure_hpa": [nan, 960.2226455441, 479.2321233729, 119.2511238423,
                             970.1414803793, nan, 250.2041555163, nan, nan],
        "standard_atmosphere_pressure_hpa": [nan, 863.0123515592, 402.4248696262, 78.23362546739,
                                             875.1251899590, nan, 209.3878910733, nan, nan],
        "blended_pressure_5km_hpa": [nan, 882.4544103562, 479.2321233729, 119.2511238423,
                                     894.1284480431, nan, 250.2041555163, nan, nan],
    }  # fmt: skip
    for name, values in expected.items():
        np.testing.assert_allclose(batch[name], values, rtol=1e-9, err_msg=name)

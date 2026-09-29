# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from healda.observations.preprocessing import features as v1
from healda.observations.preprocessing import features_v2 as v2


def _make_obs_data(n, device, include_lat=False):
    g = torch.Generator(device=device)
    g.manual_seed(42)

    height = torch.rand(n, device=device, generator=g) * 50000
    pressure = torch.rand(n, device=device, generator=g) * 1100
    scan_angle = torch.rand(n, device=device, generator=g) * 100 - 50
    sat_zenith_angle = torch.rand(n, device=device, generator=g) * 120 - 60
    sol_zenith_angle = torch.rand(n, device=device, generator=g) * 160 + 10

    # Conv/sat split: NaN height -> satellite, valid height -> conventional
    is_sat = torch.rand(n, device=device, generator=g) < 0.4
    height[is_sat] = float("nan")
    pressure[is_sat] = float("nan")
    scan_angle[~is_sat] = float("nan")
    sat_zenith_angle[~is_sat] = float("nan")
    sol_zenith_angle[~is_sat] = float("nan")

    data = dict(
        target_time_sec=torch.full(
            (n,), 1_700_000_000, dtype=torch.int64, device=device
        ),
        time=torch.full(
            (n,), 1_700_000_100_000_000_000, dtype=torch.int64, device=device
        ),
        lon=torch.rand(n, device=device, generator=g) * 360 - 180,
        height=height,
        pressure=pressure,
        scan_angle=scan_angle,
        sat_zenith_angle=sat_zenith_angle,
        sol_zenith_angle=sol_zenith_angle,
    )
    if include_lat:
        data["lat"] = torch.rand(n, device=device, generator=g) * 180 - 90
    return data


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel"
)
@pytest.mark.parametrize("n", [0, 1, 137, 10_000])
def test_v1_triton_matches_reference(n):
    device = torch.device("cuda")
    data = _make_obs_data(max(n, 1), device)
    if n == 0:
        data = {k: v[:0] for k, v in data.items()}

    ref = v1._compute_unified_metadata_reference(**data)
    triton_out = v1.compute_unified_metadata(**data)

    assert ref.shape == triton_out.shape == (n, v1.N_FEATURES)
    if n > 0:
        torch.testing.assert_close(ref, triton_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel"
)
@pytest.mark.parametrize("n", [0, 1, 137, 10_000])
def test_v2_triton_matches_reference(n):
    device = torch.device("cuda")
    data = _make_obs_data(max(n, 1), device, include_lat=True)
    if n == 0:
        data = {k: val[:0] for k, val in data.items()}

    ref = v2._compute_unified_metadata_reference(**data)
    triton_out = v2.compute_unified_metadata(**data)

    assert ref.shape == triton_out.shape == (n, v2.N_FEATURES)
    if n > 0:
        torch.testing.assert_close(ref, triton_out, atol=1e-5, rtol=1e-5)

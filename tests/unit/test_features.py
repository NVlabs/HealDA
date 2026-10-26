# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from earth2grid import healpix

from healda.observations.preprocessing import features


def _make_obs_data(n, device):
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
        lat=torch.rand(n, device=device, generator=g) * 180 - 90,
        height=height,
        pressure=pressure,
        scan_angle=scan_angle,
        sat_zenith_angle=sat_zenith_angle,
        sol_zenith_angle=sol_zenith_angle,
    )
    return data


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel"
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("position", [False, True])
@pytest.mark.parametrize("n", [0, 1, 137, 10_000])
def test_featurize_matches_gather_and_torch_features(n, position, sort):
    device = torch.device("cuda")
    data = _make_obs_data(max(n, 1), device)
    g = torch.Generator(device=device)
    g.manual_seed(7)
    columns = {
        "latitude": data["lat"],
        "longitude": data["lon"],
        "absolute_obs_time": data["time"]
        + torch.randint(-(10**13), 10**13, (max(n, 1),), device=device, generator=g),
        "target_time_sec": data["target_time_sec"],
        "height": data["height"],
        "pressure": data["pressure"],
        "scan_angle": data["scan_angle"],
        "sat_zenith_angle": data["sat_zenith_angle"],
        "sol_zenith_angle": data["sol_zenith_angle"],
        "observation": torch.randn(max(n, 1), device=device, generator=g),
    }
    for name in (
        "global_channel_id",
        "local_channel_id",
        "platform_id",
        "observation_type",
    ):
        columns[name] = torch.randint(
            0, 2**16, (max(n, 1),), device=device, generator=g, dtype=torch.int32
        ).to(torch.uint16)
    columns = {k: v[:n] for k, v in columns.items()}
    order = torch.randperm(n, device=device, generator=g) if sort else None
    rows = order if sort else torch.arange(n, device=device)
    grid = healpix.Grid(6, pixel_order=healpix.NEST)
    pix = grid.ang2pix(columns["longitude"], columns["latitude"])[rows]
    centers = grid.pix2ang(pix) if position else None

    out_columns, meta = features.featurize(columns, order, centers, hpx_level=6)

    assert meta.dtype == torch.bfloat16
    assert meta.shape == (n, features.N_FEATURES + 4 * position)
    expected = features.compute_unified_metadata(
        columns["target_time_sec"][rows],
        time=columns["absolute_obs_time"][rows],
        lon=columns["longitude"][rows],
        lat=columns["latitude"][rows],
        height=columns["height"][rows],
        pressure=columns["pressure"][rows],
        scan_angle=columns["scan_angle"][rows],
        sat_zenith_angle=columns["sat_zenith_angle"][rows],
        sol_zenith_angle=columns["sol_zenith_angle"][rows],
    )
    # rtol is one bf16 ulp: the float32 values are rounded to bf16 on store.
    torch.testing.assert_close(
        meta[:, : features.N_FEATURES].float(), expected, atol=1e-5, rtol=2**-8
    )
    if position:
        expected_position = features.position_features(
            columns["longitude"][rows], columns["latitude"][rows], *centers, 6
        ).bfloat16()
        assert torch.equal(meta[:, features.N_FEATURES :], expected_position)
    for name, values in out_columns.items():
        source = columns[name]
        source = source.int() if source.dtype == torch.uint16 else source
        assert torch.equal(
            values.view(torch.uint8), source[rows].view(torch.uint8)
        ), name

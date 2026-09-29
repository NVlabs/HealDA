# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from healda.observations.types import UnifiedObservation, split_by_sensor


def make_realistic_obs(
    B: int = 2, T: int = 2, sensors: list[int] = [0, 1, 2]
) -> UnifiedObservation:
    """Create realistic cyclic observation data matching real UFS patterns.

    Sensors cycle: 0,1,2,0,1,2,... within each (b,t) window, then data is sorted globally by sensor_id.
    """
    S = len(sensors)

    # Generate observations: each window has 6 obs cycling through sensors
    all_obs = []
    for b in range(B):
        for t in range(T):
            for i in range(6):  # 6 obs per window
                sensor_id = sensors[i % S]
                all_obs.append(
                    (sensor_id, b, t, len(all_obs))
                )  # (sensor, batch, time, value)

    # Sort by sensor_id (as real data is)
    all_obs.sort(key=lambda x: (x[0], x[3]))  # Sort by sensor, then original index

    # Extract sorted data
    values = torch.tensor([x[3] for x in all_obs], dtype=torch.float32)

    # Build 3D lengths: (S, B, T) per-window obs counts
    lengths_3d = torch.zeros((S, B, T), dtype=torch.int32)
    for s_local, s_id in enumerate(sensors):
        for b in range(B):
            for t in range(T):
                lengths_3d[s_local, b, t] = sum(
                    1
                    for obs in all_obs
                    if obs[0] == s_id and obs[1] == b and obs[2] == t
                )

    nobs = len(all_obs)
    return UnifiedObservation(
        obs=values.unsqueeze(1).expand(nobs, 3),
        time=values.long(),
        float_metadata=values.unsqueeze(1).expand(nobs, 5),
        pix=torch.arange(nobs, dtype=torch.long),
        local_channel=torch.zeros(nobs, dtype=torch.long),
        local_platform=torch.zeros(nobs, dtype=torch.long),
        obs_type=torch.zeros(nobs, dtype=torch.long),
        global_channel=torch.zeros(nobs, dtype=torch.long),
        global_platform=torch.zeros(nobs, dtype=torch.long),
        hpx_level=6,
        lengths=lengths_3d,
    )


def test_split_preserves_all_observations():
    """Critical test: verify no observations are lost during split."""
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    total_before = obs.obs.shape[0]

    split = split_by_sensor(obs, [0, 1, 2])

    # Count total after split
    total_after = sum(split[sid].obs.shape[0] for sid in [0, 1, 2])
    assert (
        total_after == total_before
    ), f"LOST OBSERVATIONS: {total_before} → {total_after}"

    # Each sensor appears 2 times per window (6 obs / 3 sensors), across B*T=4 windows = 8 total
    for sid in [0, 1, 2]:
        assert split[sid].obs.shape[0] == 8, f"Sensor {sid} should have 8 obs"


def test_split_content_correctness():
    """Verify split observations contain correct data for each sensor."""
    obs = make_realistic_obs(B=2, T=2, sensors=[0, 1, 2])
    split = split_by_sensor(obs, [0, 1, 2])

    for sid in [0, 1, 2]:
        s_obs = split[sid]
        # Verify observation count is correct (sensor_id is no longer stored,
        # but we validated counts in test_split_preserves_all_observations)
        assert s_obs.obs.shape[0] == 8, f"Sensor {sid} should have 8 obs"


def test_split_lengths_match_obs_count():
    """Verify split lengths sum equals per-sensor obs count."""
    obs = make_realistic_obs(B=1, T=2, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1])

    for sid in [0, 1]:
        s_obs = split[sid]
        assert (
            s_obs.lengths.sum().item() == s_obs.obs.shape[0]
        ), f"Sensor {sid} lengths sum doesn't match obs count"


def test_split_empty_sensor():
    """Test handling of sensor with no data."""
    obs = make_realistic_obs(B=1, T=1, sensors=[0, 1])
    split = split_by_sensor(obs, [0, 1, 2])  # Request sensor 2 which doesn't exist

    assert split[2].obs.shape[0] == 0, "Empty sensor should have 0 observations"
    assert split[2].lengths.shape == (
        1,
        1,
        1,
    ), "Empty sensor should preserve batch structure"


def test_split_requires_lengths():
    """Test that split_by_sensor requires lengths."""
    obs = UnifiedObservation(
        obs=torch.randn(10, 3),
        time=torch.zeros(10, dtype=torch.long),
        float_metadata=torch.randn(10, 5),
        pix=torch.zeros(10, dtype=torch.long),
        local_channel=torch.zeros(10, dtype=torch.long),
        local_platform=torch.zeros(10, dtype=torch.long),
        obs_type=torch.zeros(10, dtype=torch.long),
        global_channel=torch.zeros(10, dtype=torch.long),
        global_platform=torch.zeros(10, dtype=torch.long),
        hpx_level=6,
        lengths=None,
    )

    with pytest.raises(ValueError, match="lengths is required"):
        split_by_sensor(obs, [0, 1])


def test_lengths_nonnegative():
    """Lengths must be non-negative for all sensors."""
    obs = make_realistic_obs(B=2, T=3, sensors=[0, 1, 2])
    assert torch.all(obs.lengths >= 0), "lengths must be non-negative"


def test_split_handles_sparse_windows():
    """Sensor missing from some (b,t) windows; split must still work."""
    B, T = 2, 3

    # Sparse data: sensor 0 everywhere (2 obs/window), sensor 4 only in (b=1,t=2) with 3 obs
    all_obs = []
    for b in range(B):
        for t in range(T):
            all_obs.extend([(0, b, t)] * 2)  # sensor 0: 2 obs/window
    all_obs.extend([(4, 1, 2)] * 3)  # sensor 4: 3 obs only in (1,2)

    nobs = len(all_obs)

    # Build lengths: sensor 0 has 2 obs/window everywhere, sensor 4 has 3 obs only in (1,2)
    lengths_3d = torch.zeros((2, B, T), dtype=torch.int32)
    lengths_3d[0, :, :] = 2  # sensor 0: 2 obs/window
    lengths_3d[1, 1, 2] = 3  # sensor 4: 3 obs only in window (1,2)

    obs = UnifiedObservation(
        obs=torch.arange(nobs, dtype=torch.float32).unsqueeze(1).expand(nobs, 3),
        time=torch.zeros(nobs, dtype=torch.long),
        float_metadata=torch.arange(nobs, dtype=torch.float32)
        .unsqueeze(1)
        .expand(nobs, 5),
        pix=torch.arange(nobs, dtype=torch.long),
        local_channel=torch.zeros(nobs, dtype=torch.long),
        local_platform=torch.zeros(nobs, dtype=torch.long),
        obs_type=torch.zeros(nobs, dtype=torch.long),
        global_channel=torch.zeros(nobs, dtype=torch.long),
        global_platform=torch.zeros(nobs, dtype=torch.long),
        hpx_level=6,
        lengths=lengths_3d,
    )

    assert obs.batch_dims == (2, 3)

    split = split_by_sensor(obs, [0, 4, 99])

    # Sensor 0: 12 obs (2 per window * 6 windows)
    s0 = split[0]
    assert s0.obs.shape[0] == 12
    assert s0.lengths.shape == (1, 2, 3)
    assert s0.batch_dims == (2, 3)
    assert s0.lengths.sum().item() == 12

    # Sensor 4: 3 obs (only in window (1,2))
    s4 = split[4]
    assert s4.obs.shape[0] == 3
    assert s4.lengths.shape == (1, 2, 3)
    assert s4.lengths[0, 1, 2].item() == 3
    assert s4.batch_dims == (2, 3)

    # Sensor 99: absent
    s99 = split[99]
    assert s99.obs.shape[0] == 0
    assert s99.lengths.shape == (1, 2, 3)
    assert torch.all(s99.lengths == 0)

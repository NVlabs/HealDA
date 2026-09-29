# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test utilities for observation embedding tests."""

import torch
from healda.observations.types import UnifiedObservation
from healda.config.models import ModelSensorConfig


def create_unified_observation(
    nobs: int,
    batch_size: int = 1,
    time_steps: int = 1,
    meta_dim: int = 8,
    hpx_level: int = 6,
    nchannel: int = 10,
    nplatform: int = 5,
    n_embed: int = 5,
    device: str = "cpu",
    ensure_all_sensors: bool = False,
    sensor_config: dict[str, ModelSensorConfig] | None = None,
) -> UnifiedObservation:
    torch.manual_seed(0)

    # Extract sensor info
    if sensor_config is not None:
        sensors = [
            (cfg.sensor_id, cfg.nchannel, list(cfg.platform_ids))
            for cfg in sensor_config.values()
        ]
    else:
        sensors = [(0, nchannel, list(range(nplatform)))]

    sensor_ids = [s[0] for s in sensors]
    n_sensors = len(sensor_ids)
    npix = 12 * 4**hpx_level

    if nobs == 0:
        return UnifiedObservation(
            obs=torch.empty(0, device=device),
            time=torch.empty(0, dtype=torch.long, device=device),
            float_metadata=torch.empty((0, meta_dim), device=device),
            pix=torch.empty(0, dtype=torch.long, device=device),
            local_channel=torch.empty(0, dtype=torch.long, device=device),
            local_platform=torch.empty(0, dtype=torch.long, device=device),
            obs_type=torch.empty(0, dtype=torch.long, device=device),
            global_channel=torch.empty(0, dtype=torch.long, device=device),
            global_platform=torch.empty(0, dtype=torch.long, device=device),
            hpx_level=hpx_level,
            lengths=torch.zeros(
                (n_sensors, batch_size, time_steps), dtype=torch.long, device=device
            ),
        )

    # Generate random observations
    def random_obs_for_sensor(sid):
        """Generate one observation for a given sensor."""
        _, nchan, plat_ids = next(s for s in sensors if s[0] == sid)
        model_platform = torch.randint(0, len(plat_ids), (1,)).item()
        return {
            "obs": torch.randn(1).item() * 0.5,
            "pix": torch.randint(0, npix, (1,)).item(),
            "platform": model_platform,
            "global_platform": plat_ids[model_platform],
            "channel": torch.randint(0, nchan, (1,)).item(),
            "embed_id": torch.randint(0, n_embed, (1,)).item(),
            "float_meta": torch.randn(meta_dim) * 0.8,
            "sensor_id": sid,
        }

    # Generate observations
    observations = []
    if ensure_all_sensors:
        # One per sensor first, then random
        for sid in sensor_ids:
            observations.append(random_obs_for_sensor(sid))
        for _ in range(nobs - len(sensor_ids)):
            sid = sensor_ids[torch.randint(0, len(sensor_ids), (1,)).item()]
            observations.append(random_obs_for_sensor(sid))
    else:
        for _ in range(nobs):
            sid = sensor_ids[torch.randint(0, len(sensor_ids), (1,)).item()]
            observations.append(random_obs_for_sensor(sid))

    # Sort by sensor_id (required for per-sensor processing)
    observations.sort(key=lambda x: x["sensor_id"])

    obs = torch.tensor([o["obs"] for o in observations], dtype=torch.float32)
    float_metadata = torch.stack([o["float_meta"] for o in observations])
    sensor_id_tensor = torch.tensor(
        [o["sensor_id"] for o in observations], dtype=torch.long
    )

    pix_tensor = torch.tensor([o["pix"] for o in observations], dtype=torch.long)
    channel_tensor = torch.tensor(
        [o["channel"] for o in observations], dtype=torch.long
    )
    platform_tensor = torch.tensor(
        [o["platform"] for o in observations], dtype=torch.long
    )
    global_platform_tensor = torch.tensor(
        [o["global_platform"] for o in observations], dtype=torch.long
    )

    lengths = torch.zeros((n_sensors, batch_size, time_steps), dtype=torch.long)
    for s_local, sid in enumerate(sensor_ids):
        lengths[s_local, 0, 0] = (sensor_id_tensor == sid).sum().item()

    time = torch.zeros(nobs, dtype=torch.long)
    return UnifiedObservation(
        obs=obs.to(device),
        time=time.to(device),
        float_metadata=float_metadata.to(device),
        pix=pix_tensor.to(device),
        local_channel=channel_tensor.to(device),
        local_platform=platform_tensor.to(device),
        obs_type=torch.zeros(nobs, dtype=torch.long, device=device),
        global_channel=channel_tensor.to(device),
        global_platform=global_platform_tensor.to(device),
        hpx_level=hpx_level,
        lengths=lengths.to(device),
    )

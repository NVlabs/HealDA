# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
from healda.models.obs_embedding.point_embed_v2 import (
    SensorEmbedder,
    MultiSensorObsEmbedding,
)
from healda.observations.types import UnifiedObservation
from healda.config.models import ModelSensorConfig, SensorEmbedderConfig
from tests.unit.utils.obs_test_utils import create_unified_observation

# ============================================================================
# Test Utilities
# ============================================================================


def check_all_params_have_gradients(model: torch.nn.Module) -> tuple[bool, list[str]]:
    params_without_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grads.append(name)

    return len(params_without_grads) == 0, params_without_grads


# ============================================================================
# SensorEmbedder Tests
# ============================================================================


@pytest.mark.parametrize("nobs", [0, 100])
@pytest.mark.parametrize("tokenizer_type", ["concat", "film"])
def test_sensor_embedder(nobs, tokenizer_type):
    """Test SensorEmbedder shapes, values, and gradients (includes empty obs for DDP compatibility)."""
    torch.manual_seed(42)

    config = SensorEmbedderConfig(embed_dim=16, fusion_dim=32)
    meta_dim = config.meta_dim
    output_dim = 32
    hpx_level = 5
    nchannel = 10
    nplatform = 5
    batch_size = 2

    embedder = SensorEmbedder(
        platform_ids=list(range(nplatform)),
        sensor_embed_dim=config.embed_dim,
        output_dim=output_dim,
        meta_dim=meta_dim,
        n_embed=100,
        hpx_level=hpx_level,
        nchannel=nchannel,
        tokenizer_type=tokenizer_type,
    )
    embedder.train()

    obs = create_unified_observation(
        nobs=nobs,
        batch_size=batch_size,
        time_steps=1,
        hpx_level=hpx_level,
        meta_dim=meta_dim,
        nchannel=nchannel,
        nplatform=nplatform,
        n_embed=100,
    )

    output = embedder(obs)

    npix = 12 * 4**hpx_level
    assert output.shape == (batch_size, 1, npix, output_dim)
    assert torch.isfinite(output).all()

    loss = output.sum()
    loss.backward()
    all_have_grads, missing = check_all_params_have_gradients(embedder)
    assert all_have_grads, f"Parameters without gradients (nobs={nobs}): {missing}"


def test_sensor_embedder_no_batching():
    """Test SensorEmbedder with offsets=None (no explicit batching)."""
    torch.manual_seed(42)

    config = SensorEmbedderConfig(embed_dim=16)
    meta_dim = config.meta_dim
    output_dim = 128
    hpx_level = 5
    nobs = 50
    nplatform = 5

    embedder = SensorEmbedder(
        platform_ids=list(range(nplatform)),
        sensor_embed_dim=config.embed_dim,
        output_dim=output_dim,
        meta_dim=meta_dim,
        n_embed=100,
        hpx_level=hpx_level,
    )

    obs = create_unified_observation(
        nobs=nobs,
        batch_size=1,
        time_steps=1,
        hpx_level=hpx_level,
        meta_dim=meta_dim,
        n_embed=100,
    )
    obs = UnifiedObservation(
        obs=obs.obs,
        time=obs.time,
        float_metadata=obs.float_metadata,
        pix=obs.pix,
        local_channel=obs.local_channel,
        local_platform=obs.local_platform,
        obs_type=obs.obs_type,
        global_channel=obs.global_channel,
        global_platform=obs.global_platform,
        hpx_level=obs.hpx_level,
        lengths=None,
    )

    with torch.no_grad():
        output = embedder(obs)

    npix = 12 * 4**hpx_level
    assert output.shape == (npix, output_dim)


# ============================================================================
# MultiSensorObsEmbedding Tests
# ============================================================================


@pytest.mark.parametrize("num_sensors", [1, 2])
def test_multisensor_obs_embedding(num_sensors):
    """Test MultiSensorObsEmbedding with different sensor counts."""
    torch.manual_seed(42)

    hpx_level = 5

    all_sensor_configs = {
        "test_sensor_0": ModelSensorConfig(
            sensor_id=0, nchannel=10, platform_ids=tuple(range(5))
        ),
        "test_sensor_1": ModelSensorConfig(
            sensor_id=1, nchannel=10, platform_ids=tuple(range(5))
        ),
    }
    sensor_config = dict(list(all_sensor_configs.items())[:num_sensors])

    sensor_embedder_config = SensorEmbedderConfig(embed_dim=16, fusion_dim=32)
    meta_dim = sensor_embedder_config.meta_dim

    embedder = MultiSensorObsEmbedding(
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hpx_level=hpx_level,
    )

    obs = create_unified_observation(
        nobs=100,
        batch_size=2,
        time_steps=1,
        hpx_level=hpx_level,
        meta_dim=meta_dim,
        sensor_config=sensor_config,
        n_embed=100,
        ensure_all_sensors=True,
    )

    with torch.no_grad():
        output = embedder(obs)

    npix = 12 * 4**hpx_level
    assert output.shape == (2, sensor_embedder_config.fusion_dim, 1, npix)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output))


@pytest.mark.parametrize("nobs", [0, 50])
@pytest.mark.parametrize("tokenizer_type", ["concat", "film"])
def test_multisensor_gradients(nobs, tokenizer_type):
    """Test gradient flow through MultiSensorObsEmbedding (includes empty obs for DDP compatibility)."""
    torch.manual_seed(42)

    hpx_level = 5

    sensor_config = {
        "test_sensor_0": ModelSensorConfig(
            sensor_id=0, nchannel=10, platform_ids=tuple(range(5))
        ),
        "test_sensor_1": ModelSensorConfig(
            sensor_id=1, nchannel=10, platform_ids=tuple(range(5))
        ),
    }

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=16,
        fusion_dim=32,
        tokenizer_type=tokenizer_type,
    )
    meta_dim = sensor_embedder_config.meta_dim

    embedder = MultiSensorObsEmbedding(
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        hpx_level=hpx_level,
    )
    embedder.train()
    embedder.zero_grad()

    obs = create_unified_observation(
        nobs=nobs,
        batch_size=2,
        time_steps=1,
        hpx_level=hpx_level,
        meta_dim=meta_dim,
        sensor_config=sensor_config,
        n_embed=100,
        ensure_all_sensors=True,
    )

    output = embedder(obs)
    assert torch.isfinite(output).all()
    loss = output.sum()
    loss.backward()

    # Check gradients (critical for DDP - must work with empty obs)
    all_have_grads, missing = check_all_params_have_gradients(embedder)
    assert all_have_grads, f"Parameters without gradients (nobs={nobs}): {missing}"

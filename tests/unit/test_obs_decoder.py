# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from healda.models.obs_embedding.decoder import ObsDecoder


def test_obs_decoder_forward_pass(regtest):
    """Test ObsDecoder forward pass with realistic inputs."""
    torch.manual_seed(42)

    hpx_in_level = 2
    hpx_fine_level = 3
    batch_size = 2
    time_steps = 1
    spatial_size = 12 * 4**hpx_in_level  # = 192
    nobs = 50
    latent_dim = 64
    metadata_dim = 32
    max_embed_id = 10
    obs_dim = 1
    hidden_dim = 32

    # Create decoder
    decoder = ObsDecoder(
        latent_dim=latent_dim,
        metadata_dim=metadata_dim,
        max_embed_id=max_embed_id,
        hpx_fine_level=hpx_fine_level,
        hpx_in_level=hpx_in_level,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )

    # Create input tensors
    latent = torch.randn(batch_size, time_steps, spatial_size, latent_dim)
    batch_idx = torch.randint(0, batch_size * time_steps, (nobs,))
    metadata = torch.randn(nobs, metadata_dim)

    # Create valid pixel indices - obs_hpx_level must be >= hpx_fine_level
    obs_hpx_level = 4
    npix_obs = 12 * 4**obs_hpx_level
    pix = torch.randint(0, npix_obs, (nobs,))

    # Create embedding IDs for platform, channel, obs_type
    platform = torch.randint(0, max_embed_id, (nobs,))
    channel = torch.randint(0, max_embed_id, (nobs,))
    obs_type = torch.randint(0, max_embed_id, (nobs,))

    # Forward pass in eval mode for deterministic output
    decoder.eval()
    with torch.no_grad():
        output = decoder(
            latent, batch_idx, metadata, pix, platform, channel, obs_type, obs_hpx_level
        )

        # Verify output shape and properties
        expected_shape = (nobs, obs_dim)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.dtype == torch.float32

        # Test that output is deterministic with same inputs
        output2 = decoder(
            latent, batch_idx, metadata, pix, platform, channel, obs_type, obs_hpx_level
        )
        assert torch.allclose(output, output2, atol=1e-6)

        # Regression test on summary statistics. We intentionally avoid hashing
        # the raw output bytes: bit-level differences across CUDA/hardware/library
        # versions flip the hash even when the numerics are equivalent.
        print(f"Output shape: {output.shape}", file=regtest)
        print(f"Output mean: {output.mean().item():.5f}", file=regtest)
        print(f"Output std: {output.std().item():.5f}", file=regtest)
        print(f"Output min: {output.min().item():.5f}", file=regtest)
        print(f"Output max: {output.max().item():.5f}", file=regtest)

    # Test gradient flow (need to switch back to training mode)
    decoder.train()
    output_grad = decoder(
        latent, batch_idx, metadata, pix, platform, channel, obs_type, obs_hpx_level
    )
    loss = output_grad.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    assert decoder.latent_proj.weight.grad is not None
    assert decoder.embed_platform.weight.grad is not None
    assert decoder.metadata_proj.weight.grad is not None
    assert decoder.mlp[0].weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from healda.models.healpix_layers import HPXPatchDecode, HPXPatchEmbed, Subdomain
from healda.models.dit import DiT
from healda.config.models import ModelSensorConfig, SensorEmbedderConfig
from healda.observations.packing import pack_observations_by_pixel
from healda.observations.types import UnifiedObservation
import pytest

from tests.unit.utils.obs_test_utils import create_unified_observation


def test_patch_decode():
    level_fine = 6
    level_coarse = 4
    c_out = 12
    decode = HPXPatchDecode(
        in_channels=5,
        out_channels=c_out,
        level_coarse=level_coarse,
        level_fine=level_fine,
    )
    x = torch.randn(1, 1, 12 * 16 * 16, 5)
    out = decode(x)
    assert out.shape == (1, c_out, 1, 12 * 4**level_fine)


def test_patch_embed():
    n = 1
    t = 1
    level_fine = 6
    level_coarse = 4
    c_out = 12
    embed = HPXPatchEmbed(
        in_channels=5,
        out_channels=c_out,
        level_coarse=level_coarse,
        level_fine=level_fine,
    )
    doy = torch.ones([n, t])
    second = torch.ones(
        [n, t],
    )
    x = torch.randn(n, 5, t, 12 * 4**level_fine)
    out = embed(x, second_of_day=second, day_of_year=doy)
    assert out.shape == (n, t, 12 * 4**level_coarse, c_out)


@pytest.mark.parametrize("checkpoint", [0, 1, 2])
def test_dit(checkpoint):
    torch.manual_seed(0)
    net = DiT(
        num_layers=1,
        level_in=4,
        level_model=4,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    )
    net.gradient_checkpointing = checkpoint
    device = "cuda"
    net.to(device)

    noise_labels = torch.zeros([1], device=device)
    class_labels = torch.empty([1, 0], device=device).int()
    n, t = 1, 1
    img = torch.ones(n, 3, t, net.domain.numel(), device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
    )
    assert out.out.shape == (n, 3, t, net.domain.numel())

    out.out.sum().backward()


def test_dit_localize():
    net = DiT(
        num_layers=1,
        level_in=4,
        num_attention_heads=2,
        level_model=4,
        in_channels=3,
        out_channels=3,
    )
    device = "cuda"
    net.to(device)

    noise_labels = torch.ones([1], device=device)
    class_labels = torch.empty([1, 0], device=device).int()
    n, t = 1, 1
    img = torch.ones(n, 3, t, net.domain.numel(), device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        level_localize=3,
    )
    assert out.out.shape == (n, 3, t, net.domain.numel())


@pytest.mark.parametrize("t", [1, 2])
@pytest.mark.parametrize("autocast", [True, False])
def test_dit_with_observations(t, device, autocast):
    """Test DiT model forward pass with observation data."""
    n = 1

    # Create sensor_embedder_config
    sensor_config = {
        "sensor_1": ModelSensorConfig(
            sensor_id=1,
            nchannel=8,
            platform_ids=tuple(range(10)),
        ),
        "sensor_2": ModelSensorConfig(
            sensor_id=4,
            nchannel=4,
            platform_ids=tuple(range(5)),
        ),
    }

    sensor_embedder_config = SensorEmbedderConfig(
        embed_dim=16,
    )
    meta_dim = sensor_embedder_config.meta_dim

    obs = create_unified_observation(
        nobs=1000,
        batch_size=n,
        time_steps=t,
        meta_dim=meta_dim,
        hpx_level=6,
        n_embed=1024,
        device=device,
        sensor_config=sensor_config,
    )

    # DiT model configuration
    npix = 12 * 4**6  # level 6 HEALPix grid

    model = DiT(
        embed_v2=True,
        embed_v2_meta_dim=meta_dim,
        embed_v2_n_embed=1024,
        obs_hpx_level=obs.hpx_level,
        sensor_embedder_config=sensor_embedder_config,
        sensors=sensor_config,
        num_layers=2,
        level_in=6,
        time_length=t,
        level_model=4,
        in_channels=3,
        out_channels=3,
        temporal_attention=True,
    )
    model.to(device)

    # Input tensors from debug script
    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, npix, device=device).to(memory_format=torch.channels_last)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass - this should work without real data dependencies
    with torch.autocast(device, torch.bfloat16, enabled=autocast):
        out = model(
            img,
            noise_labels,
            class_labels=class_labels,
            day_of_year=doy,
            second_of_day=second,
            unified_obs=obs,
        )

    # Verify output shape
    assert out.out.shape == (n, 3, t, npix)

    # Verify observation processing worked (model should handle the obs without errors)
    assert obs.batch_dims == (n, t)
    assert obs.float_metadata.shape[-1] == 28
    assert obs.hpx_level == 6

    # Verify the correct embedding module was initialized based on encoder type
    assert model.embed_v2_patch is not None


def test_subdomain_dit():
    """Test DiT with subdomain argument for regional processing."""
    device = "cuda"
    level_in = 6
    level_model = 4

    # Create DiT model
    net = DiT(
        num_layers=1,
        level_in=level_in,
        level_model=level_model,
        num_attention_heads=2,
        in_channels=3,
        out_channels=3,
    )
    net.to(device)

    # Create subdomain - single face with size 32x32
    # This represents a regional domain instead of full global coverage
    subdomain = Subdomain(
        x=torch.tensor([[0]], device=device),
        y=torch.tensor([[0]], device=device),
        f=torch.tensor([[0]], device=device),
        n=32,
        level=level_in,
    )

    # Prepare inputs matching subdomain size
    n, t = 1, 1
    subdomain_npix = subdomain.n**2 * subdomain.num_faces  # 32*32*1 = 1024

    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, subdomain_npix, device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass with subdomain
    out = net(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        subdomain=subdomain,
    )

    # Verify output shape matches subdomain size
    assert out.out.shape == (n, 3, t, subdomain_npix)


def test_dit_obs_decoder(device):
    """Test DiT with obs_decoder=True returns observation predictions."""
    n, t = 1, 1
    obs_level = 8  # ObsDecoder requires hpx_fine_level=8
    meta_dim = SensorEmbedderConfig().meta_dim

    # Create unified observation
    obs = create_unified_observation(
        nobs=1000,
        batch_size=n,
        time_steps=t,
        meta_dim=meta_dim,
        hpx_level=obs_level,
        n_embed=1024,
        device=device,
    )

    # Create DiT with obs_decoder enabled
    model = DiT(
        obs_decoder=True,  # Enable obs decoder
        num_attention_heads=2,
        num_layers=1,
        level_in=6,
        level_model=4,
        in_channels=3,
        out_channels=3,
        embed_v2_meta_dim=meta_dim,
    )
    model.to(device)

    # Forward pass inputs
    noise_labels = torch.ones([n], device=device)
    class_labels = torch.empty([n, 0], device=device).int()
    img = torch.ones(n, 3, t, 12 * 4**6, device=device)
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    # Forward pass
    out = model(
        img,
        noise_labels,
        class_labels=class_labels,
        day_of_year=doy,
        second_of_day=second,
        unified_obs=obs,
        level_localize=3,
    )

    # Check that obs decoder output is present
    assert out.obs is not None
    assert out.obs.shape[0] == obs.obs.shape[0]  # Should match number of observations


def _forward_dit(model, img, obs=None, **extra):
    n, t = img.shape[0], img.shape[2]
    device = img.device
    return model(
        img,
        torch.zeros([n], device=device),
        class_labels=torch.empty([n, 0], device=device).int(),
        day_of_year=torch.ones([n, t], device=device),
        second_of_day=torch.ones([n, t], device=device),
        unified_obs=obs,
        **extra,
    ).out


def test_dit_batch_independence_dense(device):
    """Row i of a B=2 forward must equal a standalone B=1 forward of sample i
    (dense backbone, no obs)."""
    torch.manual_seed(0)
    net = DiT(
        num_layers=2,
        level_in=3,
        level_model=3,
        num_attention_heads=2,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    )
    net.to(device).eval()  # drop_path/dropout become identity

    npix = net.domain.numel()
    img = torch.randn(2, 3, 1, npix, device=device)
    with torch.no_grad():
        out2 = _forward_dit(net, img)
        out0 = _forward_dit(net, img[0:1])
        out1 = _forward_dit(net, img[1:2])

    torch.testing.assert_close(out2, torch.cat([out0, out1]), rtol=0, atol=5e-3)


def _single_sensor_obs(
    nobs, npix, meta_dim, nchannel, nplatform, hpx_level, seed, device
):
    g = torch.Generator().manual_seed(seed)
    ri = lambda hi: torch.randint(0, hi, (nobs,), generator=g)  # noqa: E731
    lengths = torch.zeros(1, 1, 1, dtype=torch.long)
    lengths[0, 0, 0] = nobs
    return UnifiedObservation(
        obs=torch.randn(nobs, generator=g).to(device),
        time=torch.zeros(nobs, dtype=torch.long, device=device),
        float_metadata=torch.randn(nobs, meta_dim, generator=g).to(device),
        pix=ri(npix).to(device),
        local_channel=ri(nchannel).to(device),
        local_platform=ri(nplatform).to(device),
        obs_type=torch.zeros(nobs, dtype=torch.long, device=device),
        global_channel=ri(nchannel).to(device),
        global_platform=ri(nplatform).to(device),
        hpx_level=hpx_level,
        lengths=lengths.to(device),
    )


def _stack_obs(samples, hpx_level, device):
    # Concatenate per-sample (B=1) obs into one (B=len, T=1) batch in (s,b,t) order.
    fields = (
        "obs",
        "time",
        "float_metadata",
        "pix",
        "local_channel",
        "local_platform",
        "obs_type",
        "global_channel",
        "global_platform",
    )
    merged = {f: torch.cat([getattr(s, f) for s in samples], dim=0) for f in fields}
    lengths = torch.zeros(1, len(samples), 1, dtype=torch.long, device=device)
    for b, s in enumerate(samples):
        lengths[0, b, 0] = s.obs.shape[0]
    return UnifiedObservation(hpx_level=hpx_level, lengths=lengths, **merged)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="backbone pixel cross-attention requires the CUDA Triton kernel",
)
def test_dit_obs_batch_independence():
    """Obs path (backbone_pixel_attention): row i of a B=2 forward must equal a
    standalone B=1 forward of sample i. Distinct per-sample obs surface any
    packing / cu_seqlens / pixel cross-attention leakage."""
    pytest.importorskip("triton")
    device = "cuda"
    level_in, level_model = 3, 2
    npix_in = 12 * 4**level_in
    npix_model = 12 * 4**level_model
    nchannel, nplatform = 8, 5

    # The Triton pixel-attn kernel requires q_per_kv >= 16 (tl.dot minimum).
    sec_cfg = SensorEmbedderConfig(
        embed_dim=16,
        backbone_pixel_attn_num_heads=16,
        pixel_attn_kv_heads=1,
        pixel_attn_head_dim=16,
    )
    meta_dim = sec_cfg.meta_dim
    sensors = {
        "sensor_1": ModelSensorConfig(
            sensor_id=0, nchannel=nchannel, platform_ids=tuple(range(nplatform))
        )
    }

    # Distinct obs per sample (different seeds).
    s0 = _single_sensor_obs(
        40, npix_model, meta_dim, nchannel, nplatform, level_model, 0, device
    )
    s1 = _single_sensor_obs(
        55, npix_model, meta_dim, nchannel, nplatform, level_model, 1, device
    )
    obs2 = pack_observations_by_pixel(_stack_obs([s0, s1], level_model, device))
    obs0 = pack_observations_by_pixel(_stack_obs([s0], level_model, device))
    obs1 = pack_observations_by_pixel(_stack_obs([s1], level_model, device))

    torch.manual_seed(0)
    model = DiT(
        backbone_pixel_attention=True,
        sensor_embedder_config=sec_cfg,
        sensors=sensors,
        obs_hpx_level=level_model,
        num_layers=2,
        level_in=level_in,
        level_model=level_model,
        num_attention_heads=2,
        in_channels=3,
        out_channels=3,
        time_length=1,
    )
    model.to(device).eval()

    img = torch.randn(2, 3, 1, npix_in, device=device)
    with torch.no_grad():
        out2 = _forward_dit(model, img, obs=obs2)
        out0 = _forward_dit(model, img[0:1], obs=obs0)
        out1 = _forward_dit(model, img[1:2], obs=obs1)

    torch.testing.assert_close(out2, torch.cat([out0, out1]), rtol=0, atol=5e-3)

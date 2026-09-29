# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused FiLM tokenizer Triton kernel vs the pure-PyTorch reference."""

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused FiLM Triton kernel requires CUDA"
)

from healda.observations.types import UnifiedObservation  # noqa: E402
from healda.models.obs_embedding.fused_film_triton import (  # noqa: E402
    fused_film_tokenizer_triton,
)
from healda.models.obs_embedding.point_embed_v2 import ObsTokenizerFiLM  # noqa: E402


# Production-ish dims: channel emb 16, platform emb 8 (the "ch16-plat8" run).
_OBS_TYPE_EMBED_DIM = 4
_CHANNEL_EMBED_DIM = 16
_PLATFORM_EMBED_DIM = 8
_N_EMBED = 64
_NCHANNEL = 40
_NPLATFORM = 24


def _build_tokenizer(meta_dim, out_dim, with_platform, device="cuda"):
    tok = ObsTokenizerFiLM(
        meta_dim=meta_dim,
        out_dim=out_dim,
        n_embed=_N_EMBED,
        nchannel=_NCHANNEL,
        nplatform=_NPLATFORM,
        obs_type_embed_dim=_OBS_TYPE_EMBED_DIM,
        channel_embed_dim=_CHANNEL_EMBED_DIM,
        platform_embed_dim=_PLATFORM_EMBED_DIM if with_platform else 0,
        use_fused_mlp=True,
    ).to(device)
    # Push the affine params off their init values so a zero reference can't
    # accidentally pass.
    with torch.no_grad():
        for p in tok.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return tok


def _make_obs(nobs, meta_dim, with_platform, device="cuda", seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    obs_vals = torch.randn(nobs, generator=gen)
    float_meta = torch.randn(nobs, meta_dim, generator=gen)
    obs_type = torch.randint(0, _N_EMBED, (nobs,), generator=gen)
    channel = torch.randint(0, _NCHANNEL, (nobs,), generator=gen)
    platform = torch.randint(0, _NPLATFORM, (nobs,), generator=gen)
    zeros = torch.zeros(nobs, dtype=torch.long)
    return UnifiedObservation(
        obs=obs_vals.to(device),
        time=zeros.to(device),
        float_metadata=float_meta.to(device),
        pix=zeros.to(device),
        local_channel=channel.to(device),
        local_platform=platform.to(device),
        obs_type=obs_type.to(device),
        global_channel=channel.to(device),
        global_platform=platform.to(device),
        hpx_level=5,
        lengths=None,
    )


def _reference(tok, obs):
    parts = [
        obs.float_metadata,
        tok.embed_table(obs.obs_type),
        tok.channel_embedding(obs.local_channel),
    ]
    if tok.use_platform_embedding:
        parts.append(tok.platform_embedding(obs.local_platform))
    cond = torch.cat(parts, dim=-1)
    ab = tok.cond_mlp(cond)
    alpha, beta = ab.chunk(2, dim=-1)
    return alpha * obs.obs.unsqueeze(-1) + beta


def _fused_fp32(tok, obs):
    platform_ids = obs.local_platform if tok.use_platform_embedding else None
    return fused_film_tokenizer_triton(
        obs.obs,
        obs.float_metadata,
        obs.obs_type,
        obs.local_channel,
        platform_ids,
        tok.embed_table,
        tok.channel_embedding,
        tok.platform_embedding,
        tok.cond_mlp[0],
        tok.cond_mlp[1],
        tok.cond_mlp[3],
        eps=tok.cond_mlp[1].eps,
        force_fp32=True,
    )


@pytest.mark.parametrize("with_platform", [True, False])
@pytest.mark.parametrize("nobs", [1, 37, 256])
def test_fused_forward_matches_reference_fp32(with_platform, nobs):
    torch.manual_seed(0)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform)
    obs = _make_obs(nobs, meta_dim, with_platform)

    with torch.no_grad():
        out_fused = _fused_fp32(tok, obs)
        out_ref = _reference(tok, obs)

    assert out_fused.shape == (nobs, out_dim)
    assert out_fused.dtype == torch.float32
    # force_fp32 sets the compute/output dtype to fp32, but Triton's tl.dot still
    # accumulates matmuls in TF32 on tensor cores, so this is NOT IEEE fp32: it
    # floors out around ~1e-3 scale-relative error. Measured vs the fp32 reference:
    # ~1e-3 with force_fp32 vs ~5e-3 in bf16 (~5x tighter) -- enough headroom to
    # catch a real kernel bug. atol covers the many near-zero outputs where the
    # per-element rtol is meaningless.
    torch.testing.assert_close(out_fused, out_ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("with_platform", [True, False])
def test_fused_forward_matches_reference_bf16_module(with_platform):
    """The default module path runs bf16; check it tracks the fp32 reference loosely."""
    torch.manual_seed(1)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform)
    obs = _make_obs(512, meta_dim, with_platform)

    with torch.no_grad():
        tok.use_fused_mlp = True
        out_fused = tok(obs)
        out_ref = _reference(tok, obs)

    assert out_fused.dtype == torch.bfloat16
    torch.testing.assert_close(out_fused.float(), out_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("with_platform", [True, False])
def test_fused_backward_matches_reference(with_platform):
    torch.manual_seed(2)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform)
    obs = _make_obs(384, meta_dim, with_platform)

    def run(fn):
        tok.zero_grad(set_to_none=True)
        out = fn(tok, obs)
        # Weighted sum so every output element contributes a distinct gradient.
        weight = torch.linspace(0.5, 1.5, out.shape[1], device=out.device)
        (out.float() * weight).sum().backward()
        return {name: p.grad.detach().clone() for name, p in tok.named_parameters()}

    grads_fused = run(_fused_fp32)
    grads_ref = run(_reference)

    assert grads_fused.keys() == grads_ref.keys()
    for name in grads_ref:
        ref = grads_ref[name]
        fused = grads_fused[name]
        # TF32 matmul accumulation plus atomic-add reduction order make a few
        # near-zero (cancelled) gradient entries noisy, which blows up per-element
        # relative error. Validate against the gradient *scale* instead: the max
        # absolute deviation must stay within ~0.5% of the largest gradient.
        scale = ref.abs().max().clamp_min(1e-6)
        max_abs_diff = (fused - ref).abs().max()
        assert max_abs_diff <= 5e-3 * scale, (
            f"{name}: max_abs_diff={max_abs_diff.item():.3e} "
            f"exceeds 5e-3 * scale ({scale.item():.3e})"
        )


def test_fused_empty_obs():
    torch.manual_seed(3)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform=True)
    obs = _make_obs(0, meta_dim, with_platform=True)

    tok.zero_grad(set_to_none=True)
    out = _fused_fp32(tok, obs)
    assert out.shape == (0, out_dim)

    # Backward over an empty batch must still produce (zero, finite) grads so
    # DDP/FSDP all-reduce stays in lockstep across ranks.
    out.sum().backward()
    for name, p in tok.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert torch.count_nonzero(p.grad) == 0, name


def test_no_platform_grad_when_disabled():
    torch.manual_seed(4)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform=False)
    assert tok.platform_embedding is None
    obs = _make_obs(256, meta_dim, with_platform=False)
    out = _fused_fp32(tok, obs)
    out.sum().backward()
    assert tok.embed_table.weight.grad is not None
    assert tok.channel_embedding.weight.grad is not None


def test_platform_required_when_embedding_present():
    torch.manual_seed(5)
    meta_dim, out_dim = (
        50,
        32,
    )  # production FiLM dims (features_v2 N_FEATURES, embed_dim)
    tok = _build_tokenizer(meta_dim, out_dim, with_platform=True)
    obs = _make_obs(8, meta_dim, with_platform=True)
    with pytest.raises(ValueError, match="platform required"):
        fused_film_tokenizer_triton(
            obs.obs,
            obs.float_metadata,
            obs.obs_type,
            obs.local_channel,
            None,  # platform ids omitted despite platform_embedding present
            tok.embed_table,
            tok.channel_embedding,
            tok.platform_embedding,
            tok.cond_mlp[0],
            tok.cond_mlp[1],
            tok.cond_mlp[3],
            eps=tok.cond_mlp[1].eps,
        )


def test_hidden_dim_must_be_power_of_two():
    torch.manual_seed(6)
    meta_dim = 24
    embed = torch.nn.Embedding(_N_EMBED, _OBS_TYPE_EMBED_DIM).cuda()
    chan = torch.nn.Embedding(_NCHANNEL, _CHANNEL_EMBED_DIM).cuda()
    cond_dim = meta_dim + _OBS_TYPE_EMBED_DIM + _CHANNEL_EMBED_DIM
    lin1 = torch.nn.Linear(cond_dim, 100).cuda()  # 100 is not a power of two
    ln = torch.nn.LayerNorm(100).cuda()
    lin2 = torch.nn.Linear(100, 2 * 64).cuda()
    obs = _make_obs(8, meta_dim, with_platform=False)
    with pytest.raises(ValueError, match="power of 2"):
        fused_film_tokenizer_triton(
            obs.obs,
            obs.float_metadata,
            obs.obs_type,
            obs.local_channel,
            None,
            embed,
            chan,
            None,
            lin1,
            ln,
            lin2,
        )

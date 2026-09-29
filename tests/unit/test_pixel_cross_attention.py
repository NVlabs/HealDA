# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ragged pixel cross-attention Triton kernel vs a PyTorch GQA reference."""

import math

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="pixel cross-attention Triton kernel requires CUDA",
)

from healda.models.kernels import triton_pixel_attention as pca  # noqa: E402
from healda.models.kernels.triton_pixel_attention import pixel_attention  # noqa: E402
from healda.models.obs_embedding.pixel_cross_attention import (  # noqa: E402
    PIXEL_ATTN_CUTEDSL,
    PIXEL_ATTN_TRITON,
    PixelCrossAttention,
)

# Small power-of-two dims keep every kernel launch tiny and fast.
D_HEAD = 16
TOKEN_DIM = 32

# All layouts use q_per_kv=16, so only kv=1 and kv=2 kernels get compiled
# (kv=4 runs as two kv=2 phases). n_q is the minimum allowed for each kv count.
_HEAD_LAYOUTS = [(16, 1), (32, 2), (64, 4)]


@pytest.fixture(autouse=True)
def _single_autotune_config(monkeypatch):
    # Collapse the autotuner's config sweep to one config so each kernel compiles
    # once. num_warps/num_stages are functionally irrelevant to correctness.
    # Measured: the full sweep costs ~21s per kernel per shape key (~43s fwd+bwd);
    # pinning one config keeps the tests fast.
    one = [triton.Config({"TILE_K": 32}, num_warps=4, num_stages=2)]
    for kernel in (pca._pixel_attn_gqa_fwd, pca._pixel_attn_gqa_bwd):
        monkeypatch.setattr(kernel, "configs", one, raising=False)
        if hasattr(kernel, "cache"):
            kernel.cache.clear()
    yield


def test_measured_arch_pins_triton_configs(monkeypatch):
    monkeypatch.setattr(pca, "_CONFIGS_PINNED", False)
    monkeypatch.setattr(pca, "_arch_key", lambda: "H100")

    pca.ensure_pixel_attention_configs()

    for kernel in (pca._pixel_attn_gqa_fwd, pca._pixel_attn_gqa_bwd):
        assert len(kernel.configs) == 1
        config = kernel.configs[0]
        assert config.kwargs == {"TILE_K": 64}
        assert config.num_warps == 2
        assert config.num_stages == 4


@pytest.mark.parametrize(
    "device_name,expected",
    [
        ("NVIDIA GB300", "B300"),
        ("NVIDIA B300", "B300"),
        ("NVIDIA GB200", "B200"),
        ("NVIDIA H100", "H100"),
    ],
)
def test_arch_key_normalizes_superchip_names(monkeypatch, device_name, expected):
    monkeypatch.delenv("HEALDA_GPU_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: device_name)
    assert pca._arch_key() == expected


def test_arch_key_normalizes_override(monkeypatch):
    monkeypatch.setenv("HEALDA_GPU_ARCH", "GB300")
    assert pca._arch_key() == "B300"


def _cu_seqlens(counts):
    cu = torch.zeros(len(counts) + 1, dtype=torch.int32)
    if counts:
        cu[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    return cu


def _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=None):
    n_pixels, n_q_heads, d_head = Q.shape
    q_per_kv = n_q_heads // n_kv_heads
    out = torch.zeros_like(Q)
    for p in range(n_pixels):
        start, end = int(cu[p]), int(cu[p + 1])
        if end == start:
            continue
        tok = tokens[start:end]
        K = (tok @ W_k.t()).view(-1, n_kv_heads, d_head)
        V = tok @ W_v.t()
        if B_v is not None:
            V = V + B_v
        V = V.view(-1, n_kv_heads, d_head)
        for h in range(n_q_heads):
            kv = h // q_per_kv
            scores = (K[:, kv] @ Q[p, h]) * scale
            weights = torch.softmax(scores, dim=0)
            out[p, h] = weights @ V[:, kv]
    return out


def _make_inputs(counts, n_q_heads, n_kv_heads, use_v_bias, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kv_dim = n_kv_heads * D_HEAD
    Q = torch.randn(len(counts), n_q_heads, D_HEAD, generator=gen)
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen)
    W_k = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    W_v = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    B_v = torch.randn(kv_dim, generator=gen) * 0.1 if use_v_bias else None
    cu = _cu_seqlens(counts)

    def cuda(x):
        return None if x is None else x.cuda()

    return (
        cuda(Q),
        cuda(tokens),
        cuda(W_k),
        cuda(W_v),
        cuda(B_v),
        cu.cuda(),
    )


def _assert_scale_close(actual, ref, rtol, name=""):
    # Even with force_fp32 the kernel's tl.dot accumulates in TF32 (not IEEE fp32),
    # which -- plus a few near-zero entries -- makes per-element relative error
    # noisy, so validate against the tensor's overall scale instead. Measured
    # scale-relative error vs the fp32 reference: forward ~1.4e-3 with force_fp32 vs
    # ~3e-3 in bf16 (~2x tighter); backward grads ~2e-2.
    scale = ref.abs().max().clamp_min(1e-6)
    max_abs_diff = (actual - ref).abs().max()
    assert max_abs_diff <= rtol * scale, (
        f"{name}: max_abs_diff={max_abs_diff.item():.3e} exceeds {rtol} * "
        f"scale ({scale.item():.3e})"
    )


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
@pytest.mark.parametrize("use_v_bias", [False, True])
def test_pixel_attention_forward(n_q_heads, n_kv_heads, use_v_bias):
    # Mixed ragged groups: empty, singleton, and multi-token pixels.
    counts = [0, 1, 5, 0, 12, 3]
    Q, tokens, W_k, W_v, B_v, cu = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias
    )
    scale = 1.0 / math.sqrt(D_HEAD)

    out = pixel_attention(
        Q,
        tokens,
        W_k,
        W_v,
        cu,
        n_kv_heads=n_kv_heads,
        scale=scale,
        B_v=B_v,
        force_fp32=True,
    )
    ref = _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=B_v)

    assert out.shape == ref.shape
    assert torch.count_nonzero(out[0]) == 0  # empty pixel -> zero output
    assert torch.count_nonzero(out[3]) == 0
    _assert_scale_close(out, ref, rtol=5e-3, name="forward")


def test_pixel_attention_packed_full_grid():
    # Packed full-grid layout: many pixels, almost all with zero observations.
    counts = [0] * 120
    for idx, c in [(5, 12), (37, 1), (90, 8)]:
        counts[idx] = c
    Q, tokens, W_k, W_v, B_v, cu = _make_inputs(counts, 32, 2, use_v_bias=True, seed=1)
    scale = 1.0 / math.sqrt(D_HEAD)

    out = pixel_attention(
        Q,
        tokens,
        W_k,
        W_v,
        cu,
        n_kv_heads=2,
        scale=scale,
        B_v=B_v,
        force_fp32=True,
    )
    ref = _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, 2, scale, B_v=B_v)
    _assert_scale_close(out, ref, rtol=5e-3, name="packed_full_grid")
    assert torch.count_nonzero(out[90]) > 0
    assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
def test_pixel_attention_backward(n_q_heads, n_kv_heads):
    counts = [0, 2, 9, 1, 6]
    Q, tokens, W_k, W_v, B_v, cu = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias=True, seed=3
    )
    scale = 1.0 / math.sqrt(D_HEAD)
    grad_out = torch.randn(Q.shape, generator=torch.Generator().manual_seed(7)).cuda()

    def grads_for(fn):
        leaves = {
            "Q": Q.clone().detach().requires_grad_(True),
            "tokens": tokens.clone().detach().requires_grad_(True),
            "W_k": W_k.clone().detach().requires_grad_(True),
            "W_v": W_v.clone().detach().requires_grad_(True),
            "B_v": B_v.clone().detach().requires_grad_(True),
        }
        out = fn(**leaves)
        (out * grad_out).sum().backward()
        return {k: v.grad for k, v in leaves.items()}

    triton_grads = grads_for(
        lambda Q, tokens, W_k, W_v, B_v: pixel_attention(
            Q,
            tokens,
            W_k,
            W_v,
            cu,
            n_kv_heads=n_kv_heads,
            scale=scale,
            B_v=B_v,
            force_fp32=True,
        )
    )
    ref_grads = grads_for(
        lambda Q, tokens, W_k, W_v, B_v: _ragged_gqa_reference(
            Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=B_v
        )
    )
    for name in ref_grads:
        _assert_scale_close(
            triton_grads[name], ref_grads[name], rtol=2e-2, name=f"grad_{name}"
        )


def test_pixel_attention_large_logits_single_key():
    # One key per pixel, so softmax must return exactly 1.0 and the output must be
    # V. The scaled score here is ~1e11, where a contracted score*scale - max FMA
    # leaves a positive residual of ~1e3 and overflows exp2 to inf. Rounding the
    # score-scale multiply first (as FA2 does under UNFUSE_FMA) makes the maximum
    # element's exp2 argument exactly zero.
    value = 133120.0
    Q = torch.full((1, 16, D_HEAD), value, device="cuda", dtype=torch.bfloat16)
    tokens = torch.full((1, TOKEN_DIM), value, device="cuda", dtype=torch.bfloat16)
    W_k = torch.eye(D_HEAD, TOKEN_DIM, device="cuda", dtype=torch.bfloat16)
    W_v = torch.eye(D_HEAD, TOKEN_DIM, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    leaves = [t.clone().requires_grad_(True) for t in (Q, tokens, W_k, W_v)]

    out = pixel_attention(
        *leaves,
        cu,
        n_kv_heads=1,
        scale=1.0 / math.sqrt(D_HEAD),
    )

    expected_v = (tokens @ W_v.t()).view(1, 1, D_HEAD).expand(1, 16, D_HEAD)
    assert torch.isfinite(out).all(), "single-key softmax overflowed"
    torch.testing.assert_close(out.float(), expected_v.float(), rtol=0, atol=0)

    out.float().sum().backward()
    for tensor, name in zip(leaves, ("Q", "tokens", "W_k", "W_v"), strict=True):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
def test_pixel_attention_grouping_matches_ungrouped(n_q_heads, n_kv_heads):
    # Small-pixel grouping packs several pixels into one kernel program via a CSR map.
    # Per-pixel and per-token results are bit-identical either way. B_v is the one
    # gradient the kernel reduces over programs, and grouping changes the number of
    # partials and therefore the summation order, so it gets a tolerance.
    from healda.observations.packing import build_pixel_group_map

    reduced_over_programs = {"B_v"}

    # Many small pixels (so the map actually pairs) plus a couple big + empty.
    counts = [3, 2, 4, 1, 0, 5, 2, 3, 30, 1, 2, 4, 0, 3, 2, 40, 1, 2]
    Q, tokens, W_k, W_v, B_v, cu = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias=True, seed=11
    )
    scale = 1.0 / math.sqrt(D_HEAD)
    group_map = build_pixel_group_map(cu)
    # Sanity: the map must actually group (fewer programs than nonzero pixels).
    n_nz = int((cu[1:] > cu[:-1]).sum())
    assert group_map.program_ptr.numel() - 1 < n_nz
    grad_out = torch.randn(Q.shape, generator=torch.Generator().manual_seed(13)).cuda()

    def grads_for(group_map):
        leaves = {
            "Q": Q.clone().detach().requires_grad_(True),
            "tokens": tokens.clone().detach().requires_grad_(True),
            "W_k": W_k.clone().detach().requires_grad_(True),
            "W_v": W_v.clone().detach().requires_grad_(True),
            "B_v": B_v.clone().detach().requires_grad_(True),
        }
        out = pixel_attention(
            leaves["Q"],
            leaves["tokens"],
            leaves["W_k"],
            leaves["W_v"],
            cu,
            n_kv_heads=n_kv_heads,
            scale=scale,
            B_v=leaves["B_v"],
            force_fp32=True,
            group_map=group_map,
        )
        (out * grad_out).sum().backward()
        return out, {k: v.grad for k, v in leaves.items()}

    ungrouped_output, ungrouped_grads = grads_for(None)
    grouped_output, grouped_grads = grads_for(group_map)
    torch.testing.assert_close(grouped_output, ungrouped_output, rtol=0, atol=0)
    for name in ungrouped_grads:
        tol = 1e-5 if name in reduced_over_programs else 0
        torch.testing.assert_close(
            grouped_grads[name],
            ungrouped_grads[name],
            rtol=tol,
            atol=tol,
            msg=lambda m, name=name: f"grad_{name}: {m}",
        )


def test_pixel_cross_attention_module_forward_backward():
    # Exercises the nn.Module wiring (q_proj/out_proj + reshapes) in the real
    # bf16 path. Smoke-checks shapes, finiteness, and full gradient coverage.
    torch.manual_seed(4)
    total_pixels = 20
    counts = [0, 3, 1, 0] * 5
    assert len(counts) == total_pixels
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()

    gen = torch.Generator().manual_seed(5)
    hidden = (
        torch.randn(total_pixels, module.input_dim, generator=gen)
        .cuda()
        .requires_grad_(True)
    )
    tokens = (
        torch.randn(sum(counts), TOKEN_DIM, generator=gen).cuda().requires_grad_(True)
    )
    cu = _cu_seqlens(counts).cuda()

    out = module(hidden, tokens, total_pixels, cu)
    assert out.shape == (total_pixels, module.output_dim)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_pixel_cross_attention_empty_tokens_grad():
    # No observations anywhere (this path skips the kernel entirely): every
    # projection param must still get a (zero, finite) gradient so DDP stays in
    # lockstep across ranks.
    torch.manual_seed(6)
    total_pixels = 12
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()
    hidden = torch.randn(total_pixels, module.input_dim).cuda()
    tokens = torch.zeros(0, TOKEN_DIM).cuda()
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32).cuda()

    out = module(hidden, tokens, total_pixels, cu)
    assert out.shape == (total_pixels, module.output_dim)
    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_pixel_cross_attention_rejects_unsupported_configs():
    # Document the supported head layout: q_per_kv >= 16, n_kv_heads in {1,2,even},
    # n_q_heads divisible by n_kv_heads. These raise at construction, no kernel.
    with pytest.raises(ValueError, match="below Triton tl.dot minimum"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=8, n_kv_heads=1, d_head=D_HEAD
        )
    with pytest.raises(ValueError, match="n_kv_heads"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=64, n_kv_heads=3, d_head=D_HEAD
        )
    with pytest.raises(ValueError, match="divisible"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=66, n_kv_heads=4, d_head=D_HEAD
        )


# ---------------------------------------------------------------------------
# Backend selector wiring. Triton is the baseline and the fallback floor.
# ---------------------------------------------------------------------------


def test_get_model_forwards_the_pixel_attn_selector(monkeypatch):
    """The config value must reach DiT, not just exist on the dataclass.

    The DiT constructor is intercepted rather than built: the architectures that enable pixel
    attention are too large to instantiate in a unit test.
    """
    import healda.models as models
    from healda.config.models import ModelConfigV1

    seen = {}
    monkeypatch.setattr(models.dit, "DiT", lambda *a, **kw: seen.update(kw) or object())

    for impl in (PIXEL_ATTN_TRITON, PIXEL_ATTN_CUTEDSL):
        seen.clear()
        models.get_model(
            ModelConfigV1(
                architecture="dit-5B-d128",
                backbone_pixel_attention=True,
                pixel_attn_impl=impl,
            )
        )
        assert (
            seen.get("pixel_attn_impl") == impl
        ), f"get_model dropped pixel_attn_impl={impl}, so the selector would silently do nothing"


@pytest.mark.parametrize(
    "impl,expected",
    [(PIXEL_ATTN_TRITON, True), (PIXEL_ATTN_CUTEDSL, False)],
)
def test_transform_only_builds_group_map_for_triton(impl, expected):
    import healda.models as models
    from healda.config.models import ModelConfigV1

    options = models.transform_options_for(
        ModelConfigV1(
            architecture="dit-5B-d128",
            backbone_pixel_attention=True,
            pixel_attn_impl=impl,
        )
    )
    assert options.build_attention_group_map is expected


def test_unavailable_backend_falls_back_to_triton():
    """Requesting a backend this build lacks warns and uses Triton rather than failing."""
    with pytest.warns(UserWarning, match="falling back to the Triton kernel"):
        m = PixelCrossAttention(
            token_dim=TOKEN_DIM,
            n_q_heads=32,
            n_kv_heads=2,
            d_head=D_HEAD,
            pixel_attn_impl=PIXEL_ATTN_CUTEDSL,
        )
    assert m.pixel_attn_impl == PIXEL_ATTN_TRITON


def test_selector_rejects_out_of_range():
    with pytest.raises(ValueError, match="pixel_attn_impl must be"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM,
            n_q_heads=32,
            n_kv_heads=2,
            d_head=D_HEAD,
            pixel_attn_impl=7,
        )


# The CuTe DSL kernels are generated for this head layout only.
CUTE_N_Q_HEADS, CUTE_N_KV_HEADS, CUTE_D_HEAD = 64, 2, 32


def _ragged_counts(total_pixels, seed=0):
    """Skewed token counts: empty pixels, a heavy tail, and one pixel far larger."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(1, 40, (total_pixels,), generator=g).tolist()
    counts[0] = 0
    counts[1] = 0
    counts[total_pixels // 2] = 700
    counts[-1] = 0
    return counts


def _cute_module(impl, **kwargs):
    torch.manual_seed(0)
    return PixelCrossAttention(
        token_dim=TOKEN_DIM,
        n_q_heads=CUTE_N_Q_HEADS,
        n_kv_heads=CUTE_N_KV_HEADS,
        d_head=CUTE_D_HEAD,
        pixel_attn_impl=impl,
        **kwargs,
    ).cuda()


def _cute_inputs(total_pixels, counts, dim):
    torch.manual_seed(1)
    hidden = torch.randn(total_pixels, dim, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randn(
        int(sum(counts)), TOKEN_DIM, device="cuda", dtype=torch.bfloat16
    )
    return hidden, tokens, _cu_seqlens(counts).cuda()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the pixel kernels are CUDA-only"
)
@pytest.mark.parametrize("total_pixels", [256, 4096])
@pytest.mark.parametrize("use_proj_bias", [True, False], ids=["bias", "nobias"])
def test_cutedsl_matches_triton_forward_and_backward(total_pixels, use_proj_bias):
    """Triton is the reference: it is what production runs."""
    triton_mod = _cute_module(PIXEL_ATTN_TRITON, use_proj_bias=use_proj_bias)
    cute_mod = _cute_module(PIXEL_ATTN_CUTEDSL, use_proj_bias=use_proj_bias)
    if cute_mod.pixel_attn_impl != PIXEL_ATTN_CUTEDSL:
        pytest.skip("cutedsl pixel attention is not available in this build")
    cute_mod.load_state_dict(triton_mod.state_dict())

    counts = _ragged_counts(total_pixels)
    hidden, tokens, cu = _cute_inputs(total_pixels, counts, triton_mod.input_dim)
    triton_tokens = tokens.detach().requires_grad_(True)
    cute_tokens = tokens.detach().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        triton_out = triton_mod(hidden, triton_tokens, total_pixels, cu)
        cute_out = cute_mod(hidden, cute_tokens, total_pixels, cu)

    assert torch.isfinite(cute_out.float()).all()
    scale = triton_out.float().abs().max().clamp_min(1e-6)
    assert (cute_out.float() - triton_out.float()).abs().max() / scale < 0.05

    seed = torch.randn_like(triton_out)
    triton_out.backward(seed)
    cute_out.backward(seed)

    assert torch.isfinite(cute_tokens.grad.float()).all()
    scale = triton_tokens.grad.float().abs().max().clamp_min(1e-6)
    delta = (cute_tokens.grad.float() - triton_tokens.grad.float()).abs().max()
    assert delta / scale < 0.05

    cute_params = dict(cute_mod.named_parameters())
    for name, triton_param in triton_mod.named_parameters():
        cute_grad = cute_params[name].grad
        assert torch.isfinite(cute_grad.float()).all(), name
        scale = triton_param.grad.float().abs().max().clamp_min(1e-6)
        assert (
            cute_grad.float() - triton_param.grad.float()
        ).abs().max() / scale < 0.05, name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the pixel kernels are CUDA-only"
)
@pytest.mark.parametrize("impl", [PIXEL_ATTN_TRITON, PIXEL_ATTN_CUTEDSL])
def test_empty_pixels_do_not_leak_across_boundaries(impl):
    """A pixel with no observations must attend to nothing, on either backend."""
    total_pixels = 256
    counts = _ragged_counts(total_pixels)
    empty = [i for i, c in enumerate(counts) if c == 0]
    assert empty, "fixture must contain empty pixels"

    mod = _cute_module(impl)
    if mod.pixel_attn_impl != impl:
        pytest.skip(f"pixel_attn_impl={impl} is not available in this build")
    hidden, tokens, cu = _cute_inputs(total_pixels, counts, mod.input_dim)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mod(hidden, tokens, total_pixels, cu)

    # _cute_module leaves use_proj_bias at its default of False, so out_proj has no bias and
    # an empty pixel must be exactly zero. Asserting only that the empty rows agree would pass
    # on a kernel that leaked one stale value into all of them.
    assert mod.out_proj.bias is None
    assert torch.isfinite(out.float()).all()
    assert (out[empty] == 0).all()

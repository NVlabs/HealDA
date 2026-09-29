# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL temporal-attention kernels must match the einsum path they replace, and fall back
cleanly."""

import importlib.util
import subprocess
import sys

import pytest
import torch

from healda.models.attention import (
    TEMPORAL_ATTN_CUTEDSL,
    TEMPORAL_ATTN_EINSUM,
    TemporalAttention,
)

# Scoped per test rather than module-wide: the selector-wiring, checkpoint-compatibility and
# invalid-selector tests are pure Python and should still run on a CPU-only job.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the temporal kernels are CUDA-only"
)
# Without this the backend silently falls back to einsum and the cutedsl case passes vacuously.
requires_cutedsl = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="cutedsl temporal attention is not available in this build",
)

FRAMES, TOKENS = 8, 512

# (impl, embed_dim, num_heads, warning emitted when the impl cannot serve a config).
# The CuTe DSL kernels are generated for 12 heads x 128 head dim only.
BACKENDS = [
    pytest.param(
        TEMPORAL_ATTN_CUTEDSL,
        1536,
        12,
        "cutedsl temporal kernels disabled",
        id="cutedsl",
        marks=requires_cutedsl,
    ),
]
UNSUPPORTED_CONFIGS = [
    {"linear_attention": False},
    {"use_rope": False},
    {"causal_window": 4},
    {"temporal_attn_legacy_scaling_bug": True},
]


def _module(impl, dim, heads, **kwargs):
    torch.manual_seed(0)
    return TemporalAttention(
        embed_dim=dim,
        num_heads=heads,
        linear_attention=True,
        temporal_attn_impl=impl,
        **kwargs,
    ).cuda()


@pytest.mark.parametrize("impl,dim,heads,_warn", BACKENDS)
@requires_cuda
def test_matches_einsum_forward_and_backward(impl, dim, heads, _warn):
    einsum = _module(TEMPORAL_ATTN_EINSUM, dim, heads)
    kernel = _module(impl, dim, heads)
    assert kernel.temporal_attn_impl == impl, "expected this backend to be enabled here"
    kernel.load_state_dict(einsum.state_dict())

    x = torch.randn(1, FRAMES, TOKENS, dim, device="cuda", dtype=torch.bfloat16)
    a = x.detach().requires_grad_(True)
    b = x.detach().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        want = einsum(a, is_causal=True)
        got = kernel(b, is_causal=True)
    seed = torch.randn_like(want)
    want.backward(seed)
    got.backward(seed)

    assert (got.float() - want.float()).abs().max() / want.float().abs().max() < 0.02

    grads = dict(kernel.named_parameters())
    for name, p in einsum.named_parameters():
        denom = p.grad.float().abs().max().clamp_min(1e-6)
        assert (
            grads[name].grad.float() - p.grad.float()
        ).abs().max() / denom < 0.05, name

    assert (
        b.grad.float() - a.grad.float()
    ).abs().max() / a.grad.float().abs().max() < 0.05


@pytest.mark.parametrize("impl,dim,heads,warn", BACKENDS)
@pytest.mark.parametrize("kwargs", UNSUPPORTED_CONFIGS)
@requires_cuda
def test_unsupported_configurations_fall_back(impl, dim, heads, warn, kwargs):
    """Each backend serves one configuration; anything else warns and uses the einsum path, so
    the toggle can be set globally without breaking recipes the kernel cannot serve."""
    with pytest.warns(UserWarning, match=warn):
        m = TemporalAttention(
            embed_dim=dim, num_heads=heads, temporal_attn_impl=impl, **kwargs
        )
    assert m.temporal_attn_impl == TEMPORAL_ATTN_EINSUM


# ---------------------------------------------------------------------------
# The CuTe DSL kernels bake in a shape family, so shapes outside it must fall back.
# ---------------------------------------------------------------------------

CUTE_DIM, CUTE_HEADS = 1536, 12


@pytest.mark.parametrize("embed_dim,num_heads", [(CUTE_DIM, 8), (8 * 128, CUTE_HEADS)])
@requires_cuda
@requires_cutedsl
def test_cutedsl_unsupported_shapes_fall_back(embed_dim, num_heads):
    """The kernels bake in 12 heads x 128 head dim; anything else must not dispatch."""
    with pytest.warns(UserWarning, match="cutedsl temporal kernels disabled"):
        m = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            linear_attention=True,
            temporal_attn_impl=TEMPORAL_ATTN_CUTEDSL,
        )
    assert m.temporal_attn_impl != TEMPORAL_ATTN_CUTEDSL


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, FRAMES, 128, CUTE_DIM), torch.bfloat16),  # batch != 1
        (
            (1, FRAMES // 2, 128, CUTE_DIM),
            torch.bfloat16,
        ),  # frames != FRAMES, still pow2
        ((1, FRAMES - 1, 128, CUTE_DIM), torch.bfloat16),  # frames != FRAMES, not pow2
        ((1, FRAMES, 128, CUTE_DIM), torch.float32),  # kernels are bf16-only
    ],
)
@requires_cuda
@requires_cutedsl
def test_cutedsl_runtime_fallbacks_match_einsum(shape, dtype):
    """frames, batch and dtype are only known per call, so those guards live in forward."""
    einsum = _module(TEMPORAL_ATTN_EINSUM, CUTE_DIM, CUTE_HEADS)
    cute = _module(TEMPORAL_ATTN_CUTEDSL, CUTE_DIM, CUTE_HEADS)
    cute.load_state_dict(einsum.state_dict())

    x = torch.randn(*shape, device="cuda", dtype=dtype)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)
    ):
        got, want = cute(x, is_causal=True), einsum(x, is_causal=True)

    rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
    assert rel < 0.02, f"{shape}/{dtype} fallback is numerically wrong, rel={rel}"


# ---------------------------------------------------------------------------
# Config wiring: the toggle has to survive the whole path into the model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", [TEMPORAL_ATTN_EINSUM, TEMPORAL_ATTN_CUTEDSL])
def test_get_model_forwards_the_temporal_attn_selector(monkeypatch, impl):
    """The config value must reach DiT. The constructor is intercepted because dit-5B-d128,
    the only architecture the CuTe DSL kernels serve, is too large to instantiate."""
    import healda.models as models
    from healda.config.models import ModelConfigV1

    seen = {}
    monkeypatch.setattr(models.dit, "DiT", lambda *a, **kw: seen.update(kw) or object())

    for arch in ("dit-5B-d128", "dit-l_reg_hpx6"):
        seen.clear()
        models.get_model(
            ModelConfigV1(
                architecture=arch, dit_temporal_attention=True, fused_temporal_attn=impl
            )
        )
        assert (
            seen.get("fused_temporal_attn") == impl
        ), f"{arch}: get_model dropped fused_temporal_attn"


@pytest.mark.parametrize(
    "stored,expected",
    [
        ("true", 1),
        ("false", TEMPORAL_ATTN_EINSUM),
        ("2", TEMPORAL_ATTN_CUTEDSL),
    ],
)
def test_checkpoint_configs_stored_as_bool_still_load(stored, expected):
    """fused_temporal_attn was a bool. bool is an int subclass, so old checkpoints load as 0/1."""
    from healda.config.models import ModelConfigV1

    cfg = ModelConfigV1.loads('{"fused_temporal_attn": %s}' % stored)
    assert int(cfg.fused_temporal_attn) == expected


def test_temporal_attn_impl_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be 0 or 2"):
        TemporalAttention(
            embed_dim=CUTE_DIM,
            num_heads=CUTE_HEADS,
            linear_attention=True,
            temporal_attn_impl=3,
        )


def test_removed_triton_selector_runs_einsum():
    m = TemporalAttention(
        embed_dim=CUTE_DIM,
        num_heads=CUTE_HEADS,
        linear_attention=True,
        temporal_attn_impl=1,
    )
    assert m.temporal_attn_impl == TEMPORAL_ATTN_EINSUM


@requires_cuda
def test_odd_pixel_count_falls_back():
    """The forward tiles pixels in pairs, so an odd count would leave the last pixel unwritten.

    Some odd sizes happen to agree anyway, because the unwritten output keeps whatever the
    caching allocator last put there, so this checks a size that reliably diverged (513).
    """
    einsum = _module(TEMPORAL_ATTN_EINSUM, CUTE_DIM, CUTE_HEADS)
    cute = _module(TEMPORAL_ATTN_CUTEDSL, CUTE_DIM, CUTE_HEADS)
    cute.load_state_dict(einsum.state_dict())
    for x in (513, 511):
        t = torch.randn(1, FRAMES, x, CUTE_DIM, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            want, got = einsum(t, is_causal=True), cute(t, is_causal=True)
        rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
        assert rel < 0.02, f"x={x} did not fall back, rel={rel}"


# cutlass is an optional extra, so this runs in a subprocess with it blocked -- otherwise it
# would only be meaningful on images that happen to lack it.
_NO_CUTLASS_PROBE = """
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "cutlass":
            raise ImportError("blocked by the test")


sys.meta_path.insert(0, Block())

import healda.models.dit  # noqa: F401  the import itself is the assertion
from healda.models.kernels import fused_cutedsl, unsupported_reasons

assert unsupported_reasons(num_heads=12, head_dim=128, frames=8, batch=1), (
    "a missing cutlass must be reported as a backend blocker"
)
try:
    fused_cutedsl(None, None, None)
except RuntimeError as error:
    assert "nvidia-cutlass-dsl" in str(error), error
else:
    raise AssertionError("fused_cutedsl must refuse to run without cutlass")
"""


def test_models_import_without_cutlass():
    """An eager import in healda.models.kernels made every model module unimportable on
    images without the CuTe DSL, so a missing cutlass has to read as a backend blocker."""
    done = subprocess.run(
        [sys.executable, "-c", _NO_CUTLASS_PROBE],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr

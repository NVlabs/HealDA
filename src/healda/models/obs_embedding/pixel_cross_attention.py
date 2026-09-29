# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pixel/observation cross-attention module."""

import math
import warnings
from importlib.metadata import PackageNotFoundError, version

import torch.nn as nn

from healda.models.kernels.triton_pixel_attention import (
    ensure_pixel_attention_configs,
    pixel_attention,
)

# Pixel-attention backend selector. Triton is the production baseline and the fallback floor;
# a backend that cannot serve a call falls back to it rather than failing.
PIXEL_ATTN_TRITON = 1
PIXEL_ATTN_CUTEDSL = 2


class PixelCrossAttention(nn.Module):
    """Cross-attention from pixel latents to packed per-pixel observation tokens.

    Forward expects:

    * ``hidden_states`` with shape ``[..., input_dim]`` containing one latent
      vector per pixel.
    * ``tokens`` with shape ``[total_tokens, token_dim]`` containing all
      observation tokens concatenated across pixels.
    * ``total_pixels`` equal to the flattened pixel count in ``hidden_states``.
    * ``cu_seqlens_k`` with shape ``[total_pixels + 1]`` storing prefix sums into
      ``tokens`` so pixel ``i`` attends to
      ``tokens[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]``.

    The module reshapes ``hidden_states`` to ``[total_pixels, input_dim]``,
    applies ``q_proj``, runs ragged grouped-query attention over each pixel's
    token slice, applies ``out_proj``, and returns
    ``[total_pixels, output_dim]``.
    """

    def __init__(
        self,
        token_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        input_dim=None,
        output_dim=None,
        use_proj_bias=False,
        pixel_attn_impl=PIXEL_ATTN_TRITON,
    ):
        super().__init__()

        if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
            raise ValueError(
                f"PixelCrossAttention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
            )
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        q_per_kv = n_q_heads // n_kv_heads
        if q_per_kv < 16:
            raise ValueError(
                f"n_q_heads/n_kv_heads={q_per_kv} < 16, below Triton tl.dot minimum. "
                f"For n_kv_heads={n_kv_heads}, need n_q_heads >= {n_kv_heads * 16}"
            )
        self.attn_dim = n_q_heads * d_head
        self.input_dim = self.attn_dim if input_dim is None else input_dim
        self.output_dim = self.attn_dim if output_dim is None else output_dim
        self.token_dim = token_dim
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.pixel_attn_impl = self._resolve_impl(
            pixel_attn_impl,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            token_dim=token_dim,
        )
        kv_dim = n_kv_heads * d_head
        self.q_proj = nn.Linear(self.input_dim, self.attn_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(token_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, kv_dim, bias=use_proj_bias)
        self.out_proj = nn.Linear(self.attn_dim, self.output_dim, bias=use_proj_bias)

        if self.pixel_attn_impl == PIXEL_ATTN_TRITON:
            ensure_pixel_attention_configs()

    def _forward_impl(
        self,
        hidden_states,
        tokens,
        total_pixels,
        cu_seqlens_k,
        group_map=None,
    ):
        hidden_flat = hidden_states.reshape(total_pixels, self.input_dim)

        if tokens.shape[0] == 0:
            # Keep every projection parameter in the graph even when a batch has
            # no observations, so empty groups still produce gradients (prevents issues with DDP).
            token_dummy = tokens.sum() * 0
            q_dummy = self.q_proj.weight.sum() * 0
            if self.q_proj.bias is not None:
                q_dummy = q_dummy + self.q_proj.bias.sum() * 0
            kv_dummy = self.k_proj.weight.sum() * 0 + self.v_proj.weight.sum() * 0
            if self.v_proj.bias is not None:
                kv_dummy = kv_dummy + self.v_proj.bias.sum() * 0
            out = self.out_proj(hidden_flat.new_zeros((total_pixels, self.attn_dim)))
            return out + token_dummy + q_dummy + kv_dummy

        hidden_flat = self.q_proj(hidden_flat)
        Q = hidden_flat.view(total_pixels, self.n_q_heads, self.d_head)
        Q = Q.contiguous()

        if self.pixel_attn_impl == PIXEL_ATTN_CUTEDSL:
            from healda.models.kernels import cutedsl_pixel_attention as cutedsl

            attn_out = cutedsl.pixel_attention(
                Q,
                tokens,
                self.k_proj.weight,
                self.v_proj.weight,
                cu_seqlens_k,
                n_kv_heads=self.n_kv_heads,
                scale=self.scale,
                B_v=self.v_proj.bias,
            )
        else:
            attn_out = pixel_attention(
                Q,
                tokens,
                self.k_proj.weight,
                self.v_proj.weight,
                cu_seqlens_k,
                n_kv_heads=self.n_kv_heads,
                scale=self.scale,
                B_v=self.v_proj.bias,
                group_map=group_map,
            )

        return self.out_proj(attn_out.reshape(total_pixels, self.attn_dim))

    @staticmethod
    def _resolve_impl(requested, *, n_q_heads, n_kv_heads, d_head, token_dim):
        """Pick a backend, falling back to Triton when the requested one cannot serve this shape.

        Triton is the production baseline and the floor: every configuration this module
        accepts is one Triton can serve, so a fallback is always possible.
        """
        impl = int(requested)
        if impl not in (PIXEL_ATTN_TRITON, PIXEL_ATTN_CUTEDSL):
            raise ValueError(
                f"pixel_attn_impl must be {PIXEL_ATTN_TRITON} (triton) or "
                f"{PIXEL_ATTN_CUTEDSL} (cutedsl); got {requested!r}"
            )
        if impl != PIXEL_ATTN_CUTEDSL:
            return impl
        try:
            from healda.models.kernels import cutedsl_pixel_attention as k
        except (ImportError, OSError):
            warnings.warn(
                "cutedsl pixel attention is not available in this build; "
                "falling back to the Triton kernel.",
                stacklevel=3,
            )
            return PIXEL_ATTN_TRITON
        # The kernels build against a version-sensitive API and an older DSL fails at the
        # first launch rather than at import, so the version is checked here instead.
        # The dependency is an optional extra, so the metadata may be absent even when the
        # module imports, and a locally built version may not parse as two integers. Either
        # way the point of this path is to fall back, not to abort model construction.
        try:
            dsl = version("nvidia-cutlass-dsl")
            too_old = tuple(int(p) for p in dsl.split(".")[:2]) < (4, 7)
        except (PackageNotFoundError, ValueError):
            dsl, too_old = "unknown", True
        if too_old:
            warnings.warn(
                f"cutedsl pixel attention needs nvidia-cutlass-dsl>=4.7, found {dsl}; "
                "falling back to the Triton kernel.",
                stacklevel=3,
            )
            return PIXEL_ATTN_TRITON
        reasons = k.unsupported_reasons(
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            token_dim=token_dim,
        )
        if reasons:
            warnings.warn(
                "cutedsl pixel attention requires " + ", ".join(reasons) + "; "
                "falling back to the Triton kernel.",
                stacklevel=3,
            )
            return PIXEL_ATTN_TRITON
        return impl

    def forward(
        self,
        hidden_states,
        tokens,
        total_pixels,
        cu_seqlens_k,
        group_map=None,
    ):
        return self._forward_impl(
            hidden_states,
            tokens,
            total_pixels,
            cu_seqlens_k,
            group_map=group_map,
        )

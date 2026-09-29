# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand-written custom kernels and their autograd glue.

One module per (region, backend). The consuming ``nn.Module`` picks a backend through an
integer selector and falls back when a backend cannot serve a call.

Temporal attention (``TemporalAttention(temporal_attn_impl=...)``):

* ``cutedsl_temporal_attention.fused_cutedsl`` -- single shape family, see ``unsupported_reasons``

Pixel/observation cross-attention (``PixelCrossAttention``):

* ``triton_pixel_attention.pixel_attention`` -- ragged grouped-query attention
"""

from healda.models.kernels.triton_pixel_attention import pixel_attention

# cutlass is an optional extra and is absent from some images. Importing it eagerly makes the
# whole healda.models package unimportable there, so a miss becomes a backend blocker instead.
try:
    from healda.models.kernels.cutedsl_temporal_attention import (
        fused_cutedsl,
        unsupported_reasons,
    )
except (ImportError, OSError) as exc:
    _CUTEDSL_UNAVAILABLE = exc

    def unsupported_reasons(**_kwargs) -> list[str]:
        return ["nvidia-cutlass-dsl to be installed"]

    def fused_cutedsl(*_args, **_kwargs):
        raise RuntimeError(
            "the cutedsl temporal kernels need nvidia-cutlass-dsl, which is not installed"
        ) from _CUTEDSL_UNAVAILABLE


__all__ = ["fused_cutedsl", "pixel_attention", "unsupported_reasons"]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import math
import warnings

import einops
import torch
from diffusers.models.attention import Attention

from healda.models.attention_processors import TEAttnProcessor

try:
    from transformer_engine.pytorch.attention import DotProductAttention

    _HAS_TRANSFORMER_ENGINE = True
except (ImportError, OSError):
    DotProductAttention = None
    _HAS_TRANSFORMER_ENGINE = False


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    This class provides rotary position embeddings that can be applied to query and key tensors
    in attention mechanisms to encode positional information.
    """

    def __init__(self, head_dim, base: int = 10000, max_seq_len: int = 24):
        """
        Initialize the Rotary Position Embedding.

        Args:
            head_dim (int): Dimension of each attention head
            base (int): Base for frequency calculation (default: 10000)
            max_seq_len (int): Maximum sequence length to precompute
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute frequencies for efficiency
        self._precompute_freqs()

    def _precompute_freqs(self):
        """Precompute frequency matrices for all possible sequence lengths."""
        # Create position indices up to max_seq_len
        position = torch.arange(self.max_seq_len).float()

        # Create frequency indices for pairs (head_dim//2 pairs)
        dim_indices = torch.arange(self.head_dim // 2).float()
        dim_indices = dim_indices[None, :]  # [1, head_dim//2]

        # Calculate frequencies
        freqs = 1.0 / (
            self.base ** (2 * dim_indices / self.head_dim)
        )  # [1, head_dim//2]
        freqs = position[:, None] * freqs  # [max_seq_len, head_dim//2]

        # Generate cos and sin
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    @torch.compile
    def forward(self, x):
        """
        Apply rotary position embedding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, t, x, heads, head_dim]

        Returns:
            torch.Tensor: Tensor with rotary position embedding applied
        """
        seq_len = x.shape[1]

        # Ensure we don't exceed precomputed length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Get the relevant frequency matrices
        freqs_cos = self.freqs_cos[:seq_len]  # [seq_len, head_dim]
        freqs_sin = self.freqs_sin[:seq_len]  # [seq_len, head_dim]

        return self._apply_rotary_pos_emb(x, freqs_cos, freqs_sin)

    def _apply_rotary_pos_emb(self, x, freqs_cos, freqs_sin):
        """Apply rotary position embedding to input tensor x."""
        # x: [b, t, x, heads, head_dim
        # freqs_cos, freqs_sin: [t, head_dim//2]

        # Split x into even and odd indices along the head_dim
        # Each pair of dimensions gets rotated together
        x1, x2 = x[..., 0::2], x[..., 1::2]

        cos = freqs_cos[None, :, None, None, :]
        sin = freqs_sin[None, :, None, None, :]

        # Apply rotation - each pair shares the same angle
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave the results back
        out = torch.stack([out1, out2], dim=-1)
        out = out.reshape(x.shape)

        return out


def mask_causal(attn, linear: bool = True, window: int | None = None):
    """Apply a causal mask to a ``[b tq tk x h]`` attention tensor (out-of-place).

    Zero-fill for linear attention, -inf fill for softmax attention. Masks out
    positions where tq < tk (supports asymmetric q/k lengths).

    When ``window`` is given, additionally restrict each query to a sliding
    lookback of ``window`` frames (including itself): keys older than
    ``q - window + 1`` are masked. ``window=None`` is unbounded causal attention.

    Out-of-place on purpose: an in-place ``masked_fill_`` on the non-contiguous
    einsum output (``"b q k x h"``) makes torch.compile/inductor emit a view that
    fails for some strides; the out-of-place form compiles cleanly.
    """
    tq, tk = attn.shape[1], attn.shape[2]
    mask = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).triu(diagonal=1)
    if window is not None:
        # Mask keys more than ``window - 1`` steps in the past: q - k >= window.
        too_old = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).tril(
            diagonal=-window
        )
        mask = mask | too_old
    return attn.masked_fill(
        mask.view(1, tq, tk, 1, 1), 0.0 if linear else float("-inf")
    )


# Temporal-attention implementation selector.
TEMPORAL_ATTN_EINSUM = 0
TEMPORAL_ATTN_CUTEDSL = 2

# The CuTe DSL kernels are generated for exactly this many frames.
FRAMES = 8


class TemporalAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_base=100,
        max_seq_len=100,
        linear_attention=True,
        temporal_attn_legacy_scaling_bug=False,
        causal_window: int | None = None,
        temporal_attn_impl: int = TEMPORAL_ATTN_EINSUM,
    ) -> None:
        """Temporal self-attention over the time dimension of (b, t, x, c) tensors.

        Args:
            embed_dim: Hidden dimension (split across heads).
            num_heads: Number of attention heads.
            use_rope: Apply rotary position embeddings to q/k along the time axis.
            rope_base: Base frequency for RoPE.
            max_seq_len: Maximum sequence length for RoPE cache.
            linear_attention: If True, skip softmax — attention weights are raw
                dot-products (causal masking uses zero-fill instead of -inf).
            temporal_attn_legacy_scaling_bug: When True *and* linear_attention is
                True, normalize q·k by sequence length (k.shape[1]) instead of head
                dimension (k.shape[-1]). This reproduces a bug from older
                checkpoints; set False for correct behaviour.
            causal_window: When set (and the forward pass is causal), restrict each
                frame to attend to itself and the previous ``causal_window - 1``
                frames. ``None`` is unbounded causal attention.
            temporal_attn_impl: Which implementation to use for the region between the qkv
                and output projections. ``TEMPORAL_ATTN_EINSUM`` (0) or
                ``TEMPORAL_ATTN_CUTEDSL`` (2). A CuTe DSL request that cannot be served
                falls back to the einsum path with a warning; see
                ``healda.models.kernels.unsupported_reasons``.
        """
        super().__init__()
        self._time_parallel_group = None
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.head_dim = embed_dim // num_heads
        self.linear_attention = linear_attention
        self.temporal_attn_legacy_scaling_bug = temporal_attn_legacy_scaling_bug
        self.causal_window = causal_window
        # Requirements of the fused path; anything else falls back to the einsums.
        unsupported = [
            name
            for name, ok in (
                ("linear_attention=True", linear_attention),
                ("use_rope=True", use_rope),
                ("causal_window=None", causal_window is None),
                (
                    "temporal_attn_legacy_scaling_bug=False",
                    not temporal_attn_legacy_scaling_bug,
                ),
            )
            if not ok
        ]
        impl = int(temporal_attn_impl)
        # 1 selected a backend that has been removed.
        if impl == 1:
            impl = TEMPORAL_ATTN_EINSUM
        if impl not in (TEMPORAL_ATTN_EINSUM, TEMPORAL_ATTN_CUTEDSL):
            raise ValueError(
                f"temporal_attn_impl must be 0 or 2; got {temporal_attn_impl!r}"
            )

        # frames and batch are only known per call and are re-checked in forward.
        if impl == TEMPORAL_ATTN_CUTEDSL:
            from healda.models.kernels import unsupported_reasons

            blockers = unsupported + unsupported_reasons(
                num_heads=num_heads, head_dim=self.head_dim, frames=FRAMES, batch=1
            )
            if blockers:
                impl = TEMPORAL_ATTN_EINSUM
                warnings.warn(
                    "cutedsl temporal kernels disabled: they require "
                    f"{', '.join(blockers)}. Falling back to the einsum path.",
                    stacklevel=2,
                )

        self.temporal_attn_impl = impl

        # Initialize RoPE if enabled
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim, base=rope_base, max_seq_len=max_seq_len
            )
        else:
            self.rope = None
        if impl == TEMPORAL_ATTN_CUTEDSL:
            # The kernels want contiguous fp32 trig tables for exactly FRAMES frames. Slicing
            # and casting per call costs ~3.5us against a ~50us kernel, so do it once. Buffers
            # (not plain attributes) so they follow .cuda()/.to(); non-persistent so they stay
            # out of state_dict.
            self.register_buffer(
                "_cutedsl_cos",
                self.rope.freqs_cos[:FRAMES].contiguous().float(),
                persistent=False,
            )
            self.register_buffer(
                "_cutedsl_sin",
                self.rope.freqs_sin[:FRAMES].contiguous().float(),
                persistent=False,
            )

    @torch.compile
    def forward(self, x, is_causal: bool = False):
        # ensure the data is contiguous in time first if we are using separable parallelism
        qkv = self.qkv(x)
        # frames, batch, pixel count and dtype are only known here. The kernels are generated
        # for (batch 1, FRAMES frames, bfloat16) and tile pixels in pairs, so an odd pixel
        # count would leave the last pixel unwritten. Anything else falls through to the einsums.
        if (
            self.temporal_attn_impl == TEMPORAL_ATTN_CUTEDSL
            and is_causal
            and qkv.shape[0] == 1
            and qkv.shape[1] == FRAMES
            and qkv.shape[2] % 2 == 0
            and qkv.dtype == torch.bfloat16
        ):
            from healda.models.kernels import fused_cutedsl

            out = fused_cutedsl(
                qkv, self.num_heads, self._cutedsl_cos, self._cutedsl_sin, True
            )
            return self.proj(out)
        q, k, v = einops.rearrange(
            qkv,
            "b t x (n heads c) -> n b t x heads c",
            n=3,
            heads=self.num_heads,
        )

        # Apply RoPE to queries and keys if enabled
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.linear_attention:
            scale_dim = (
                k.shape[1] if self.temporal_attn_legacy_scaling_bug else k.shape[-1]
            )
        else:
            scale_dim = k.shape[-1]

        attn = torch.einsum(
            "b q x h c, b k x h c -> b q k x h", q, k / math.sqrt(scale_dim)
        )

        if is_causal:
            attn = mask_causal(
                attn, linear=self.linear_attention, window=self.causal_window
            )
        if not self.linear_attention:
            attn = attn.softmax(2)

        out = einops.einsum(attn, v, "b q k x h, b k x h c -> b q x h c")
        out = einops.rearrange(out, "b t x h c -> b t x (h c)")
        out = self.proj(out)
        return out


def _set_transformer_engine_domain_parallel_group(attn, group):
    if group is None:
        return

    if attn.cp_stream is None:
        cp_stream = torch.cuda.Stream()
    else:
        cp_stream = attn.cp_stream

    ranks = torch.distributed.get_process_group_ranks(group)
    attn.set_context_parallel_group(group, ranks, cp_stream)


class TEMultiHeadAttentionWithQK(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        qkv_format: str = "bshd",
        bias: bool = True,
        out_bias: bool = True,
        dropout: float = 0.0,
        qk_rms_norm: bool = False,
        qk_norm_elementwise_affine: bool = False,
    ):
        super().__init__()
        if not _HAS_TRANSFORMER_ENGINE:
            raise ImportError(
                "TEMultiHeadAttentionWithQK requires Transformer Engine. "
                "Please install transformer-engine[pytorch]."
            )

        embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.qkv_format = qkv_format
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.attn = DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            attn_mask_type="no_mask",
            qkv_format=qkv_format,
        )
        # Disable dynamo on the attention call. Store via object.__setattr__ so the
        # wrapper is NOT registered as a duplicate submodule (which would add a
        # redundant ``_call_attn._extra_state`` to the state dict); the real TE state
        # lives under ``attn._extra_state``. Disabling the module (not its bound
        # forward) preserves the (query, key, value) call convention.
        object.__setattr__(self, "_call_attn", torch._dynamo.disable(self.attn))
        self.norm_q = (
            torch.nn.RMSNorm([head_dim], elementwise_affine=qk_norm_elementwise_affine)
            if qk_rms_norm
            else None
        )
        self.norm_k = (
            torch.nn.RMSNorm([head_dim], elementwise_affine=qk_norm_elementwise_affine)
            if qk_rms_norm
            else None
        )

    def set_domain_parallel_group(self, group):
        _set_transformer_engine_domain_parallel_group(self.attn, group)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "TEMultiHeadAttentionWithQK only supports self-attention."
            )

        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)

        attn_kwargs = dict(
            qkv_format=self.qkv_format,
            attn_mask_type="arbitrary" if attention_mask is not None else "no_mask",
        )
        if attention_mask is not None:
            attn_kwargs["attention_mask"] = attention_mask
        out = self._call_attn(q, k, v, **attn_kwargs)
        out = out.reshape(batch_size, sequence_length, self.embed_dim)
        return self.dropout(self.proj(out))


class TESpatialAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        out_bias: bool = True,
        qk_norm: str | None = None,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        super().__init__()
        del upcast_attention, kwargs
        if cross_attention_dim is not None:
            raise NotImplementedError(
                "TESpatialAttention does not support cross-attention."
            )
        if query_dim != heads * dim_head:
            raise ValueError(
                f"TESpatialAttention expects query_dim={heads * dim_head}, got {query_dim}."
            )
        if qk_norm not in (None, "rms_norm"):
            raise NotImplementedError(f"Unsupported qk_norm backend: {qk_norm}")
        self.attn = TEMultiHeadAttentionWithQK(
            heads,
            dim_head,
            qkv_format="bshd",
            bias=bias,
            out_bias=out_bias,
            dropout=dropout,
            qk_rms_norm=qk_norm == "rms_norm",
            qk_norm_elementwise_affine=elementwise_affine,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        **cross_attention_kwargs,
    ):
        b, t, x, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b * t, x, c)
        out = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )
        return out.reshape(b, t, x, c)

    def set_domain_parallel_group(self, group):
        self.attn.set_domain_parallel_group(group)


DISK_BLOCK_SIZE = 128


@functools.lru_cache(maxsize=None)
def healpix_disk_block_mask(
    num_tokens: int,
    radius_deg: float,
    device: str,
    pixel_order: str,
):
    """BlockMask keeping keys within ``radius_deg`` geodesic degrees of each query.

    Local attention on a haversine disk: a query attends to the cap of the sphere around it,
    not to a neighbourhood in pixel index.

    Must be built outside a compiled region: the traced mask_mod fails to lower under no_grad.
    """
    import earth2grid
    from healda.observations.types import healpix_pixel_order
    from torch.nn.attention.flex_attention import create_block_mask

    nside = int(round(math.sqrt(num_tokens / 12)))
    if 12 * nside * nside != num_tokens:
        raise ValueError(
            f"{num_tokens} tokens is not a HEALPix map size (12 * nside^2)"
        )
    level = int(round(math.log2(nside)))
    if 2**level != nside:
        raise ValueError(f"nside {nside} is not a power of two")

    grid = earth2grid.healpix.Grid(level, pixel_order=healpix_pixel_order(pixel_order))
    lon = torch.deg2rad(torch.as_tensor(grid.lon, dtype=torch.float32, device=device))
    lat = torch.deg2rad(torch.as_tensor(grid.lat, dtype=torch.float32, device=device))
    # Haversine distance monotonically follows the central angle, whose cosine is the dot
    # product of the unit vectors, so the predicate is one comparison and no trigonometry.
    xyz = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=1)
    threshold = math.cos(math.radians(radius_deg))

    def inside_disk(b, h, q_idx, kv_idx):
        return (xyz[q_idx] * xyz[kv_idx]).sum(-1) >= threshold

    # Compiled, or the predicate is materialised over the dense Q x KV grid: 36 GiB at level
    # 6. Wrapped rather than the create_block_mask(_compile=True) flag, which is deprecated.
    return torch.compile(create_block_mask)(
        inside_disk,
        B=None,
        H=None,
        Q_LEN=num_tokens,
        KV_LEN=num_tokens,
        device=device,
        BLOCK_SIZE=DISK_BLOCK_SIZE,
    )


# B300 config the default inductor sweep does not try: its tile is matched to the mask block
# size, and num_warps/num_stages are unprefixed so they reach the backward kernel too.
# num_stages=2 is a shared-memory limit, so this is 16-bit only.
_flex_attention = None


def _compiled_flex_attention():
    global _flex_attention
    if _flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _flex_attention = torch.compile(flex_attention, dynamic=False)
    return _flex_attention


FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": DISK_BLOCK_SIZE,
    "BLOCK_N": DISK_BLOCK_SIZE,
    "num_warps": 8,
    "num_stages": 2,
}


class DiskSpatialAttention(torch.nn.Module):
    """Spatial self-attention over HEALPix tokens, restricted to a geodesic disk per query.

    Local on the sphere, not in pixel index: a query attends to the haversine cap of
    ``radius_deg`` around it. ``pixel_order`` is the order the caller stores tokens in; the
    mask is built to match.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        radius_deg: float,
        pixel_order: str,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        out_bias: bool = True,
        qk_norm: str | None = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        del upcast_attention
        if cross_attention_dim is not None:
            raise NotImplementedError("DiskSpatialAttention is self-attention only.")

        self.num_heads = heads
        self.head_dim = dim_head
        self.embed_dim = query_dim
        self.radius_deg = radius_deg
        self.pixel_order = pixel_order
        self.qkv = torch.nn.Linear(query_dim, query_dim * 3, bias=bias)
        self.proj = torch.nn.Linear(query_dim, query_dim, bias=out_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm_q = (
            torch.nn.RMSNorm([dim_head], elementwise_affine=elementwise_affine)
            if qk_norm == "rms_norm"
            else None
        )
        self.norm_k = (
            torch.nn.RMSNorm([dim_head], elementwise_affine=elementwise_affine)
            if qk_norm == "rms_norm"
            else None
        )
        self._block_mask = None

    def prepare_block_mask(self, num_tokens: int, device) -> None:
        self._block_mask = healpix_disk_block_mask(
            num_tokens, self.radius_deg, str(device), self.pixel_order
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        **cross_attention_kwargs,
    ):
        del cross_attention_kwargs
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "DiskSpatialAttention only supports self-attention."
            )
        if attention_mask is not None:
            raise NotImplementedError(
                "DiskSpatialAttention supplies its own mask; an additional "
                "attention_mask would have to be composed into the block mask."
            )

        b, t, x, c = hidden_states.shape
        qkv = self.qkv(hidden_states.reshape(b * t, x, c))
        # (bt, x, 3, h, d) -> three of (bt, h, x, d), the layout flex_attention wants.
        q, k, v = qkv.view(b * t, x, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)

        # On q, not on the input: under autocast the input stays fp32 while the projections
        # produce bf16. FLEX_KERNEL_OPTIONS is a 16-bit-only tile and fp32 fails inside
        # inductor autotuning with "No valid triton configs".
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"DiskSpatialAttention needs bf16/fp16, got {q.dtype}; enable "
                "autocast(bfloat16) or use attention_backend='diffusers'."
            )

        if self._block_mask is None:
            self.prepare_block_mask(x, hidden_states.device)
        out = _compiled_flex_attention()(
            q, k, v, block_mask=self._block_mask, kernel_options=FLEX_KERNEL_OPTIONS
        )
        out = out.transpose(1, 2).reshape(b * t, x, self.embed_dim)
        return self.dropout(self.proj(out)).reshape(b, t, x, c)

    def set_domain_parallel_group(self, group):
        raise NotImplementedError(
            "DiskSpatialAttention has no domain-parallel path; the block mask is built for "
            "the whole sphere and would have to be sliced per rank."
        )


class SpatialAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        **cross_attention_kwargs,
    ):
        b, t, x, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b * t, x, c)
        out = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return out.reshape(b, t, x, c)

    def set_domain_parallel_group(self, group):
        processor = TEAttnProcessor(domain_parallel_group=group)
        self.set_processor(processor)


class SpatioTemporalAttention(Attention):
    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        **cross_attention_kwargs,
    ):
        b, t, x, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t * x, c)
        out = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return out.reshape(b, t, x, c)

    def set_domain_parallel_group(self, group):
        processor = TEAttnProcessor(domain_parallel_group=group)
        self.set_processor(processor)

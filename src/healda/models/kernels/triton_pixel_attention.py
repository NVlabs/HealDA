# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Triton-backed pixel cross-attention layer.

This module implements pixel/observation cross-attention.

1. Triton forward and backward kernels for ragged grouped-query attention.
2. ``torch.library.custom_op`` wrappers so the kernels participate in autograd and
   fake-tensor tracing for ``torch.compile``.
3. ``PixelCrossAttention``, an ``nn.Module`` that performs

       q_proj -> attention -> out_proj

   from pixel latents to the observation tokens assigned to each pixel.

The public ``pixel_attention()`` helper operates on a packed ragged layout:

* ``Q`` has shape ``[total_pixels, n_q_heads, d_head]``.
* ``tokens`` has shape ``[total_tokens, token_dim]`` and contains all
  observation tokens for all pixels concatenated together.
* ``cu_seqlens_k`` has shape ``[total_pixels + 1]`` and stores prefix sums that
  delimit which token rows belong to each pixel.
* ``W_k`` and ``W_v`` have shape ``[n_kv_heads * d_head, token_dim]``.

For a given pixel, the kernel projects only that pixel's token slice into keys
and values, applies grouped-query attention from that pixel's query heads, and
returns ``[n_q_heads, d_head]`` attention output for that pixel. The full module
just wraps that primitive with query and output projections.

Implementation notes
--------------------
* The kernel streams over the token dimension in tiles and keeps online softmax
  statistics, so it never materializes the full attention score matrix for a
  ragged pixel group.
* ``n_q_heads`` must be divisible by ``n_kv_heads``. The current Triton path
  also requires ``q_per_kv = n_q_heads / n_kv_heads >= 16`` because of the
  ``tl.dot`` tile shape used by the kernel.
* For ``n_kv_heads <= 2`` the module launches one grouped kernel per pixel. For
  larger even ``n_kv_heads`` it splits the head dimension into two-KV-head
  phases and concatenates the per-phase outputs back in the original order.
* K bias is accepted for API compatibility but dropped from the computation.
  For a fixed query, it adds the same scalar offset to every key logit, so
  softmax cancels it exactly and the analytic gradient should be zero. In
  practice, keeping it only introduced finite-precision noise, resulting in
  nonzero gradients on that bias. V bias is applied inside the Triton path.
* The Triton kernels assume packed contiguous tensors, so the Python wrappers
  materialize contiguous views before launch when needed.
* Launch configurations are pinned on measured GPU architectures. Other
  architectures use Triton's in-memory autotuning.

PyTorch integration
-------------------
``PixelCrossAttention`` keeps parameter ownership in standard ``nn.Linear``
modules. The Triton kernels consume the raw ``q_proj``/``k_proj``/``v_proj`` and
``out_proj`` tensors, while the custom-op registrations save the tensors needed
for backward so gradients flow back to the original module parameters through
ordinary PyTorch autograd.
"""

import math
import os

import torch
import triton
import triton.language as tl


# Base-2 softmax: exp(x) == exp2(x * log2e), log(x) == log2(x) / log2e.
# Folding log2e into the score scale lets the kernels use the faster MUFU
# exp2/log2 hardware instructions. The LSE is stored in the log2 domain so the
# forward (producer) and backward (consumer) agree on the convention.
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _mul_rn_f32(x, y):
    # Prevent FMA contraction so forward and backward use identical rounded scores.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


def _autotune_configs():
    return [
        triton.Config({"TILE_K": tk}, num_warps=nw, num_stages=ns)
        for tk in [32, 64, 128]
        for nw in [1, 2, 4, 8]
        for ns in [1, 2, 4]
    ]


_ARCH_CONFIGS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("B300", "fwd"): (64, 2, 4),
    ("B300", "bwd"): (64, 2, 4),
    ("H100", "fwd"): (64, 2, 4),
    ("H100", "bwd"): (64, 2, 4),
}
_ARCH_KEYS = {
    "GB300": "B300",
    "GB200": "B200",
    "B300": "B300",
    "B200": "B200",
    "H200": "H200",
    "H100": "H100",
    "A100": "A100",
}


def _arch_key() -> str | None:
    override = os.environ.get("HEALDA_GPU_ARCH")
    if override:
        return _ARCH_KEYS.get(override, override)
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    return next(
        (canonical for detected, canonical in _ARCH_KEYS.items() if detected in name),
        None,
    )


def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


_PRINTED_AUTOTUNE_CHOICES = set()
_AUTOTUNE_REPORTER = None


def set_pixel_attn_autotune_reporter(reporter):
    global _AUTOTUNE_REPORTER
    _AUTOTUNE_REPORTER = reporter


def _maybe_print_autotune_choice(
    kind,
    autotuner,
    q_per_kv,
    n_kv_heads,
    compute_dtype,
):
    if os.environ.get("HEALDA_PRINT_PIXEL_ATTN_AUTOTUNE") != "1":
        return
    best_config = getattr(autotuner, "best_config", None)
    if best_config is None:
        return
    key = (kind, q_per_kv, n_kv_heads, str(compute_dtype))
    if key in _PRINTED_AUTOTUNE_CHOICES:
        return
    _PRINTED_AUTOTUNE_CHOICES.add(key)
    message = (
        "[pixel_attn autotune] "
        f"{kind} key=(Q_PER_KV={q_per_kv}, N_KV_HEADS={n_kv_heads}, "
        f"COMPUTE_DTYPE={compute_dtype}) "
        f"best_config={best_config}"
    )
    if _AUTOTUNE_REPORTER is not None:
        _AUTOTUNE_REPORTER(message)
    else:
        print(message, flush=True)


# ─── Grouped-query fused kernels ─────────────────────────────────────
# One program per pixel group, all KV heads processed together.
# Handles n_kv_heads={1,2,4} via constexpr branching with explicit
# per-head accumulators. Tokens loaded ONCE per tile, d_tokens uses
# plain tl.store (no per-head atomic contention).
#
# Wk and Wv passed as separate pointers


@triton.jit
def _gqa_fwd_head(
    tokens_tile,
    wk_h,
    wv_h,
    bv_h,
    q_h,
    kv_mask,
    scale,
    m_h,
    l_h,
    acc_h,
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Head selection already happened in the caller: q_h holds the Q_PER_KV
    # query heads assigned to one KV head, and wk_h/wv_h/bv_h are that KV
    # head's projection parameters.
    k_h = tl.dot(tokens_tile, tl.trans(wk_h)).to(COMPUTE_DTYPE)
    # Add the fp32 bias before casting the projection.
    v_acc = tl.dot(tokens_tile, tl.trans(wv_h))
    if USE_V_BIAS:
        v_acc += bv_h[None, :]
    v_h = v_acc.to(COMPUTE_DTYPE)

    # Fold log2e into the score scale so scores live in the log2 domain and the
    # softmax can use exp2 (faster MUFU instruction) instead of exp.
    scores = _mul_rn_f32(
        tl.dot(q_h.to(COMPUTE_DTYPE), tl.trans(k_h)).to(tl.float32),
        scale * LOG2E,
    )
    scores = tl.where(kv_mask[None, :], scores, float("-inf"))

    # Online softmax over KV tiles (as in FlashAttention). We keep a running max (m_h), denominator
    # (l_h), and weighted value sum (acc_h) for each query row so we never
    # have to materialize the full attention matrix across all keys.
    m_tile = tl.max(scores, axis=1)
    m_new = tl.maximum(m_h, m_tile)
    # If this tile raises the running max, rescale the previous partial sums
    # into the new log-sum-exp coordinate system before adding this tile.
    corr = tl.exp2(m_h - m_new)
    exp_s = tl.exp2(scores - m_new[:, None])

    l_h = l_h * corr + tl.sum(exp_s, axis=1)
    acc_h = tl.dot(exp_s.to(COMPUTE_DTYPE), v_h, acc=acc_h * corr[:, None])
    m_h = m_new
    return m_h, l_h, acc_h


@triton.jit
def _gqa_bwd_head(
    tokens_tile,
    wk_h,
    wv_h,
    bv_h,
    q_h,
    dout_h,
    D_h,
    lse_h,
    kv_mask,
    scale,
    dq_h,
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
):
    # HYBRID unfuse: keep the cheap K/V recompute + dtokens IN-kernel (read
    # tokens once, project in registers per tile) but DROP the loop-carried
    # [32,32] fp32 weight-grad accumulators (dWk/dWv/dBv) that pinned 255 regs /
    # 28M spills. Instead this returns the per-tile dk/dv, which the caller
    # stores; dWk/dWv/dBv are recovered as dense GEMMs after the kernel. This
    # keeps the fused kernel's *minimal* HBM footprint (no K/V materialization)
    # while removing the spill source -> far less extra traffic than full unfuse.
    k_h = tl.dot(tokens_tile, tl.trans(wk_h)).to(COMPUTE_DTYPE)
    # Recompute V exactly as in the forward.
    v_acc = tl.dot(tokens_tile, tl.trans(wv_h))
    if USE_V_BIAS:
        v_acc += bv_h[None, :]
    v_h = v_acc.to(COMPUTE_DTYPE)

    # Match the forward's non-contracted score scaling.
    scores = _mul_rn_f32(
        tl.dot(q_h.to(COMPUTE_DTYPE), tl.trans(k_h)).to(tl.float32),
        scale * LOG2E,
    )
    if MASKED:
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))
    weights = tl.exp2(scores - lse_h[:, None])

    # D_h = rowsum(dO * O) is the FA "delta"; dout/out are tile-invariant, so the
    # caller computes it ONCE per program and passes it in (not recomputed per tile).
    dv_tile = tl.dot(tl.trans(weights.to(COMPUTE_DTYPE)), dout_h.to(COMPUTE_DTYPE))
    pt = tl.dot(dout_h, tl.trans(v_h)).to(tl.float32)
    # ds is the gradient w.r.t. the raw logits q·k (natural score scale).
    ds = weights * (pt - D_h[:, None]) * scale
    ds_cast = ds.to(COMPUTE_DTYPE)
    dk_tile = tl.dot(tl.trans(ds_cast), q_h)
    dq_h = tl.dot(ds_cast, k_h, acc=dq_h)

    dk_cast = dk_tile.to(COMPUTE_DTYPE)
    dv_cast = dv_tile.to(COMPUTE_DTYPE)

    d_tok = tl.dot(dv_cast, wv_h, acc=tl.dot(dk_cast, wk_h))

    return dq_h, d_tok, dk_cast, dv_cast


@triton.jit
def _gqa_bwd_tile(
    Tokens_ptr, dTokens_ptr, dK_ptr, dV_ptr, kv_start, tile_off, seqlen_k, scale,
    wk0, wv0, bv0, q0, dout0, D0, lse0, dq0,
    wk1, wv1, bv1, q1, dout1, D1, lse1, dq1,
    dbv0, dbv1,
    USE_V_BIAS: tl.constexpr, Q_PER_KV: tl.constexpr, N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr, TOKEN_DIM: tl.constexpr, KV_DIM: tl.constexpr,
    TILE_K: tl.constexpr, COMPUTE_DTYPE: tl.constexpr, MASKED: tl.constexpr,
):
    # dq{0,1}/dbv{0,1} are carried across tiles; MASKED specializes the single ragged tail.
    offs_kv = tl.arange(0, TILE_K)
    offs_d = tl.arange(0, D_HEAD)
    offs_td = tl.arange(0, TOKEN_DIM)
    if MASKED:
        kv_mask = offs_kv < (seqlen_k - tile_off)
    else:
        kv_mask = tl.full((TILE_K,), 1, tl.int1)
    tok_base = (kv_start + tile_off) * TOKEN_DIM
    if MASKED:
        tokens_tile = tl.load(
            Tokens_ptr + tok_base + offs_kv[:, None] * TOKEN_DIM + offs_td[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        ).to(COMPUTE_DTYPE)
    else:
        tokens_tile = tl.load(
            Tokens_ptr + tok_base + offs_kv[:, None] * TOKEN_DIM + offs_td[None, :]
        ).to(COMPUTE_DTYPE)

    # Combined [dK | dV] rows: store dK through dK_ptr and dV through the same
    # base plus KV_DIM, matching the campaign spelling (dvp = dK_ptr, dvx = KV_DIM).
    kv_row_off = (
        (kv_start + tile_off) * (2 * KV_DIM)
        + offs_kv[:, None] * (2 * KV_DIM)
        + offs_d[None, :]
    )
    dvp = dK_ptr
    dvx: tl.constexpr = KV_DIM
    dq0, dt0, dk0, dv0 = _gqa_bwd_head(
        tokens_tile, wk0, wv0, bv0, q0, dout0, D0, lse0, kv_mask, scale, dq0,
        USE_V_BIAS, Q_PER_KV, D_HEAD, TOKEN_DIM, TILE_K, COMPUTE_DTYPE, MASKED,
    )
    d_tok_sum = dt0
    if USE_V_BIAS:
        dbv0 += tl.sum(dv0.to(tl.float32), axis=0)
    if MASKED:
        tl.store(dK_ptr + kv_row_off + 0 * D_HEAD, dk0, mask=kv_mask[:, None])
        tl.store(dvp + kv_row_off + dvx + 0 * D_HEAD, dv0, mask=kv_mask[:, None])
    else:
        tl.store(dK_ptr + kv_row_off + 0 * D_HEAD, dk0)
        tl.store(dvp + kv_row_off + dvx + 0 * D_HEAD, dv0)
    if N_KV_HEADS >= 2:
        dq1, dt1, dk1, dv1 = _gqa_bwd_head(
            tokens_tile, wk1, wv1, bv1, q1, dout1, D1, lse1, kv_mask, scale, dq1,
            USE_V_BIAS, Q_PER_KV, D_HEAD, TOKEN_DIM, TILE_K, COMPUTE_DTYPE, MASKED,
        )
        d_tok_sum += dt1
        if USE_V_BIAS:
            dbv1 += tl.sum(dv1.to(tl.float32), axis=0)
        if MASKED:
            tl.store(dK_ptr + kv_row_off + 1 * D_HEAD, dk1, mask=kv_mask[:, None])
            tl.store(dvp + kv_row_off + dvx + 1 * D_HEAD, dv1, mask=kv_mask[:, None])
        else:
            tl.store(dK_ptr + kv_row_off + 1 * D_HEAD, dk1)
            tl.store(dvp + kv_row_off + dvx + 1 * D_HEAD, dv1)

    if MASKED:
        tl.store(
            dTokens_ptr + tok_base + offs_kv[:, None] * TOKEN_DIM + offs_td[None, :],
            d_tok_sum,
            mask=kv_mask[:, None],
        )
    else:
        tl.store(
            dTokens_ptr + tok_base + offs_kv[:, None] * TOKEN_DIM + offs_td[None, :],
            d_tok_sum,
        )
    return dq0, dq1, dbv0, dbv1


# ─── Unified GQA kernel: n_kv_heads={1,2} via constexpr branching ───


@triton.autotune(
    configs=_autotune_configs(),
    # Sequence length does not change the measured winner.
    key=[
        "Q_PER_KV",
        "N_KV_HEADS",
        "COMPUTE_DTYPE",
        "n_pix",
        "GROUPED",
    ],
)
@triton.jit
def _pixel_attn_gqa_fwd(
    Q_ptr,
    Tokens_ptr,
    Wk_ptr,
    Wv_ptr,
    Bv_ptr,
    Out_ptr,
    LSE_ptr,
    cu_seqlens_ptr,
    ProgPtr_ptr,
    ProgPix_ptr,
    scale,
    n_pix,  # autotune-key only: grid size (T1 vs T2 want different configs)
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GROUPED: tl.constexpr,
):
    # CSR program map: program p handles pixels prog_pix[prog_ptr[p]:prog_ptr[p+1]]
    # (1 = ungrouped, 2 = paired small pixels). Output/LSE stay pixel-id indexed, so the
    # layout is identical to the ungrouped kernel. Weights are loaded ONCE here and
    # shared across the program's pixels -- the amortization that makes pairing win.
    # GROUPED=False (no map given) -> program p IS pixel p; skip the map loads so the
    # default/ungrouped path has no CSR overhead vs the pre-grouping kernel.
    prog = tl.program_id(0)
    if GROUPED:
        start = tl.load(ProgPtr_ptr + prog).to(tl.int64)
        end = tl.load(ProgPtr_ptr + prog + 1).to(tl.int64)
    else:
        start = prog.to(tl.int64)
        end = start + 1

    N_Q: tl.constexpr = N_KV_HEADS * Q_PER_KV
    offs_qh = tl.arange(0, BLOCK_Q)
    qh_mask = offs_qh < Q_PER_KV
    offs_d = tl.arange(0, D_HEAD)
    offs_td = tl.arange(0, TOKEN_DIM)
    # Wk/Wv are stored per KV head, so wk0/wv0 are the parameters for KV head 0.
    wk0 = tl.load(
        Wk_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    wv0 = tl.load(
        Wv_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        bv0 = tl.load(Bv_ptr + 0 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0).to(
            tl.float32
        )
    else:
        bv0 = tl.zeros((D_HEAD,), dtype=tl.float32)
    if N_KV_HEADS >= 2:
        wk1 = tl.load(
            Wk_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        wv1 = tl.load(
            Wv_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        if USE_V_BIAS:
            bv1 = tl.load(
                Bv_ptr + 1 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0
            ).to(tl.float32)
        else:
            bv1 = tl.zeros((D_HEAD,), dtype=tl.float32)
    else:
        wk1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        wv1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        bv1 = tl.zeros((D_HEAD,), dtype=tl.float32)

    # Per-pixel forward is inlined so weights are loaded once per program and then
    # shared by the pixels in the CSR group.
    for i in range(start, end):
        pix = tl.load(ProgPix_ptr + i).to(tl.int64) if GROUPED else i
        kv_start = tl.load(cu_seqlens_ptr + pix).to(tl.int64)
        kv_end = tl.load(cu_seqlens_ptr + pix + 1).to(tl.int64)
        seqlen_k = kv_end - kv_start
        # Empty pixels would divide zero accumulators by a zero softmax denominator.
        if seqlen_k > 0:
            q_base = pix * N_Q * D_HEAD
            # Queries are laid out as [kv_head_0's Q_PER_KV queries][kv_head_1's
            # Q_PER_KV queries]... . q0 selects the first query-head group.
            q0 = tl.load(
                Q_ptr
                + q_base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            m0 = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
            l0 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
            acc0 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
            if N_KV_HEADS >= 2:
                q1 = tl.load(
                    Q_ptr
                    + q_base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                m1 = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
                l1 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
                acc1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)

            for tile_off in range(0, seqlen_k, TILE_K):
                offs_kv = tl.arange(0, TILE_K)
                kv_mask = offs_kv < (seqlen_k - tile_off)
                tok_base = (kv_start + tile_off) * TOKEN_DIM
                tokens_tile = tl.load(
                    Tokens_ptr
                    + tok_base
                    + offs_kv[:, None] * TOKEN_DIM
                    + offs_td[None, :],
                    mask=kv_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                m0, l0, acc0 = _gqa_fwd_head(
                    tokens_tile,
                    wk0,
                    wv0,
                    bv0,
                    q0,
                    kv_mask,
                    scale,
                    m0,
                    l0,
                    acc0,
                    USE_V_BIAS,
                    Q_PER_KV,
                    D_HEAD,
                    TILE_K,
                    COMPUTE_DTYPE,
                )
                if N_KV_HEADS >= 2:
                    m1, l1, acc1 = _gqa_fwd_head(
                        tokens_tile,
                        wk1,
                        wv1,
                        bv1,
                        q1,
                        kv_mask,
                        scale,
                        m1,
                        l1,
                        acc1,
                        USE_V_BIAS,
                        Q_PER_KV,
                        D_HEAD,
                        TILE_K,
                        COMPUTE_DTYPE,
                    )

            out_base = pix * N_Q * D_HEAD
            lse_base = pix * N_Q
            tl.store(
                Out_ptr
                + out_base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                acc0 / l0[:, None],
                mask=qh_mask[:, None],
            )
            # LSE in the log2 domain: m0 is the log2-scaled running max and
            # log2(l0) keeps the denominator in the same domain.
            tl.store(
                LSE_ptr + lse_base + 0 * Q_PER_KV + offs_qh,
                m0 + tl.log2(l0),
                mask=qh_mask,
            )
            if N_KV_HEADS >= 2:
                tl.store(
                    Out_ptr
                    + out_base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    acc1 / l1[:, None],
                    mask=qh_mask[:, None],
                )
                tl.store(
                    LSE_ptr + lse_base + 1 * Q_PER_KV + offs_qh,
                    m1 + tl.log2(l1),
                    mask=qh_mask,
                )


@triton.autotune(
    configs=_autotune_configs(),
    # Sequence length does not change the measured winner.
    key=[
        "Q_PER_KV",
        "N_KV_HEADS",
        "COMPUTE_DTYPE",
        "n_pix",
        "GROUPED",
    ],
)
@triton.jit
def _pixel_attn_gqa_bwd(
    Q_ptr,
    Tokens_ptr,
    Wk_ptr,
    Wv_ptr,
    Bv_ptr,
    Out_ptr,
    LSE_ptr,
    dOut_ptr,
    dQ_ptr,
    dTokens_ptr,
    dK_ptr,
    dV_ptr,  # same buffer as dK_ptr; COMBINED rows use dK_ptr + KV_DIM
    dBv_part_ptr,
    cu_seqlens_ptr,
    ProgPtr_ptr,
    ProgPix_ptr,
    scale,
    n_pix,  # autotune-key only: grid size (T1 vs T2 want different configs)
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GROUPED: tl.constexpr,
):
    # CSR program map (see forward kernel). Weights loaded once per program and
    # shared across its 1-2 pixels; dK/dV/dTokens are written per global token row
    # so the pixel-id indirection never reorders any output. GROUPED=False -> program
    # p is pixel p (no map loads; same cost as the pre-grouping kernel).
    prog = tl.program_id(0)
    if GROUPED:
        start = tl.load(ProgPtr_ptr + prog).to(tl.int64)
        end = tl.load(ProgPtr_ptr + prog + 1).to(tl.int64)
    else:
        start = prog.to(tl.int64)
        end = start + 1

    N_Q: tl.constexpr = N_KV_HEADS * Q_PER_KV
    offs_qh = tl.arange(0, BLOCK_Q)
    qh_mask = offs_qh < Q_PER_KV
    offs_d = tl.arange(0, D_HEAD)
    offs_td = tl.arange(0, TOKEN_DIM)
    wk0 = tl.load(
        Wk_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    wv0 = tl.load(
        Wv_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        bv0 = tl.load(Bv_ptr + 0 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0).to(
            tl.float32
        )
    else:
        bv0 = tl.zeros((D_HEAD,), dtype=tl.float32)
    if N_KV_HEADS >= 2:
        wk1 = tl.load(
            Wk_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        wv1 = tl.load(
            Wv_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        if USE_V_BIAS:
            bv1 = tl.load(
                Bv_ptr + 1 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0
            ).to(tl.float32)
        else:
            bv1 = tl.zeros((D_HEAD,), dtype=tl.float32)
    else:
        wk1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        wv1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        bv1 = tl.zeros((D_HEAD,), dtype=tl.float32)

    # Reduced over this program's tiles, then written out as a per-program partial.
    dbv0 = tl.zeros((D_HEAD,), dtype=tl.float32)
    dbv1 = tl.zeros((D_HEAD,), dtype=tl.float32)

    # Per-pixel backward is inlined so weights are amortized across the CSR group.
    for i in range(start, end):
        pix = tl.load(ProgPix_ptr + i).to(tl.int64) if GROUPED else i
        kv_start = tl.load(cu_seqlens_ptr + pix).to(tl.int64)
        kv_end = tl.load(cu_seqlens_ptr + pix + 1).to(tl.int64)
        seqlen_k = kv_end - kv_start
        if seqlen_k > 0:
            base = pix * N_Q * D_HEAD
            lse_b = pix * N_Q
            q0 = tl.load(
                Q_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            dout0 = tl.load(
                dOut_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            out0 = tl.load(
                Out_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            lse0 = tl.load(
                LSE_ptr + lse_b + 0 * Q_PER_KV + offs_qh, mask=qh_mask, other=0.0
            )
            dq0 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
            D0 = tl.sum(dout0.to(tl.float32) * out0.to(tl.float32), axis=1)

            if N_KV_HEADS >= 2:
                q1 = tl.load(
                    Q_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                dout1 = tl.load(
                    dOut_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                out1 = tl.load(
                    Out_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                lse1 = tl.load(
                    LSE_ptr + lse_b + 1 * Q_PER_KV + offs_qh, mask=qh_mask, other=0.0
                )
                dq1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
                D1 = tl.sum(dout1.to(tl.float32) * out1.to(tl.float32), axis=1)
            else:
                q1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=COMPUTE_DTYPE)
                dout1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=COMPUTE_DTYPE)
                lse1 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
                dq1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
                D1 = tl.zeros((BLOCK_Q,), dtype=tl.float32)

            # Run full tiles unmasked and specialize the single ragged tail.
            # Keep the tail as a loop: the equivalent `if` miscompiles its gradients.
            n_full = (seqlen_k // TILE_K) * TILE_K
            for tile_off in range(0, n_full, TILE_K):
                dq0, dq1, dbv0, dbv1 = _gqa_bwd_tile(
                    Tokens_ptr, dTokens_ptr, dK_ptr, dV_ptr, kv_start, tile_off,
                    seqlen_k, scale,
                    wk0, wv0, bv0, q0, dout0, D0, lse0, dq0,
                    wk1, wv1, bv1, q1, dout1, D1, lse1, dq1,
                    dbv0, dbv1,
                    USE_V_BIAS, Q_PER_KV, N_KV_HEADS, D_HEAD, TOKEN_DIM, KV_DIM,
                    TILE_K, COMPUTE_DTYPE, MASKED=False,
                )
            for tile_off in range(n_full, seqlen_k, TILE_K):
                dq0, dq1, dbv0, dbv1 = _gqa_bwd_tile(
                    Tokens_ptr, dTokens_ptr, dK_ptr, dV_ptr, kv_start, tile_off,
                    seqlen_k, scale,
                    wk0, wv0, bv0, q0, dout0, D0, lse0, dq0,
                    wk1, wv1, bv1, q1, dout1, D1, lse1, dq1,
                    dbv0, dbv1,
                    USE_V_BIAS, Q_PER_KV, N_KV_HEADS, D_HEAD, TOKEN_DIM, KV_DIM,
                    TILE_K, COMPUTE_DTYPE, MASKED=True,
                )

            tl.store(
                dQ_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                dq0,
                mask=qh_mask[:, None],
            )
            if N_KV_HEADS >= 2:
                tl.store(
                    dQ_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    dq1,
                    mask=qh_mask[:, None],
                )

    # [program, KV_DIM] partials, reduced over programs on the host.
    if USE_V_BIAS:
        b_off = prog * KV_DIM + offs_d
        tl.store(dBv_part_ptr + b_off + 0 * D_HEAD, dbv0)
        if N_KV_HEADS >= 2:
            tl.store(dBv_part_ptr + b_off + 1 * D_HEAD, dbv1)


# ─── Custom op registration ──────────────────────────────────────────
# The custom-op boundary lets PyTorch treat the Triton launch as a single op for
# autograd/fake tensor purposes while we keep the real launch logic in Python.


def _gqa_fwd_impl(
    Q,
    tokens,
    W_k,
    W_v,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    # Empty CSR map => ungrouped: one program per pixel, kernel derives pixel =
    # program_id (GROUPED=False) and skips the per-program map loads.
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    n_q_heads = Q.shape[1]
    d_head = Q.shape[2]
    block_q = max(16, _next_power_of_2(q_per_kv))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    # The Triton kernels below use flat pointer math for packed [group, head, d]
    # storage and do not take explicit tensor strides. Multi-phase q/head slices
    # are views with the original group stride, so materialize packed inputs here.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_v = B_v.contiguous()
    Out = torch.zeros_like(Q)
    # Empty pixels are skipped by the kernel.
    LSE = torch.full(
        (n_groups, n_q_heads), float("-inf"), device=Q.device, dtype=torch.float32
    )
    _pixel_attn_gqa_fwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
    )
    _maybe_print_autotune_choice(
        "fwd",
        _pixel_attn_gqa_fwd,
        q_per_kv,
        n_kv_heads,
        compute_dtype,
    )
    return Out, LSE


def _gqa_bwd_impl(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    d_head = Q.shape[2]
    kv_dim = n_kv_heads * d_head
    block_q = max(16, _next_power_of_2(q_per_kv))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    torch_compute_dtype = torch.float32 if force_fp32 else torch.bfloat16
    # Backward sees the original saved inputs from the custom op; for multi-phase
    # q/head slicing those can be non-contiguous views, which breaks the kernel's
    # flat indexing unless we repack them first.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_v = B_v.contiguous()
    Out = Out.contiguous()
    LSE = LSE.contiguous()
    dOut = dOut.contiguous()
    # Empty pixels leave dQ untouched; every packed token is written exactly once.
    dQ = torch.zeros_like(Q)
    d_tokens = torch.empty_like(tokens)
    # Emit per-token [dK | dV] rows for one dense weight-gradient GEMM.
    dKV = torch.empty(
        tokens.shape[0], 2 * kv_dim, device=Q.device, dtype=torch_compute_dtype
    )
    dBv_part = (
        torch.empty(n_programs, kv_dim, device=Q.device, dtype=torch.float32)
        if use_v_bias
        else Q.new_empty((0,))
    )
    _pixel_attn_gqa_bwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        Out,
        LSE,
        dOut,
        dQ,
        d_tokens,
        dKV,
        dKV,
        dBv_part,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        KV_DIM=kv_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
    )
    _maybe_print_autotune_choice(
        "bwd",
        _pixel_attn_gqa_bwd,
        q_per_kv,
        n_kv_heads,
        compute_dtype,
    )
    # Keep the GEMM accumulator in fp32 before splitting [dW_k | dW_v].
    tokens_compute = tokens.to(torch_compute_dtype)
    dW_kv = torch.mm(dKV.t(), tokens_compute, out_dtype=torch.float32)
    dW_k = dW_kv[:kv_dim].clone()
    dW_v = dW_kv[kv_dim:].clone()
    if use_v_bias:
        dB_v = dBv_part.sum(dim=0)
    else:
        # B_v is the empty placeholder here; the gradient has to match it, or the fake and
        # real kernels disagree on the output shape.
        dB_v = torch.zeros_like(B_v, dtype=torch.float32)
    dW_k = dW_k if W_k.dtype == dW_k.dtype else dW_k.to(W_k.dtype)
    dW_v = dW_v if W_v.dtype == dW_v.dtype else dW_v.to(W_v.dtype)
    if B_v.dtype != dB_v.dtype:
        dB_v = dB_v.to(B_v.dtype)
    return dQ, d_tokens, dW_k, dW_v, dB_v


@torch.library.custom_op("healda::pixel_attn_fwd", mutates_args=())
def pixel_attn_fwd(
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_impl(
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_fwd.register_fake
def _fake_fwd(
    Q,
    tokens,
    W_k,
    W_v,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    # Fake registrations mirror output metadata so torch.compile/export can trace
    # through the custom op without running the Triton kernel.
    n_groups, n_q_heads, d_head = Q.shape
    return Q.new_empty((n_groups, n_q_heads, d_head)), Q.new_empty(
        (n_groups, n_q_heads), dtype=torch.float32
    )


@torch.library.custom_op("healda::pixel_attn_bwd", mutates_args=())
def pixel_attn_bwd(
    dOut: torch.Tensor,
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_v: torch.Tensor,
    Out: torch.Tensor,
    LSE: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return _gqa_bwd_impl(
        dOut,
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_bwd.register_fake
def _fake_bwd(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    return (
        Q.new_empty(Q.shape),
        tokens.new_empty(tokens.shape),
        W_k.new_empty(W_k.shape),
        W_v.new_empty(W_v.shape),
        B_v.new_empty(B_v.shape),
    )


def _setup_context(ctx, inputs, output):
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    ) = inputs
    Out, LSE = output
    # Save the packed tensors the Triton backward expects rather than rebuilding
    # projections during the autograd callback.
    ctx.save_for_backward(
        Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, prog_ptr, prog_pix
    )
    ctx.scale = scale
    ctx.q_per_kv = q_per_kv
    ctx.token_dim = token_dim
    ctx.n_kv_heads = n_kv_heads
    ctx.use_v_bias = use_v_bias
    ctx.force_fp32 = force_fp32


def _backward(ctx, grad_Out, grad_LSE):
    del grad_LSE
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
    ) = ctx.saved_tensors
    dQ, d_tokens, dW_k, dW_v, dB_v = pixel_attn_bwd(
        grad_Out,
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        ctx.scale,
        ctx.q_per_kv,
        ctx.token_dim,
        ctx.n_kv_heads,
        ctx.use_v_bias,
        ctx.force_fp32,
    )
    # One grad slot per fwd input: 5 real grads then None for
    # cu_seqlens_k, prog_ptr, prog_pix, scale, q_per_kv, token_dim,
    # n_kv_heads, use_v_bias, force_fp32.
    return (
        dQ,
        d_tokens,
        dW_k,
        dW_v,
        dB_v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


pixel_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def _pixel_attention_gqa(
    Q,
    tokens,
    W_k,
    W_v,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    n_kv_heads,
    scale,
    force_fp32=False,
):
    n_q_heads = Q.shape[1]
    q_per_kv = n_q_heads // n_kv_heads
    token_dim = tokens.shape[1]
    use_v_bias = B_v is not None
    if B_v is None:
        # The custom op has a fixed tensor schema, so use an empty placeholder when
        # the bias is logically absent.
        B_v = W_v.new_empty((0,))
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_v = B_v.contiguous()
    Out, _LSE = pixel_attn_fwd(
        Q,
        tokens,
        W_k,
        W_v,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )
    return Out


def pixel_attention(
    Q,
    tokens,
    W_k,
    W_v,
    cu_seqlens_k,
    n_kv_heads=1,
    scale=None,
    B_v=None,
    force_fp32=False,
    group_map=None,
):
    # ``group_map`` is an optional CSR map that packs multiple
    # small pixels into one kernel program (built once per batch in the dataloader,
    # carried on AttentionPacking). When absent we pass an empty map, which the
    # kernel treats as one-program-per-pixel with no CSR overhead (ungrouped path).
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    n_q_heads = Q.shape[1]
    if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
        raise ValueError(
            f"pixel_attention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
        )
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )
    kv_dim = n_kv_heads * Q.shape[-1]
    token_dim = tokens.shape[1]
    if W_k.shape != (kv_dim, token_dim) or W_v.shape != (kv_dim, token_dim):
        raise ValueError(
            f"Expected W_k/W_v shape {(kv_dim, token_dim)}, "
            f"got W_k={tuple(W_k.shape)}, W_v={tuple(W_v.shape)}"
        )
    if B_v is not None and B_v.shape != (kv_dim,):
        raise ValueError(f"Expected B_v shape {(kv_dim,)}, got B_v={tuple(B_v.shape)}")
    # K bias only adds a per-query constant shift to the logits, which softmax
    # cancels exactly. Dropping it avoids carrying a mathematically redundant
    # term that can still pick up small finite-precision gradient noise.
    # Only V bias is material to the kernel path.

    if group_map is None:
        prog_ptr = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
        prog_pix = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
    else:
        prog_ptr = group_map.program_ptr
        prog_pix = group_map.program_pixels

    if n_kv_heads <= 2:
        return _pixel_attention_gqa(
            Q,
            tokens,
            W_k,
            W_v,
            B_v,
            cu_seqlens_k,
            prog_ptr,
            prog_pix,
            n_kv_heads,
            scale,
            force_fp32=force_fp32,
        )

    # For larger grouped-query layouts, run the same kernel in two-KV-head
    # phases and concatenate the head blocks back in the original order.
    n_phases = n_kv_heads // 2
    q_per_phase = n_q_heads // n_phases
    d_head = Q.shape[-1]
    kv_rows_per_phase = 2 * d_head
    outs = []
    for p in range(n_phases):
        q_slice = Q[:, p * q_per_phase : (p + 1) * q_per_phase]
        wk_slice = W_k[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        wv_slice = W_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        bv_slice = (
            None
            if B_v is None
            else B_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        )
        outs.append(
            _pixel_attention_gqa(
                q_slice,
                tokens,
                wk_slice,
                wv_slice,
                bv_slice,
                cu_seqlens_k,
                prog_ptr,
                prog_pix,
                2,
                scale,
                force_fp32=force_fp32,
            )
        )
    return torch.cat(outs, dim=1)



_AUTOTUNERS = {
    "pixel_attn_gqa_fwd": _pixel_attn_gqa_fwd,
    "pixel_attn_gqa_bwd": _pixel_attn_gqa_bwd,
}
_CONFIGS_PINNED = False


def ensure_pixel_attention_configs():
    """Pin measured Triton launch configurations once per process."""
    global _CONFIGS_PINNED
    if _CONFIGS_PINNED:
        return
    _CONFIGS_PINNED = True
    arch = _arch_key()
    if arch is None:
        return
    for name, tuner in _AUTOTUNERS.items():
        kind = "bwd" if name.endswith("bwd") else "fwd"
        pinned = _ARCH_CONFIGS.get((arch, kind))
        if pinned is None:
            continue
        tile_k, warps, stages = pinned
        tuner.configs = [
            triton.Config({"TILE_K": tile_k}, num_warps=warps, num_stages=stages)
        ]

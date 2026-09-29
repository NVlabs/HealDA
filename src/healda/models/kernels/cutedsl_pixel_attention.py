# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL pixel/observation cross-attention, forward and backward.

Covers the same region as ``triton_pixel_attention.pixel_attention``: ragged grouped-query
attention from one latent per pixel to that pixel's slice of packed observation tokens.

Projection folding is what makes it fast. Because

    score[h, t] = Q[h, :] . (W_k[kv(h)] @ tok[t]) = (Q[h, :] @ W_k[kv(h)]) . tok[t]

the per-token K projection disappears: Q is pre-multiplied by W_k once per pixel and the score
GEMM runs against the raw packed tokens. The same holds for V, so per-token work drops from
8192 to 4096 MAC and K and V become the same shared-memory tile.

Per-pixel token counts are extremely right-skewed -- one pixel can hold 25k tokens against a
mean of 710 -- so a pixel-granular work queue is bounded below by the largest single pixel.
Both forward variants split long pixels into tile-aligned chunks that different CTAs process
independently, each publishing its own online-softmax state and folding siblings in with the
stable merge, and hoist long pixels into a priority list drained before the index-ordered sweep.

Every tile subtracts its exact data-dependent row maximum. Real scores reach well past the
point where a zero-anchored exponential overflows, so the max subtraction is load-bearing
rather than a formality.

Two forward variants serve the two token-density regimes and are selected by mean tokens per
pixel; they share a skeleton but not their tiling, staging or tail handling, so they are kept
whole rather than merged behind a flag.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp

from healda.models.kernels.cutedsl_arch import unsupported_arch_reasons

# Baked into the tile shapes and shared-memory layouts of both directions.
N_Q_HEADS = 64
N_KV_HEADS = 2
D_HEAD = 32
TOKEN_DIM = D_HEAD

# Observation count changes every step while the pixel count is fixed by the model config.
# TT is only the declared extent of the token tensor, so pinning it keeps the compile cache
# keyed on the pixel count alone. Verified correct well past this bound.
TT_LAYOUT_BOUND = 1 << 27

LOG2_E = 1.4426950408889634
NEG_BIG = -1.0e30              # finite "-inf" seed for the running row max

NHEAD = N_Q_HEADS              # query heads per pixel (== tile_m)
DHEAD = D_HEAD                 # head dim == token dim
NKV = N_KV_HEADS
MAXSLOT = 64                   # 63 pixels exceed the split threshold at the largest
                               # production shape; one slot of headroom
PREP_THREADS = 256

# Forward.
TILE_N_L5 = 64
TILE_N_L6 = 32
MAXCH = 8                      # max chunks a split pixel is cut into
SCR_PER_CH = 2560              # per split chunk: 2048 acc_O + n_rows*nthread {l, m}

# Backward.
SCORE_N = 16
BWD_NWARP = 4
BWD_NTHREAD = BWD_NWARP * 32
FIN_THREADS = 1024
FIN_WARPS = FIN_THREADS // 32
SCR_WK = 0
SCR_WV = 2048
SCR_BV = 4096
SCR_PER_CTA = 4224
MAXCHUNK = 4
NBUCK = 8
MBUCK = 8 + 2 * MAXSLOT
RSTATE = 2048                  # one pixel's fp32 R state, thread-interleaved


# Empty list == this configuration is supported.
def unsupported_reasons(*, n_q_heads, n_kv_heads, d_head, token_dim):
    reasons = unsupported_arch_reasons()
    for name, got, want in (("n_q_heads", n_q_heads, N_Q_HEADS),
                            ("n_kv_heads", n_kv_heads, N_KV_HEADS),
                            ("d_head", d_head, D_HEAD),
                            ("token_dim", token_dim, TOKEN_DIM)):
        if got != want:
            reasons.append(f"{name}=={want} (got {got})")
    return reasons


# ------------------------ forward: long-sequence variant
# ---------------------------------------------------------------------------

def smem_layout_atom_long(dtype, k_dim):
    dtype_byte = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte
    smem_k_block_size = (
        128 if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    swizzle_bits = (
        4 if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if k_dim % 32 == 0 else 16, smem_k_block_size), order=(1, 0)
        ),
    )


def convert_layout_acc_mn_long(acc_layout):
    """((2,2), MMA_M, MMA_N) -> ((2, MMA_M), (2, MMA_N))."""
    cm = cute.make_layout(acc_layout.shape)
    shape = (
        (cm.shape[0][1], cm.shape[1]),
        (cm.shape[0][0], *cm.shape[0][2:], cm.shape[2]),
        *cm.shape[3:],
    )
    stride = (
        (cm.stride[0][1], cm.stride[1]),
        (cm.stride[0][0], *cm.stride[0][2:], cm.stride[2]),
        *cm.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def acc_mn_view_long(acc):
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn_long(acc.layout))


def convert_layout_acc_frgA_long(acc_layout):
    """C fragment of an (M,N) tile -> A fragment of an (M,K=N) mma."""
    if cute.rank(acc_layout.shape[0]) == 3:
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        return cute.make_layout(
            ((l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
             l.shape[1],
             (l.shape[0][2][1], l.shape[2])),
            stride=((l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                    l.stride[1],
                    (l.stride[0][2][1], l.stride[2])),
        )
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
        stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
    )


def transpose_view_long(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def head_perm_layout_long(nwarp):
    """Relabel heads so each two-warp path warp owns one 32-head KV group."""
    if nwarp == 2:
        return cute.make_layout(((16, 2, 2), DHEAD), stride=((1, 32, 16), NHEAD))
    return cute.make_layout((NHEAD, DHEAD), stride=(1, NHEAD))


def head_of_row_long(r, nwarp):
    if nwarp == 2:
        return (r % 16) + ((r // 16) % 2) * 32 + (r // 32) * 16
    return r


def warp_sum4_long(x):
    x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=2)
    return x





@cute.jit
def gemm_rs_long(tiled_mma, acc, tCrA, tCrB, tCsB, smem_thr_copy_B):
    """A already in registers, B streamed from smem via ldmatrix."""
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_view[None, None, 0])
    nk = cute.size(tCrA.shape[2])
    for k in cutlass.range_constexpr(nk):
        if const_expr(k < nk - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def gemm_ss_long(tiled_mma, acc, tCrA, tCrB, tCsA, tCsB, smem_thr_copy_A, smem_thr_copy_B):
    tCrA_view = smem_thr_copy_A.retile(tCrA)
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_view[None, None, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_view[None, None, 0])
    nk = cute.size(tCsA.shape[2])
    for k in cutlass.range_constexpr(nk):
        if const_expr(k < nk - 1):
            cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_view[None, None, k + 1])
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def attention_tile_long(
    tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, col_limit,
    tSrQ, tSrTok, tSsTok, cpB_n, row_max, row_sum, n_rows,
    acc_O, acc_O_mn, tOrTok, tOsTok, cpB_t,
    mask: cutlass.Constexpr, first: cutlass.Constexpr,
):
    """One score tile with exact FP32 online-softmax state."""
    acc_S.fill(0.0)
    gemm_rs_long(tiled_mma, acc_S, tSrQ, tSrTok, tSsTok, cpB_n)
    if const_expr(mask):
        for i in cutlass.range_constexpr(n_S):
            acc_S[i] = acc_S[i] if t0ScS[i][1] < col_limit else -Float32.inf

    for r in cutlass.range_constexpr(n_rows):
        mt = acc_S_mn[r, None].load().reduce(
            cute.ReductionOp.MAX, Float32(NEG_BIG), 0)
        o = cute.arch.shuffle_sync_bfly(mt, offset=1)
        mt = o if o > mt else mt
        o = cute.arch.shuffle_sync_bfly(mt, offset=2)
        mt = o if o > mt else mt
        if const_expr(first):
            mn = mt
            row_max[r] = mt
        else:
            mn = mt if mt > row_max[r] else row_max[r]
            sc = cute.math.exp2(row_max[r] - mn, fastmath=True)
            row_max[r] = mn
            row_sum[r] = row_sum[r] * sc
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * sc)
        p_row = cute.math.exp2(acc_S_mn[r, None].load() - mn, fastmath=True)
        row_sum[r] = p_row.reduce(cute.ReductionOp.ADD, row_sum[r], 0)
        acc_S_mn[r, None].store(p_row)

    rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
    rP.store(acc_S.load().to(cutlass.BFloat16))
    tOrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA_long(rP.layout))
    gemm_rs_long(tiled_mma, acc_O, tOrP, tOrTok, tOsTok, cpB_t)


@cute.jit
def run_tiles_full_long(
    tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, tSrQ, tSrTok, tSsTok_ld, cpB_n,
    row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok, tOsTok_ld, cpB_t,
    gmem_copy_load, tKgTok, tKsTok, t0KcTok, Lc, n_tiles, n_full, col_base,
    tok_row_base,
    n_cpy_m: cutlass.Constexpr, tile_n: cutlass.Constexpr,
    NSTAGE: cutlass.Constexpr,
):
    """Unrolled full-width sweep used by the short-group T32 specialization."""
    acc_O.fill(0.0)
    row_sum.fill(0.0)
    row_max.fill(NEG_BIG)

    if n_full > 0:
        cute.arch.cp_async_wait_group(NSTAGE - 2)
        cute.arch.barrier()
        nl = Int32(NSTAGE - 1)
        if nl < n_tiles:
            rows_valid = Lc - nl * tile_n - tok_row_base
            for m in cutlass.range_constexpr(n_cpy_m):
                if t0KcTok[0, m, 0][0] < rows_valid:
                    cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                              tKsTok[None, m, None, (NSTAGE - 1) % NSTAGE])
        cute.arch.cp_async_commit_group()
        attention_tile_long(
            tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, Int32(0),
            tSrQ, tSrTok, tSsTok_ld[None, None, None, 0], cpB_n,
            row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
            tOsTok_ld[None, None, None, 0], cpB_t, False, True,
        )

        for nn in cutlass.range(n_full - 1):
            n = nn + Int32(1)
            stage = n % NSTAGE
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()
            nl = n + NSTAGE - 1
            if nl < n_tiles:
                rows_valid = Lc - nl * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                                  tKsTok[None, m, None, nl % NSTAGE])
            cute.arch.cp_async_commit_group()
            attention_tile_long(
                tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, Int32(0),
                tSrQ, tSrTok, tSsTok_ld[None, None, None, stage], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, stage], cpB_t, False, False,
            )

        if n_full < n_tiles:
            n = n_full
            stage = n % NSTAGE
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()
            nl = n + NSTAGE - 1
            if nl < n_tiles:
                rows_valid = Lc - nl * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                                  tKsTok[None, m, None, nl % NSTAGE])
            cute.arch.cp_async_commit_group()
            attention_tile_long(
                tiled_mma, acc_S, acc_S_mn, n_S, t0ScS,
                Lc - n * tile_n - col_base,
                tSrQ, tSrTok, tSsTok_ld[None, None, None, stage], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, stage], cpB_t, True, False,
            )
    else:
        if n_tiles > Int32(0):
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()
            cute.arch.cp_async_commit_group()
            attention_tile_long(
                tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, Lc - col_base,
                tSrQ, tSrTok, tSsTok_ld[None, None, None, 0], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, 0], cpB_t, True, True,
            )


@cute.jit
def run_tiles_long(
    tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, tSrQ, tSrTok, tSsTok_ld, cpB_n,
    row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok, tOsTok_ld,
    cpB_t, half,
    gmem_copy_load, tKgTok, tKsTok, t0KcTok, Lc, n_tiles, n_full, col_base,
    tok_row_base,
    n_cpy_m: cutlass.Constexpr, tile_n: cutlass.Constexpr,
    tile_h: cutlass.Constexpr, NSTAGE: cutlass.Constexpr,
):
    """Sweep one work item's token tiles through the cp.async pipeline.

    Steady-state tiles keep the full `tile_n` score width -- that is where the
    MMA/exp2 dependency chains are long enough to hide their own latency.  The
    single ragged tail instead runs in 16-column pieces: two pieces for T32 and
    four for T64.  Sixteen is the BF16 MMA K quantum for the probability-value
    product, while four-way T64 tails reduce L5 padding without changing the
    steady-state path.  The narrow fragments are live only inside the tail, so
    they reuse the steady-state accumulator's registers.
    """
    (acc_Sh, acc_Sh_mn, n_Sh, t0ScSh, tSrTokh, tSsTokh_ld,
     tOrTokh, tOsTokh_ld, n_rowsh, col_baseh) = half
    acc_O.fill(0.0)
    row_sum.fill(0.0)
    row_max.fill(NEG_BIG)

    if n_full > Int32(0):
        cute.arch.cp_async_wait_group(NSTAGE - 2)
        cute.arch.barrier()

        nl = Int32(NSTAGE - 1)
        if nl < n_tiles:
            rows_valid = Lc - nl * tile_n - tok_row_base
            for m in cutlass.range_constexpr(n_cpy_m):
                if t0KcTok[0, m, 0][0] < rows_valid:
                    cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                              tKsTok[None, m, None, nl % NSTAGE])
        cute.arch.cp_async_commit_group()

        attention_tile_long(
            tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, Int32(0),
            tSrQ, tSrTok, tSsTok_ld[None, None, None, 0], cpB_n,
            row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
            tOsTok_ld[None, None, None, 0], cpB_t, False, True,
        )

        for nn in cutlass.range(n_full - Int32(1)):
            n = nn + Int32(1)
            stage = n % NSTAGE
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()

            nl = n + NSTAGE - 1
            if nl < n_tiles:
                rows_valid = Lc - nl * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                                  tKsTok[None, m, None, nl % NSTAGE])
            cute.arch.cp_async_commit_group()

            attention_tile_long(
                tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, Int32(0),
                tSrQ, tSrTok, tSsTok_ld[None, None, None, stage], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, stage], cpB_t, False, False,
            )

    if n_full < n_tiles:
        n = n_full
        stage = n % NSTAGE
        cute.arch.cp_async_wait_group(NSTAGE - 2)
        cute.arch.barrier()

        nl = n + NSTAGE - 1
        if nl < n_tiles:
            rows_valid = Lc - nl * tile_n - tok_row_base
            for m in cutlass.range_constexpr(n_cpy_m):
                if t0KcTok[0, m, 0][0] < rows_valid:
                    cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                              tKsTok[None, m, None, nl % NSTAGE])
        cute.arch.cp_async_commit_group()

        rem = Lc - n * tile_n
        if n_full > Int32(0):
            attention_tile_long(
                tiled_mma, acc_Sh, acc_Sh_mn, n_Sh, t0ScSh, rem - col_baseh,
                tSrQ, tSrTokh,
                tSsTokh_ld[None, None, None, stage * (tile_n // tile_h)], cpB_n,
                row_max, row_sum, n_rowsh, acc_O, acc_O_mn, tOrTokh,
                tOsTokh_ld[None, None, None, stage * (tile_n // tile_h)], cpB_t,
                True, False,
            )
        else:
            attention_tile_long(
                tiled_mma, acc_Sh, acc_Sh_mn, n_Sh, t0ScSh, rem - col_baseh,
                tSrQ, tSrTokh,
                tSsTokh_ld[None, None, None, stage * (tile_n // tile_h)], cpB_n,
                row_max, row_sum, n_rowsh, acc_O, acc_O_mn, tOrTokh,
                tOsTokh_ld[None, None, None, stage * (tile_n // tile_h)], cpB_t,
                True, True,
            )
        if rem > Int32(tile_h):
            attention_tile_long(
                tiled_mma, acc_Sh, acc_Sh_mn, n_Sh, t0ScSh,
                rem - Int32(tile_h) - col_baseh,
                tSrQ, tSrTokh,
                tSsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 1],
                cpB_n,
                row_max, row_sum, n_rowsh, acc_O, acc_O_mn,
                tOrTokh,
                tOsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 1], cpB_t,
                True, False,
            )
        if rem > Int32(2 * tile_h):
            attention_tile_long(
                tiled_mma, acc_Sh, acc_Sh_mn, n_Sh, t0ScSh,
                rem - Int32(2 * tile_h) - col_baseh,
                tSrQ, tSrTokh,
                tSsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 2], cpB_n,
                row_max, row_sum, n_rowsh, acc_O, acc_O_mn,
                tOrTokh,
                tOsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 2], cpB_t,
                True, False,
            )
        if rem > Int32(3 * tile_h):
            attention_tile_long(
                tiled_mma, acc_Sh, acc_Sh_mn, n_Sh, t0ScSh,
                rem - Int32(3 * tile_h) - col_baseh,
                tSrQ, tSrTokh,
                tSsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 3], cpB_n,
                row_max, row_sum, n_rowsh, acc_O, acc_O_mn,
                tOrTokh,
                tOsTokh_ld[None, None, None,
                            stage * (tile_n // tile_h) + 3], cpB_t,
                True, False,
            )


# --------------------------------------------------------------------------- #
# prep kernel: build the priority work list from cu_seqlens
# --------------------------------------------------------------------------- #
@cute.kernel
def prep_kernel_long(
    mCu: cute.Tensor,       # (TP+1,) i32
    mWit: cute.Tensor,      # (MAXITEM*4,) i32
    mMeta: cute.Tensor,     # (8+2*MAXSLOT,) i32 -- [0]=items [1]=slots [2]=cursor
    TP: Int32,
    CH: Int32,
    CH2: Int32,
    tile_n: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    p = bidx * PREP_THREADS + tidx
    if p < TP:
        L = mCu[p + 1] - mCu[p]
        if L > CH2:
            nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
            split = False
            tpc = nt
            nc = Int32(1)
            slot = Int32(-1)
            if L > CH:
                s = cute.arch.atomic_add(mMeta.iterator + 1, Int32(1))
                if s < Int32(MAXSLOT):
                    slot = s
                    nc0 = (L + CH - Int32(1)) // CH
                    nc0 = cutlass.min(nc0, Int32(MAXCH))
                    tpc = (nt + nc0 - Int32(1)) // nc0
                    nc = (nt + tpc - Int32(1)) // tpc
                    split = True
            if split:
                base = cute.arch.atomic_add(mMeta.iterator + 0, nc)
                mMeta[8 + slot] = nc
                mMeta[8 + MAXSLOT + slot] = nc
                for c in cutlass.range(nc):
                    lo = c * tpc
                    hi = cutlass.min(lo + tpc, nt)
                    w4 = (base + c) * Int32(4)
                    mWit[w4] = p
                    mWit[w4 + 1] = lo
                    mWit[w4 + 2] = hi
                    mWit[w4 + 3] = slot * Int32(MAXCH) + c
            else:
                base = cute.arch.atomic_add(mMeta.iterator + 0, Int32(1))
                w4 = base * Int32(4)
                mWit[w4] = p
                mWit[w4 + 1] = Int32(0)
                mWit[w4 + 2] = nt
                mWit[w4 + 3] = Int32(-1)


# --------------------------------------------------------------------------- #
# main kernel
# --------------------------------------------------------------------------- #
@cute.kernel
def pca_kernel_long(
    mQ: cute.Tensor,        # (TP*64, 32)   bf16
    mTok: cute.Tensor,      # (TT, 32)      bf16
    mWkf: cute.Tensor,      # (2048,)       bf16   W_k flattened
    mWvf: cute.Tensor,      # (2048,)       bf16   W_v flattened
    mBv: cute.Tensor,       # (64,)         bf16
    mCu: cute.Tensor,       # (TP+1,)       i32
    mOut: cute.Tensor,      # (TP*64, 32)   bf16
    mLSE: cute.Tensor,      # (TP, 64)      f32
    mMeta: cute.Tensor,     # (8+MAXSLOT,)  i32
    mWit: cute.Tensor,      # (MAXITEM*4,)  i32
    mScr: cute.Tensor,      # (MAXSLOT*MAXCH*SCR_PER_CH,) f32
    scale_log2: Float32,
    TP: Int32,
    CH2: Int32,
    tile_n: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
    nwarp: cutlass.Constexpr,
    NSTAGE: cutlass.Constexpr,
    sched: cutlass.Constexpr,
    sQ_layout: cute.ComposedLayout,
    sTok_layout: cute.ComposedLayout,
    sTokH_layout: cute.ComposedLayout,
    sW_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    gmem_copy_load: cute.TiledCopy,
    gmem_copy_store: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(tidx // 32)
    nthread = nwarp * 32
    kv = warp_idx // (nwarp // 2)
    hperm = head_perm_layout_long(nwarp)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQ.get_tensor(sQ_layout)
    sO = storage.sQ.get_tensor(sO_layout)      # Q is dead by the time O is staged
    sTok = storage.sTok.get_tensor(sTok_layout)
    # Same bytes, re-tiled at the score granularity.  tile_to_shape of the same
    # 8x32 swizzle atom over (tile_c, D, NSTAGE*nsub) is linearly identical to
    # (tile_n, D, NSTAGE), so cp.async and the MMA read one buffer through two
    # views with no copy and no extra SMEM.
    sTokH = storage.sTok.get_tensor(sTokH_layout)
    sWk = storage.sWk.get_tensor(sW_layout)
    sWv = storage.sWv.get_tensor(sW_layout)
    sBv = storage.sBv.get_tensor(cute.make_layout(NHEAD))
    sLSE = storage.sLSE.get_tensor(cute.make_layout(NHEAD))
    sIdxT = storage.sIdx.get_tensor(cute.make_layout(4))
    sTokFlat = storage.sTok.get_tensor(cute.make_layout(cute.cosize(sTok_layout)))

    # ---- one-time CTA setup -------------------------------------------------
    # Lane-contiguous, not thread-major: with `lin = tidx * 32 + i` every warp
    # instruction touched 32 addresses 64 B apart, and NCU charged this pair of
    # lines 4.55 M excessive global sectors -- 17% of the whole kernel's sector
    # traffic -- plus 0.45 M excessive shared wavefronts on the store side.
    # Striding by `nthread` makes each warp read 64 contiguous bytes instead.
    for i in cutlass.range_constexpr(2048 // nthread):
        lin = i * nthread + tidx
        c = (lin % 1024) // DHEAD
        d = lin % DHEAD
        h = lin // 1024
        sWk[c, d, h] = mWkf[lin]
        sWv[c, d, h] = mWvf[lin]
    nzero = cute.cosize(sTok_layout) // nthread
    for i in cutlass.range_constexpr(nzero):
        sTokFlat[i * nthread + tidx] = cutlass.BFloat16(0.0)
    if tidx < NHEAD:
        sBv[tidx] = mBv[tidx].to(Float32)

    # ---- static partitions --------------------------------------------------
    thr_mma = tiled_mma.get_slice(tidx)
    gmem_thr_load = gmem_copy_load.get_slice(tidx)
    gmem_thr_store = gmem_copy_store.get_slice(tidx)

    sWkT = transpose_view_long(sWk)                      # (d, c, kv) for the B operand
    sTokT = transpose_view_long(sTok)                    # (d, tok, stage)
    sTokHT = transpose_view_long(sTokH)                  # (d, tok_h, sub)

    ldm_n = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                cutlass.BFloat16)
    ldm_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                                cutlass.BFloat16)
    cpA_n = cute.make_tiled_copy_A(ldm_n, tiled_mma).get_slice(tidx)
    cpB_n = cute.make_tiled_copy_B(ldm_n, tiled_mma).get_slice(tidx)
    cpB_t = cute.make_tiled_copy_B(ldm_t, tiled_mma).get_slice(tidx)

    # Qeff = Q @ W_k[kv]   (M=64, N=32, K=32)
    tQrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
    tQsQ_ld = cpA_n.partition_S(sQ)
    tQrWk = thr_mma.make_fragment_B(thr_mma.partition_B(sWkT[None, None, 0]))
    tQsWk_ld = cpB_t.partition_S(sWkT)

    # S = Qeff @ tok^T     (M=64, N=tile_n, K=32)
    tSrTok = thr_mma.make_fragment_B(thr_mma.partition_B(sTok[None, None, 0]))
    tSsTok_ld = cpB_n.partition_S(sTok)
    # The ragged tail is never live at the same time as the steady-state tile,
    # so its three fragments are aliased onto the steady-state registers rather
    # than allocated beside them.  Their compact layouts share the leading
    # strides and only shrink the MMA_N / MMA_K extent, so element i of the tail
    # fragment is element i of the full one.  This body sits exactly on the
    # 128-register cliff that 8 CTAs/SM needs, and NCU attributes 9.6 M
    # instructions -- 6.4% of the kernel -- to swizzled smem addresses that are
    # loop-invariant but get rematerialised per work item for want of registers.
    _tSrTokh = thr_mma.make_fragment_B(thr_mma.partition_B(sTokH[None, None, 0]))
    tSrTokh = cute.make_tensor(tSrTok.iterator, _tSrTokh.layout) \
        if const_expr(nwarp == 2) else _tSrTokh
    tSsTokh_ld = cpB_n.partition_S(sTokH)

    # acc = P @ tok        (M=64, N=32, K=tile_n)
    tOrTok = thr_mma.make_fragment_B(thr_mma.partition_B(sTokT[None, None, 0]))
    tOsTok_ld = cpB_t.partition_S(sTokT)
    _tOrTokh = thr_mma.make_fragment_B(thr_mma.partition_B(sTokHT[None, None, 0]))
    tOrTokh = cute.make_tensor(tOrTok.iterator, _tOrTokh.layout) \
        if const_expr(nwarp == 2) else _tOrTokh
    tOsTokh_ld = cpB_t.partition_S(sTokHT)

    # out = accn @ W_v[kv] (M=64, N=32, K=32)
    tErWv = thr_mma.make_fragment_B(thr_mma.partition_B(sWv[None, None, 0]))
    tEsWv_ld = cpB_n.partition_S(sWv)

    acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((NHEAD, tile_n)), Float32)
    _acc_Sh = cute.make_rmem_tensor(thr_mma.partition_shape_C((NHEAD, tile_h)),
                                    Float32)
    acc_Sh = cute.make_tensor(acc_S.iterator, _acc_Sh.layout) \
        if const_expr(nwarp == 2) else _acc_Sh
    acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((NHEAD, DHEAD)), Float32)
    acc_E = acc_O
    acc_Q = acc_O
    acc_S_mn = acc_mn_view_long(acc_S)
    acc_Sh_mn = acc_mn_view_long(acc_Sh)
    n_Sh = cute.size(acc_Sh)
    n_rowsh = cute.size(acc_Sh_mn.shape[0])
    acc_O_mn = acc_mn_view_long(acc_O)
    n_rows = cute.size(acc_S_mn.shape[0])
    n_S = cute.size(acc_S)
    n_E = cute.size(acc_E)

    row_sum = cute.make_rmem_tensor(n_rows, Float32)
    row_max = cute.make_rmem_tensor(n_rows, Float32)

    thr0_mma = tiled_mma.get_slice(0)
    cS = cute.make_identity_tensor((NHEAD, tile_n))
    cSh = cute.make_identity_tensor((NHEAD, tile_h))
    t0ScSh = thr0_mma.partition_C(cSh)
    col_baseh = thr_mma.partition_C(cSh)[0][1]
    tScS = thr_mma.partition_C(cS)
    tScS_mn = acc_mn_view_long(tScS)
    t0ScS = thr0_mma.partition_C(cS)
    cO = cute.make_identity_tensor((NHEAD, DHEAD))
    t0OcO = thr0_mma.partition_C(cO)
    col_base = tScS[0][1]
    bv_base = kv * DHEAD + thr_mma.partition_C(cO)[0][1]
    half = (acc_Sh, acc_Sh_mn, n_Sh, t0ScSh, tSrTokh, tSsTokh_ld,
            tOrTokh, tOsTokh_ld, n_rowsh, col_baseh)
    lse_row = cute.make_rmem_tensor(n_rows, Int32)
    for r in cutlass.range_constexpr(n_rows):
        lse_row[r] = head_of_row_long(tScS_mn[r, 0][0], nwarp)

    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                  num_bits_per_copy=32)
    st_thr = cute.make_tiled_copy_C(st_atom, tiled_mma).get_slice(tidx)
    st_sO = st_thr.partition_D(sO)

    cTok = cute.make_identity_tensor((tile_n, DHEAD))
    tKcTok = gmem_thr_load.partition_S(cTok)
    t0KcTok = gmem_copy_load.get_slice(0).partition_S(cTok)
    n_cpy_m = cute.size(tKcTok.shape[1])
    tok_row_base = tKcTok[0, 0, 0][0]

    tQsQ = gmem_thr_load.partition_D(sQ)
    tKsTok = gmem_thr_load.partition_D(sTok)
    tOsO = gmem_thr_store.partition_S(sO)

    cute.arch.barrier()

    # ---- work loop ----------------------------------------------------------
    # Two work domains behind one flat index: [0, n_prio) are descriptors from
    # the prep kernel (split chunks + long pixels, drained first) and
    # [n_prio, n_prio+TP) is the plain index sweep.  Index-sweep entries that
    # were already covered by a descriptor collapse to a zero-tile item with the
    # epilogue disabled, which keeps the loop body branch-free.
    bidx, _, _ = cute.arch.block_idx()
    nblocks, _, _ = cute.arch.grid_dim()
    if const_expr(sched):
        n_prio = mMeta[0]
        W = n_prio + TP
    else:
        n_prio = Int32(0)
        W = TP
    w = Int32(bidx)

    while w < W:
        if const_expr(sched):
            # Clamping the descriptor index keeps the speculative load inside
            # the (small, L1-resident) priority list instead of streaming cold
            # lines out of the unused tail of the array on every sweep item.
            is_prio = w < n_prio
            w4 = (w if is_prio else Int32(0)) * Int32(4)
            p = mWit[w4] if is_prio else (w - n_prio)
            lo = mWit[w4 + 1] if is_prio else Int32(0)
            hi0 = mWit[w4 + 2] if is_prio else Int32(0)
            sidx = mWit[w4 + 3] if is_prio else Int32(-1)
            base = mCu[p]
            L = mCu[p + 1] - base
            nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
            okv = Int32(1) if is_prio else (Int32(1) if L <= CH2 else Int32(0))
            hi = hi0 if is_prio else (nt if okv == Int32(1) else Int32(0))
            n_tiles = hi - lo
            k0 = base + lo * tile_n
            Lc = L - lo * tile_n
        else:
            p = w
            base = mCu[p]
            L = mCu[p + 1] - base
            n_tiles = (L + Int32(tile_n - 1)) // Int32(tile_n)
            k0 = base
            Lc = L

        gQ = cute.composition(cute.local_tile(mQ, (NHEAD, DHEAD), (p, 0)), hperm)
        cute.copy(gmem_copy_load, gmem_thr_load.partition_S(gQ), tQsQ)
        cute.arch.cp_async_commit_group()

        mTokP = cute.domain_offset((k0, 0), mTok)
        gTok = cute.local_tile(mTokP, (tile_n, DHEAD), (None, 0))
        tKgTok = gmem_thr_load.partition_S(gTok)
        for s in cutlass.range_constexpr(NSTAGE - 1):
            if s < n_tiles:
                rows_valid = Lc - s * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, s],
                                  tKsTok[None, m, None, s])
            cute.arch.cp_async_commit_group()
        if tidx == 0:
            if const_expr(sched):
                sIdxT[1] = okv
            if w < nblocks:
                sIdxT[0] = w + nblocks
            else:
                ticket = cute.arch.atomic_add(mMeta.iterator + 2, Int32(1))
                sIdxT[0] = Int32(2) * nblocks + ticket

        # --- Qeff prologue ---
        cute.arch.cp_async_wait_group(NSTAGE - 1)
        cute.arch.barrier()
        acc_Q.fill(0.0)
        gemm_ss_long(tiled_mma, acc_Q, tQrQ, tQrWk, tQsQ_ld, tQsWk_ld[None, None, None, kv],
                cpA_n, cpB_t)
        rQeff = cute.make_fragment_like(acc_Q, cutlass.BFloat16)
        rQeff.store((acc_Q.load() * scale_log2).to(cutlass.BFloat16))
        tSrQ = cute.make_tensor(rQeff.iterator, convert_layout_acc_frgA_long(rQeff.layout))

        n_full = cutlass.min(n_tiles, Lc // Int32(tile_n))
        if const_expr(tile_h == tile_n):
            run_tiles_full_long(
                tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, tSrQ, tSrTok,
                tSsTok_ld, cpB_n, row_max, row_sum, n_rows, acc_O, acc_O_mn,
                tOrTok, tOsTok_ld, cpB_t, gmem_copy_load, tKgTok, tKsTok,
                t0KcTok, Lc, n_tiles, n_full, col_base, tok_row_base,
                n_cpy_m, tile_n, NSTAGE)
        else:
            args_t = (tiled_mma, acc_S, acc_S_mn, n_S, t0ScS, tSrQ, tSrTok,
                      tSsTok_ld, cpB_n, row_max, row_sum, n_rows,
                      acc_O, acc_O_mn, tOrTok, tOsTok_ld, cpB_t, half,
                      gmem_copy_load, tKgTok, tKsTok, t0KcTok, Lc, n_tiles,
                      n_full, col_base, tok_row_base, n_cpy_m, tile_n, tile_h,
                      NSTAGE)
            run_tiles_long(*args_t)

        # --- cross-CTA combine for split chunks ---
        # Each chunk publishes its own (m, l, O) state; whichever chunk retires
        # last (the one that sees the slot counter hit zero) reads the siblings
        # back and folds them in with the stable merge
        #     m = max(m_a, m_b),  l = l_a 2**(m_a-m) + l_b 2**(m_b-m),  likewise O.
        if const_expr(sched):
          sch = sidx // Int32(MAXCH)
          sb = sidx * Int32(SCR_PER_CH)
          if sidx >= Int32(0):
            for i in cutlass.range_constexpr(n_E):
                mScr[sb + Int32(i * nthread) + tidx] = acc_O[i]
            for r in cutlass.range_constexpr(n_rows):
                mScr[sb + Int32(2048 + r * nthread) + tidx] = row_sum[r]
                mScr[sb + Int32(2048 + (n_rows + r) * nthread) + tidx] = row_max[r]
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == 0:
                old = cute.arch.atomic_add(mMeta.iterator + (Int32(8) + sch),
                                           Int32(-1), sem="acq_rel", scope="gpu")
                sIdxT[1] = Int32(1) if old == Int32(1) else Int32(0)
            cute.arch.barrier()

        if const_expr(sched):
            do_epi = sIdxT[1] == Int32(1)
        else:
            do_epi = True
        if do_epi:
            if const_expr(sched):
                if sidx >= Int32(0):
                    cute.arch.fence_acq_rel_gpu()
                    nch = mMeta[8 + MAXSLOT + sch]
                    myc = sidx - sch * Int32(MAXCH)
                    oth = cute.make_fragment_like(acc_O, Float32)
                    oth_mn = acc_mn_view_long(oth)
                    for c in cutlass.range(nch):
                        if c != myc:
                            ob = (sch * Int32(MAXCH) + c) * Int32(SCR_PER_CH)
                            for i in cutlass.range_constexpr(n_E):
                                oth[i] = mScr[ob + Int32(i * nthread) + tidx]
                            for r in cutlass.range_constexpr(n_rows):
                                l2 = mScr[ob + Int32(2048 + r * nthread) + tidx]
                                m2 = mScr[ob + Int32(2048 + (n_rows + r) * nthread)
                                          + tidx]
                                mn = m2 if m2 > row_max[r] else row_max[r]
                                s1 = cute.math.exp2(row_max[r] - mn, fastmath=True)
                                s2 = cute.math.exp2(m2 - mn, fastmath=True)
                                row_max[r] = mn
                                row_sum[r] = row_sum[r] * s1 + l2 * s2
                                acc_O_mn[r, None].store(
                                    acc_O_mn[r, None].load() * s1
                                    + oth_mn[r, None].load() * s2)

            # --- epilogue: normalize, apply W_v + B_v, store ---
            for r in cutlass.range_constexpr(n_rows):
                rs = warp_sum4_long(row_sum[r])
                row_sum[r] = rs
                inv = 0.0 if rs == 0.0 else cute.arch.rcp_approx(rs)
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

            bv_gate = 0.0 if row_sum[0] == 0.0 else 1.0
            rAcc = cute.make_fragment_like(acc_O, cutlass.BFloat16)
            rAcc.store(acc_O.load().to(cutlass.BFloat16))
            tErA = cute.make_tensor(rAcc.iterator, convert_layout_acc_frgA_long(rAcc.layout))
            for i in cutlass.range_constexpr(n_E):
                acc_E[i] = sBv[bv_base + t0OcO[i][1]] * bv_gate
            gemm_rs_long(tiled_mma, acc_E, tErA, tErWv, tEsWv_ld[None, None, None, kv], cpB_n)

            # LSE lands on permuted, non-contiguous head indices; addressing
            # `mLSE[p, lse_row[r]]` from a per-thread register index cost NCU
            # 29 instructions per pixel per warp for four 4-byte stores.
            # Scatter into SMEM instead (all four lanes of a quad hold the same
            # warp-reduced value and the same address) and let the 64-head row
            # leave as one burst alongside the already-staged Out tile.  sLSE
            # lands inside the existing alignment padding ahead of sQ, so the
            # per-CTA SMEM footprint is unchanged and residency is unchanged.
            # Four MMA lanes in each quad hold identical row statistics.
            # Remap logical rows across those lanes so each head issues one
            # log2 and one SMEM store instead of four duplicates.
            qr = tidx % 4
            if qr < Int32(n_rows):
                lse_l = row_sum[0]
                lse_m = row_max[0]
                lse_h = lse_row[0]
                for r in cutlass.range_constexpr(1, n_rows):
                    if qr == Int32(r):
                        lse_l = row_sum[r]
                        lse_m = row_max[r]
                        lse_h = lse_row[r]
                sLSE[lse_h] = lse_m + cute.math.log2(lse_l, fastmath=True)

            rO = cute.make_fragment_like(acc_E, cutlass.BFloat16)
            rO.store(acc_E.load().to(cutlass.BFloat16))
            cute.copy(st_atom, st_thr.retile(rO), st_sO)
            cute.arch.barrier()
            if tidx < NHEAD:
                mLSE[p, tidx] = sLSE[tidx]
            gO = cute.composition(cute.local_tile(mOut, (NHEAD, DHEAD), (p, 0)),
                                  hperm)
            rOs = cute.make_fragment_like(tOsO, cutlass.BFloat16)
            cute.autovec_copy(tOsO, rOs)
            cute.copy(gmem_copy_store, rOs, gmem_thr_store.partition_D(gO))
        cute.arch.barrier()
        w = sIdxT[0]
        # The next iteration's thread 0 overwrites this slot near the top of the loop with no
        # intervening rendezvous, so every thread must finish reading it first.
        cute.arch.barrier()


# --------------------------------------------------------------------------- #
# host
# --------------------------------------------------------------------------- #
def _gt_long(ptr, dtype, shape, stride, align):
    return cute.make_tensor(
        cute.make_ptr(dtype, ptr, cute.AddressSpace.gmem, assumed_align=align),
        cute.make_layout(shape, stride=stride))


@cute.jit
def pca_launch_long(pQ: cutlass.Int64, pTok: cutlass.Int64, pWk: cutlass.Int64,
               pWv: cutlass.Int64, pBv: cutlass.Int64, pCu: cutlass.Int64,
               pOut: cutlass.Int64, pLSE: cutlass.Int64,
               pMeta: cutlass.Int64, pWit: cutlass.Int64, pScr: cutlass.Int64,
               scale_log2: Float32,
               CH: Int32, CH2: Int32,
               TP: cutlass.Constexpr, TT: cutlass.Constexpr,
               MAXITEM: cutlass.Constexpr,
               tile_n: cutlass.Constexpr, tile_h: cutlass.Constexpr,
               nwarp: cutlass.Constexpr,
               NSTAGE: cutlass.Constexpr, cta_sm: cutlass.Constexpr,
               sched: cutlass.Constexpr,
               grid_x: cutlass.Constexpr, prep_grid: cutlass.Constexpr,
               stream):
    dt = cutlass.BFloat16
    mQ = _gt_long(pQ, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mTok = _gt_long(pTok, dt, (TT, DHEAD), (DHEAD, 1), 16)
    mWkf = _gt_long(pWk, dt, (2048,), (1,), 16)
    mWvf = _gt_long(pWv, dt, (2048,), (1,), 16)
    mBv = _gt_long(pBv, dt, (NHEAD,), (1,), 16)
    mCu = _gt_long(pCu, Int32, (TP + 1,), (1,), 4)
    mOut = _gt_long(pOut, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mLSE = _gt_long(pLSE, Float32, (TP, NHEAD), (NHEAD, 1), 16)
    mMeta = _gt_long(pMeta, Int32, (8 + 2 * MAXSLOT if sched else 8,), (1,), 4)
    mWit = _gt_long(pWit, Int32, (MAXITEM * 4 if sched else 1,), (1,), 4)
    mScr = _gt_long(pScr, Float32,
               (MAXSLOT * MAXCH * SCR_PER_CH if sched else 1,), (1,), 4)
    atom = smem_layout_atom_long(dt, DHEAD)
    sQ_layout = cute.tile_to_shape(atom, (NHEAD, DHEAD), (0, 1))
    sO_layout = cute.tile_to_shape(atom, (NHEAD, DHEAD), (0, 1))
    sTok_layout = cute.tile_to_shape(atom, (tile_n, DHEAD, NSTAGE), (0, 1, 2))
    # Same bytes as sTok_layout, re-tiled at half width: tile_to_shape of the
    # same 8x32 swizzle atom over (tile_n/2, D, 2*NSTAGE) is linearly identical
    # to (tile_n, D, NSTAGE), so the tail reads the staged tile through a second
    # view with no copy and no extra SMEM.
    sTokH_layout = cute.tile_to_shape(
        atom, (tile_h, DHEAD, NSTAGE * (tile_n // tile_h)), (0, 1, 2))
    sW_layout = cute.tile_to_shape(atom, (DHEAD, DHEAD, NKV), (0, 1, 2))

    copy_bits = 128
    elems = copy_bits // dt.width
    dim1 = DHEAD // elems
    nthread = nwarp * 32
    t_layout = cute.make_ordered_layout((nthread // dim1, dim1), order=(1, 0))
    v_layout = cute.make_layout((1, elems))
    atom_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dt,
        num_bits_per_copy=copy_bits)
    atom_univ = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dt,
                                    num_bits_per_copy=copy_bits)
    gmem_copy_load = cute.make_tiled_copy_tv(atom_async, t_layout, v_layout)
    gmem_copy_store = cute.make_tiled_copy_tv(atom_univ, t_layout, v_layout)

    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)),
        (nwarp, 1, 1),
        permutation_mnk=(nwarp * 16, 16, 16),
    )

    @cute.struct
    class SharedStorage:
        sIdx: cute.struct.MemRange[Int32, 4]
        sBv: cute.struct.MemRange[Float32, NHEAD]
        sLSE: cute.struct.MemRange[Float32, NHEAD]
        # 512 B is the swizzle period of the (8, 32) bf16 atom and a multiple
        # of the 128 B bank row, so it preserves both the swizzle pattern and
        # the bank mapping while removing 752 B of alignment padding per CTA
        # (two-warp block 21.50 -> 20.50 KiB, four-warp T64 29.50 -> 28.50).
        sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 512]
        sWk: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 512]
        sWv: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 512]
        sTok: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sTok_layout)], 512]

    if const_expr(sched):
        prep_kernel_long(mCu, mWit, mMeta, Int32(TP), CH, CH2, tile_n).launch(
            grid=(prep_grid, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)

    pca = pca_kernel_long(
        mQ, mTok, mWkf, mWvf, mBv, mCu, mOut, mLSE, mMeta, mWit, mScr,
        scale_log2, Int32(TP), CH2, tile_n, tile_h, nwarp, NSTAGE, sched,
        sQ_layout, sTok_layout, sTokH_layout, sW_layout, sO_layout,
        gmem_copy_load, gmem_copy_store, tiled_mma, SharedStorage,
    )
    # The persistent grid is sized at exactly cta_sm CTAs per SM, so the
    # register budget has to be pinned to that residency on every path -- left
    # unbounded, ptxas picks a register count for the four-warp bodies that
    # makes part of the grid non-resident and the sweep tail lumpy.
    pca.launch(grid=(grid_x, 1, 1), block=(nthread, 1, 1),
               smem=SharedStorage.size_in_bytes(), stream=stream,
               max_number_threads=[nthread, 1, 1], min_blocks_per_mp=cta_sm)


_CACHE_long = {}
_STREAM_long = {}
_NSM_long = []


def _build_long(TP, TT, tt_c, dev):
    if not _NSM_long:
        _NSM_long.append(torch.cuda.get_device_properties(dev).multi_processor_count)
    nsm = _NSM_long[0]
    mean_len = TT / max(TP, 1)
    long_groups = mean_len >= 300.0
    nwarp = 4 if long_groups else 2
    tile_n = 64 if long_groups else 32
    # L5 amortizes the narrow exact tail; L6 keeps a single full-width update.
    tile_h = 16 if long_groups else 32
    nstage = 4
    # Constrain the two-warp body to the resident-CTA register budget.
    # Two-warp L6 body: residency wins (8 CTAs/SM, 128 registers; 7 CTAs at
    # 146 registers measured 4% slower).  Four-warp bodies: the register cap
    # wins (4 CTAs/SM at 128 registers; 5 CTAs at 102 spills).
    cta_sm = 4 if long_groups else 6
    sched = mean_len >= 60.0
    grid_x = min(TP, cta_sm * nsm)
    # Split states are written, never accumulated, so the scratch needs no
    # zero-fill pass: each chunk owns its own (m, l, O) slice.
    nscr = MAXSLOT * MAXCH * SCR_PER_CH
    prep_grid = (TP + PREP_THREADS - 1) // PREP_THREADS

    MAXITEM = TP + MAXCH * MAXSLOT

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    fn = cute.compile(
        pca_launch_long, z, z, z, z, z, z, z, z, z, z, z, Float32(0.0),
        Int32(0), Int32(0), TP, tt_c, MAXITEM, tile_n, tile_h, nwarp, nstage, cta_sm,
        sched, grid_x, prep_grid, stream)
    nmeta = 8 + 2 * MAXSLOT if sched else 8
    nwit = MAXITEM * 4 if sched else 1
    nscr_words = nscr if sched else 1
    return fn, nmeta, nwit + nscr_words, nwit * 4, grid_x, tile_n


def run_long(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale,
        q_per_kv, token_dim, n_kv_heads, use_v_bias, force_fp32,
        Out, LSE):
    TP = Q.shape[0]
    TT = tokens.shape[0]
    # sched is the only build input that still depends on the observation count, and it is a
    # coarse threshold rather than the count itself.
    sched = TT / max(TP, 1) >= 60.0
    tt_c = TT_LAYOUT_BOUND if TT <= TT_LAYOUT_BOUND else TT
    key = (TP, sched, tt_c)
    ent = _CACHE_long.get(key)
    if ent is None:
        ent = _build_long(TP, TT, tt_c, Q.device)
        _CACHE_long[key] = ent
    fn, nmeta, nwork, scr_offset, grid_x, tile_n = ent
    CH = max(TT // (2 * grid_x), 8 * tile_n)
    CH2 = max(TT // (8 * grid_x), 2 * tile_n)

    dev = Q.device
    meta = torch.zeros(nmeta, dtype=torch.int32, device=dev)
    work = torch.empty(nwork, dtype=torch.int32, device=dev)
    pw = work.data_ptr()

    cs = torch.cuda.current_stream().cuda_stream
    stream = _STREAM_long.get(cs)
    if stream is None:
        stream = cuda_driver.CUstream(cs)
        _STREAM_long[cs] = stream

    I = cutlass.Int64
    fn(I(Q.data_ptr()), I(tokens.data_ptr()), I(W_k.data_ptr()),
       I(W_v.data_ptr()), I(B_v.data_ptr()), I(cu_seqlens_k.data_ptr()),
       I(Out.data_ptr()), I(LSE.data_ptr()), I(meta.data_ptr()),
       I(pw), I(pw + scr_offset), Float32(float(scale) * LOG2_E),
       Int32(CH), Int32(CH2), stream)


# ------------------------ forward: uniform variant
# ---------------------------------------------------------------------------

def smem_layout_atom_uniform(dtype, k_dim):
    dtype_byte = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte
    smem_k_block_size = (
        128 if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    swizzle_bits = (
        4 if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if k_dim % 32 == 0 else 16, smem_k_block_size), order=(1, 0)
        ),
    )


def convert_layout_acc_mn_uniform(acc_layout):
    """((2,2), MMA_M, MMA_N) -> ((2, MMA_M), (2, MMA_N))."""
    cm = cute.make_layout(acc_layout.shape)
    shape = (
        (cm.shape[0][1], cm.shape[1]),
        (cm.shape[0][0], *cm.shape[0][2:], cm.shape[2]),
        *cm.shape[3:],
    )
    stride = (
        (cm.stride[0][1], cm.stride[1]),
        (cm.stride[0][0], *cm.stride[0][2:], cm.stride[2]),
        *cm.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def acc_mn_view_uniform(acc):
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn_uniform(acc.layout))


def convert_layout_acc_frgA_uniform(acc_layout):
    """C fragment of an (M,N) tile -> A fragment of an (M,K=N) mma."""
    if cute.rank(acc_layout.shape[0]) == 3:
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        return cute.make_layout(
            ((l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
             l.shape[1],
             (l.shape[0][2][1], l.shape[2])),
            stride=((l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                    l.stride[1],
                    (l.stride[0][2][1], l.stride[2])),
        )
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
        stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
    )


def transpose_view_uniform(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def head_perm_layout_uniform(nwarp):
    """Relabel heads so each two-warp path warp owns one 32-head KV group."""
    if nwarp == 2:
        return cute.make_layout(((16, 2, 2), DHEAD), stride=((1, 32, 16), NHEAD))
    return cute.make_layout((NHEAD, DHEAD), stride=(1, NHEAD))


def head_of_row_uniform(r, nwarp):
    if nwarp == 2:
        return (r % 16) + ((r // 16) % 2) * 32 + (r // 32) * 16
    return r


def warp_sum4_uniform(x):
    x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=2)
    return x


# --------------------------------------------------------------------------- #
# gemm helpers
# --------------------------------------------------------------------------- #
@cute.jit
def gemm_rs_uniform(tiled_mma, acc, tCrA, tCrB, tCsB, smem_thr_copy_B):
    """A already in registers, B streamed from smem via ldmatrix."""
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_view[None, None, 0])
    nk = cute.size(tCrA.shape[2])
    for k in cutlass.range_constexpr(nk):
        if const_expr(k < nk - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def gemm_ss_uniform(tiled_mma, acc, tCrA, tCrB, tCsA, tCsB, smem_thr_copy_A, smem_thr_copy_B):
    tCrA_view = smem_thr_copy_A.retile(tCrA)
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_view[None, None, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_view[None, None, 0])
    nk = cute.size(tCsA.shape[2])
    for k in cutlass.range_constexpr(nk):
        if const_expr(k < nk - 1):
            cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_view[None, None, k + 1])
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def attention_tile_uniform(
    tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, col_limit,
    tSrQ, tSrTok, tSsTok, cpB_n, row_max, row_sum, n_rows,
    acc_O, acc_O_mn, tOrTok, tOsTok, cpB_t,
    mask: cutlass.Constexpr, first: cutlass.Constexpr,
):
    """One exact online-softmax score tile followed by its PV update."""
    acc_S.fill(0.0)
    gemm_rs_uniform(tiled_mma, acc_S, tSrQ, tSrTok, tSsTok, cpB_n)
    if const_expr(mask):
        for i in cutlass.range_constexpr(n_S):
            acc_S[i] = acc_S[i] if t0ScS[i][1] < col_limit else -Float32.inf

    for r in cutlass.range_constexpr(n_rows):
        mt = acc_S_mn[r, None].load().reduce(
            cute.ReductionOp.MAX, Float32(NEG_BIG), 0)
        o = cute.arch.shuffle_sync_bfly(mt, offset=1)
        mt = o if o > mt else mt
        o = cute.arch.shuffle_sync_bfly(mt, offset=2)
        mt = o if o > mt else mt
        if const_expr(first):
            # The incoming state is exactly (NEG_BIG, 0, 0).  Updating it to
            # this tile's exact maximum therefore needs no exp2 correction and
            # no O rescale: both multiplications would only produce zero.
            mn = mt
            row_max[r] = mt
        else:
            mn = mt if mt > row_max[r] else row_max[r]
            sc = cute.math.exp2(row_max[r] - mn, fastmath=True)
            row_max[r] = mn
            row_sum[r] = row_sum[r] * sc
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * sc)
        p_row = cute.math.exp2(acc_S_mn[r, None].load() - mn, fastmath=True)
        if const_expr(first):
            row_sum[r] = p_row.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        else:
            row_sum[r] = p_row.reduce(cute.ReductionOp.ADD, row_sum[r], 0)
        acc_S_mn[r, None].store(p_row)

    rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
    rP.store(acc_S.load().to(cutlass.BFloat16))
    tOrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA_uniform(rP.layout))
    gemm_rs_uniform(tiled_mma, acc_O, tOrP, tOrTok, tOsTok, cpB_t)


@cute.jit
def run_tiles_uniform(
    tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, tSrQ, tSrTok, tSsTok_ld, cpB_n,
    row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok, tOsTok_ld, cpB_t,
    gmem_copy_load, tKgTok, tKsTok, t0KcTok, Lc, n_tiles, n_full, col_base,
    tok_row_base,
    n_cpy_m: cutlass.Constexpr, tile_n: cutlass.Constexpr,
    NSTAGE: cutlass.Constexpr,
):
    """Sweep one work item's token tiles with exact per-tile max updates."""
    acc_O.fill(0.0)
    if n_tiles == Int32(0):
        row_sum.fill(0.0)
        row_max.fill(NEG_BIG)

    if n_full > 0:
        cute.arch.cp_async_wait_group(NSTAGE - 2)
        cute.arch.barrier()
        nl = Int32(NSTAGE - 1)
        if nl < n_tiles:
            rows_valid = Lc - nl * tile_n - tok_row_base
            for m in cutlass.range_constexpr(n_cpy_m):
                if t0KcTok[0, m, 0][0] < rows_valid:
                    cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                              tKsTok[None, m, None, (NSTAGE - 1) % NSTAGE])
        cute.arch.cp_async_commit_group()
        attention_tile_uniform(
            tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, Int32(0),
            tSrQ, tSrTok, tSsTok_ld[None, None, None, 0], cpB_n,
            row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
            tOsTok_ld[None, None, None, 0], cpB_t, False, True,
        )

        for nn in cutlass.range(n_full - 1):
            n = nn + Int32(1)
            stage = n % NSTAGE
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()

            nl = n + NSTAGE - 1
            if nl < n_tiles:
                rows_valid = Lc - nl * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                                  tKsTok[None, m, None, nl % NSTAGE])
            cute.arch.cp_async_commit_group()

            attention_tile_uniform(
                tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, Int32(0),
                tSrQ, tSrTok, tSsTok_ld[None, None, None, stage], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, stage], cpB_t, False, False,
            )

        if n_full < n_tiles:
            n = n_full
            stage = n % NSTAGE
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()

            nl = n + NSTAGE - 1
            if nl < n_tiles:
                rows_valid = Lc - nl * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, nl],
                                  tKsTok[None, m, None, nl % NSTAGE])
            cute.arch.cp_async_commit_group()

            col_limit = Lc - n * tile_n - col_base
            attention_tile_uniform(
                tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, col_limit,
                tSrQ, tSrTok, tSsTok_ld[None, None, None, stage], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, stage], cpB_t, True, False,
            )
    else:
        # A real sub-tile item has one masked tile.  A zero-tile item is either
        # an empty pixel or the duplicate sweep entry for a priority descriptor;
        # leave its initialized state untouched instead of doing wasted QK/PV.
        if n_tiles > Int32(0):
            cute.arch.cp_async_wait_group(NSTAGE - 2)
            cute.arch.barrier()
            cute.arch.cp_async_commit_group()
            col_limit = Lc - col_base
            attention_tile_uniform(
                tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, col_limit,
                tSrQ, tSrTok, tSsTok_ld[None, None, None, 0], cpB_n,
                row_max, row_sum, n_rows, acc_O, acc_O_mn, tOrTok,
                tOsTok_ld[None, None, None, 0], cpB_t, True, True,
            )


# --------------------------------------------------------------------------- #
# prep kernel: build the priority work list from cu_seqlens
# --------------------------------------------------------------------------- #
@cute.kernel
def prep_kernel_uniform(
    mCu: cute.Tensor,       # (TP+1,) i32
    mWit: cute.Tensor,      # (MAXITEM*4,) i32
    mMeta: cute.Tensor,     # (8+2*MAXSLOT,) i32 -- [0]=items [1]=slots [2]=cursor
    TP: Int32,
    CH: Int32,
    CH2: Int32,
    tile_n: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    p = bidx * PREP_THREADS + tidx
    if p < TP:
        L = mCu[p + 1] - mCu[p]
        if L > CH2:
            nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
            split = False
            tpc = nt
            nc = Int32(1)
            slot = Int32(-1)
            if L > CH:
                s = cute.arch.atomic_add(mMeta.iterator + 1, Int32(1))
                if s < Int32(MAXSLOT):
                    slot = s
                    nc0 = (L + CH - Int32(1)) // CH
                    nc0 = cutlass.min(nc0, Int32(MAXCH))
                    tpc = (nt + nc0 - Int32(1)) // nc0
                    nc = (nt + tpc - Int32(1)) // tpc
                    split = True
            if split:
                base = cute.arch.atomic_add(mMeta.iterator + 0, nc)
                mMeta[8 + slot] = nc
                mMeta[8 + MAXSLOT + slot] = nc
                for c in cutlass.range(nc):
                    lo = c * tpc
                    hi = cutlass.min(lo + tpc, nt)
                    w4 = (base + c) * Int32(4)
                    mWit[w4] = p
                    mWit[w4 + 1] = lo
                    mWit[w4 + 2] = hi
                    mWit[w4 + 3] = slot * Int32(MAXCH) + c
            else:
                base = cute.arch.atomic_add(mMeta.iterator + 0, Int32(1))
                w4 = base * Int32(4)
                mWit[w4] = p
                mWit[w4 + 1] = Int32(0)
                mWit[w4 + 2] = nt
                mWit[w4 + 3] = Int32(-1)


# --------------------------------------------------------------------------- #
# main kernel
# --------------------------------------------------------------------------- #
@cute.kernel
def pca_kernel_uniform(
    mQ: cute.Tensor,        # (TP*64, 32)   bf16
    mTok: cute.Tensor,      # (TT, 32)      bf16
    mWkf: cute.Tensor,      # (2048,)       bf16   W_k flattened
    mWvf: cute.Tensor,      # (2048,)       bf16   W_v flattened
    mBv: cute.Tensor,       # (64,)         bf16
    mCu: cute.Tensor,       # (TP+1,)       i32
    mOut: cute.Tensor,      # (TP*64, 32)   bf16
    mLSE: cute.Tensor,      # (TP, 64)      f32
    mMeta: cute.Tensor,     # (8+MAXSLOT,)  i32
    mWit: cute.Tensor,      # (MAXITEM*4,)  i32
    mScr: cute.Tensor,      # (MAXSLOT*MAXCH*SCR_PER_CH,) f32
    scale_log2: Float32,
    TP: Int32,
    CH2: Int32,
    tile_n: cutlass.Constexpr,
    nwarp: cutlass.Constexpr,
    NSTAGE: cutlass.Constexpr,
    sched: cutlass.Constexpr,
    sQ_layout: cute.ComposedLayout,
    sTok_layout: cute.ComposedLayout,
    sW_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    gmem_copy_load: cute.TiledCopy,
    gmem_copy_store: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(tidx // 32)
    nthread = nwarp * 32
    kv = warp_idx // (nwarp // 2)
    hperm = head_perm_layout_uniform(nwarp)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQ.get_tensor(sQ_layout)
    sO = storage.sQ.get_tensor(sO_layout)      # Q is dead by the time O is staged
    sTok = storage.sTok.get_tensor(sTok_layout)
    sWk = storage.sWk.get_tensor(sW_layout)
    sWv = storage.sWv.get_tensor(sW_layout)
    sBv = storage.sBv.get_tensor(cute.make_layout(NHEAD))
    sLSE = storage.sLSE.get_tensor(cute.make_layout(NHEAD))
    sIdxT = storage.sIdx.get_tensor(cute.make_layout(4))
    sTokFlat = storage.sTok.get_tensor(cute.make_layout(cute.cosize(sTok_layout)))

    # ---- one-time CTA setup -------------------------------------------------
    # Thread-major indexing here (lane t reads element t + i*nthread) so the
    # weight preload is one fully coalesced 128B burst per warp per step.  The
    # lane-major form it replaces made every 2-byte load its own sector: NCU
    # attributed 4.55M excessive global sectors -- 17% of the whole kernel's
    # sector traffic -- to these two lines alone, plus 0.45M excessive shared
    # wavefronts on the paired SMEM stores.
    for i in cutlass.range_constexpr(2048 // nthread):
        lin = i * nthread + tidx
        c = (lin % 1024) // DHEAD
        d = lin % DHEAD
        h = lin // 1024
        sWk[c, d, h] = mWkf[lin]
        sWv[c, d, h] = mWvf[lin]
    nzero = cute.cosize(sTok_layout) // nthread
    for i in cutlass.range_constexpr(nzero):
        sTokFlat[i * nthread + tidx] = cutlass.BFloat16(0.0)
    if tidx < NHEAD:
        sBv[tidx] = mBv[tidx].to(Float32)

    # ---- static partitions --------------------------------------------------
    thr_mma = tiled_mma.get_slice(tidx)
    gmem_thr_load = gmem_copy_load.get_slice(tidx)
    gmem_thr_store = gmem_copy_store.get_slice(tidx)

    sWkT = transpose_view_uniform(sWk)                      # (d, c, kv) for the B operand
    sTokT = transpose_view_uniform(sTok)                    # (d, tok, stage)

    ldm_n = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                cutlass.BFloat16)
    ldm_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                                cutlass.BFloat16)
    cpA_n = cute.make_tiled_copy_A(ldm_n, tiled_mma).get_slice(tidx)
    cpB_n = cute.make_tiled_copy_B(ldm_n, tiled_mma).get_slice(tidx)
    cpB_t = cute.make_tiled_copy_B(ldm_t, tiled_mma).get_slice(tidx)

    # Qeff = Q @ W_k[kv]   (M=64, N=32, K=32)
    tQrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
    tQsQ_ld = cpA_n.partition_S(sQ)
    tQrWk = thr_mma.make_fragment_B(thr_mma.partition_B(sWkT[None, None, 0]))
    tQsWk_ld = cpB_t.partition_S(sWkT)

    # S = Qeff @ tok^T     (M=64, N=tile_n, K=32)
    tSrTok = thr_mma.make_fragment_B(thr_mma.partition_B(sTok[None, None, 0]))
    tSsTok_ld = cpB_n.partition_S(sTok)

    # acc = P @ tok        (M=64, N=32, K=tile_n)
    tOrTok = thr_mma.make_fragment_B(thr_mma.partition_B(sTokT[None, None, 0]))
    tOsTok_ld = cpB_t.partition_S(sTokT)

    # out = accn @ W_v[kv] (M=64, N=32, K=32)
    tErWv = thr_mma.make_fragment_B(thr_mma.partition_B(sWv[None, None, 0]))
    tEsWv_ld = cpB_n.partition_S(sWv)

    acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((NHEAD, tile_n)), Float32)
    acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((NHEAD, DHEAD)), Float32)
    acc_E = acc_O
    acc_Q = acc_O
    acc_S_mn = acc_mn_view_uniform(acc_S)
    acc_O_mn = acc_mn_view_uniform(acc_O)
    n_rows = cute.size(acc_S_mn.shape[0])
    n_cols = cute.size(acc_S_mn.shape[1])
    n_S = cute.size(acc_S)
    n_E = cute.size(acc_E)

    row_sum = cute.make_rmem_tensor(n_rows, Float32)
    row_max = cute.make_rmem_tensor(n_rows, Float32)

    thr0_mma = tiled_mma.get_slice(0)
    cS = cute.make_identity_tensor((NHEAD, tile_n))
    tScS = thr_mma.partition_C(cS)
    tScS_mn = acc_mn_view_uniform(tScS)
    t0ScS = thr0_mma.partition_C(cS)
    cO = cute.make_identity_tensor((NHEAD, DHEAD))
    t0OcO = thr0_mma.partition_C(cO)
    col_base = tScS[0][1]
    bv_base = kv * DHEAD + thr_mma.partition_C(cO)[0][1]
    lse_row = cute.make_rmem_tensor(n_rows, Int32)
    for r in cutlass.range_constexpr(n_rows):
        lse_row[r] = head_of_row_uniform(tScS_mn[r, 0][0], nwarp)

    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                  num_bits_per_copy=32)
    st_thr = cute.make_tiled_copy_C(st_atom, tiled_mma).get_slice(tidx)
    st_sO = st_thr.partition_D(sO)

    cTok = cute.make_identity_tensor((tile_n, DHEAD))
    tKcTok = gmem_thr_load.partition_S(cTok)
    t0KcTok = gmem_copy_load.get_slice(0).partition_S(cTok)
    n_cpy_m = cute.size(tKcTok.shape[1])
    tok_row_base = tKcTok[0, 0, 0][0]

    tQsQ = gmem_thr_load.partition_D(sQ)
    tKsTok = gmem_thr_load.partition_D(sTok)
    tOsO = gmem_thr_store.partition_S(sO)

    cute.arch.barrier()

    # ---- work loop ----------------------------------------------------------
    # Two work domains behind one flat index: [0, n_prio) are descriptors from
    # the prep kernel (split chunks + long pixels, drained first) and
    # [n_prio, n_prio+TP) is the plain index sweep.  Index-sweep entries that
    # were already covered by a descriptor collapse to a zero-tile item with the
    # epilogue disabled, which keeps the loop body branch-free.
    bidx, _, _ = cute.arch.block_idx()
    nblocks, _, _ = cute.arch.grid_dim()
    if const_expr(sched):
        n_prio = mMeta[0]
        W = n_prio + TP
    else:
        n_prio = Int32(0)
        W = TP
    w = Int32(bidx)

    while w < W:
        if const_expr(sched):
            # Clamping the descriptor index keeps the speculative load inside
            # the (small, L1-resident) priority list instead of streaming cold
            # lines out of the unused tail of the array on every sweep item.
            is_prio = w < n_prio
            w4 = (w if is_prio else Int32(0)) * Int32(4)
            p = mWit[w4] if is_prio else (w - n_prio)
            lo = mWit[w4 + 1] if is_prio else Int32(0)
            hi0 = mWit[w4 + 2] if is_prio else Int32(0)
            sidx = mWit[w4 + 3] if is_prio else Int32(-1)
            base = mCu[p]
            L = mCu[p + 1] - base
            nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
            okv = Int32(1) if is_prio else (Int32(1) if L <= CH2 else Int32(0))
            hi = hi0 if is_prio else (nt if okv == Int32(1) else Int32(0))
            n_tiles = hi - lo
            k0 = base + lo * tile_n
            Lc = L - lo * tile_n
        else:
            p = w
            base = mCu[p]
            L = mCu[p + 1] - base
            n_tiles = (L + Int32(tile_n - 1)) // Int32(tile_n)
            k0 = base
            Lc = L

        gQ = cute.composition(cute.local_tile(mQ, (NHEAD, DHEAD), (p, 0)), hperm)
        cute.copy(gmem_copy_load, gmem_thr_load.partition_S(gQ), tQsQ)
        cute.arch.cp_async_commit_group()

        mTokP = cute.domain_offset((k0, 0), mTok)
        gTok = cute.local_tile(mTokP, (tile_n, DHEAD), (None, 0))
        tKgTok = gmem_thr_load.partition_S(gTok)
        for s in cutlass.range_constexpr(NSTAGE - 1):
            if s < n_tiles:
                rows_valid = Lc - s * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_load, tKgTok[None, m, None, s],
                                  tKsTok[None, m, None, s])
            cute.arch.cp_async_commit_group()
        if tidx == 0:
            if const_expr(sched):
                sIdxT[1] = okv
            if w < nblocks:
                sIdxT[0] = w + nblocks
            else:
                ticket = cute.arch.atomic_add(mMeta.iterator + 2, Int32(1))
                sIdxT[0] = Int32(2) * nblocks + ticket

        # --- Qeff prologue ---
        cute.arch.cp_async_wait_group(NSTAGE - 1)
        cute.arch.barrier()
        acc_Q.fill(0.0)
        gemm_ss_uniform(tiled_mma, acc_Q, tQrQ, tQrWk, tQsQ_ld, tQsWk_ld[None, None, None, kv],
                cpA_n, cpB_t)
        rQeff = cute.make_fragment_like(acc_Q, cutlass.BFloat16)
        rQeff.store((acc_Q.load() * scale_log2).to(cutlass.BFloat16))
        tSrQ = cute.make_tensor(rQeff.iterator, convert_layout_acc_frgA_uniform(rQeff.layout))

        n_full = cutlass.min(n_tiles, Lc // Int32(tile_n))
        args_t = (tiled_mma, acc_S, acc_S_mn, n_S, n_cols, t0ScS, tSrQ, tSrTok,
                  tSsTok_ld, cpB_n, row_max, row_sum, n_rows, acc_O, acc_O_mn,
                  tOrTok, tOsTok_ld, cpB_t, gmem_copy_load, tKgTok, tKsTok,
                  t0KcTok, Lc, n_tiles, n_full, col_base, tok_row_base,
                  n_cpy_m, tile_n, NSTAGE)
        run_tiles_uniform(*args_t)

        # --- cross-CTA combine for split chunks ---
        # Each chunk publishes its own (m, l, O) state; whichever chunk retires
        # last (the one that sees the slot counter hit zero) reads the siblings
        # back and folds them in with the stable merge
        #     m = max(m_a, m_b),  l = l_a 2**(m_a-m) + l_b 2**(m_b-m),  likewise O.
        if const_expr(sched):
          sch = sidx // Int32(MAXCH)
          sb = sidx * Int32(SCR_PER_CH)
          if sidx >= Int32(0):
            for i in cutlass.range_constexpr(n_E):
                mScr[sb + Int32(i * nthread) + tidx] = acc_O[i]
            for r in cutlass.range_constexpr(n_rows):
                mScr[sb + Int32(2048 + r * nthread) + tidx] = row_sum[r]
                mScr[sb + Int32(2048 + (n_rows + r) * nthread) + tidx] = row_max[r]
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == 0:
                old = cute.arch.atomic_add(mMeta.iterator + (Int32(8) + sch),
                                           Int32(-1), sem="acq_rel", scope="gpu")
                sIdxT[1] = Int32(1) if old == Int32(1) else Int32(0)
            cute.arch.barrier()

        if const_expr(sched):
            do_epi = sIdxT[1] == Int32(1)
        else:
            do_epi = True
        if do_epi:
            if const_expr(sched):
                if sidx >= Int32(0):
                    cute.arch.fence_acq_rel_gpu()
                    nch = mMeta[8 + MAXSLOT + sch]
                    myc = sidx - sch * Int32(MAXCH)
                    oth = cute.make_fragment_like(acc_O, Float32)
                    oth_mn = acc_mn_view_uniform(oth)
                    for c in cutlass.range(nch):
                        if c != myc:
                            ob = (sch * Int32(MAXCH) + c) * Int32(SCR_PER_CH)
                            for i in cutlass.range_constexpr(n_E):
                                oth[i] = mScr[ob + Int32(i * nthread) + tidx]
                            for r in cutlass.range_constexpr(n_rows):
                                l2 = mScr[ob + Int32(2048 + r * nthread) + tidx]
                                m2 = mScr[ob + Int32(2048 + (n_rows + r) * nthread)
                                          + tidx]
                                mn = m2 if m2 > row_max[r] else row_max[r]
                                s1 = cute.math.exp2(row_max[r] - mn, fastmath=True)
                                s2 = cute.math.exp2(m2 - mn, fastmath=True)
                                row_max[r] = mn
                                row_sum[r] = row_sum[r] * s1 + l2 * s2
                                acc_O_mn[r, None].store(
                                    acc_O_mn[r, None].load() * s1
                                    + oth_mn[r, None].load() * s2)

            # --- epilogue: normalize, apply W_v + B_v, store ---
            for r in cutlass.range_constexpr(n_rows):
                rs = warp_sum4_uniform(row_sum[r])
                row_sum[r] = rs
                inv = 0.0 if rs == 0.0 else cute.arch.rcp_approx(rs)
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

            bv_gate = 0.0 if row_sum[0] == 0.0 else 1.0
            rAcc = cute.make_fragment_like(acc_O, cutlass.BFloat16)
            rAcc.store(acc_O.load().to(cutlass.BFloat16))
            tErA = cute.make_tensor(rAcc.iterator, convert_layout_acc_frgA_uniform(rAcc.layout))
            for i in cutlass.range_constexpr(n_E):
                acc_E[i] = sBv[bv_base + t0OcO[i][1]] * bv_gate
            gemm_rs_uniform(tiled_mma, acc_E, tErA, tErWv, tEsWv_ld[None, None, None, kv], cpB_n)

            # Four MMA lanes in each quad hold identical row statistics.
            # Select one distinct logical row per lane so each head pays for
            # one log2 and one SMEM store instead of four duplicates.
            qr = tidx % 4
            if qr < Int32(n_rows):
                lse_l = row_sum[0]
                lse_m = row_max[0]
                lse_h = lse_row[0]
                for r in cutlass.range_constexpr(1, n_rows):
                    if qr == Int32(r):
                        lse_l = row_sum[r]
                        lse_m = row_max[r]
                        lse_h = lse_row[r]
                sLSE[lse_h] = lse_m + cute.math.log2(lse_l, fastmath=True)

            rO = cute.make_fragment_like(acc_E, cutlass.BFloat16)
            rO.store(acc_E.load().to(cutlass.BFloat16))
            cute.copy(st_atom, st_thr.retile(rO), st_sO)
            cute.arch.barrier()
            if tidx < NHEAD:
                mLSE[p, tidx] = sLSE[tidx]
            gO = cute.composition(cute.local_tile(mOut, (NHEAD, DHEAD), (p, 0)),
                                  hperm)
            rOs = cute.make_fragment_like(tOsO, cutlass.BFloat16)
            cute.autovec_copy(tOsO, rOs)
            cute.copy(gmem_copy_store, rOs, gmem_thr_store.partition_D(gO))
        cute.arch.barrier()
        w = sIdxT[0]
        # The next iteration's thread 0 overwrites this slot near the top of the loop with no
        # intervening rendezvous, so every thread must finish reading it first.
        cute.arch.barrier()


# --------------------------------------------------------------------------- #
# host
# --------------------------------------------------------------------------- #
def _gt_uniform(ptr, dtype, shape, stride, align):
    return cute.make_tensor(
        cute.make_ptr(dtype, ptr, cute.AddressSpace.gmem, assumed_align=align),
        cute.make_layout(shape, stride=stride))


@cute.jit
def pca_launch_uniform(pQ: cutlass.Int64, pTok: cutlass.Int64, pWk: cutlass.Int64,
               pWv: cutlass.Int64, pBv: cutlass.Int64, pCu: cutlass.Int64,
               pOut: cutlass.Int64, pLSE: cutlass.Int64,
               pMeta: cutlass.Int64, pWit: cutlass.Int64, pScr: cutlass.Int64,
               scale_log2: Float32,
               CH: Int32, CH2: Int32,
               TP: cutlass.Constexpr, TT: cutlass.Constexpr,
               MAXITEM: cutlass.Constexpr,
               tile_n: cutlass.Constexpr, nwarp: cutlass.Constexpr,
               NSTAGE: cutlass.Constexpr, cta_sm: cutlass.Constexpr,
               sched: cutlass.Constexpr,
               grid_x: cutlass.Constexpr, prep_grid: cutlass.Constexpr,
               stream):
    dt = cutlass.BFloat16
    mQ = _gt_uniform(pQ, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mTok = _gt_uniform(pTok, dt, (TT, DHEAD), (DHEAD, 1), 16)
    mWkf = _gt_uniform(pWk, dt, (2048,), (1,), 16)
    mWvf = _gt_uniform(pWv, dt, (2048,), (1,), 16)
    mBv = _gt_uniform(pBv, dt, (NHEAD,), (1,), 16)
    mCu = _gt_uniform(pCu, Int32, (TP + 1,), (1,), 4)
    mOut = _gt_uniform(pOut, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mLSE = _gt_uniform(pLSE, Float32, (TP, NHEAD), (NHEAD, 1), 16)
    mMeta = _gt_uniform(pMeta, Int32, (8 + 2 * MAXSLOT if sched else 8,), (1,), 4)
    mWit = _gt_uniform(pWit, Int32, (MAXITEM * 4 if sched else 1,), (1,), 4)
    mScr = _gt_uniform(pScr, Float32,
               (MAXSLOT * MAXCH * SCR_PER_CH if sched else 1,), (1,), 4)
    atom = smem_layout_atom_uniform(dt, DHEAD)
    sQ_layout = cute.tile_to_shape(atom, (NHEAD, DHEAD), (0, 1))
    sO_layout = cute.tile_to_shape(atom, (NHEAD, DHEAD), (0, 1))
    sTok_layout = cute.tile_to_shape(atom, (tile_n, DHEAD, NSTAGE), (0, 1, 2))
    sW_layout = cute.tile_to_shape(atom, (DHEAD, DHEAD, NKV), (0, 1, 2))

    copy_bits = 128
    elems = copy_bits // dt.width
    dim1 = DHEAD // elems
    nthread = nwarp * 32
    t_layout = cute.make_ordered_layout((nthread // dim1, dim1), order=(1, 0))
    v_layout = cute.make_layout((1, elems))
    atom_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dt,
        num_bits_per_copy=copy_bits)
    atom_univ = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dt,
                                    num_bits_per_copy=copy_bits)
    gmem_copy_load = cute.make_tiled_copy_tv(atom_async, t_layout, v_layout)
    gmem_copy_store = cute.make_tiled_copy_tv(atom_univ, t_layout, v_layout)

    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)),
        (nwarp, 1, 1),
        permutation_mnk=(nwarp * 16, 16, 16),
    )

    @cute.struct
    class SharedStorage:
        sIdx: cute.struct.MemRange[Int32, 4]
        sBv: cute.struct.MemRange[Float32, NHEAD]
        sLSE: cute.struct.MemRange[Float32, NHEAD]
        sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 512]
        sWk: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 512]
        sWv: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 512]
        sTok: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sTok_layout)], 512]

    if const_expr(sched):
        prep_kernel_uniform(mCu, mWit, mMeta, Int32(TP), CH, CH2, tile_n).launch(
            grid=(prep_grid, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)

    pca = pca_kernel_uniform(
        mQ, mTok, mWkf, mWvf, mBv, mCu, mOut, mLSE, mMeta, mWit, mScr,
        scale_log2, Int32(TP), CH2, tile_n, nwarp, NSTAGE, sched,
        sQ_layout, sTok_layout, sW_layout, sO_layout,
        gmem_copy_load, gmem_copy_store, tiled_mma, SharedStorage,
    )
    if const_expr(nwarp == 2):
        pca.launch(grid=(grid_x, 1, 1), block=(nthread, 1, 1),
                   smem=SharedStorage.size_in_bytes(), stream=stream,
                   max_number_threads=[nthread, 1, 1], min_blocks_per_mp=cta_sm)
    else:
        pca.launch(grid=(grid_x, 1, 1), block=(nthread, 1, 1),
                   smem=SharedStorage.size_in_bytes(), stream=stream)


_CACHE_uniform = {}
_STREAM_uniform = {}
_NSM_uniform = []


def _build_uniform(TP, TT, tt_c, dev):
    if not _NSM_uniform:
        _NSM_uniform.append(torch.cuda.get_device_properties(dev).multi_processor_count)
    nsm = _NSM_uniform[0]
    mean_len = TT / max(TP, 1)
    # Uniform two-warp T32 avoids redundant token ldmatrix traffic across both
    # L5 and L6 while keeping enough registers for the exact online state.
    nwarp = 2
    tile_n = 32
    nstage = 4
    # Six CTAs/SM avoids the register spills seen at the eight-CTA cap.
    cta_sm = 6
    # Priority + chunk-splitting scheduling now pays for its prep kernel even on
    # the shortest grid (L6-2015, mean 115 tok/px: 0.341 -> 0.333 ms), whose
    # heaviest pixel is still 1.5x an even per-CTA share.
    sched = mean_len >= 60.0
    grid_x = min(TP, cta_sm * nsm)
    # Split states are written, never accumulated, so the scratch needs no
    # zero-fill pass: each chunk owns its own (m, l, O) slice.
    nscr = MAXSLOT * MAXCH * SCR_PER_CH
    prep_grid = (TP + PREP_THREADS - 1) // PREP_THREADS

    MAXITEM = TP + MAXCH * MAXSLOT

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    fn = cute.compile(
        pca_launch_uniform, z, z, z, z, z, z, z, z, z, z, z, Float32(0.0),
        Int32(0), Int32(0), TP, tt_c, MAXITEM, tile_n, nwarp, nstage, cta_sm, sched,
        grid_x, prep_grid, stream)
    nmeta = 8 + 2 * MAXSLOT if sched else 8
    nwit = MAXITEM * 4 if sched else 1
    nscr_words = nscr if sched else 1
    return fn, nmeta, nwit + nscr_words, nwit * 4, grid_x, tile_n


def run_uniform(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale,
        q_per_kv, token_dim, n_kv_heads, use_v_bias, force_fp32,
        Out, LSE):
    TP = Q.shape[0]
    TT = tokens.shape[0]
    # sched is the only build input that still depends on the observation count, and it is a
    # coarse threshold rather than the count itself.
    sched = TT / max(TP, 1) >= 60.0
    tt_c = TT_LAYOUT_BOUND if TT <= TT_LAYOUT_BOUND else TT
    key = (TP, sched, tt_c)
    ent = _CACHE_uniform.get(key)
    if ent is None:
        ent = _build_uniform(TP, TT, tt_c, Q.device)
        _CACHE_uniform[key] = ent
    fn, nmeta, nwork, scr_offset, grid_x, tile_n = ent
    CH = max(TT // (2 * grid_x), 8 * tile_n)
    CH2 = max(TT // (8 * grid_x), 2 * tile_n)

    dev = Q.device
    meta = torch.zeros(nmeta, dtype=torch.int32, device=dev)
    work = torch.empty(nwork, dtype=torch.int32, device=dev)
    pw = work.data_ptr()

    cs = torch.cuda.current_stream().cuda_stream
    stream = _STREAM_uniform.get(cs)
    if stream is None:
        stream = cuda_driver.CUstream(cs)
        _STREAM_uniform[cs] = stream

    I = cutlass.Int64
    fn(I(Q.data_ptr()), I(tokens.data_ptr()), I(W_k.data_ptr()),
       I(W_v.data_ptr()), I(B_v.data_ptr()), I(cu_seqlens_k.data_ptr()),
       I(Out.data_ptr()), I(LSE.data_ptr()), I(meta.data_ptr()),
       I(pw), I(pw + scr_offset), Float32(float(scale) * LOG2_E),
       Int32(CH), Int32(CH2), stream)


# ------------------------ backward
# ---------------------------------------------------------------------------

def smem_layout_atom_bwd(dtype, k_dim):
    dtype_byte = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte
    smem_k_block_size = (
        128 if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    swizzle_bits = (
        4 if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if k_dim % 32 == 0 else 16, smem_k_block_size), order=(1, 0)
        ),
    )


def convert_layout_acc_mn_bwd(acc_layout):
    """((2,2), MMA_M, MMA_N) -> ((2, MMA_M), (2, MMA_N))."""
    cm = cute.make_layout(acc_layout.shape)
    shape = (
        (cm.shape[0][1], cm.shape[1]),
        (cm.shape[0][0], *cm.shape[0][2:], cm.shape[2]),
        *cm.shape[3:],
    )
    stride = (
        (cm.stride[0][1], cm.stride[1]),
        (cm.stride[0][0], *cm.stride[0][2:], cm.stride[2]),
        *cm.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def acc_mn_view_bwd(acc):
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn_bwd(acc.layout))


def convert_layout_acc_frgA_bwd(acc_layout):
    """C fragment of an (M,N) tile -> A fragment of an (M,K=N) mma."""
    if cute.rank(acc_layout.shape[0]) == 3:
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        return cute.make_layout(
            ((l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
             l.shape[1],
             (l.shape[0][2][1], l.shape[2])),
            stride=((l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                    l.stride[1],
                    (l.stride[0][2][1], l.stride[2])),
        )
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
        stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
    )


def transpose_view_bwd(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


# --------------------------------------------------------------------------- #
# gemm helpers
# --------------------------------------------------------------------------- #
@cute.jit
def gemm_rs_bwd(tiled_mma, acc, tCrA, tCrB, tCsB, smem_thr_copy_B):
    """A already in registers, B streamed from smem via ldmatrix."""
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    nk = cute.size(tCrA.shape[2])
    # The B fragment already spans the whole K range, so issuing every ldmatrix
    # before the first MMA costs no register state and exposes the loads to the
    # memory pipe while the tensor pipe is still draining the previous tile.
    for k in cutlass.range_constexpr(nk):
        cute.copy(smem_thr_copy_B, tCsB[None, None, k], tCrB_view[None, None, k])
    for k in cutlass.range_constexpr(nk):
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def gemm_rs2_bwd(tiled_mma, acc0, acc1, tCrA0, tCrA1, tCrB, tCsB, smem_thr_copy_B):
    """Two A operands share one streamed B tile (score + dPe, R + Z)."""
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    nk = cute.size(tCrA0.shape[2])
    for k in cutlass.range_constexpr(nk):
        cute.copy(smem_thr_copy_B, tCsB[None, None, k], tCrB_view[None, None, k])
    for k in cutlass.range_constexpr(nk):
        cute.gemm(tiled_mma, acc0, tCrA0[None, None, k], tCrB[None, None, k], acc0)
        cute.gemm(tiled_mma, acc1, tCrA1[None, None, k], tCrB[None, None, k], acc1)


@cute.jit
def gemm_ss_kchunk_bwd(tiled_mma, acc, sA, sB, smem_thr_copy_A, smem_thr_copy_B,
                   thr_mma, m_ext, n_ext, k_chunk: cutlass.Constexpr,
                   nchunk: cutlass.Constexpr):
    """Both operands from smem, blocked over K so only one chunk of A/B
    fragments is live at a time (keeps the long-K d_tokens gemm off the
    register-spill cliff)."""
    for c in cutlass.range_constexpr(nchunk):
        sAc = cute.local_tile(sA, (m_ext, k_chunk), (0, c))
        sBc = cute.local_tile(sB, (n_ext, k_chunk), (0, c))
        rA = thr_mma.make_fragment_A(thr_mma.partition_A(sAc))
        rB = thr_mma.make_fragment_B(thr_mma.partition_B(sBc))
        rA_view = smem_thr_copy_A.retile(rA)
        rB_view = smem_thr_copy_B.retile(rB)
        tsA = smem_thr_copy_A.partition_S(sAc)
        tsB = smem_thr_copy_B.partition_S(sBc)
        for k in cutlass.range_constexpr(cute.size(rA.shape[2])):
            cute.copy(smem_thr_copy_A, tsA[None, None, k], rA_view[None, None, k])
            cute.copy(smem_thr_copy_B, tsB[None, None, k], rB_view[None, None, k])
            cute.gemm(tiled_mma, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_sr_k_bwd(tiled_mma, acc, sA, tCrB, smem_thr_copy_A, thr_mma,
              m_ext, k_chunk: cutlass.Constexpr, nchunk: cutlass.Constexpr):
    """A streamed from smem one K-chunk at a time, B already resident in
    registers (hoisted once per pixel)."""
    for c in cutlass.range_constexpr(nchunk):
        sAc = cute.local_tile(sA, (m_ext, k_chunk), (0, c))
        rA = thr_mma.make_fragment_A(thr_mma.partition_A(sAc))
        rA_view = smem_thr_copy_A.retile(rA)
        tsA = smem_thr_copy_A.partition_S(sAc)
        nk = cute.size(rA.shape[2])
        # Issue the chunk's independent ldmatrix operations before either MMA.
        # The fragment already spans the whole K=32 chunk, so this adds no
        # register state while exposing both loads to the memory pipe before
        # the tensor pipe consumes them.
        for k in cutlass.range_constexpr(nk):
            cute.copy(smem_thr_copy_A, tsA[None, None, k], rA_view[None, None, k])
        for k in cutlass.range_constexpr(nk):
            cute.gemm(tiled_mma, acc, rA[None, None, k],
                      tCrB[None, None, c * nk + k], acc)


@cute.jit
def gemm_ss_bwd(tiled_mma, acc, tCrA, tCrB, tCsA, tCsB, smem_thr_copy_A, smem_thr_copy_B):
    tCrA_view = smem_thr_copy_A.retile(tCrA)
    tCrB_view = smem_thr_copy_B.retile(tCrB)
    nk = cute.size(tCsA.shape[2])
    for k in cutlass.range_constexpr(nk):
        cute.copy(smem_thr_copy_A, tCsA[None, None, k], tCrA_view[None, None, k])
        cute.copy(smem_thr_copy_B, tCsB[None, None, k], tCrB_view[None, None, k])
    for k in cutlass.range_constexpr(nk):
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


# --------------------------------------------------------------------------- #
# prep kernel: heavy pixels first
# --------------------------------------------------------------------------- #
@cute.kernel
def prep_kernel_bwd(
    mCu: cute.Tensor,       # (TP+1,) i32
    mWit: cute.Tensor,      # (MAXITEM*4,) i32: pixel, lo, hi, split slot
    mBuck: cute.Tensor,     # (NBUCK*TP,) i32
    mMeta: cute.Tensor,
    TP: Int32,
    CH: Int32,              # split threshold (tokens)
    CH2: Int32,             # priority threshold (tokens)
    tile_n: cutlass.Constexpr,
    maxchunk: cutlass.Constexpr,
    buck: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    p = bidx * PREP_THREADS + tidx
    if p < TP:
        L = mCu[p + 1] - mCu[p]
        if const_expr(buck):
            if L <= CH2:
                b = Int32(0)
                for k in cutlass.range_constexpr(1, NBUCK):
                    if L <= (CH2 // Int32(1 << k)):
                        b = Int32(k)
                pos = cute.arch.atomic_add(
                    mMeta.iterator + (Int32(MBUCK) + b), Int32(1))
                mBuck[b * TP + pos] = p
        if L > CH2:
            nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
            nc = Int32(1)
            slot = Int32(-1)
            if L > CH:
                s = cute.arch.atomic_add(mMeta.iterator + 2, Int32(1))
                if s < Int32(MAXSLOT):
                    slot = s
                    nc = cutlass.min((L + CH - Int32(1)) // CH, Int32(maxchunk))
            tpc = (nt + nc - Int32(1)) // nc
            nc = (nt + tpc - Int32(1)) // tpc
            if nc == Int32(1):
                slot = Int32(-1)
            if slot >= Int32(0):
                mMeta[8 + slot] = nc
                mMeta[8 + MAXSLOT + slot] = nc
            base = cute.arch.atomic_add(mMeta.iterator + 0, nc)
            for c in cutlass.range(nc):
                lo = c * tpc
                hi = cutlass.min(lo + tpc, nt)
                w2 = (base + c) * Int32(4)
                mWit[w2] = p
                # Keep both int32 endpoints losslessly.  The public ragged ABI
                # does not impose a 16-bit bound on the number of token tiles.
                mWit[w2 + 1] = lo
                mWit[w2 + 2] = hi
                mWit[w2 + 3] = (slot * Int32(MAXCHUNK) + c) if slot >= Int32(0) else Int32(-1)


@cute.kernel
def bucket_prefix_kernel_bwd(mMeta: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        racc = mMeta[0]
        for b in cutlass.range_constexpr(NBUCK):
            mMeta[MBUCK + NBUCK + b] = racc
            racc = racc + mMeta[MBUCK + b]
        mMeta[0] = racc


@cute.kernel
def bucket_compact_kernel_bwd(
    mCu: cute.Tensor,
    mWit: cute.Tensor,
    mBuck: cute.Tensor,
    mMeta: cute.Tensor,
    TP: Int32,
    tile_n: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, b, _ = cute.arch.block_idx()
    pos = bx * PREP_THREADS + tidx
    if pos < mMeta[MBUCK + b]:
        p = mBuck[b * TP + pos]
        out = mMeta[MBUCK + NBUCK + b] + pos
        L = mCu[p + 1] - mCu[p]
        nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
        w4 = out * Int32(4)
        mWit[w4] = p
        mWit[w4 + 1] = Int32(0)
        mWit[w4 + 2] = nt
        mWit[w4 + 3] = Int32(-1)


# --------------------------------------------------------------------------- #
# finalize: reduce per-CTA dW/dB partials
# --------------------------------------------------------------------------- #
@cute.kernel
def fin_kernel_bwd(
    mScr: cute.Tensor,      # (grid_x * SCR_PER_CTA,) f32
    mdWk: cute.Tensor,      # (2048,) bf16
    mdWv: cute.Tensor,      # (2048,) bf16
    mdBv: cute.Tensor,      # (64,)   bf16
    FinStorage: cutlass.Constexpr,
    grid_x: Int32,
):
    """Column reduction of the per-CTA dW/dB scratch.

    32 outputs per block, FIN_WARPS warps each sweeping a 1/FIN_WARPS stride of
    the CTA range, so the global loads stay 128B-coalesced across the lane
    dimension and the long CTA loop has FIN_WARPS independent chains in flight.
    Only 130 blocks are available (32 outputs each), so the per-block warp count
    -- not the grid -- is what keeps the SMs busy.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    fst = smem.allocate(FinStorage)
    sSum = fst.buf.get_tensor(cute.make_ordered_layout((FIN_WARPS, 32), order=(1, 0)))
    lane = tidx % 32
    wid = tidx // 32
    o = bidx * Int32(32) + lane
    a = Float32(0.0)
    if o < Int32(4096):
        for c in cutlass.range(wid, grid_x, FIN_WARPS):
            a = a + mScr[c * Int32(SCR_PER_CTA) + o]
    elif o < Int32(4160):
        j = o - Int32(4096) + Int32(SCR_BV)
        for c in cutlass.range(wid, grid_x, FIN_WARPS):
            base = c * Int32(SCR_PER_CTA) + j
            a = a + mScr[base] + mScr[base + Int32(64)]
    sSum[wid, lane] = a
    cute.arch.barrier()
    if wid == 0:
        t = sSum[0, lane]
        for q in cutlass.range_constexpr(1, FIN_WARPS):
            t = t + sSum[q, lane]
        if o < Int32(2048):
            mdWk[o] = t.to(cutlass.BFloat16)
        elif o < Int32(4096):
            mdWv[o - Int32(2048)] = t.to(cutlass.BFloat16)
        elif o < Int32(4160):
            mdBv[o - Int32(4096)] = t.to(cutlass.BFloat16)


# --------------------------------------------------------------------------- #
# main backward kernel
# --------------------------------------------------------------------------- #
@cute.kernel
def bwd_kernel_bwd(
    mdO: cute.Tensor,       # (TP*64, 32) bf16
    mQ: cute.Tensor,        # (TP*64, 32) bf16
    mTok: cute.Tensor,      # (TT, 32)    bf16
    mWkf: cute.Tensor,      # (2048,)     bf16
    mWvf: cute.Tensor,      # (2048,)     bf16
    mBv: cute.Tensor,       # (64,)       bf16
    mOut: cute.Tensor,      # (TP*64, 32) bf16
    mLSE: cute.Tensor,      # (TP, 64)    f32
    mCu: cute.Tensor,       # (TP+1,)     i32
    mdQ: cute.Tensor,       # (TP*64, 32) bf16
    mdTok: cute.Tensor,     # (TT, 32)    bf16
    mScr: cute.Tensor,      # (grid*SCR_PER_CTA,) f32
    mMeta: cute.Tensor,     # (8 + 2*MAXSLOT,) i32
    mWit: cute.Tensor,      # (MAXITEM*4,) i32: pixel, lo, hi, split slot
    mScrR: cute.Tensor,     # split-pixel R partials, f32
    scale: Float32,
    sl: Float32,
    TP: Int32,
    CH2: Int32,
    tile_n: cutlass.Constexpr,
    NSTAGE: cutlass.Constexpr,
    sched: cutlass.Constexpr,
    sQ_layout: cute.ComposedLayout,
    sTok_layout: cute.ComposedLayout,
    sW_layout: cute.ComposedLayout,
    sQZ_layout: cute.ComposedLayout,
    sScore_layout: cute.ComposedLayout,
    sDT_layout: cute.ComposedLayout,
    gmem_copy_load: cute.TiledCopy,
    gmem_copy_store: cute.TiledCopy,
    gmem_copy_tok: cute.TiledCopy,
    mma_h: cute.TiledMma,
    mma_w: cute.TiledMma,
    mma_w1: cute.TiledMma,
    SUP: cutlass.Constexpr,
    alias_dt: cutlass.Constexpr,
    direct_dt: cutlass.Constexpr,
    resident_b: cutlass.Constexpr,
    use_v_bias: cutlass.Constexpr,
    SharedStorage: cutlass.Constexpr,
):
    sup_n = tile_n * SUP
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    nblocks, _, _ = cute.arch.grid_dim()
    warp_idx = cute.arch.make_warp_uniform(tidx // 32)
    kv = warp_idx // 2

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQ.get_tensor(sQ_layout)
    sdO = storage.sdO.get_tensor(sQ_layout)
    sQZ = storage.sQZ.get_tensor(sQZ_layout)
    sWk = storage.sWk.get_tensor(sW_layout)
    sWv = storage.sWv.get_tensor(sW_layout)
    sTok = storage.sTok.get_tensor(sTok_layout)
    sScore = storage.sScr.get_tensor(sScore_layout)
    sOut = storage.sScr.get_tensor(sQ_layout)       # alias, prologue only
    sDQ = storage.sScr.get_tensor(sQ_layout)        # alias, epilogue only
    sRZ = storage.sScr.get_tensor(sQZ_layout)       # alias, epilogue only
    sDT = (storage.sScr.get_tensor(sDT_layout) if alias_dt
           else storage.sDT.get_tensor(sDT_layout))
    sD = storage.sD.get_tensor(cute.make_layout(NHEAD))
    sBv = storage.sBv.get_tensor(cute.make_layout(NHEAD))
    sIdx = storage.sIdx.get_tensor(cute.make_layout(4))
    sTokFlat = storage.sTok.get_tensor(cute.make_layout(cute.cosize(sTok_layout)))

    # ---- one-time CTA setup -------------------------------------------------
    for i in cutlass.range_constexpr(2048 // BWD_NTHREAD):
        lin = tidx + i * BWD_NTHREAD
        c = (lin % 1024) // DHEAD
        d = lin % DHEAD
        h = lin // 1024
        sWk[c, d, h] = mWkf[lin]
        sWv[c, d, h] = mWvf[lin]
    nzero = cute.cosize(sTok_layout) // BWD_NTHREAD
    for i in cutlass.range_constexpr(nzero):
        sTokFlat[tidx * nzero + i] = cutlass.BFloat16(0.0)
    if tidx < NHEAD:
        if const_expr(use_v_bias):
            sBv[tidx] = mBv[tidx].to(Float32)
        else:
            sBv[tidx] = Float32(0.0)

    # ---- static partitions --------------------------------------------------
    thr_h = mma_h.get_slice(tidx)
    thr_w = mma_w.get_slice(tidx)
    thr0_h = mma_h.get_slice(0)
    gmem_thr_load = gmem_copy_load.get_slice(tidx)
    gmem_thr_tok = gmem_copy_tok.get_slice(tidx)
    gmem_thr_store = gmem_copy_store.get_slice(tidx)

    ldm_n = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                cutlass.BFloat16)
    ldm_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                                cutlass.BFloat16)
    cpA_n = cute.make_tiled_copy_A(ldm_n, mma_h).get_slice(tidx)
    cpB_n = cute.make_tiled_copy_B(ldm_n, mma_h).get_slice(tidx)
    cpB_t = cute.make_tiled_copy_B(ldm_t, mma_h).get_slice(tidx)
    wA_t = cute.make_tiled_copy_A(ldm_t, mma_w).get_slice(tidx)
    wB_t = cute.make_tiled_copy_B(ldm_t, mma_w).get_slice(tidx)
    lane = tidx % 32
    thr_w1 = mma_w1.get_slice(lane)
    wA_t1 = cute.make_tiled_copy_A(ldm_t, mma_w1).get_slice(lane)
    wB_t1 = cute.make_tiled_copy_B(ldm_t, mma_w1).get_slice(lane)

    sWkT = transpose_view_bwd(sWk)                 # (c, d, kv)
    sWvT = transpose_view_bwd(sWv)                 # (c, d, kv)

    tPrA = thr_h.make_fragment_A(thr_h.partition_A(sQ))
    tPsQ = cpA_n.partition_S(sQ)
    tPsdO = cpA_n.partition_S(sdO)
    tPrB = thr_h.make_fragment_B(thr_h.partition_B(sWkT[None, None, 0]))
    tPsWk = cpB_t.partition_S(sWkT)
    tPsWv = cpB_t.partition_S(sWvT)

    tQrB = thr_h.make_fragment_B(thr_h.partition_B(sWk[None, None, 0]))
    tQsWk = cpB_n.partition_S(sWk)

    acc_pro = cute.make_rmem_tensor(thr_h.partition_shape_C((NHEAD, DHEAD)), Float32)
    acc_R = cute.make_rmem_tensor(thr_h.partition_shape_C((NHEAD, DHEAD)), Float32)
    acc_Z = cute.make_rmem_tensor(thr_h.partition_shape_C((NHEAD, DHEAD)), Float32)
    # Warp w owns exactly one of the four 32x32 parameter-gradient GEMMs:
    #   w = 0,1 -> dW_k for kv group w      (Q^T . R)
    #   w = 2,3 -> dW_v for kv group w - 2  (dOut^T . Z)
    # The weight gradients are reduced over every token in the batch, so a BF16
    # loop-carried accumulator is re-rounded once per tile.  That made dW_k/dW_v
    # differ by 1-2e-2 between identical runs; FP32 makes them reproducible for
    # +7% backward latency.
    dw_dtype = Float32
    acc_dW = cute.make_rmem_tensor(
        thr_w1.partition_shape_C((DHEAD, DHEAD)), dw_dtype)
    acc_dW.fill(0.0)
    acc_w = cute.make_rmem_tensor(
        thr_w1.partition_shape_C((DHEAD, DHEAD)), Float32)
    n_w = cute.size(acc_dW)
    bv_part = Float32(0.0)

    # The dW scratch offsets are only needed once, at CTA teardown; with one
    # 32x32 GEMM per warp the table would be 32 live registers for the whole
    # kernel, so it is recomputed at the flush instead of pre-materialised.
    cWt = cute.make_identity_tensor((DHEAD, DHEAD))
    tWc = thr_w1.partition_C(cWt)

    cS = cute.make_identity_tensor((NHEAD, SCORE_N))
    tScS = thr_h.partition_C(cS)
    tScS_mn = acc_mn_view_bwd(tScS)
    n_rows = cute.size(tScS_mn.shape[0])
    t0cS = thr0_h.partition_C(cS)
    col_base = tScS[0][1]
    lse_row = cute.make_rmem_tensor(n_rows, Int32)
    for r in cutlass.range_constexpr(n_rows):
        lse_row[r] = tScS_mn[r, 0][0]
    rowv = cute.make_rmem_tensor(n_rows, Float32)
    rowd = cute.make_rmem_tensor(n_rows, Float32)

    st_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    stC_h = cute.make_tiled_copy_C(st_atom, mma_h).get_slice(tidx)
    stC_w = cute.make_tiled_copy_C(st_atom, mma_w).get_slice(tidx)
    direct_atom = st_atom
    directC_w = stC_w
    if const_expr(direct_dt):
        direct_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
            num_bits_per_copy=32)
        directC_w = cute.make_tiled_copy_C(direct_atom, mma_w).get_slice(tidx)
    # Two 16-column (A, P) score slots per 32-token tile.
    st_sScTop = []
    st_sScBot = []
    for hh in cutlass.range_constexpr(2 * SUP):
        st_sScTop.append(
            stC_h.partition_D(cute.local_tile(sScore, (NHEAD, SCORE_N), (0, hh))))
        st_sScBot.append(
            stC_h.partition_D(cute.local_tile(sScore, (NHEAD, SCORE_N), (1, hh))))
    st_sQZ0 = stC_h.partition_D(cute.local_tile(sQZ, (NHEAD, DHEAD), (0, 0)))
    st_sQZ1 = stC_h.partition_D(cute.local_tile(sQZ, (NHEAD, DHEAD), (1, 0)))
    st_sDQ = stC_h.partition_D(sDQ)
    st_sRZ0 = stC_h.partition_D(cute.local_tile(sRZ, (NHEAD, DHEAD), (0, 0)))
    st_sRZ1 = stC_h.partition_D(cute.local_tile(sRZ, (NHEAD, DHEAD), (1, 0)))
    st_sDT = stC_w.partition_D(sDT)

    sScoreT = transpose_view_bwd(sScore)            # (sup_n, 128)
    sQZT = transpose_view_bwd(sQZ)                  # (32, 128)
    tDrB = thr_w.make_fragment_B(thr_w.partition_B(sQZT))

    cTok = cute.make_identity_tensor((tile_n, DHEAD))
    tKcTok = gmem_thr_tok.partition_S(cTok)
    t0KcTok = gmem_copy_tok.get_slice(0).partition_S(cTok)
    n_cpy_m = cute.size(tKcTok.shape[1])
    tok_row_base = tKcTok[0, 0, 0][0]
    cTokS = cute.make_identity_tensor((sup_n, DHEAD))
    tOcTok = gmem_thr_store.partition_S(cTokS)
    t0OcTok = gmem_copy_store.get_slice(0).partition_S(cTokS)
    n_st_m = cute.size(tOcTok.shape[1])
    st_row_base = tOcTok[0, 0, 0][0]

    tQsQ = gmem_thr_load.partition_D(sQ)
    tQsdO = gmem_thr_load.partition_D(sdO)
    tQsOut = gmem_thr_load.partition_D(sOut)
    tKsTok = gmem_thr_tok.partition_D(sTok)
    tOsDQ = gmem_thr_store.partition_S(sDQ)
    tOsDT = gmem_thr_store.partition_S(sDT)

    dq = tidx % 4
    dq8 = dq * 8
    dr = tidx // 4
    bo = tidx % NHEAD
    bh0 = (bo // 32) * 32 + (tidx // NHEAD) * 16
    bd = bo % 32

    cute.arch.barrier()

    # ---- work loop ----------------------------------------------------------
    if const_expr(sched):
        n_prio = mMeta[0]
        W = n_prio if const_expr(sched == 2) else (n_prio + TP)
    else:
        n_prio = Int32(0)
        W = TP
    w = Int32(bidx)

    # The four scheduler descriptor words plus the two cu_seqlens endpoints form
    # a two-level dependent chain of scattered global loads whose latency used to
    # sit on the work loop's back edge.  They are carried in registers instead
    # and issued one whole item ahead, right after the token rendezvous.
    wc0 = cutlass.min(w, W - Int32(1))
    if const_expr(sched):
        pr0 = wc0 < n_prio
        i0 = (wc0 if pr0 else Int32(0)) * Int32(4)
        d_p = mWit[i0] if pr0 else (wc0 - n_prio)
        d_lo = mWit[i0 + 1] if pr0 else Int32(0)
        d_hi = mWit[i0 + 2] if pr0 else Int32(-1)
        d_senc = mWit[i0 + 3] if pr0 else Int32(-1)
    else:
        d_p = wc0
        d_lo = Int32(0)
        d_hi = Int32(-1)
        d_senc = Int32(-1)
    d_base = mCu[d_p]
    d_end = mCu[d_p + 1]

    while w < W:
        p = d_p
        senc = d_senc
        base = d_base
        L = d_end - base
        nt = (L + Int32(tile_n - 1)) // Int32(tile_n)
        lo = d_lo
        # d_hi < 0 marks a sweep item: it owns the whole pixel unless the pixel
        # was large enough to have been handed to the priority list.
        if const_expr(sched):
            hi = d_hi if d_hi >= Int32(0) else (nt if L <= CH2 else Int32(0))
        else:
            hi = nt
        n_tiles = hi - lo
        k0 = base + lo * Int32(tile_n)
        Lc = L - lo * Int32(tile_n)
        is_split = senc >= Int32(0)
        sidx = (senc // Int32(MAXCHUNK)) if is_split else Int32(-1)
        cid = (senc % Int32(MAXCHUNK)) if is_split else Int32(0)

        gQ = cute.local_tile(mQ, (NHEAD, DHEAD), (p, 0))
        gdO = cute.local_tile(mdO, (NHEAD, DHEAD), (p, 0))
        gOut = cute.local_tile(mOut, (NHEAD, DHEAD), (p, 0))
        if n_tiles > Int32(0):
            # Scattered per-head LSE gather: not needed until the first score
            # tile, so it is issued ahead of the staging copies and covered by
            # the whole prologue.
            for r in cutlass.range_constexpr(n_rows):
                rowv[r] = mLSE[p, lse_row[r]]
            cute.copy(gmem_copy_load, gmem_thr_load.partition_S(gQ), tQsQ)
            cute.copy(gmem_copy_load, gmem_thr_load.partition_S(gdO), tQsdO)
            cute.copy(gmem_copy_load, gmem_thr_load.partition_S(gOut), tQsOut)
        cute.arch.cp_async_commit_group()

        mTokP = cute.domain_offset((k0, 0), mTok)
        gTok = cute.local_tile(mTokP, (tile_n, DHEAD), (None, 0))
        tKgTok = gmem_thr_tok.partition_S(gTok)
        for s in cutlass.range_constexpr(NSTAGE - 1):
            if s < n_tiles:
                rows_valid = Lc - s * tile_n - tok_row_base
                for m in cutlass.range_constexpr(n_cpy_m):
                    if t0KcTok[0, m, 0][0] < rows_valid:
                        cute.copy(gmem_copy_tok, tKgTok[None, m, None, s],
                                  tKsTok[None, m, None, s])
            cute.arch.cp_async_commit_group()

        if tidx == 0:
            if w < nblocks:
                sIdx[0] = w + nblocks
            else:
                ticket = cute.arch.atomic_add(mMeta.iterator + 1, Int32(1))
                sIdx[0] = Int32(2) * nblocks + ticket

        # ---------------- prologue ----------------
        # The rendezvous is hoisted out of the n_tiles branch so the next item's
        # descriptor loads can be issued here, at the top of the token loop,
        # rather than on the loop's back edge.
        cute.arch.cp_async_wait_group(NSTAGE - 1)
        cute.arch.barrier()
        wn = sIdx[0]
        wnc = cutlass.min(wn, W - Int32(1))
        if const_expr(sched):
            prn = wnc < n_prio
            i1 = (wnc if prn else Int32(0)) * Int32(4)
            d_p = mWit[i1] if prn else (wnc - n_prio)
            d_lo = mWit[i1 + 1] if prn else Int32(0)
            d_hi = mWit[i1 + 2] if prn else Int32(-1)
            d_senc = mWit[i1 + 3] if prn else Int32(-1)
        else:
            d_p = wnc
            d_lo = Int32(0)
            d_hi = Int32(-1)
            d_senc = Int32(-1)
        d_base = mCu[d_p]
        d_end = mCu[d_p + 1]

        if n_tiles > Int32(0):
            for it in cutlass.range_constexpr(2):
                hh = dr + Int32(32 * it)
                gO = cute.local_tile(sdO, (1, 8), (hh, dq))
                gU = cute.local_tile(sOut, (1, 8), (hh, dq))
                rO8 = cute.make_fragment_like(gO, cutlass.BFloat16)
                rU8 = cute.make_fragment_like(gU, cutlass.BFloat16)
                cute.autovec_copy(gO, rO8)
                cute.autovec_copy(gU, rU8)
                dacc = Float32(0.0)
                if const_expr(use_v_bias):
                    for i in cutlass.range_constexpr(8):
                        dacc = dacc + rO8[0, i].to(Float32) * (
                            rU8[0, i].to(Float32) - sBv[Int32(it * 32) + dq8 + i])
                else:
                    for i in cutlass.range_constexpr(8):
                        dacc = dacc + rO8[0, i].to(Float32) * rU8[0, i].to(Float32)
                dacc = dacc + cute.arch.shuffle_sync_bfly(dacc, offset=1)
                dacc = dacc + cute.arch.shuffle_sync_bfly(dacc, offset=2)
                if dq == Int32(0):
                    sD[hh] = dacc
            if const_expr(use_v_bias):
                if cid == Int32(0):
                    for j in cutlass.range_constexpr(16):
                        bv_part = bv_part + sdO[bh0 + j, bd].to(Float32)

            acc_pro.fill(0.0)
            gemm_ss_bwd(mma_h, acc_pro, tPrA, tPrB, tPsQ,
                    tPsWk[None, None, None, kv], cpA_n, cpB_t)
            rQs = cute.make_fragment_like(acc_pro, cutlass.BFloat16)
            rQs.store((acc_pro.load() * sl).to(cutlass.BFloat16))
            tSrQ = cute.make_tensor(rQs.iterator, convert_layout_acc_frgA_bwd(rQs.layout))
            rQe = cute.make_fragment_like(acc_pro, cutlass.BFloat16)
            rQe.store((acc_pro.load() * scale).to(cutlass.BFloat16))

            acc_pro.fill(0.0)
            gemm_ss_bwd(mma_h, acc_pro, tPrA, tPrB, tPsdO,
                    tPsWv[None, None, None, kv], cpA_n, cpB_t)
            rdZ = cute.make_fragment_like(acc_pro, cutlass.BFloat16)
            rdZ.store(acc_pro.load().to(cutlass.BFloat16))
            tSrdZ = cute.make_tensor(rdZ.iterator, convert_layout_acc_frgA_bwd(rdZ.layout))

            cute.copy(st_atom, stC_h.retile(rQe), st_sQZ0)
            cute.copy(st_atom, stC_h.retile(rdZ), st_sQZ1)

            acc_R.fill(0.0)
            acc_Z.fill(0.0)
            # sD is produced by the quad leaders of every warp but consumed by
            # threads owning different head rows, so the read has to sit behind
            # a CTA rendezvous.  The resident-B publication barrier already
            # exists at exactly this point; reuse it instead of adding one.
            cute.arch.barrier()
            if const_expr(resident_b):
                # Long pixels reuse (Qs'|dZ) across many dTokens groups.  The
                # 3-CTA specialization keeps its four K=32 chunks resident.
                cute.copy(wB_t, wB_t.partition_S(sQZT), wB_t.retile(tDrB))
            for r in cutlass.range_constexpr(n_rows):
                rowd[r] = sD[lse_row[r]]

            # The score/R/Z work is done one tile_n tile at a time, but the
            # A/P transposes are buffered for SUP tiles and consumed by a single
            # wider d_tokens GEMM.  Its B operand (Qs'|dZ) is then fetched once
            # per SUP tiles instead of once per tile, which is the largest
            # single item of shared-memory traffic in the token loop.
            n_full = cutlass.min(n_tiles, Lc // Int32(tile_n))
            n_grp = n_full // Int32(SUP)
            for g in cutlass.range(0, n_grp):
                for h in cutlass.range_constexpr(SUP):
                    n = g * Int32(SUP) + h
                    stage = n % NSTAGE
                    cute.arch.cp_async_wait_group(NSTAGE - 2)
                    cute.arch.barrier()
                    nl = n + NSTAGE - 1
                    if nl < n_tiles:
                        rows_valid = Lc - nl * tile_n - tok_row_base
                        for m in cutlass.range_constexpr(n_cpy_m):
                            if t0KcTok[0, m, 0][0] < rows_valid:
                                cute.copy(gmem_copy_tok, tKgTok[None, m, None, nl],
                                          tKsTok[None, m, None, nl % NSTAGE])
                    cute.arch.cp_async_commit_group()
                    score_tile_bwd(
                        mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
                        st_sScTop[2 * h], st_sScBot[2 * h],
                        st_sScTop[2 * h + 1], st_sScBot[2 * h + 1],
                        acc_R, acc_Z,
                        tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
                        stage, Int32(tile_n), tile_n, NSTAGE, False,
                    )
                dtok_flush_bwd(
                    mma_w, thr_w, wA_t, wB_t, st_atom, stC_w,
                    direct_atom, directC_w,
                    gmem_copy_store, gmem_thr_store, t0OcTok, st_row_base,
                    n_st_m, st_sDT, tOsDT, sScoreT, sQZT, tDrB, mdTok,
                    k0 + g * Int32(SUP * tile_n), Int32(sup_n),
                    tile_n, sup_n, alias_dt, direct_dt, resident_b, False,
                )

            # Trailing group: up to one full tile plus up to one masked tile.
            # (`n_tiles > n0` covers both; Python `or` would collapse the two
            # dynamic predicates into the first one.)
            n0 = n_grp * Int32(SUP)
            has_a = n_full > n0
            has_b = n_tiles > n_full
            if n_tiles > n0:
                if const_expr(SUP == 1):
                    tile_step_bwd(
                        mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
                        st_sScTop[0], st_sScBot[0], st_sScTop[1], st_sScBot[1],
                        acc_R, acc_Z,
                        tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
                        gmem_copy_tok, tKgTok, tKsTok, t0KcTok, tok_row_base,
                        n_cpy_m, n0, n_tiles, Lc, Lc - n0 * Int32(tile_n),
                        tile_n, NSTAGE, True,
                    )
                elif has_a:
                    tile_step_bwd(
                        mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
                        st_sScTop[0], st_sScBot[0], st_sScTop[1], st_sScBot[1],
                        acc_R, acc_Z,
                        tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
                        gmem_copy_tok, tKgTok, tKsTok, t0KcTok, tok_row_base,
                        n_cpy_m, n0, n_tiles, Lc, Int32(tile_n),
                        tile_n, NSTAGE, False,
                    )
                    if has_b:
                        tile_step_bwd(
                            mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
                            st_sScTop[2], st_sScBot[2], st_sScTop[3], st_sScBot[3],
                            acc_R, acc_Z,
                            tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
                            gmem_copy_tok, tKgTok, tKsTok, t0KcTok,
                            tok_row_base, n_cpy_m, n0 + Int32(1), n_tiles, Lc,
                            Lc - (n0 + Int32(1)) * Int32(tile_n),
                            tile_n, NSTAGE, True,
                        )
                else:
                    tile_step_bwd(
                        mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
                        st_sScTop[0], st_sScBot[0], st_sScTop[1], st_sScBot[1],
                        acc_R, acc_Z,
                        tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
                        gmem_copy_tok, tKgTok, tKsTok, t0KcTok, tok_row_base,
                        n_cpy_m, n0, n_tiles, Lc, Lc - n0 * Int32(tile_n),
                        tile_n, NSTAGE, True,
                    )
                # A split pixel's chunk can end on a tile boundary well before
                # its own Lc, so the group is bounded by tiles as well as by
                # tokens -- otherwise the stale half of sScore would be written
                # over the next chunk's rows.
                dtok_flush_bwd(
                    mma_w, thr_w, wA_t, wB_t, st_atom, stC_w,
                    direct_atom, directC_w,
                    gmem_copy_store, gmem_thr_store, t0OcTok, st_row_base,
                    n_st_m, st_sDT, tOsDT, sScoreT, sQZT, tDrB, mdTok,
                    k0 + n0 * Int32(tile_n),
                    cutlass.min(Lc - n0 * Int32(tile_n),
                                (n_tiles - n0) * Int32(tile_n)),
                    tile_n, sup_n, alias_dt, direct_dt, resident_b, True,
                )

            # ---------------- epilogue ----------------
            # A split pixel's chunks each own a token range, so d_tokens and the
            # dW/dB partials are already complete per chunk.  Only R (which dQ
            # needs whole) is reduced: every chunk deposits its fp32 partial in
            # the same per-thread interleaving and the last arriver folds them.
            rRloc = cute.make_fragment_like(acc_R, cutlass.BFloat16)
            rRloc.store((acc_R.load() * scale).to(cutlass.BFloat16))
            own = Int32(1)
            if const_expr(sched):
                if senc >= Int32(0):
                    cb = senc * Int32(RSTATE)
                    for i in cutlass.range_constexpr(cute.size(acc_R)):
                        mScrR[cb + Int32(i * BWD_NTHREAD) + tidx] = acc_R[i]
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier()
                    if tidx == 0:
                        old = cute.arch.atomic_add(
                            mMeta.iterator + (Int32(8 + MAXSLOT) + sidx),
                            Int32(-1), sem="acq_rel", scope="gpu")
                        sIdx[1] = Int32(1) if old == Int32(1) else Int32(0)
                    cute.arch.barrier()
                    if sIdx[1] == Int32(1):
                        ncs = mMeta[8 + sidx]
                        acc_R.fill(0.0)
                        for c in cutlass.range(ncs):
                            cb2 = (sidx * Int32(MAXCHUNK) + c) * Int32(RSTATE)
                            for i in cutlass.range_constexpr(cute.size(acc_R)):
                                acc_R[i] = acc_R[i] + mScrR[
                                    cb2 + Int32(i * BWD_NTHREAD) + tidx]
                    own = sIdx[1]

            rZ = cute.make_fragment_like(acc_Z, cutlass.BFloat16)
            rZ.store(acc_Z.load().to(cutlass.BFloat16))
            if own == Int32(1):
                # Only the last split chunk owns the whole-pixel R reduction.
                # Non-owners still contribute local R/Z to dW, but must not pay
                # for a dQ projection whose result would be discarded.
                rR = cute.make_fragment_like(acc_R, cutlass.BFloat16)
                rR.store((acc_R.load() * scale).to(cutlass.BFloat16))
                tQrA = cute.make_tensor(
                    rR.iterator, convert_layout_acc_frgA_bwd(rR.layout))
                acc_pro.fill(0.0)
                gemm_rs_bwd(mma_h, acc_pro, tQrA, tQrB,
                        tQsWk[None, None, None, kv], cpB_n)
                rDQ = cute.make_fragment_like(acc_pro, cutlass.BFloat16)
                rDQ.store(acc_pro.load().to(cutlass.BFloat16))
                cute.arch.barrier()
                cute.copy(st_atom, stC_h.retile(rDQ), st_sDQ)
                cute.arch.barrier()
                gdQ = cute.local_tile(mdQ, (NHEAD, DHEAD), (p, 0))
                rOs = cute.make_fragment_like(tOsDQ, cutlass.BFloat16)
                cute.autovec_copy(tOsDQ, rOs)
                cute.copy(gmem_copy_store, rOs, gmem_thr_store.partition_D(gdQ))
                cute.arch.barrier()

            cute.copy(st_atom, stC_h.retile(rRloc), st_sRZ0)
            cute.copy(st_atom, stC_h.retile(rZ), st_sRZ1)
            cute.arch.barrier()
            # Each warp owns one whole 32x32x32 GEMM, so its operand pair is
            # ldmatrix-ed exactly once instead of once per warp-column.
            acc_w.store(acc_dW.load().to(Float32))
            if warp_idx == Int32(0):
                dw_gemm_bwd(mma_w1, thr_w1, wA_t1, wB_t1, acc_w, sQ, sRZ, 0, 0)
            elif warp_idx == Int32(1):
                dw_gemm_bwd(mma_w1, thr_w1, wA_t1, wB_t1, acc_w, sQ, sRZ, 1, 1)
            elif warp_idx == Int32(2):
                dw_gemm_bwd(mma_w1, thr_w1, wA_t1, wB_t1, acc_w, sdO, sRZ, 0, 2)
            else:
                dw_gemm_bwd(mma_w1, thr_w1, wA_t1, wB_t1, acc_w, sdO, sRZ, 1, 3)
            acc_dW.store(acc_w.load().to(dw_dtype))
        else:
            if L == Int32(0):
                gdQ0 = cute.local_tile(mdQ, (NHEAD, DHEAD), (p, 0))
                rzero = cute.make_fragment_like(tOsDQ, cutlass.BFloat16)
                rzero.fill(0.0)
                cute.copy(gmem_copy_store, rzero, gmem_thr_store.partition_D(gdQ0))

        cute.arch.barrier()
        w = wn

    # ---- flush per-CTA dW / dB partials -------------------------------------
    # warp 0/1 -> dW_k[kv], warp 2/3 -> dW_v[kv]; the four warps together tile
    # the 4096-float dW scratch exactly once.
    obase = bidx * Int32(SCR_PER_CTA)
    dw_base = (Int32(SCR_WK) if warp_idx < Int32(2) else Int32(SCR_WV)) \
        + (warp_idx % Int32(2)) * Int32(1024)
    for i in cutlass.range_constexpr(n_w):
        mScr[obase + dw_base + Int32(tWc[i][0] * DHEAD + tWc[i][1])] = \
            acc_dW[i].to(Float32)
    mScr[obase + Int32(SCR_BV) + tidx] = bv_part


@cute.jit
def dw_gemm_bwd(mma_w1, thr_w1, wA_t1, wB_t1, acc, sA, sRZ,
            kv: cutlass.Constexpr, bj: cutlass.Constexpr):
    """Single-warp 32x32x32 parameter-gradient accumulation, A^T . B."""
    sAj = transpose_view_bwd(cute.local_tile(sA, (DHEAD, DHEAD), (kv, 0)))
    sBj = transpose_view_bwd(cute.local_tile(sRZ, (DHEAD, DHEAD), (bj, 0)))
    aF = thr_w1.make_fragment_A(thr_w1.partition_A(sAj))
    bF = thr_w1.make_fragment_B(thr_w1.partition_B(sBj))
    gemm_ss_bwd(mma_w1, acc, aF, bF, wA_t1.partition_S(sAj),
            wB_t1.partition_S(sBj), wA_t1, wB_t1)


@cute.jit
def score_tile_bwd(
    mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
    st_sScTop0, st_sScBot0, st_sScTop1, st_sScBot1, acc_R, acc_Z,
    tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
    stage, rem,
    tile_n: cutlass.Constexpr, NSTAGE: cutlass.Constexpr,
    mask: cutlass.Constexpr,
):
    """One 32-token tile as two 16-token score/softmax/RZ subtiles."""
    st_top = (st_sScTop0, st_sScTop1)
    st_bot = (st_sScBot0, st_sScBot1)
    for sub in cutlass.range_constexpr(2):
        # Keep the cp.async stage mode projected through the 2-D subtile.
        sTokH = cute.local_tile(sTok, (SCORE_N, DHEAD), (sub, 0, None))
        sTokHT = transpose_view_bwd(sTokH)
        rBs = thr_h.make_fragment_B(thr_h.partition_B(sTokH[None, None, 0]))
        acc_S = cute.make_rmem_tensor(
            thr_h.partition_shape_C((NHEAD, SCORE_N)), Float32)
        acc_P = cute.make_rmem_tensor(
            thr_h.partition_shape_C((NHEAD, SCORE_N)), Float32)
        acc_S.fill(0.0)
        acc_P.fill(0.0)
        gemm_rs2_bwd(mma_h, acc_S, acc_P, tSrQ, tSrdZ, rBs,
                 cpB_n.partition_S(sTokH)[None, None, None, stage], cpB_n)

        if const_expr(mask):
            col_limit = rem - Int32(sub * SCORE_N) - col_base
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                acc_S[i] = (acc_S[i] if t0cS[i][1] < col_limit
                            else -Float32.inf)

        acc_S_mn = acc_mn_view_bwd(acc_S)
        acc_P_mn = acc_mn_view_bwd(acc_P)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            pr = cute.math.exp2(
                acc_S_mn[r, None].load() - rowv[r], fastmath=True)
            acc_S_mn[r, None].store(pr)
            acc_P_mn[r, None].store(
                pr * (acc_P_mn[r, None].load() - rowd[r]))

        rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
        rP.store(acc_S.load().to(cutlass.BFloat16))
        rA = cute.make_fragment_like(acc_P, cutlass.BFloat16)
        rA.store(acc_P.load().to(cutlass.BFloat16))
        tOrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA_bwd(rP.layout))
        tOrA = cute.make_tensor(rA.iterator, convert_layout_acc_frgA_bwd(rA.layout))
        rBo = thr_h.make_fragment_B(thr_h.partition_B(sTokHT[None, None, 0]))
        gemm_rs2_bwd(mma_h, acc_Z, acc_R, tOrP, tOrA, rBo,
                 cpB_t.partition_S(sTokHT)[None, None, None, stage], cpB_t)

        cute.copy(st_atom, stC_h.retile(rA), st_top[sub])
        cute.copy(st_atom, stC_h.retile(rP), st_bot[sub])


@cute.jit
def tile_step_bwd(
    mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
    st_sScTop0, st_sScBot0, st_sScTop1, st_sScBot1, acc_R, acc_Z,
    tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
    gmem_copy_tok, tKgTok, tKsTok, t0KcTok, tok_row_base, n_cpy_m,
    n, n_tiles, Lc, rem,
    tile_n: cutlass.Constexpr, NSTAGE: cutlass.Constexpr,
    mask: cutlass.Constexpr,
):
    """cp.async rendezvous + prefetch + one score tile (trailing-group form)."""
    stage = n % NSTAGE
    cute.arch.cp_async_wait_group(NSTAGE - 2)
    cute.arch.barrier()
    nl = n + NSTAGE - 1
    if nl < n_tiles:
        rows_valid = Lc - nl * tile_n - tok_row_base
        for m in cutlass.range_constexpr(n_cpy_m):
            if t0KcTok[0, m, 0][0] < rows_valid:
                cute.copy(gmem_copy_tok, tKgTok[None, m, None, nl],
                          tKsTok[None, m, None, nl % NSTAGE])
    cute.arch.cp_async_commit_group()
    score_tile_bwd(
        mma_h, thr_h, cpB_n, cpB_t, st_atom, stC_h, sTok,
        st_sScTop0, st_sScBot0, st_sScTop1, st_sScBot1, acc_R, acc_Z,
        tSrQ, tSrdZ, rowv, rowd, t0cS, col_base,
        stage, rem, tile_n, NSTAGE, mask,
    )


@cute.jit
def dtok_flush_bwd(
    mma_w, thr_w, wA_t, wB_t, st_atom, stC_w, direct_atom, directC_w,
    gmem_copy_store, gmem_thr_store, t0OcTok, st_row_base, n_st_m,
    st_sDT, tOsDT, sScoreT, sQZT, tDrB, mdTok, tok0, drows,
    tile_n: cutlass.Constexpr, sup_n: cutlass.Constexpr,
    alias_dt: cutlass.Constexpr, direct_dt: cutlass.Constexpr,
    resident_b: cutlass.Constexpr,
    mask: cutlass.Constexpr,
):
    """d_tokens for a whole SUP-tile group of tokens.

    Rows of sScore past `drows` hold stale (but finite) data from the previous
    group; they only feed accumulator rows that the predicated store drops, so
    no clearing is needed.
    """
    cute.arch.barrier()
    acc_dT = cute.make_rmem_tensor(thr_w.partition_shape_C((sup_n, DHEAD)), Float32)
    acc_dT.fill(0.0)
    if const_expr(resident_b):
        gemm_sr_k_bwd(mma_w, acc_dT, sScoreT, tDrB, wA_t, thr_w,
                  sup_n, 32, 4)
    else:
        gemm_ss_kchunk_bwd(mma_w, acc_dT, sScoreT, sQZT, wA_t, wB_t, thr_w,
                       sup_n, DHEAD, 32, 4)
    rDT = cute.make_fragment_like(acc_dT, cutlass.BFloat16)
    rDT.store(acc_dT.load().to(cutlass.BFloat16))

    # domain_offset lowers the element offset tok0*DHEAD in 32 bits, which wraps
    # above 2**31/DHEAD tokens.  Rebase on a 64-bit byte address instead: the
    # widening multiply is once per SUP-tile group, not in the inner loop.
    mdTokP = cute.make_tensor(
        cute.make_ptr(
            cutlass.BFloat16,
            mdTok.iterator.toint() + cutlass.Int64(tok0) * cutlass.Int64(2 * DHEAD),
            cute.AddressSpace.gmem, assumed_align=16),
        mdTok.layout)
    gdT = cute.local_tile(mdTokP, (sup_n, DHEAD), (0, 0))
    if const_expr(mask or alias_dt or not direct_dt):
        if const_expr(alias_dt):
            cute.arch.barrier()
        cute.copy(st_atom, stC_w.retile(rDT), st_sDT)
        cute.arch.barrier()
        tOgDT = gmem_thr_store.partition_D(gdT)
        rOs = cute.make_fragment_like(tOsDT, cutlass.BFloat16)
        cute.autovec_copy(tOsDT, rOs)
        if const_expr(mask):
            rows_valid = drows - st_row_base
            for m in cutlass.range_constexpr(n_st_m):
                if t0OcTok[0, m, 0][0] < rows_valid:
                    cute.copy(gmem_copy_store, rOs[None, m, None],
                              tOgDT[None, m, None])
        else:
            cute.copy(gmem_copy_store, rOs, tOgDT)
    else:
        cute.copy(direct_atom, directC_w.retile(rDT), directC_w.partition_D(gdT))


# --------------------------------------------------------------------------- #
# host-side trace
# --------------------------------------------------------------------------- #
@cute.jit
def bwd_launch_bwd(pdO: cutlass.Int64, pQ: cutlass.Int64, pTok: cutlass.Int64,
               pWk: cutlass.Int64, pWv: cutlass.Int64, pBv: cutlass.Int64,
               pOut: cutlass.Int64, pLSE: cutlass.Int64, pCu: cutlass.Int64,
               pdQ: cutlass.Int64, pdTok: cutlass.Int64, pdWk: cutlass.Int64,
               pdWv: cutlass.Int64, pdBv: cutlass.Int64, pScr: cutlass.Int64,
               pMeta: cutlass.Int64, pWit: cutlass.Int64, pBuck: cutlass.Int64,
               pScrR: cutlass.Int64,
               TT: Int32, NSCR: Int32, NMETA: Int32, NWIT: Int32, NBUCKT: Int32,
               NSCRR: Int32,
               scale: Float32, sl: Float32, TP: Int32, CH: Int32, CH2: Int32,
               tile_n: cutlass.Constexpr, NSTAGE: cutlass.Constexpr,
               sched: cutlass.Constexpr, grid_x: cutlass.Constexpr,
               prep_grid: cutlass.Constexpr, maxchunk: cutlass.Constexpr,
               cta_sm: cutlass.Constexpr, SUP: cutlass.Constexpr,
               direct_dt: cutlass.Constexpr, resident_b: cutlass.Constexpr,
               use_v_bias: cutlass.Constexpr, buck: cutlass.Constexpr,
               stream):
    dt = cutlass.BFloat16
    # Operands arrive as raw device addresses rather than through from_dlpack: 19 dlpack
    # capsules per call cost ~84 us of host time, and the wrapper must detach every input
    # first because from_dlpack rejects tensors that require grad. The extents stay RUNTIME
    # values (Int32), exactly as from_dlpack bound them, so one compiled kernel still serves
    # every observation count and the compile cache is unchanged.
    mdO = _gt_long(pdO, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mQ = _gt_long(pQ, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mTok = _gt_long(pTok, dt, (TT, DHEAD), (DHEAD, 1), 16)
    mWkf = _gt_long(pWk, dt, (NKV * DHEAD * TOKEN_DIM,), (1,), 16)
    mWvf = _gt_long(pWv, dt, (NKV * DHEAD * TOKEN_DIM,), (1,), 16)
    mBv = _gt_long(pBv, dt, (NKV * DHEAD,), (1,), 16)
    mOut = _gt_long(pOut, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mLSE = _gt_long(pLSE, Float32, (TP, NHEAD), (NHEAD, 1), 16)
    mCu = _gt_long(pCu, Int32, (TP + 1,), (1,), 4)
    mdQ = _gt_long(pdQ, dt, (TP * NHEAD, DHEAD), (DHEAD, 1), 16)
    mdTok = _gt_long(pdTok, dt, (TT, DHEAD), (DHEAD, 1), 16)
    mdWk = _gt_long(pdWk, dt, (NKV * DHEAD * TOKEN_DIM,), (1,), 16)
    mdWv = _gt_long(pdWv, dt, (NKV * DHEAD * TOKEN_DIM,), (1,), 16)
    mdBv = _gt_long(pdBv, dt, (NKV * DHEAD,), (1,), 16)
    mScr = _gt_long(pScr, Float32, (NSCR,), (1,), 16)
    mMeta = _gt_long(pMeta, Int32, (NMETA,), (1,), 4)
    mWit = _gt_long(pWit, Int32, (NWIT,), (1,), 4)
    mBuck = _gt_long(pBuck, Int32, (NBUCKT,), (1,), 4)
    mScrR = _gt_long(pScrR, Float32, (NSCRR,), (1,), 16)
    sup_n = tile_n * SUP
    # Folding sDT into the scratch arena buys the shared memory the wider score
    # buffer needs, at the cost of one extra rendezvous per flush -- worth it
    # only when that flush is amortised over SUP > 1 token tiles.
    alias_dt = SUP > 1
    atom32 = smem_layout_atom_bwd(dt, DHEAD)
    # Each 16-column score subtile starts on its own swizzle-atom boundary.
    atomT = smem_layout_atom_bwd(dt, SCORE_N)
    sQ_layout = cute.tile_to_shape(atom32, (NHEAD, DHEAD), (0, 1))
    sQZ_layout = cute.tile_to_shape(atom32, (2 * NHEAD, DHEAD), (0, 1))
    sW_layout = cute.tile_to_shape(atom32, (DHEAD, DHEAD, NKV), (0, 1, 2))
    sTok_layout = cute.tile_to_shape(atom32, (tile_n, DHEAD, NSTAGE), (0, 1, 2))
    sScore_layout = cute.tile_to_shape(atomT, (2 * NHEAD, sup_n), (0, 1))
    sDT_layout = cute.tile_to_shape(atom32, (sup_n, DHEAD), (0, 1))

    # sScr is the shared scratch arena: it aliases sOut (prologue), sScore and
    # sDT (token loop) and sDQ / sRZ (epilogue).  Folding sDT in here keeps the
    # CTA inside the 4-blocks-per-SM shared-memory budget now that the score
    # buffer spans a whole SUP-tile group.
    scr_elems = max(cute.cosize(sScore_layout), cute.cosize(sQZ_layout),
                    cute.cosize(sDT_layout) if alias_dt else 0)
    sdt_elems = 1 if alias_dt else cute.cosize(sDT_layout)

    @cute.struct
    class SharedStorage:
        sIdx: cute.struct.MemRange[Int32, 4]
        sD: cute.struct.MemRange[Float32, NHEAD]
        sBv: cute.struct.MemRange[Float32, NHEAD]
        sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 1024]
        sdO: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 1024]
        sQZ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQZ_layout)], 1024]
        sWk: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 1024]
        sWv: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sW_layout)], 1024]
        sTok: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sTok_layout)], 1024]
        sScr: cute.struct.Align[cute.struct.MemRange[dt, scr_elems], 1024]
        sDT: cute.struct.Align[cute.struct.MemRange[dt, sdt_elems], 1024]

    copy_bits = 128
    elems = copy_bits // dt.width
    dim1 = DHEAD // elems
    t_layout = cute.make_ordered_layout((BWD_NTHREAD // dim1, dim1), order=(1, 0))
    v_layout = cute.make_layout((1, elems))
    atom_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dt,
        num_bits_per_copy=copy_bits)
    atom_univ = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dt,
                                    num_bits_per_copy=copy_bits)
    gmem_copy_load = cute.make_tiled_copy_tv(atom_async, t_layout, v_layout)
    gmem_copy_tok = cute.make_tiled_copy_tv(atom_async, t_layout, v_layout)
    gmem_copy_store = cute.make_tiled_copy_tv(atom_univ, t_layout, v_layout)

    mma_h = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)), (BWD_NWARP, 1, 1),
        permutation_mnk=(BWD_NWARP * 16, 16, 16))
    mma_w = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)), (2, 2, 1),
        permutation_mnk=(32, 32, 16))
    # One 32x32x32 parameter-gradient GEMM per warp: with all four warps on one
    # GEMM each operand tile is fetched twice, with one warp per GEMM it is
    # fetched once, halving epilogue ldmatrix traffic for the same MMA count.
    mma_w1 = cute.make_tiled_mma(
        warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)), (1, 1, 1),
        permutation_mnk=(32, 32, 16))

    if const_expr(sched):
        prep_kernel_bwd(mCu, mWit, mBuck, mMeta, TP, CH, CH2, tile_n,
                    maxchunk, buck).launch(
            grid=(prep_grid, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)
        if const_expr(buck):
            bucket_prefix_kernel_bwd(mMeta).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
            bucket_compact_kernel_bwd(mCu, mWit, mBuck, mMeta, TP, tile_n).launch(
                grid=(prep_grid, NBUCK, 1), block=(PREP_THREADS, 1, 1),
                stream=stream)

    bwd_kernel_bwd(
        mdO, mQ, mTok, mWkf, mWvf, mBv, mOut, mLSE, mCu, mdQ, mdTok,
        mScr, mMeta, mWit, mScrR, scale, sl, TP, CH2,
        tile_n, NSTAGE, sched,
        sQ_layout, sTok_layout, sW_layout, sQZ_layout, sScore_layout, sDT_layout,
        gmem_copy_load, gmem_copy_store, gmem_copy_tok, mma_h, mma_w, mma_w1,
        SUP, alias_dt, direct_dt, resident_b, use_v_bias, SharedStorage,
    ).launch(grid=(grid_x, 1, 1), block=(BWD_NTHREAD, 1, 1),
             smem=SharedStorage.size_in_bytes(), stream=stream,
             max_number_threads=[BWD_NTHREAD, 1, 1], min_blocks_per_mp=cta_sm)

    fin_grid = (4160 + 31) // 32

    @cute.struct
    class FinStorage:
        buf: cute.struct.Align[cute.struct.MemRange[Float32, FIN_WARPS * 32], 16]

    fin_kernel_bwd(mScr, mdWk, mdWv, mdBv, FinStorage, Int32(grid_x)).launch(
        grid=(fin_grid, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream,
        smem=FinStorage.size_in_bytes())


_CACHE_bwd = {}


def run_bwd(dOut, Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, scale,
        q_per_kv, token_dim, n_kv_heads, use_v_bias, force_fp32,
        dQ, d_tokens, dW_k, dW_v, dB_v):
    TP = Q.shape[0]
    TT = tokens.shape[0]
    dev = Q.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    tile_n = 32
    # The 2-tile score buffer halves how often the d_tokens B operand is fetched, which cuts
    # register spill traffic by 58% (4.76M -> 1.99M spill requests). On B300 it costs 4 KB of
    # shared memory and still leaves room for the fourth cp.async stage at 3 CTAs/SM, so both
    # are taken unconditionally: 1.055x weighted backward, faster on all eight shapes.
    #
    # Tuned on B300 (148 SMs, 228 KiB smem/SM). The B200 build these kernels came from gated
    # them instead, on the grounds that short pixels cannot give up the fourth cp.async stage
    # because their token pipeline never leaves its startup ramp:
    #     sup = 2 if (TP > 12288 and TT / max(TP, 1) >= 300.0) else 1
    #     nstage = 3 if sup == 2 else 4
    # Re-measure before assuming the unconditional form is right on another part.
    sup = 2
    nstage = 4
    resident_b = True
    cta_sm = 3
    sched = True
    grid_x = min(max(TP, 1), cta_sm * nsm)
    maxchunk = MAXCHUNK
    prep_grid = max(1, (TP + PREP_THREADS - 1) // PREP_THREADS)
    CH = max(TT // (2 * grid_x), 16 * tile_n)
    CH2 = max(TT // (8 * grid_x), 2 * tile_n)
    direct_dt = TP < 90000
    buck = TP < 60 * grid_x
    sched = 2 if buck else True
    MAXITEM = TP + MAXSLOT * MAXCHUNK

    q2 = Q.view(TP * NHEAD, DHEAD)
    do2 = dOut.view(TP * NHEAD, DHEAD)
    ou2 = Out.view(TP * NHEAD, DHEAD)
    dq2 = dQ.view(TP * NHEAD, DHEAD)

    meta_n = (MBUCK + 2 * NBUCK) if buck else (8 + 2 * MAXSLOT)
    meta = torch.zeros(meta_n, dtype=torch.int32, device=dev)
    scr = torch.empty(grid_x * SCR_PER_CTA, dtype=torch.float32, device=dev)
    wit = torch.empty(MAXITEM * 4, dtype=torch.int32, device=dev)
    buckt = (torch.empty(NBUCK * TP, dtype=torch.int32, device=dev)
             if buck else wit)
    scrR = torch.empty(MAXSLOT * MAXCHUNK * RSTATE, dtype=torch.float32, device=dev)

    def _p(t):
        return cutlass.Int64(t.data_ptr())

    # Bind the flattened views to locals: reshape on a non-contiguous tensor returns a COPY,
    # and the kernel must not be handed the address of a temporary that is already freed.
    Wkf, Wvf = W_k.reshape(-1), W_v.reshape(-1)
    dWkf, dWvf = dW_k.reshape(-1), dW_v.reshape(-1)

    args = (
        _p(do2), _p(q2), _p(tokens), _p(Wkf), _p(Wvf), _p(B_v), _p(ou2), _p(LSE), _p(cu_seqlens_k),
        _p(dq2), _p(d_tokens), _p(dWkf), _p(dWvf), _p(dB_v),
        _p(scr), _p(meta), _p(wit), _p(buckt), _p(scrR),
        Int32(TT), Int32(scr.numel()), Int32(meta.numel()), Int32(wit.numel()),
        Int32(buckt.numel()), Int32(scrR.numel()),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    use_v_bias = bool(use_v_bias)
    # The observation count is deliberately absent: TT is passed as a RUNTIME extent, so one
    # compiled kernel serves every step. Verified by reusing a kernel built at 5.6M tokens for
    # 8.7M (+54%) with correct gradients.
    key = (grid_x, TP, tile_n, nstage, sched, prep_grid, maxchunk,
           cta_sm, sup, direct_dt, resident_b, use_v_bias, buck)
    fn = _CACHE_bwd.get(key)
    if fn is None:
        fn = cute.compile(
            bwd_launch_bwd, *args, Float32(0.0), Float32(0.0), Int32(TP), Int32(CH),
            Int32(CH2), tile_n, nstage, sched, grid_x, prep_grid, maxchunk,
            cta_sm, sup, direct_dt, resident_b, use_v_bias, buck, stream)
        _CACHE_bwd[key] = fn

    fn(*args, Float32(float(scale)), Float32(float(scale) * LOG2_E),
       Int32(TP), Int32(CH), Int32(CH2), stream)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

# Tokens per pixel above which the long-sequence variant wins.
LONG_MEAN_LEN = 300.0


def run_forward(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale, q_per_kv, n_kv_heads, Out, LSE):
    args = (Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale, q_per_kv, TOKEN_DIM,
            n_kv_heads, True, False, Out, LSE)
    if tokens.shape[0] / max(Q.shape[0], 1) >= LONG_MEAN_LEN:
        run_long(*args)
    else:
        run_uniform(*args)


def run_backward(dOut, Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, scale,
                 q_per_kv, n_kv_heads, dQ, d_tokens, dW_k, dW_v, dB_v):
    run_bwd(dOut, Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, scale, q_per_kv,
            TOKEN_DIM, n_kv_heads, True, False, dQ, d_tokens, dW_k, dW_v, dB_v)


@torch.library.custom_op("healda::cutedsl_pixel_attn_fwd", mutates_args=())
def _op_fwd(
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    n_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    TP, n_q_heads = Q.shape[0], Q.shape[1]
    Out = torch.empty(TP, n_q_heads, D_HEAD, device=Q.device, dtype=Q.dtype)
    LSE = torch.empty(TP, n_q_heads, device=Q.device, dtype=torch.float32)
    run_forward(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale,
                n_q_heads // n_kv_heads, n_kv_heads, Out, LSE)
    return Out, LSE


@_op_fwd.register_fake
def _fake_fwd(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale, n_kv_heads):
    return (
        torch.empty_like(Q),
        Q.new_empty((Q.shape[0], Q.shape[1]), dtype=torch.float32),
    )


@torch.library.custom_op("healda::cutedsl_pixel_attn_bwd", mutates_args=())
def _op_bwd(
    dOut: torch.Tensor,
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_v: torch.Tensor,
    Out: torch.Tensor,
    LSE: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    n_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The kernel takes raw device addresses, so anything still attached to the autograd graph
    # is detached first.
    Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k = (
        t.detach() for t in (Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k)
    )
    # The backward writes every element of all five gradients -- it stores, never accumulates
    # (the only atomics in the kernel are on the scheduler's counter block, not on these
    # buffers), and it explicitly zeroes dQ for pixels with no observations. Allocating empty
    # therefore skips five memsets, one of them over the full token tensor.
    dQ = torch.empty_like(Q)
    d_tokens = torch.empty_like(tokens)
    dW_k = torch.empty_like(W_k)
    dW_v = torch.empty_like(W_v)
    dB_v = torch.empty_like(B_v)
    run_backward(dOut.detach().contiguous(), Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k,
                 scale, Q.shape[1] // n_kv_heads, n_kv_heads,
                 dQ, d_tokens, dW_k, dW_v, dB_v)
    return dQ, d_tokens, dW_k, dW_v, dB_v


@_op_bwd.register_fake
def _fake_bwd(dOut, Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, scale, n_kv_heads):
    return (torch.empty_like(Q), torch.empty_like(tokens), torch.empty_like(W_k),
            torch.empty_like(W_v), torch.empty_like(B_v))


def _setup_context(ctx, inputs, output):
    Q, tokens, W_k, W_v, B_v, cu_seqlens_k, scale, n_kv_heads = inputs
    ctx.save_for_backward(Q, tokens, W_k, W_v, B_v, cu_seqlens_k, output[0], output[1])
    ctx.scale, ctx.n_kv_heads = scale, n_kv_heads


def _backward_fn(ctx, dOut, dLSE):
    Q, tokens, W_k, W_v, B_v, cu_seqlens_k, Out, LSE = ctx.saved_tensors
    dQ, d_tokens, dW_k, dW_v, dB_v = _op_bwd(
        dOut, Q, tokens, W_k, W_v, B_v, Out, LSE, cu_seqlens_k, ctx.scale, ctx.n_kv_heads)
    # One slot per fwd input: five real grads, then None for cu_seqlens_k, scale, n_kv_heads.
    return dQ, d_tokens, dW_k, dW_v, dB_v, None, None, None


_op_fwd.register_autograd(_backward_fn, setup_context=_setup_context)


def _check_inputs(Q, tokens, cu_seqlens_k):
    """The kernels take raw pointers and re-declare dtype, stride and bounds themselves.

    Nothing downstream can detect a mismatch: an int64 ``cu_seqlens_k`` is read as int32,
    giving arbitrary token offsets, and a non-contiguous input is read as if it were packed.
    Both surface as an illegal access or silent garbage rather than an error, so they are
    rejected here. Everything checked is tensor metadata: no device work, no synchronisation.

    Not checked: that cu_seqlens_k is non-decreasing and ends at the token count. Those need
    device reductions and a readback, which measured 19-26% of the kernel -- far too expensive
    for the calling convention to police on every step.
    """
    for name, t in (("Q", Q), ("tokens", tokens)):
        if t.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must be bfloat16, got {t.dtype}; the kernel reads it as bfloat16 and "
                "autocast does not retype it"
            )
    if cu_seqlens_k.dtype != torch.int32:
        raise ValueError(
            f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}; the kernel reads it as int32"
        )
    for name, t in (("Q", Q), ("tokens", tokens), ("cu_seqlens_k", cu_seqlens_k)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens_k.numel() != Q.shape[0] + 1:
        raise ValueError(
            f"cu_seqlens_k must have one entry per pixel plus one, got "
            f"{cu_seqlens_k.numel()} for {Q.shape[0]} pixels"
        )


def pixel_attention(Q, tokens, W_k, W_v, cu_seqlens_k, n_kv_heads, scale, B_v=None):
    """Entry point matching ``triton_pixel_attention.pixel_attention``.

    The kernels read the projection weights as bf16 off a raw pointer, so they are cast here
    rather than relying on autocast, which does not retype parameters.

    The forward applies the V bias unconditionally, so a model configured without one is
    served by adding an explicit zero bias rather than by a flag the kernel ignores.
    """
    reasons = unsupported_reasons(
        n_q_heads=Q.shape[1], n_kv_heads=n_kv_heads, d_head=Q.shape[2],
        token_dim=tokens.shape[1],
    )
    if reasons:
        raise ValueError(
            "cutedsl pixel attention requires " + ", ".join(reasons)
            + "; the kernels bake this layout into their tile shapes"
        )
    if B_v is None:
        B_v = W_v.new_zeros(n_kv_heads * D_HEAD)
    _check_inputs(Q, tokens, cu_seqlens_k)
    return _op_fwd(Q, tokens, W_k.to(torch.bfloat16), W_v.to(torch.bfloat16),
                   B_v.to(torch.bfloat16), cu_seqlens_k, scale, n_kv_heads)[0]

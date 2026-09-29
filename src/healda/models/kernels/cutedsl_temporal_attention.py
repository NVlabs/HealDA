# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL temporal-attention kernels.

Fused implementation of the temporal-attention body -- RoPE, the causal t x t contraction, the
weighted sum over v, and the head merge.

1. ``run_forward`` / ``run_backward``: persistent CuTe DSL kernels using TMA loads into swizzled
   shared memory, m16n8k16 BF16 MMA, and an stmatrix epilogue. Two independent 8-frame problems
   are folded into each 16-row MMA tile (pixels in the forward, heads in the backward) and the
   off-diagonal blocks discarded, because one 8-frame problem cannot fill the tile.
2. ``torch.library.custom_op`` wrappers, so the launches participate in autograd and fake-tensor
   tracing under ``torch.compile`` while staying opaque to inductor.
3. ``fused_cutedsl``, the entry point called by ``TemporalAttention``.

Generated for one shape family only: 12 heads, 128 head dim, 8 frames, batch 1, causal, bf16,
and an even pixel count (the forward tiles pixels in pairs). The statically-known constraints are
in ``unsupported_reasons``; frames, batch, pixel count and dtype are only known per call and are
checked at the dispatch site in ``TemporalAttention.forward``.

Tuning constants (forward ``nblk``, backward ``nbx``) are tuned per architecture, not derived;
see ``_pick_nbx``.
"""

import functools
import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync, warp

# The backward kernel shadows `warp` with a local warp index, hence the `cwarp` alias.
from cutlass.cute.nvgpu import warp as cwarp
from cutlass.cute.runtime import from_dlpack

from healda.models.kernels.cutedsl_arch import unsupported_arch_reasons

HN = 12
CD = 128
QKVW = 3 * HN * CD  # 4608
OUTW = HN * CD  # 1536
FWD_WARPS = 4
FWD_THREADS = FWD_WARPS * 32
FWD_SCALE = 1.0 / math.sqrt(CD)

_MKT = getattr(cute, "make_rmem_tensor", None) or getattr(cute, "make_fragment")
_MKTL = getattr(cute, "make_rmem_tensor_like", None) or getattr(
    cute, "make_fragment_like"
)

TPITCH = 68  # 4-float-strided rows -> conflict-free trig fetch
TILE = 2048  # elements in one 16x128 bf16 SMEM tile


def _sw():
    # 128 B swizzle: XOR the row index into the 16 B unit index inside each
    # 128 B row.  With a 64-element (128 B) inner block and a 64-element row
    # stride, the 8 rows of any ldmatrix 8x8 block land in 8 distinct 16 B
    # units -> conflict free, and it is exactly the layout TMA can fill.
    return cute.make_swizzle(3, 4, 3)


def _aff_tma():
    # (t, x, (c_in, c_blk)) ; row r = t + 8*x sits at stride 64
    return cute.make_layout((8, 2, (64, 2)), stride=(64, 512, (1, 1024)))


def _aff_cmp4():
    # (warp, (t,x), (c_in,c_blk)) -> a (16,128) row-major-equivalent tile per warp
    return cute.make_layout(
        (FWD_WARPS, (8, 2), (64, 2)), stride=(TILE, (64, 512), (1, 1024))
    )


def _aff_t():
    # transpose view of one tile: (128 channels, 16 rows)
    return cute.make_layout(((64, 2), (8, 2)), stride=((1, 1024), (64, 512)))


def _acc_to_frgA(acc_layout):
    lay = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((lay.shape[0], lay.shape[2][0]), lay.shape[1], lay.shape[2][1]),
        stride=((lay.stride[0], lay.stride[2][0]), lay.stride[1], lay.stride[2][1]),
    )


@cute.kernel
def _healda_kernel(
    tma_atom: cute.CopyAtom,
    tma_g: cute.Tensor,
    o5: cute.Tensor,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    ostride: cutlass.Constexpr,
    npp: cutlass.Constexpr,
    nblk: cutlass.Constexpr,
    npairs: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    warp_id = tidx // 32
    lane = tidx % 32
    gid = lane // 4
    tig = lane % 4

    sw = _sw()
    smem = cutlass_utils.SmemAllocator()
    sQ4 = smem.allocate_tensor(
        cutlass.BFloat16, _aff_cmp4(), byte_alignment=1024, swizzle=sw
    )
    sK4 = smem.allocate_tensor(
        cutlass.BFloat16, _aff_cmp4(), byte_alignment=1024, swizzle=sw
    )
    sV4 = smem.allocate_tensor(
        cutlass.BFloat16, _aff_cmp4(), byte_alignment=1024, swizzle=sw
    )
    mbar = smem.allocate_array(
        cutlass.Int64, num_elems=2 * FWD_WARPS, byte_alignment=16
    )

    sA = sQ4[(warp_id, None, None)]
    sB = sK4[(warp_id, None, None)]
    sC = sV4[(warp_id, None, None)]
    sVt = cute.make_tensor(sC.iterator, _aff_t())

    sQ3 = cute.make_tensor(sA.iterator, _aff_tma())
    sK3 = cute.make_tensor(sB.iterator, _aff_tma())
    sV3 = cute.make_tensor(sC.iterator, _aff_tma())

    bqk = mbar + 2 * warp_id
    bv = mbar + 2 * warp_id + 1
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(bqk, 1)
        cute.arch.mbarrier_init(bv, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    head = bidx * FWD_WARPS + warp_id
    pair0 = bidy

    gT = cute.group_modes(cute.local_tile(tma_g, (8, 2, CD), (None, None, None)), 0, 3)
    tAsQ, tAgQ = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), cute.group_modes(sQ3, 0, 3), gT
    )
    tAsK, tAgK = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), cute.group_modes(sK3, 0, 3), gT
    )
    tAsV, tAgV = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), cute.group_modes(sV3, 0, 3), gT
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bqk, 8192)
    cute.copy(tma_atom, tAgQ[(None, 0, pair0, head)], tAsQ, tma_bar_ptr=bqk)
    cute.copy(tma_atom, tAgK[(None, 0, pair0, HN + head)], tAsK, tma_bar_ptr=bqk)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bv, 4096)
    cute.copy(tma_atom, tAgV[(None, 0, pair0, 2 * HN + head)], tAsV, tma_bar_ptr=bv)

    # RoPE tables: stage cos/sin through SMEM once per CTA.
    sTrig = smem.allocate_tensor(
        Float32,
        cute.make_layout((8, TPITCH, 2), stride=(TPITCH, 1, 8 * TPITCH)),
        byte_alignment=16,
    )
    for kk in cutlass.range_constexpr(4):
        idx = tidx + FWD_THREADS * kk
        rr = idx // 64
        cc = idx % 64
        sTrig[rr, cc, 0] = mCos[rr, cc]
        sTrig[rr, cc, 1] = mSin[rr, cc]
    cute.arch.barrier()

    rCos = _MKT((8, 2), Float32)
    rSin = _MKT((8, 2), Float32)
    for kb in cutlass.range_constexpr(8):
        for a2 in cutlass.range_constexpr(2):
            m = 8 * kb + 4 * a2 + tig
            rCos[kb, a2] = sTrig[gid, m, 0]
            rSin[kb, a2] = sTrig[gid, m, 1]

    op = warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16))
    mma_qk = cute.make_tiled_mma(op)
    mma_pv = cute.make_tiled_mma(op)
    thr_qk = mma_qk.get_slice(lane)
    thr_pv = mma_pv.get_slice(lane)

    ld_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16
    )
    ldt_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16
    )
    cp_Q = cute.make_tiled_copy_A(ld_atom, mma_qk).get_slice(lane)
    cp_K = cute.make_tiled_copy_B(ld_atom, mma_qk).get_slice(lane)
    cp_V = cute.make_tiled_copy_B(ldt_atom, mma_pv).get_slice(lane)

    tSrQ = mma_qk.make_fragment_A(thr_qk.partition_A(sA))
    tSrK = mma_qk.make_fragment_B(thr_qk.partition_B(sB))
    tOrV = mma_pv.make_fragment_B(thr_pv.partition_B(sVt))
    acc_S = _MKT(mma_qk.partition_shape_C((16, 16)), Float32)
    acc_O = _MKT(mma_pv.partition_shape_C((16, CD)), Float32)

    st_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2), cutlass.BFloat16
    )
    tc_r2s = cute.make_tiled_copy_S(
        st_atom, cute.make_tiled_copy_C_atom(st_atom, mma_pv)
    )
    thr_r2s = tc_r2s.get_slice(lane)
    tRS_sO = thr_r2s.partition_D(sC)

    s2g_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    thr_layout = cute.make_ordered_layout((2, 16), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, 8), order=(1, 0))
    tiled_s2g = cute.make_tiled_copy_tv(s2g_atom, thr_layout, val_layout)
    thr_s2g = tiled_s2g.get_slice(lane)
    tS = thr_s2g.partition_S(sC)

    ocopy = cute.make_layout(((8, 2), CD), stride=((ostride, OUTW), 1))
    p0 = gid >= 2 * tig
    p1 = gid >= 2 * tig + 1

    if cutlass.const_expr(npp > 0):
        nit = npp
    else:
        nit = (npairs - pair0 + nblk - 1) // nblk

    for it in cutlass.range(nit):
        pair = pair0 + it * nblk
        ph = it % 2
        acc_S.fill(0.0)
        acc_O.fill(0.0)

        cute.arch.mbarrier_wait(bqk, ph)
        cute.copy(cp_Q, cp_Q.partition_S(sA), cp_Q.retile(tSrQ))
        cute.copy(cp_K, cp_K.partition_S(sB), cp_K.retile(tSrK))
        cute.arch.sync_warp()

        if it + 1 < nit:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bqk, 8192)
            cute.copy(
                tma_atom, tAgQ[(None, 0, pair + nblk, head)], tAsQ, tma_bar_ptr=bqk
            )
            cute.copy(
                tma_atom,
                tAgK[(None, 0, pair + nblk, HN + head)],
                tAsK,
                tma_bar_ptr=bqk,
            )

        for kb in cutlass.range_constexpr(8):
            for a2 in cutlass.range_constexpr(2):
                cv = rCos[kb, a2]
                sv = rSin[kb, a2]
                for a1 in cutlass.range_constexpr(2):
                    e = tSrQ[(0, a1, a2), 0, kb].to(Float32)
                    o = tSrQ[(1, a1, a2), 0, kb].to(Float32)
                    tSrQ[(0, a1, a2), 0, kb] = (e * cv - o * sv).to(cutlass.BFloat16)
                    tSrQ[(1, a1, a2), 0, kb] = (e * sv + o * cv).to(cutlass.BFloat16)
                for n in cutlass.range_constexpr(2):
                    e = tSrK[(0, a2), n, kb].to(Float32)
                    o = tSrK[(1, a2), n, kb].to(Float32)
                    tSrK[(0, a2), n, kb] = (e * cv - o * sv).to(cutlass.BFloat16)
                    tSrK[(1, a2), n, kb] = (e * sv + o * cv).to(cutlass.BFloat16)

        for kb in cutlass.range_constexpr(8):
            cute.gemm(mma_qk, acc_S, tSrQ[None, None, kb], tSrK[None, None, kb], acc_S)

        for n in cutlass.range_constexpr(2):
            for v1 in cutlass.range_constexpr(2):
                for v0 in cutlass.range_constexpr(2):
                    if v1 != n:
                        acc_S[(v0, v1), 0, n] = Float32(0.0)
                    elif v0 == 0:
                        acc_S[(v0, v1), 0, n] = (
                            acc_S[(v0, v1), 0, n] * FWD_SCALE if p0 else Float32(0.0)
                        )
                    else:
                        acc_S[(v0, v1), 0, n] = (
                            acc_S[(v0, v1), 0, n] * FWD_SCALE if p1 else Float32(0.0)
                        )

        rP = _MKTL(acc_S, cutlass.BFloat16)
        rP.store(acc_S.load().to(cutlass.BFloat16))
        tOrP = cute.make_tensor(rP.iterator, _acc_to_frgA(rP.layout))

        cute.arch.mbarrier_wait(bv, ph)
        cute.copy(cp_V, cp_V.partition_S(sVt), cp_V.retile(tOrV))
        cute.gemm(mma_pv, acc_O, tOrP[None, None, 0], tOrV[None, None, 0], acc_O)

        rO = _MKTL(acc_O, cutlass.BFloat16)
        rO.store(acc_O.load().to(cutlass.BFloat16))
        cute.arch.sync_warp()
        cute.copy(tc_r2s, tc_r2s.retile(rO), tRS_sO)
        cute.arch.sync_warp()

        frg = _MKTL(tS)
        cute.copy(tiled_s2g, tS, frg)
        cute.arch.sync_warp()

        gO = cute.make_tensor(o5[(None, pair, None, head, None)].iterator, ocopy)
        cute.copy(tiled_s2g, frg, thr_s2g.partition_D(gO))

        if it + 1 < nit:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bv, 4096)
            cute.copy(
                tma_atom,
                tAgV[(None, 0, pair + nblk, 2 * HN + head)],
                tAsV,
                tma_bar_ptr=bv,
            )


@cute.jit
def _healda_host(
    mQKV: cute.Tensor,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    mOut: cute.Tensor,
    X: cutlass.Constexpr,
    nblk: cutlass.Constexpr,
):
    g3 = cute.make_tensor(
        mQKV.iterator,
        cute.make_layout((8, X, QKVW), stride=(X * QKVW, QKVW, 1)),
    )
    o5 = cute.make_tensor(
        mOut.iterator,
        cute.make_layout(
            (8, X // 2, 2, HN, CD), stride=(X * OUTW, 2 * OUTW, OUTW, CD, 1)
        ),
    )
    smem_tma = cute.make_composed_layout(_sw(), 0, _aff_tma())
    tma_atom, tma_g = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), g3, smem_tma, (8, 2, CD)
    )
    npp = (X // 2) // nblk if (X // 2) % nblk == 0 else 0
    _healda_kernel(tma_atom, tma_g, o5, mCos, mSin, X * OUTW, npp, nblk, X // 2).launch(
        grid=(HN // FWD_WARPS, nblk, 1), block=(FWD_THREADS, 1, 1)
    )


_FWD_CACHE = {}


@torch.no_grad()
def run_forward(qkv, cos, sin, out):
    x = qkv.shape[2]
    key = (x, qkv.dtype, out.dtype)
    fn = _FWD_CACHE.get(key)
    if fn is None:
        nblk = 98 if x > 6144 else 96
        q_ = from_dlpack(qkv, assumed_align=16, enable_tvm_ffi=True)
        c_ = from_dlpack(cos, assumed_align=16, enable_tvm_ffi=True)
        s_ = from_dlpack(sin, assumed_align=16, enable_tvm_ffi=True)
        o_ = from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
        fn = cute.compile(
            _healda_host, q_, c_, s_, o_, x, nblk, options="--enable-tvm-ffi"
        )
        _FWD_CACHE[key] = fn
    fn(qkv, cos, sin, out)


# ------------------------------------------------------------------------------------------------
# backward
# ------------------------------------------------------------------------------------------------

T = 8
H = 12
C = 128
HALF = C // 2
QKV_WIDTH = 3 * H * C
OUT_WIDTH = H * C

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

SUBC = 8  # channels owned by one lane
NSUB = C // SUBC  # 16 lanes per pixel
NPAIR = SUBC // 2
PIX = 32 // NSUB  # 2 heads per warp -> 16 MMA rows
M = PIX * T
PADK = C + 8
BWD_WARPS = 2
BWD_THREADS = BWD_WARPS * 32
HPB = PIX * BWD_WARPS  # 4 consecutive heads per CTA
HGROUPS = H // HPB
BWD_SCALE = 2.0**-3.5  # 1/sqrt(C); 1.0/math.sqrt(C) rounds 2 ULP away from this


@cute.kernel
def _bwd(
    qkv: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    grad_out: cute.Tensor,
    grad_qkv: cute.Tensor,
    QC: cutlass.Constexpr,
    REM: cutlass.Constexpr,
    NBX: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    hg, bx, _ = cute.arch.block_idx()
    # Trip count = ceil((X - bx) / NBX): the first REM columns walk one extra
    # pixel.  With NBX chosen so the grid is just under 148*8 CTAs, every SM
    # gets a full 8 resident CTAs AND the per-CTA pixel counts differ by at
    # most one, instead of 1152 CTAs leaving 32 SMs a whole CTA short.
    if cutlass.const_expr(REM == 0):
        NPP = QC
    else:
        NPP = cutlass.Int32(QC)
        if bx < REM:
            NPP = cutlass.Int32(QC + 1)
    warp = tidx // 32
    lane = tidx % 32
    gid = lane // 4
    tig = lane % 4
    p = lane // NSUB
    sub = lane % NSUB
    head = hg * HPB + warp * PIX + p

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate_tensor(
        BF16,
        cute.make_layout(
            (BWD_WARPS, 2, M, PADK), stride=(2 * M * PADK, M * PADK, PADK, 1)
        ),
        byte_alignment=128,
    )
    tA = st[(warp, 0, None, None)]
    tB = st[(warp, 1, None, None)]
    mlay = cute.make_layout((PIX, T, T), stride=(T * PADK, PADK, 1))
    wA = cute.make_tensor(tA.iterator + C, mlay)
    wB = cute.make_tensor(tB.iterator + C, mlay)
    vA = cute.make_tensor(
        tA.iterator, cute.make_layout((M, NSUB, SUBC), stride=(PADK, SUBC, 1))
    )
    vB = cute.make_tensor(
        tB.iterator, cute.make_layout((M, NSUB, SUBC), stride=(PADK, SUBC, 1))
    )
    sA = cute.make_tensor(tA.iterator, cute.make_layout((M, C), stride=(PADK, 1)))
    sB = cute.make_tensor(tB.iterator, cute.make_layout((M, C), stride=(PADK, 1)))

    # ---- CTA-shared BF16 interleaved [cos,sin] table (2 KiB) --------------
    # SMEM has slack (block limit 9) while registers cap occupancy at 8 CTAs, so this
    # table costs no occupancy.
    strig = smem.allocate_tensor(
        BF16,
        cute.make_layout((T, NSUB, 2 * NPAIR), stride=(NSUB * 2 * NPAIR, 2 * NPAIR, 1)),
        byte_alignment=16,
    )
    tfill = cute.make_tensor(
        strig.iterator, cute.make_layout((BWD_THREADS, 16), stride=(16, 1))
    )
    cp_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BF16, num_bits_per_copy=128)
    vA0 = cute.make_tensor(
        tA.iterator, cute.make_layout((M, NSUB, SUBC), stride=(PADK, SUBC, 1))
    )
    vB0 = cute.make_tensor(
        tB.iterator, cute.make_layout((M, NSUB, SUBC), stride=(PADK, SUBC, 1))
    )
    # Prefetch the first pixel's V/grad_out before the RoPE-table fill so the dependent
    # cos/sin loads and the CTA barrier hide behind the cp.async.
    # the two dependent cos/sin global loads and the CTA barrier are covered by
    # the cp.async instead of sitting in front of it.  With npp=8 this prologue
    # is paid once per 8 pixel-iterations, so it is a real slice of runtime.
    for t in cutlass.range_constexpr(T):
        row = p * T + t
        cute.copy(cp_atom, qkv[(t, bx, 2, head, sub, None)], vA0[(row, sub, None)])
        cute.copy(cp_atom, grad_out[(t, bx, head, sub, None)], vB0[(row, sub, None)])
    cute.arch.cp_async_commit_group()
    rc8 = cute.make_rmem_tensor((8,), F32)
    rs8 = cute.make_rmem_tensor((8,), F32)
    rt16 = cute.make_rmem_tensor((16,), BF16)
    cute.autovec_copy(cos[(tidx, None)], rc8)
    cute.autovec_copy(sin[(tidx, None)], rs8)
    for i in cutlass.range_constexpr(8):
        rt16[2 * i] = rc8[i].to(BF16)
        rt16[2 * i + 1] = rs8[i].to(BF16)
    cute.autovec_copy(rt16, tfill[(tidx, None)])
    cute.arch.barrier()

    # ---------------- MMA machinery ---------------------------------------
    op = cwarp.MmaF16BF16Op(BF16, F32, (16, 8, 16))
    mma = cute.make_tiled_mma(
        op, cute.make_layout((1, 1, 1)), permutation_mnk=(M, M, 16)
    )
    thr = mma.get_slice(lane)
    ldA = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cwarp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False), BF16
        ),
        mma,
    )
    ldB = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cwarp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False), BF16
        ),
        mma,
    )
    thrA = ldA.get_slice(lane)
    thrB = ldB.get_slice(lane)

    up0 = cutlass.Float32(0.0)
    up1 = cutlass.Float32(0.0)
    if 2 * tig >= gid:
        up0 = cutlass.Float32(BWD_SCALE)
    if 2 * tig + 1 >= gid:
        up1 = cutlass.Float32(BWD_SCALE)

    def contract(
        sa, sb, dst, upper, mma, ldA, ldB, thr, thrA, thrB, gid, tig, up0, up1
    ):
        fa = mma.make_fragment_A(thr.partition_A(sa))
        fb = mma.make_fragment_B(thr.partition_B(sb))
        tsa = thrA.partition_S(sa)
        tsb = thrB.partition_S(sb)
        tra = thrA.retile(fa)
        trb = thrB.retile(fb)
        acc = mma.make_fragment_C(mma.partition_shape_C((M, M)))
        acc.fill(0.0)
        for kb in cutlass.range_constexpr(C // 16):
            cute.copy(ldA, tsa[None, None, kb], tra[None, None, kb])
            cute.copy(ldB, tsb[None, None, kb], trb[None, None, kb])
            cute.gemm(mma, acc, fa[None, None, kb], fb[None, None, kb], acc)
        for i in cutlass.range_constexpr(8):
            e = i % 4
            nt = i // 4
            if cutlass.const_expr((e >> 1) == nt):
                col = 2 * tig + (e & 1)
                if cutlass.const_expr(upper):
                    msk = up0 if cutlass.const_expr((e & 1) == 0) else up1
                    dst[(nt, gid, col)] = (acc[i] * msk).to(BF16)
                else:
                    vb = (acc[i] * BWD_SCALE).to(BF16)
                    if gid >= col:
                        dst[(nt, gid, col)] = vb
                    if gid > col:
                        dst[(nt, col, gid)] = vb

    # ---------------- stage 1: V and G, then dS = G V^T --------------------
    r_g = cute.make_rmem_tensor(cute.make_layout((T, SUBC), stride=(SUBC, 1)), BF16)
    r_kq = cute.make_rmem_tensor(cute.make_layout((T, SUBC), stride=(SUBC, 1)), BF16)
    r_qr = cute.make_rmem_tensor(cute.make_layout((T, SUBC), stride=(SUBC, 1)), BF16)
    r_a = cute.make_rmem_tensor((SUBC,), BF16)
    r_b = cute.make_rmem_tensor((SUBC,), BF16)
    tr8 = cute.make_rmem_tensor((2 * NPAIR,), BF16)
    out = cute.make_rmem_tensor((SUBC,), BF16)
    accb = cute.make_rmem_tensor((SUBC,), BF16)
    mrow = cute.make_rmem_tensor((T,), BF16)

    for it in cutlass.range(NPP):
        x = bx + it * NBX
        xn = x + NBX
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        contract(sB, sA, wB, False, mma, ldA, ldB, thr, thrA, thrB, gid, tig, up0, up1)
        for t in cutlass.range_constexpr(T):
            cute.autovec_copy(vB[(p * T + t, sub, None)], r_g[(t, None)])
        cute.arch.sync_warp()

        # ---------------- stage 2: rotated Q and K, then W^T = Kr Qr^T ---------
        for t in cutlass.range_constexpr(T):
            row = p * T + t
            cute.autovec_copy(strig[(t, sub, None)], tr8)
            cute.autovec_copy(qkv[(t, x, 0, head, sub, None)], r_a)
            cute.autovec_copy(qkv[(t, x, 1, head, sub, None)], r_b)
            for j in cutlass.range_constexpr(NPAIR):
                cv = tr8[2 * j]
                sv = tr8[2 * j + 1]
                qe = r_a[2 * j]
                qo = r_a[2 * j + 1]
                ke = r_b[2 * j]
                ko = r_b[2 * j + 1]
                r_a[2 * j] = qe * cv - qo * sv
                r_a[2 * j + 1] = qe * sv + qo * cv
                r_b[2 * j] = ke * cv - ko * sv
                r_b[2 * j + 1] = ke * sv + ko * cv
            cute.autovec_copy(r_a, vA[(row, sub, None)])
            cute.autovec_copy(r_b, vB[(row, sub, None)])
        cute.arch.sync_warp()
        contract(sB, sA, wA, True, mma, ldA, ldB, thr, thrA, thrB, gid, tig, up0, up1)
        cute.arch.sync_warp()

        # ---------------- SIMT gradient contractions --------------------------
        # Accumulate the three tiny K=8 contractions directly in packed BF16 so a
        # channel pair is one instruction and no bf16->f32 conversion is emitted.

        # dV[s] = sum_{t>=s} W^T[s,t] * G[t]   (whole row in one 16-byte load)
        for s in cutlass.range_constexpr(T):
            cute.autovec_copy(wA[(p, s, None)], mrow)
            accb.store(r_g[(s, None)].load() * mrow[s])
            for t in cutlass.range_constexpr(s + 1, T):
                accb.store(accb.load() + r_g[(t, None)].load() * mrow[t])
            cute.autovec_copy(accb, grad_qkv[(s, x, 2, head, sub, None)])

        # dQ/dK epilogue. `contract` stores dS symmetrically, so wB row r carries both the
        # dQ weights dS[r][s<=r] and the dK weights dS[t>=r][r]. NBX == 394 (B200) selects a
        # fused variant that drives both outputs from one wB row load.
        # wB row r carries BOTH the dQ weights dS[r][s<=r] and the dK weights
        # dS[t>=r][r].  Reading Kr and Qr up front lets one r-loop drive both
        # outputs off one wB row load and one trig row load instead of two of
        # each, removing 16 LDS.128 per tile, and lets both cp.async prefetches
        # issue as soon as the tiles are dead rather than half an epilogue apart.
        for t in cutlass.range_constexpr(T):
            cute.autovec_copy(vB[(p * T + t, sub, None)], r_kq[(t, None)])
            cute.autovec_copy(vA[(p * T + t, sub, None)], r_qr[(t, None)])
        cute.arch.sync_warp()
        if it + 1 < NPP:
            for t in cutlass.range_constexpr(T):
                cute.copy(
                    cp_atom,
                    grad_out[(t, xn, head, sub, None)],
                    vB[(p * T + t, sub, None)],
                )
                cute.copy(
                    cp_atom,
                    qkv[(t, xn, 2, head, sub, None)],
                    vA[(p * T + t, sub, None)],
                )
            cute.arch.cp_async_commit_group()
        for r in cutlass.range_constexpr(T):
            cute.autovec_copy(strig[(r, sub, None)], tr8)
            cute.autovec_copy(wB[(p, r, None)], mrow)
            if cutlass.const_expr(NBX == 394):
                out.store(r_kq[(0, None)].load() * mrow[0])
                accb.store(r_qr[(r, None)].load() * mrow[r])
                for u in cutlass.range_constexpr(1, T):
                    if cutlass.const_expr(u <= r):
                        out.store(out.load() + r_kq[(u, None)].load() * mrow[u])
                    if cutlass.const_expr(r + u < T):
                        accb.store(
                            accb.load() + r_qr[(r + u, None)].load() * mrow[r + u]
                        )
                for j in cutlass.range_constexpr(NPAIR):
                    cv = tr8[2 * j]
                    sv = tr8[2 * j + 1]
                    qe = out[2 * j]
                    qo = out[2 * j + 1]
                    ke = accb[2 * j]
                    ko = accb[2 * j + 1]
                    out[2 * j] = qe * cv + qo * sv
                    out[2 * j + 1] = qo * cv - qe * sv
                    accb[2 * j] = ke * cv + ko * sv
                    accb[2 * j + 1] = ko * cv - ke * sv
                cute.autovec_copy(out, grad_qkv[(r, x, 0, head, sub, None)])
                cute.autovec_copy(accb, grad_qkv[(r, x, 1, head, sub, None)])
            else:
                accb.store(r_kq[(0, None)].load() * mrow[0])
                for s in cutlass.range_constexpr(1, r + 1):
                    accb.store(accb.load() + r_kq[(s, None)].load() * mrow[s])
                for j in cutlass.range_constexpr(NPAIR):
                    cv = tr8[2 * j]
                    sv = tr8[2 * j + 1]
                    e = accb[2 * j]
                    o = accb[2 * j + 1]
                    out[2 * j] = e * cv + o * sv
                    out[2 * j + 1] = o * cv - e * sv
                cute.autovec_copy(out, grad_qkv[(r, x, 0, head, sub, None)])
                accb.store(r_qr[(r, None)].load() * mrow[r])
                for t in cutlass.range_constexpr(r + 1, T):
                    accb.store(accb.load() + r_qr[(t, None)].load() * mrow[t])
                for j in cutlass.range_constexpr(NPAIR):
                    cv = tr8[2 * j]
                    sv = tr8[2 * j + 1]
                    e = accb[2 * j]
                    o = accb[2 * j + 1]
                    out[2 * j] = e * cv + o * sv
                    out[2 * j + 1] = o * cv - e * sv
                cute.autovec_copy(out, grad_qkv[(r, x, 1, head, sub, None)])


@cute.jit
def _launch(
    qkv_mem: cute.Tensor,
    cos_mem: cute.Tensor,
    sin_mem: cute.Tensor,
    grad_mem: cute.Tensor,
    out_mem: cute.Tensor,
    x_size: cutlass.Constexpr,
    nbx: cutlass.Constexpr,
):
    qkv = cute.make_tensor(
        qkv_mem.iterator,
        cute.make_layout(
            (T, x_size, 3, H, NSUB, SUBC),
            stride=(x_size * QKV_WIDTH, QKV_WIDTH, H * C, C, SUBC, 1),
        ),
    )
    grad = cute.make_tensor(
        grad_mem.iterator,
        cute.make_layout(
            (T, x_size, H, NSUB, SUBC),
            stride=(x_size * OUT_WIDTH, OUT_WIDTH, C, SUBC, 1),
        ),
    )
    out = cute.make_tensor(
        out_mem.iterator,
        cute.make_layout(
            (T, x_size, 3, H, NSUB, SUBC),
            stride=(x_size * QKV_WIDTH, QKV_WIDTH, H * C, C, SUBC, 1),
        ),
    )
    tbl = cute.make_layout(
        (BWD_THREADS, (T * HALF) // BWD_THREADS), stride=((T * HALF) // BWD_THREADS, 1)
    )
    cos = cute.make_tensor(cos_mem.iterator, tbl)
    sin = cute.make_tensor(sin_mem.iterator, tbl)

    _bwd(qkv, cos, sin, grad, out, x_size // nbx, x_size % nbx, nbx).launch(
        grid=(HGROUPS, nbx, 1),
        block=(BWD_THREADS, 1, 1),
        min_blocks_per_mp=8,
    )


# Persistent-CTA count for the backward, keyed on (small x_size, large x_size).
# B200 sizes for a single resident wave (3*394 <= 148 SMs * 8 blocks/SM); B300 is faster
# oversubscribed. Both are tuned values with a broad optimum, not derived.
# nbx > x_size faults with an illegal address, hence the clamp in _pick_nbx.
_NBX_BY_ARCH = {
    (10, 0): (394, 1536),  # B200
    (10, 3): (2048, 8192),  # B300
}
_NBX_DEFAULT = (2048, 8192)  # unknown arch: use the B300 sizing


@functools.lru_cache(maxsize=None)
def _pick_nbx(x_size: int) -> int:
    """Resolved once per problem shape; this sits on the per-launch path."""
    small, large = _NBX_BY_ARCH.get(torch.cuda.get_device_capability(), _NBX_DEFAULT)
    return min(small if x_size <= 4096 else large, x_size)


_BWD_CACHE = {}


def run_backward(qkv, cos, sin, grad_out, grad_qkv):
    x_size = int(qkv.shape[2])
    nbx = _pick_nbx(x_size)
    key = (x_size, nbx)
    fn = _BWD_CACHE.get(key)
    if fn is None:
        args = (
            from_dlpack(qkv, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(cos, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(sin, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(grad_out, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(grad_qkv, assumed_align=16, enable_tvm_ffi=True),
        )
        fn = cute.compile(_launch, *args, x_size, nbx, options="--enable-tvm-ffi")
        _BWD_CACHE[key] = fn
    fn(qkv, cos, sin, grad_out, grad_qkv)


# ------------------------------------------------------------------------------------------------
# autograd glue
# ------------------------------------------------------------------------------------------------

# Tile shapes, SMEM layouts, mbarrier counts and the head->warp mapping are baked to these.
HEADS = 12
HEAD_DIM = 128
FRAMES = 8


# Empty list == this configuration is supported.
def unsupported_reasons(
    *, num_heads: int, head_dim: int, frames: int, batch: int
) -> list[str]:
    reasons = unsupported_arch_reasons()
    if num_heads != HEADS:
        reasons.append(f"num_heads=={HEADS} (got {num_heads})")
    if head_dim != HEAD_DIM:
        reasons.append(f"head_dim=={HEAD_DIM} (got {head_dim})")
    if frames != FRAMES:
        reasons.append(f"frames=={FRAMES} (got {frames})")
    if batch != 1:
        # both kernels view the qkv buffer as (frames, x, width); a batch axis would alias.
        reasons.append(f"batch==1 (got {batch})")
    return reasons


def _out_shape(qkv: torch.Tensor) -> tuple[int, ...]:
    b, frames, x_size, width = qkv.shape
    return b, frames, x_size, width // 3


@torch.library.custom_op("healda::cutedsl_temporal_attn_fwd", mutates_args=())
def _op_fwd(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = torch.empty(_out_shape(qkv), device=qkv.device, dtype=qkv.dtype)
    run_forward(qkv, cos, sin, out)
    return out


@_op_fwd.register_fake
def _(qkv, cos, sin):
    return qkv.new_empty(_out_shape(qkv))


@torch.library.custom_op("healda::cutedsl_temporal_attn_bwd", mutates_args=())
def _op_bwd(
    qkv: torch.Tensor, grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    grad_qkv = torch.empty_like(qkv)
    run_backward(qkv, cos, sin, grad_out.contiguous(), grad_qkv)
    return grad_qkv


@_op_bwd.register_fake
def _(qkv, grad_out, cos, sin):
    return torch.empty_like(qkv)


def _setup_context(ctx, inputs, output):
    qkv, cos, sin = inputs
    ctx.save_for_backward(qkv, cos, sin)


def _backward_fn(ctx, grad_out):
    qkv, cos, sin = ctx.saved_tensors
    return _op_bwd(qkv, grad_out, cos, sin), None, None


_op_fwd.register_autograd(_backward_fn, setup_context=_setup_context)


def fused_cutedsl(
    qkv: torch.Tensor, heads: int, cos: torch.Tensor, sin: torch.Tensor, causal: bool
) -> torch.Tensor:
    """``heads`` and ``causal`` are validated rather than used: the kernels are generated for HEADS
    heads with the causal mask baked into their loop bounds.
    """
    if heads != HEADS:
        raise ValueError(f"cute dsl kernels are built for {HEADS} heads, got {heads}")
    if not causal:
        raise ValueError("cute dsl temporal-attention kernels are causal-only")
    return _op_fwd(qkv, cos, sin)

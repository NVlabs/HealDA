# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL backward of the fused FiLM observation tokenizer.

Same contract as ``fused_film_triton._launch_fused_film_bwd``: given the tokenizer inputs and
``grad_out`` it returns the embedding-table gradients and the gradients of the conditioning MLP
(W1, b1, LayerNorm weight and bias, W2, b2).

1. ``_film_bwd_rows``: one warp per 16-row tile, walking tiles persistently. W1^T, W2 and the
   embedding rows of W1 sit in padded (bank-conflict-free) shared memory for the CTA's lifetime.
   ``h = cond W1``, ``d_act = d_ab W2^T`` and ``d_emb = dh W1_emb^T`` are m16n8k16 bf16 MMAs;
   LayerNorm, SiLU and their backward run on the fp32 accumulator fragments (row reductions
   over the 4-lane quad). Accumulator element (v0, v1, n) of a thread is row gid + 8 * v1,
   column 8 * n + 2 * tig + v0. The conditioning gather is branch-free so a row's loads are in
   flight together. Embedding-table gradients are summed over runs of equal id before one
   atomic per run. The per-row factors cond, act, d_ab and dh are written in bf16.
2. ``film_bwd``: runs the rows in chunks of ``CHUNK`` so the factor buffers stay bounded, and
   forms dW1 = cond^T dh and dW2 = act^T d_ab as fp32-accumulated GEMMs per chunk.
3. ``torch.library.custom_op`` wrapper, called from the FiLM tokenizer's autograd backward.

Generated for one tokenizer layout (4/16/8-wide obs-type/channel/platform embeddings, hidden
64, 32 output channels) and compiled per metadata width, 32 to 64 features; see
``unsupported_reasons``.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from healda.models.kernels.cutedsl_arch import unsupported_arch_reasons

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32

NOBS, NCH, NPL = 4, 16, 8
EMB = NOBS + NCH + NPL  # 28 embedding columns of cond
TABLE_ROWS = (
    1024,
    2048,
    1024,
)  # obs type, channel, platform; concatenated in one buffer
OBS_ROWS, CH_ROWS, _ = TABLE_ROWS
TAB_BASE = (0, OBS_ROWS * NOBS, OBS_ROWS * NOBS + CH_ROWS * NCH)
KC = 96  # cond padded to the MMA K step
HID = 64
OUT = 32
MLP = 2 * OUT  # alpha | beta
EMBP = 32
ROWS = 16
WARPS = 8
THREADS = WARPS * 32
NVEC = 3 * HID + MLP  # db1 | dln_w | dln_b | db2

# Padded SMEM row strides: 208 B and 144 B rows put the 8 rows of an ldmatrix in 8 banks.
S_KC = 104
# One tile's inputs in 16 B chunks (cp.async units).
GRAD_CHUNKS = ROWS * OUT // 8  # 64 x 8 bf16
ID_CHUNKS = 3 * ROWS // 4  # 12 x 4 i32
OBS_CHUNKS = ROWS // 4  # 4 x 4 f32
S_64 = 72
# Metadata widths the lane mapping covers: lane l owns metadata columns l and 32 + l.
META_RANGE = (32, 64)


def _widths(meta):
    # cond width, the row stride of the stored cond (16 B rows), and one tile's metadata in
    # 16 B chunks
    cond = meta + EMB
    return cond, -(-cond // 8) * 8, ROWS * meta // 4


def _row_major(shape):
    return cute.make_ordered_layout(shape, order=tuple(reversed(range(len(shape)))))


def _tile(smem, rows, cols, stride, dtype=BF16):
    return smem.allocate_tensor(
        dtype, cute.make_layout((rows, cols), stride=(stride, 1)), byte_alignment=16
    )


@cute.jit
def _quad_sum(x):
    x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
    return x + cute.arch.shuffle_sync_bfly(x, offset=2)


@cute.jit
def _gid_sum(x):
    x = x + cute.arch.shuffle_sync_bfly(x, offset=4)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=8)
    return x + cute.arch.shuffle_sync_bfly(x, offset=16)


@cute.jit
def _store_rows(tile, tile_stride: cutlass.Constexpr, dst, chunks: cutlass.Constexpr, row0,
                n_rows, lane):  # fmt: skip
    # 16-row SMEM tile -> rows row0.. of dst, viewed as (N, chunks, 8), in 16 B vectors.
    src = cute.make_tensor(
        tile.iterator, cute.make_layout((ROWS, chunks, 8), stride=(tile_stride, 8, 1))
    )
    for i in cutlass.range_constexpr((ROWS * chunks + 31) // 32):
        idx = lane + 32 * i
        rr = idx // chunks
        k = idx % chunks
        if idx < ROWS * chunks and row0 + rr < n_rows:
            cute.autovec_copy(src[(rr, k, None)], dst[(row0 + rr, k, None)])


@cute.jit
def _warp_gemm(mma, lane, s_a, s_b, acc, kblocks: cutlass.Constexpr):
    ld = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BF16
    )
    thr = mma.get_slice(lane)
    cp_a = cute.make_tiled_copy_A(ld, mma).get_slice(lane)
    cp_b = cute.make_tiled_copy_B(ld, mma).get_slice(lane)
    acc.fill(0.0)
    # One 16-wide K block at a time keeps only that block's fragments in registers.
    for kb in cutlass.range_constexpr(kblocks):
        a_k = cute.local_tile(s_a, (ROWS, 16), (0, kb))
        b_k = cute.local_tile(s_b, (cute.size(s_b, mode=[0]), 16), (0, kb))
        fa = mma.make_fragment_A(thr.partition_A(a_k))
        fb = mma.make_fragment_B(thr.partition_B(b_k))
        cute.copy(cp_a, cp_a.partition_S(a_k), cp_a.retile(fa))
        cute.copy(cp_b, cp_b.partition_S(b_k), cp_b.retile(fb))
        cute.gemm(mma, acc, fa, fb, acc)


@cute.jit
def _prefetch(tile, buf, n_tiles, n_rows, lane, w, meta_t, grad_t, obs_t, ids_t, b_meta,
              b_grad, b_obs, b_ids, meta_chunks: cutlass.Constexpr):  # fmt: skip
    # cp.async one tile's inputs into buffer buf, then commit the group (empty when there is no
    # such tile, so every iteration waits on the same group count). Ids are padded on the host;
    # metadata, grads and obs only for a full tile.
    cp_f32 = cute.make_copy_atom(cpasync.CopyG2SOp(), F32, num_bits_per_copy=128)
    cp_bf16 = cute.make_copy_atom(cpasync.CopyG2SOp(), BF16, num_bits_per_copy=128)
    cp_i32 = cute.make_copy_atom(cpasync.CopyG2SOp(), I32, num_bits_per_copy=128)
    if tile < n_tiles:
        if lane < ID_CHUNKS:
            cute.copy(cp_i32, ids_t[(tile, lane, None)], b_ids[(w, buf, lane, None)])
        if (tile + 1) * ROWS <= n_rows:
            for i in cutlass.range_constexpr((meta_chunks + 31) // 32):
                c = lane + 32 * i
                if c < meta_chunks:
                    cute.copy(
                        cp_f32, meta_t[(tile, c, None)], b_meta[(w, buf, c, None)]
                    )
            for i in cutlass.range_constexpr(GRAD_CHUNKS // 32):
                cute.copy(
                    cp_bf16,
                    grad_t[(tile, lane + 32 * i, None)],
                    b_grad[(w, buf, lane + 32 * i, None)],
                )
            if lane < OBS_CHUNKS:
                cute.copy(
                    cp_f32, obs_t[(tile, lane, None)], b_obs[(w, buf, lane, None)]
                )
    cute.arch.cp_async_commit_group()


@cute.kernel
def _film_bwd_rows(
    grad: cute.Tensor,  # (N, OUT) bf16
    obs: cute.Tensor,  # (N,) f32
    meta: cute.Tensor,  # (N, META) f32
    grad_t: cute.Tensor,  # the same three, viewed as (tiles, chunks, 16 B)
    obs_t: cute.Tensor,
    meta_t: cute.Tensor,
    ids_t: cute.Tensor,  # (tiles, ID_CHUNKS, 4) i32: per tile 16 obs-type, channel, platform ids
    tab: cute.Tensor,  # the three embedding tables concatenated, flat f32
    w1: cute.Tensor,  # (COND, HID) f32
    w2: cute.Tensor,  # (HID, MLP) f32
    vecs: cute.Tensor,  # (3, HID) f32: b1, ln_w, ln_b
    cond_out: cute.Tensor,  # (N, CSTR // 8, 8) bf16
    act_out: cute.Tensor,  # (N, HID // 8, 8) bf16
    dab_out: cute.Tensor,  # (N, MLP // 8, 8) bf16
    dh_out: cute.Tensor,  # (N, HID // 8, 8) bf16
    vec_part: cute.Tensor,  # (n_warps, NVEC) f32
    g_tab: cute.Tensor,  # their gradients, same layout
    n_rows: I32,
    META: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    COND, CSTR, META_CHUNKS = _widths(META)
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    n_cta, _, _ = cute.arch.grid_dim()
    w = tidx // 32
    lane = tidx % 32
    gid = lane // 4
    tig = lane % 4

    smem = cutlass.utils.SmemAllocator()
    s_w1t = _tile(smem, HID, KC, S_KC)  # [h, c] = W1[c, h]
    s_w2 = _tile(smem, HID, MLP, S_64)  # [h, m] = W2[h, m]
    s_w1e = _tile(smem, EMBP, HID, S_64)  # [e, h] = W1[META + e, h]
    s_vec = _tile(smem, 3, HID, HID, F32)
    s_cond = smem.allocate_tensor(
        BF16, cute.make_layout((WARPS, ROWS, KC), stride=(ROWS * S_KC, S_KC, 1)), 16
    )
    s_dab = smem.allocate_tensor(
        BF16, cute.make_layout((WARPS, ROWS, MLP), stride=(ROWS * S_64, S_64, 1)), 16
    )
    s_dh = smem.allocate_tensor(
        BF16, cute.make_layout((WARPS, ROWS, HID), stride=(ROWS * S_64, S_64, 1)), 16
    )
    s_emb = smem.allocate_tensor(
        F32,
        cute.make_layout(
            (WARPS, ROWS, EMBP + 1), stride=(ROWS * (EMBP + 1), EMBP + 1, 1)
        ),
        16,
    )
    # Per warp, two buffers of one tile's inputs, filled by cp.async one tile ahead.
    b_meta = smem.allocate_tensor(F32, _row_major((WARPS, 2, META_CHUNKS, 4)), 16)
    b_grad = smem.allocate_tensor(BF16, _row_major((WARPS, 2, GRAD_CHUNKS, 8)), 16)
    b_obs = smem.allocate_tensor(F32, _row_major((WARPS, 2, OBS_CHUNKS, 4)), 16)
    b_ids = smem.allocate_tensor(I32, _row_major((WARPS, 2, ID_CHUNKS, 4)), 16)

    for i in cutlass.range_constexpr(HID * KC // THREADS):
        idx = tidx + THREADS * i
        h, c = idx // KC, idx % KC
        v = F32(0.0)
        if c < COND:
            v = w1[c, h]
        s_w1t[h, c] = v.to(BF16)
    for i in cutlass.range_constexpr(HID * MLP // THREADS):
        idx = tidx + THREADS * i
        s_w2[idx // MLP, idx % MLP] = w2[idx // MLP, idx % MLP].to(BF16)
    for i in cutlass.range_constexpr(EMBP * HID // THREADS):
        idx = tidx + THREADS * i
        e, h = idx // HID, idx % HID
        v = F32(0.0)
        if e < EMB:
            v = w1[META + e, h]
        s_w1e[e, h] = v.to(BF16)
    for i in cutlass.range_constexpr(3 * HID // THREADS + 1):
        idx = tidx + THREADS * i
        if idx < 3 * HID:
            s_vec[idx // HID, idx % HID] = vecs[idx // HID, idx % HID]
    cute.arch.barrier()

    sc = s_cond[(w, None, None)]
    sd = s_dab[(w, None, None)]
    sh = s_dh[(w, None, None)]
    se = s_emb[(w, None, None)]

    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BF16, F32, (16, 8, 16)))
    acc_h = cute.make_rmem_tensor(mma.partition_shape_C((ROWS, HID)), F32)
    acc_d = cute.make_rmem_tensor(mma.partition_shape_C((ROWS, HID)), F32)
    acc_e = cute.make_rmem_tensor(mma.partition_shape_C((ROWS, EMBP)), F32)
    # SiLU derivative per accumulator element, kept from the forward for the backward.
    dsilu = cute.make_rmem_tensor((2, 8, 2), F32)
    sum_db1 = cute.make_rmem_tensor((8, 2), F32)
    sum_lnw = cute.make_rmem_tensor((8, 2), F32)
    sum_lnb = cute.make_rmem_tensor((8, 2), F32)
    sum_db1.fill(0.0)
    sum_lnw.fill(0.0)
    sum_lnb.fill(0.0)
    sum_da = F32(0.0)
    sum_db = F32(0.0)
    mean = cute.make_rmem_tensor((2,), F32)
    rstd = cute.make_rmem_tensor((2,), F32)
    s1 = cute.make_rmem_tensor((2,), F32)
    s2 = cute.make_rmem_tensor((2,), F32)

    # Lane l < 28 owns embedding column l: table k, column col, row width, table base offset.
    emb_k = 0 if lane < NOBS else (1 if lane < NOBS + NCH else 2)
    emb_col = (
        lane
        if lane < NOBS
        else (lane - NOBS if lane < NOBS + NCH else lane - NOBS - NCH)
    )
    emb_col = emb_col if emb_col < NPL or emb_k != 2 else NPL - 1
    emb_width = NOBS if lane < NOBS else (NCH if lane < NOBS + NCH else NPL)
    emb_base = (
        TAB_BASE[0]
        if lane < NOBS
        else (TAB_BASE[1] if lane < NOBS + NCH else TAB_BASE[2])
    )

    n_warps = n_cta * WARPS
    gw = cta * WARPS + w
    n_tiles = (n_rows + ROWS - 1) // ROWS
    _prefetch(gw, 0, n_tiles, n_rows, lane, w, meta_t, grad_t, obs_t, ids_t, b_meta, b_grad,
              b_obs, b_ids, META_CHUNKS)  # fmt: skip
    for tile in cutlass.range(gw, n_tiles, n_warps):
        row0 = tile * ROWS
        buf = ((tile - gw) // n_warps) % 2
        _prefetch(tile + n_warps, 1 - buf, n_tiles, n_rows, lane, w, meta_t, grad_t, obs_t,
                  ids_t, b_meta, b_grad, b_obs, b_ids, META_CHUNKS)  # fmt: skip
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_warp()
        full = row0 + ROWS <= n_rows
        # Flat views of the current buffer (row-major: rows of the tile back to back).
        slot = w * 2 + buf
        cur_meta = cute.make_tensor(
            b_meta.iterator + slot * ROWS * META, (ROWS * META,)
        )
        cur_grad = cute.make_tensor(b_grad.iterator + slot * ROWS * OUT, (ROWS * OUT,))
        cur_obs = cute.make_tensor(b_obs.iterator + slot * ROWS, (ROWS,))
        cur_ids = cute.make_tensor(
            b_ids.iterator + slot * 3 * ROWS,
            cute.make_layout((3, ROWS), stride=(ROWS, 1)),
        )
        # Lane l owns meta columns l and 32 + l (l < META - 32) and embedding column l
        # (l < 28). Full tiles read metadata and grads from the prefetched buffer; the final
        # partial tile of a call reads them directly, clamped, so nothing past the inputs is
        # touched.
        for rr in cutlass.range_constexpr(ROWS):
            r = row0 + rr
            ok = r < n_rows
            rc = r if ok else n_rows - 1
            m0 = F32(0.0)
            m1 = F32(0.0)
            g = F32(0.0)
            ob = F32(0.0)
            c1 = lane + 32 if lane < META - 32 else META - 1
            if full:
                m0 = cur_meta[rr * META + lane]
                m1 = cur_meta[rr * META + c1]
                g = cur_grad[rr * OUT + lane].to(F32)
                ob = cur_obs[rr]
            else:
                m0 = meta[rc, lane]
                m1 = meta[rc, c1]
                g = grad[rc, lane].to(F32)
                ob = obs[rc]
            eid = cur_ids[emb_k, rr]
            ve = tab[emb_base + (eid if eid > 0 else 0) * emb_width + emb_col]
            zero = F32(0.0)
            sc[rr, lane] = (m0 if ok else zero).to(BF16)
            if lane < META - 32:
                sc[rr, 32 + lane] = (m1 if ok else zero).to(BF16)
            if lane < EMB:
                sc[rr, META + lane] = (ve if ok else zero).to(BF16)
            for j in cutlass.range_constexpr((KC - COND + 31) // 32):
                if lane + 32 * j < KC - COND:
                    sc[rr, COND + lane + 32 * j] = zero.to(BF16)
            # d_ab = [grad * obs | grad]; lane owns one output channel.
            g = g if ok else zero
            da = g * ob
            sum_da = sum_da + da
            sum_db = sum_db + g
            sd[rr, lane] = da.to(BF16)
            sd[rr, OUT + lane] = g.to(BF16)
        cute.arch.sync_warp()
        _store_rows(sc, S_KC, cond_out, CSTR // 8, row0, n_rows, lane)
        _store_rows(sd, S_64, dab_out, MLP // 8, row0, n_rows, lane)

        # ---- h = cond W1 + b1; LayerNorm + SiLU replay ----
        _warp_gemm(mma, lane, sc, s_w1t, acc_h, KC // 16)
        cute.arch.sync_warp()
        sa = sc  # the cond tile is consumed; act is staged in its first 64 columns
        for v1 in cutlass.range_constexpr(2):
            tot = F32(0.0)
            for n in cutlass.range_constexpr(HID // 8):
                for v0 in cutlass.range_constexpr(2):
                    col = 8 * n + 2 * tig + v0
                    hv = acc_h[(v0, v1), 0, n] + s_vec[0, col]
                    acc_h[(v0, v1), 0, n] = hv
                    tot = tot + hv
            mean[v1] = _quad_sum(tot) * (1.0 / HID)
            var = F32(0.0)
            for n in cutlass.range_constexpr(HID // 8):
                for v0 in cutlass.range_constexpr(2):
                    dv = acc_h[(v0, v1), 0, n] - mean[v1]
                    var = var + dv * dv
            rstd[v1] = cute.math.rsqrt(_quad_sum(var) * (1.0 / HID) + EPS)
            for n in cutlass.range_constexpr(HID // 8):
                for v0 in cutlass.range_constexpr(2):
                    col = 8 * n + 2 * tig + v0
                    xh = (acc_h[(v0, v1), 0, n] - mean[v1]) * rstd[v1]
                    nm = xh * s_vec[1, col] + s_vec[2, col]
                    sig = cute.math.rcp(
                        1.0 + cute.math.exp(-nm, fastmath=True), fastmath=True
                    )
                    sa[gid + 8 * v1, col] = (nm * sig).to(BF16)
                    acc_h[(v0, v1), 0, n] = xh  # the backward reads x_hat from here
                    dsilu[v1, n, v0] = sig * (1.0 + nm * (1.0 - sig))

        # ---- d_act = d_ab W2^T; LayerNorm/SiLU backward -> dh ----
        _warp_gemm(mma, lane, sd, s_w2, acc_d, MLP // 16)
        for v1 in cutlass.range_constexpr(2):
            a1 = F32(0.0)
            a2 = F32(0.0)
            for n in cutlass.range_constexpr(HID // 8):
                for v0 in cutlass.range_constexpr(2):
                    col = 8 * n + 2 * tig + v0
                    xh = acc_h[(v0, v1), 0, n]
                    dn = acc_d[(v0, v1), 0, n] * dsilu[v1, n, v0]
                    sum_lnw[n, v0] = sum_lnw[n, v0] + dn * xh
                    sum_lnb[n, v0] = sum_lnb[n, v0] + dn
                    dx = dn * s_vec[1, col]
                    acc_d[(v0, v1), 0, n] = dx
                    a1 = a1 + dx
                    a2 = a2 + dx * xh
            s1[v1] = _quad_sum(a1)
            s2[v1] = _quad_sum(a2)
            for n in cutlass.range_constexpr(HID // 8):
                for v0 in cutlass.range_constexpr(2):
                    col = 8 * n + 2 * tig + v0
                    xh = acc_h[(v0, v1), 0, n]
                    dh = rstd[v1] * (
                        acc_d[(v0, v1), 0, n] - (s1[v1] + xh * s2[v1]) * (1.0 / HID)
                    )
                    sum_db1[n, v0] = sum_db1[n, v0] + dh
                    sh[gid + 8 * v1, col] = dh.to(BF16)
        cute.arch.sync_warp()

        _store_rows(sa, S_KC, act_out, HID // 8, row0, n_rows, lane)
        _store_rows(sh, S_64, dh_out, HID // 8, row0, n_rows, lane)

        # ---- embedding-table gradients, one atomic per run of equal id ----
        _warp_gemm(mma, lane, sh, s_w1e, acc_e, HID // 16)
        for v1 in cutlass.range_constexpr(2):
            for n in cutlass.range_constexpr(EMBP // 8):
                for v0 in cutlass.range_constexpr(2):
                    se[gid + 8 * v1, 8 * n + 2 * tig + v0] = acc_e[(v0, v1), 0, n]
        cute.arch.sync_warp()
        if lane < EMB:
            run = F32(0.0)
            for rr2 in cutlass.range_constexpr(ROWS):
                cur = cur_ids[emb_k, rr2]
                if cur >= 0:
                    run = run + se[rr2, lane]
                    nxt = I32(-2)
                    if cutlass.const_expr(rr2 + 1 < ROWS):
                        nxt = cur_ids[emb_k, rr2 + 1]
                    if nxt != cur:
                        off = emb_base + cur * emb_width + emb_col
                        cute.arch.atomic_add(g_tab.iterator + off, run)
                        run = F32(0.0)
        cute.arch.sync_warp()

    # ---- per-warp partial column sums ----
    for n in cutlass.range_constexpr(HID // 8):
        for v0 in cutlass.range_constexpr(2):
            col = 8 * n + 2 * tig + v0
            a = _gid_sum(sum_db1[n, v0])
            b = _gid_sum(sum_lnw[n, v0])
            c = _gid_sum(sum_lnb[n, v0])
            if gid == 0:
                vec_part[gw, col] = a
                vec_part[gw, HID + col] = b
                vec_part[gw, 2 * HID + col] = c
    vec_part[gw, 3 * HID + lane] = sum_da
    vec_part[gw, 3 * HID + OUT + lane] = sum_db


def _row_chunks(mem, n_rows, width):
    return cute.make_tensor(
        mem.iterator, cute.make_layout((n_rows, width // 8, 8), stride=(width, 8, 1))
    )


def _tiles(mem, n_tiles, chunks, width):
    return cute.make_tensor(
        mem.iterator,
        cute.make_layout((n_tiles, chunks, width), stride=(chunks * width, width, 1)),
    )


@cute.jit
def _launch(
    grad, obs, meta, ids, tab, w1, w2, vecs, cond_mem, act_mem, dab_mem, dh_mem, vec_part,
    g_tab, n_rows: I32, n_cta: cutlass.Constexpr, META: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):  # fmt: skip
    _, CSTR, META_CHUNKS = _widths(META)
    n_tiles = (n_rows + ROWS - 1) // ROWS
    grad_t = _tiles(grad, n_tiles, GRAD_CHUNKS, 8)
    obs_t = _tiles(obs, n_tiles, OBS_CHUNKS, 4)
    meta_t = _tiles(meta, n_tiles, META_CHUNKS, 4)
    ids_t = _tiles(ids, n_tiles, ID_CHUNKS, 4)
    cond_out = _row_chunks(cond_mem, n_rows, CSTR)
    act_out = _row_chunks(act_mem, n_rows, HID)
    dab_out = _row_chunks(dab_mem, n_rows, MLP)
    dh_out = _row_chunks(dh_mem, n_rows, HID)
    _film_bwd_rows(
        grad, obs, meta, grad_t, obs_t, meta_t, ids_t, tab, w1, w2, vecs, cond_out, act_out,
        dab_out, dh_out, vec_part, g_tab, n_rows, META, EPS,
    ).launch(grid=(n_cta, 1, 1), block=(THREADS, 1, 1))  # fmt: skip


CTAS_PER_SM = 4
CHUNK = 1 << 22  # rows per factor-buffer pass (~2.3 GB of bf16 buffers)
_COMPILED = {}


def _dlpack_rows(t):
    # Only the observation count varies between calls: mark that mode dynamic, keep strides.
    d = from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
    return d.mark_compact_shape_dynamic(mode=0)


def _tile_ids(obs_type_id, channel, platform):
    # (tiles, 3, ROWS) int32, tile-major so one tile's ids are 192 contiguous bytes; rows past
    # the end are -1.
    m = obs_type_id.shape[0]
    n_tiles = -(-m // ROWS)
    ids = torch.full(
        (3, n_tiles * ROWS), -1, device=obs_type_id.device, dtype=torch.int32
    )
    ids[0, :m], ids[1, :m], ids[2, :m] = obs_type_id, channel, platform
    return ids.view(3, n_tiles, ROWS).permute(1, 0, 2).contiguous()


def film_bwd(grad_out, obs, float_meta, obs_type_id, channel, platform, embed_weight,
             chan_embed_weight, platform_embed_weight, w1, b1, ln_w, ln_b, w2, eps):  # fmt: skip
    n, meta = float_meta.shape
    cond_w, cstr, _ = _widths(meta)
    dev = obs.device
    tables = (embed_weight, chan_embed_weight, platform_embed_weight)
    if n == 0:
        f32 = dict(device=dev, dtype=torch.float32)
        return (
            *(torch.zeros_like(t) for t in tables),
            torch.zeros(cond_w, HID, **f32), torch.zeros(HID, **f32),
            torch.zeros(HID, **f32), torch.zeros(HID, **f32),
            torch.zeros(HID, MLP, **f32), torch.zeros(MLP, **f32),
        )  # fmt: skip
    vecs = torch.stack([b1, ln_w, ln_b]).float().contiguous()
    w1f, w2f = w1.float().contiguous(), w2.float().contiguous()
    grad_out = grad_out.contiguous()
    # The kernel reads metadata and obs as compact f32 rows in 16 B vectors.
    obs = obs.float().contiguous()
    float_meta = float_meta.float().contiguous()
    # Rows go through in chunks so the bf16 factor buffers stay at CHUNK rows.
    chunk = min(CHUNK, n)
    cond = torch.empty(chunk, cstr, device=dev, dtype=torch.bfloat16)
    act = torch.empty(chunk, HID, device=dev, dtype=torch.bfloat16)
    dab = torch.empty(chunk, MLP, device=dev, dtype=torch.bfloat16)
    dh = torch.empty(chunk, HID, device=dev, dtype=torch.bfloat16)
    n_cta = CTAS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count
    vec_part = torch.empty(n_cta * WARPS, NVEC, device=dev, dtype=torch.float32)
    tab = torch.cat([t.reshape(-1) for t in tables]).float()
    g_tab = torch.zeros_like(tab)
    dw1 = torch.zeros(cond_w, HID, device=dev, dtype=torch.float32)
    dw2 = torch.zeros(HID, MLP, device=dev, dtype=torch.float32)
    vec = torch.zeros(NVEC, device=dev, dtype=torch.float32)
    for s0 in range(0, n, chunk):
        m = min(chunk, n - s0)
        rows = slice(s0, s0 + m)
        tensors = (grad_out[rows], obs[rows], float_meta[rows],
                   _tile_ids(obs_type_id[rows], channel[rows], platform[rows]), tab, w1f, w2f,
                   vecs, cond[:m], act[:m], dab[:m], dh[:m], vec_part, g_tab)  # fmt: skip
        key = (n_cta, meta, float(eps))
        fn = _COMPILED.get(key)
        if fn is None:
            args = [_dlpack_rows(t) for t in tensors]
            fn = cute.compile(
                _launch, *args, m, n_cta, meta, float(eps), options="--enable-tvm-ffi"
            )
            _COMPILED[key] = fn
        fn(*tensors, m)
        dw1 += torch.mm(cond[:m, :cond_w].t(), dh[:m], out_dtype=torch.float32)
        dw2 += torch.mm(act[:m].t(), dab[:m], out_dtype=torch.float32)
        vec += vec_part.sum(0)
    # Separate tensors: a custom op's outputs may not alias one another.
    db1, dln_w, dln_b, db2 = (v.clone() for v in vec.split([HID, HID, HID, MLP]))
    g_obs, g_ch, g_pl = (
        g.view_as(t).to(t.dtype, copy=True)
        for g, t in zip(g_tab.split([t.numel() for t in tables]), tables)
    )
    return g_obs, g_ch, g_pl, dw1, db1, dln_w, dln_b, dw2, db2


# ------------------------------------------------------------------------------------------------
# custom op
# ------------------------------------------------------------------------------------------------


# Empty list == this configuration is supported.
def unsupported_reasons(
    *,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    hidden_dim: int,
    out_dim: int,
    table_rows: tuple[int, int, int],
) -> list[str]:
    reasons = unsupported_arch_reasons()
    lo, hi = META_RANGE
    if not lo <= meta_dim <= hi:
        reasons.append(f"{lo} <= meta_dim <= {hi} (got {meta_dim})")
    expected = {
        "obs_embed_dim": (obs_embed_dim, NOBS),
        "chan_embed_dim": (chan_embed_dim, NCH),
        "platform_embed_dim": (platform_embed_dim, NPL),
        "hidden_dim": (hidden_dim, HID),
        "out_dim": (out_dim, OUT),
        "table_rows": (tuple(table_rows), TABLE_ROWS),
    }
    reasons += [
        f"{k}=={want} (got {got})" for k, (got, want) in expected.items() if got != want
    ]
    return reasons


@torch.library.custom_op("healda::cutedsl_film_bwd", mutates_args=())
def cutedsl_film_bwd(
    grad_out: torch.Tensor,
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:  # fmt: skip
    """Returns (grad_embed, grad_chan_embed, grad_platform_embed, dw1, db1, dln_w, dln_b, dw2,
    db2), as ``fused_film_triton.fused_film_bwd``."""
    return film_bwd(
        grad_out, obs, float_meta, obs_type_id, channel, platform, embed_weight,
        chan_embed_weight, platform_embed_weight, w1, b1, ln_w, ln_b, w2, eps,
    )  # fmt: skip


@cutedsl_film_bwd.register_fake
def _(grad_out, obs, float_meta, obs_type_id, channel, platform, embed_weight,
      chan_embed_weight, platform_embed_weight, w1, b1, ln_w, ln_b, w2, b2, eps):  # fmt: skip
    f32 = dict(dtype=torch.float32)
    return (
        torch.empty_like(embed_weight),
        torch.empty_like(chan_embed_weight),
        torch.empty_like(platform_embed_weight),
        w1.new_empty(w1.shape, **f32),
        b1.new_empty(b1.shape, **f32),
        ln_w.new_empty(ln_w.shape, **f32),
        ln_b.new_empty(ln_b.shape, **f32),
        w2.new_empty(w2.shape, **f32),
        b2.new_empty(b2.shape, **f32),
    )

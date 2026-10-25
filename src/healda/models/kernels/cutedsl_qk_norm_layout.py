# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL q/k RMSNorm fused with the split of a packed qkv projection into attention layout.

``qkv`` is the (B, X, 3 * H * D) output of the qkv projection. The forward writes
``rms_norm(q)``, ``rms_norm(k)`` and ``v`` as (B, H, X, D), the layout flex_attention reads, in
one pass over ``qkv``. The norm has no weight and runs in fp32. The backward recomputes the
per-row statistics from ``qkv`` and writes the gradient in the packed layout.

1. ``_fwd_kernel`` / ``_bwd_kernel``: SIMT kernels. Each lane owns 8 channels (one 16 B vector)
   of a 128-channel head row, 16 lanes cover the row and a warp covers two consecutive tokens
   of one head, so packed-qkv accesses are 256 B segments and head-major accesses are 512 B
   runs. Row reductions are butterfly shuffles over the 16 lanes.
2. ``torch.library.custom_op`` wrappers, so the launches participate in autograd and fake-tensor
   tracing under ``torch.compile`` while staying opaque to inductor.
3. ``qk_norm_split_heads``, the entry point called by ``DiskSpatialAttention``.

Generated for 12 heads of 128 channels in bf16; see ``unsupported_reasons``.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from healda.models.kernels.cutedsl_arch import unsupported_arch_reasons

HN = 12
CD = 128
QKVW = 3 * HN * CD  # 4608
SUBC = 8  # channels owned by one lane
NSUB = CD // SUBC  # 16 lanes per head row
WARPS = 8
THREADS = WARPS * 32
ROWS_PER_CTA = THREADS // NSUB  # 16 tokens

F32 = cutlass.Float32
BF16 = cutlass.BFloat16


@cute.jit
def _row_sum(x):
    # Butterfly over the 16 lanes of one head row; every lane ends with the full sum.
    x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=2)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=4)
    x = x + cute.arch.shuffle_sync_bfly(x, offset=8)
    return x


@cute.jit
def _rstd(row, eps):
    ss = F32(0.0)
    for i in cutlass.range_constexpr(SUBC):
        f = row[i].to(F32)
        ss = ss + f * f
    return cute.math.rsqrt(_row_sum(ss) * (1.0 / CD) + eps)


@cute.jit
def _thread_row():
    # (head, token, batch, lane-in-row) of the calling thread.
    tidx, _, _ = cute.arch.thread_idx()
    head, bx, b = cute.arch.block_idx()
    return head, bx * ROWS_PER_CTA + tidx // NSUB, b, tidx % NSUB


@cute.kernel
def _fwd_kernel(
    qkv: cute.Tensor,
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    v_out: cute.Tensor,
    X: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    head, x, b, sub = _thread_row()
    if x < X:
        row = cute.make_rmem_tensor((SUBC,), BF16)
        out = cute.make_rmem_tensor((SUBC,), BF16)
        for slot in cutlass.range_constexpr(2):
            dst = q_out if cutlass.const_expr(slot == 0) else k_out
            cute.autovec_copy(qkv[(b, x, slot, head, sub, None)], row)
            rstd = _rstd(row, EPS)
            for i in cutlass.range_constexpr(SUBC):
                out[i] = (row[i].to(F32) * rstd).to(BF16)
            cute.autovec_copy(out, dst[(b, head, x, sub, None)])
        cute.autovec_copy(qkv[(b, x, 2, head, sub, None)], row)
        cute.autovec_copy(row, v_out[(b, head, x, sub, None)])


@cute.kernel
def _bwd_kernel(
    qkv: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dqkv: cute.Tensor,
    X: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    # y = x * rstd  =>  dx = rstd * (dy - y * mean(dy * y))
    head, x, b, sub = _thread_row()
    if x < X:
        row = cute.make_rmem_tensor((SUBC,), BF16)
        dy = cute.make_rmem_tensor((SUBC,), BF16)
        y = cute.make_rmem_tensor((SUBC,), F32)
        for slot in cutlass.range_constexpr(2):
            grad = dq if cutlass.const_expr(slot == 0) else dk
            cute.autovec_copy(qkv[(b, x, slot, head, sub, None)], row)
            cute.autovec_copy(grad[(b, head, x, sub, None)], dy)
            rstd = _rstd(row, EPS)
            dot = F32(0.0)
            for i in cutlass.range_constexpr(SUBC):
                y[i] = row[i].to(F32) * rstd
                dot = dot + dy[i].to(F32) * y[i]
            mean_dot = _row_sum(dot) * (1.0 / CD)
            for i in cutlass.range_constexpr(SUBC):
                row[i] = (rstd * (dy[i].to(F32) - y[i] * mean_dot)).to(BF16)
            cute.autovec_copy(row, dqkv[(b, x, slot, head, sub, None)])
        cute.autovec_copy(dv[(b, head, x, sub, None)], dy)
        cute.autovec_copy(dy, dqkv[(b, x, 2, head, sub, None)])


def _packed_view(mem, B, X):
    # (B, X, 3 * HN * CD) as (b, x, slot, head, lane, channel)
    return cute.make_tensor(
        mem.iterator,
        cute.make_layout(
            (B, X, 3, HN, NSUB, SUBC), stride=(X * QKVW, QKVW, HN * CD, CD, SUBC, 1)
        ),
    )


def _split_view(mem, B, X):
    # (B, HN, X, CD) as (b, head, x, lane, channel)
    return cute.make_tensor(
        mem.iterator,
        cute.make_layout(
            (B, HN, X, NSUB, SUBC), stride=(HN * X * CD, X * CD, CD, SUBC, 1)
        ),
    )


def _grid(B, X):
    return ((HN, (X + ROWS_PER_CTA - 1) // ROWS_PER_CTA, B), (THREADS, 1, 1))


@cute.jit
def _fwd_launch(
    qkv_mem: cute.Tensor,
    q_mem: cute.Tensor,
    k_mem: cute.Tensor,
    v_mem: cute.Tensor,
    B: cutlass.Constexpr,
    X: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    q, k, v = (_split_view(m, B, X) for m in (q_mem, k_mem, v_mem))
    grid, block = _grid(B, X)
    _fwd_kernel(_packed_view(qkv_mem, B, X), q, k, v, X, EPS).launch(
        grid=grid, block=block
    )


@cute.jit
def _bwd_launch(
    qkv_mem: cute.Tensor,
    dq_mem: cute.Tensor,
    dk_mem: cute.Tensor,
    dv_mem: cute.Tensor,
    dqkv_mem: cute.Tensor,
    B: cutlass.Constexpr,
    X: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    dq, dk, dv = (_split_view(m, B, X) for m in (dq_mem, dk_mem, dv_mem))
    grid, block = _grid(B, X)
    _bwd_kernel(
        _packed_view(qkv_mem, B, X), dq, dk, dv, _packed_view(dqkv_mem, B, X), X, EPS
    ).launch(grid=grid, block=block)


_COMPILED = {}


@torch.no_grad()
def _run(launcher, *tensors, eps: float):
    b, x_size, _ = tensors[0].shape
    key = (launcher, b, x_size, eps)
    fn = _COMPILED.get(key)
    if fn is None:
        args = [from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) for t in tensors]
        fn = cute.compile(launcher, *args, b, x_size, eps, options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    fn(*tensors)


# ------------------------------------------------------------------------------------------------
# autograd glue
# ------------------------------------------------------------------------------------------------


# Empty list == this configuration is supported.
def unsupported_reasons(
    *, num_heads: int, head_dim: int, norm_has_weight: bool
) -> list[str]:
    reasons = unsupported_arch_reasons()
    if num_heads != HN:
        reasons.append(f"num_heads=={HN} (got {num_heads})")
    if head_dim != CD:
        reasons.append(f"head_dim=={CD} (got {head_dim})")
    if norm_has_weight:
        reasons.append("a q/k RMSNorm without a weight")
    return reasons


def _split_shape(qkv: torch.Tensor) -> tuple[int, ...]:
    b, x_size, _ = qkv.shape
    return b, HN, x_size, CD


@torch.library.custom_op("healda::cutedsl_qk_norm_split_heads_fwd", mutates_args=())
def _op_fwd(
    qkv: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = (qkv.new_empty(_split_shape(qkv)) for _ in range(3))
    _run(_fwd_launch, qkv, q, k, v, eps=eps)
    return q, k, v


@_op_fwd.register_fake
def _(qkv, eps):
    return tuple(qkv.new_empty(_split_shape(qkv)) for _ in range(3))


@torch.library.custom_op("healda::cutedsl_qk_norm_split_heads_bwd", mutates_args=())
def _op_bwd(
    qkv: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, eps: float
) -> torch.Tensor:
    dqkv = torch.empty_like(qkv)
    _run(
        _bwd_launch, qkv, dq.contiguous(), dk.contiguous(), dv.contiguous(), dqkv,
        eps=eps,
    )  # fmt: skip
    return dqkv


@_op_bwd.register_fake
def _(qkv, dq, dk, dv, eps):
    return torch.empty_like(qkv)


def _setup_context(ctx, inputs, output):
    qkv, eps = inputs
    ctx.save_for_backward(qkv)
    ctx.eps = eps


def _backward_fn(ctx, dq, dk, dv):
    (qkv,) = ctx.saved_tensors
    return _op_bwd(qkv, dq, dk, dv, ctx.eps), None


_op_fwd.register_autograd(_backward_fn, setup_context=_setup_context)


def qk_norm_split_heads(
    qkv: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, X, 3*HN*CD) bf16 -> q, k, v of (B, HN, X, CD), with q and k RMS-normalised."""
    return _op_fwd(qkv.contiguous(), eps)

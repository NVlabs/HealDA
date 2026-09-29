# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch


class _ReduceScatterSum(torch.autograd.Function):
    # Use tensor-form collectives with explicit backward to avoid graph breaks from list-based collectives.
    @staticmethod
    def forward(ctx, tensor, group):
        world_size = torch.distributed.get_world_size(group)
        if tensor.shape[0] % world_size != 0:
            raise ValueError(
                f"Leading dimension {tensor.shape[0]} is not divisible by world size {world_size}."
            )

        shard_shape = (tensor.shape[0] // world_size, *tensor.shape[1:])
        shard = torch.empty(shard_shape, device=tensor.device, dtype=tensor.dtype)
        torch.distributed.reduce_scatter_tensor(
            shard,
            tensor.contiguous(),
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        ctx.group = group
        ctx.world_size = world_size
        return shard

    @staticmethod
    def backward(ctx, grad_output):
        grad_shape = (grad_output.shape[0] * ctx.world_size, *grad_output.shape[1:])
        grad_input = torch.empty(
            grad_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        torch.distributed.all_gather_into_tensor(
            grad_input, grad_output.contiguous(), group=ctx.group
        )
        return grad_input, None


def _compute_row_major_strides(shape):
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return strides


def scatter_mean(
    tensor: torch.Tensor,
    index: torch.Tensor,
    shape: tuple[int, ...],
    fill_value: float = float("nan"),
    group: torch.distributed.ProcessGroup | None = None,
    shard_dim: int = 0,
) -> torch.Tensor:
    """Scatter-mean values onto a multi-dimensional grid

    Args:
        tensor: [N, c] observation feature vectors
        index: [N, d] 1D grid cell index for each value in tensor
        shape: d-tuple. The size of the non value dimensions of the output array
        group: If provided, aggregates observations across all ranks in the group.
            The output tensor will be sharded across the ranks along ``shard_dim``.
        shard_dim: Dimension to shard the output along. Only used if ``group`` is provided.
            Must be < len(shape).

    Returns:
        aggregated: [*shape, c] with mean-aggregated values,
            filled with fill_value at grid cells with no values. sharded along ``shard_dim``.
        present: (*shape) bool mask indicating which grid cells have values
    """
    strides = _compute_row_major_strides(shape)
    # manually implement the dot product since matmul doesn't support long tensors on cuda
    # avoids RuntimeError: "addmv_impl_cuda" not implemented for 'Long'
    grid_indices_flat = (index * torch.tensor(strides, device=index.device)).sum(dim=-1)
    grid_size = math.prod(shape)

    device = tensor.device
    dtype = tensor.dtype
    embedding_dim = tensor.shape[1]

    values_sum = torch.zeros((grid_size, embedding_dim), device=device, dtype=dtype)
    counts = torch.zeros(grid_size, device=device, dtype=torch.int32)

    values_sum.index_add_(0, grid_indices_flat, tensor)

    ones = torch.ones_like(grid_indices_flat, dtype=torch.int32)
    counts.index_add_(0, grid_indices_flat, ones)

    values_sum = values_sum.view(*shape, embedding_dim)
    counts = counts.view(shape)

    if group is not None:
        world_size = torch.distributed.get_world_size(group)
        if shape[shard_dim] % world_size != 0:
            raise ValueError(
                f"Dimension {shard_dim} of shape {shape} (size {shape[shard_dim]}) is not divisible by world size {world_size}."
            )

        # Move shard_dim to front for reduce_scatter
        values_sum_permuted = torch.movedim(values_sum, shard_dim, 0)
        counts_permuted = torch.movedim(counts, shard_dim, 0)

        values_sum_shard = _ReduceScatterSum.apply(values_sum_permuted, group)

        counts_shard = torch.empty(
            (
                counts_permuted.shape[0] // world_size,
                *counts_permuted.shape[1:],
            ),
            device=counts_permuted.device,
            dtype=counts_permuted.dtype,
        )
        torch.distributed.reduce_scatter_tensor(
            counts_shard,
            counts_permuted.contiguous(),
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        values_sum = torch.movedim(values_sum_shard, 0, shard_dim)
        counts = torch.movedim(counts_shard, 0, shard_dim)

    # Compute mean: sum / count, with NaN for empty cells
    present = counts > 0  # [grid_size] bool
    counts_expanded = counts.unsqueeze(-1).to(dtype)  # [grid_size, 1]
    values_mean = values_sum / counts_expanded.clamp_min(1)
    values_mean = values_mean.masked_fill(~present.unsqueeze(-1), fill_value)

    return values_mean, present

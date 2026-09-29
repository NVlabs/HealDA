# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
from healda.models.scatter_mean import scatter_mean
from healda.utils.distributed import init

from tests.unit.test_distributed import requires_multi_gpu


def test_scatter_mean_basic():
    """Test scatter_mean with simple known values"""
    # Create test data:
    # - 5 observations with 2 features each
    # - Scatter into a 3x2 grid (6 cells total)
    # - Some cells will have multiple values (need averaging)
    # - Some cells will be empty (should get fill_value)

    tensor = torch.tensor(
        [
            [1.0, 10.0],  # goes to cell (0, 0)
            [2.0, 20.0],  # goes to cell (0, 1)
            [3.0, 30.0],  # goes to cell (0, 0) - same as first, should average
            [4.0, 40.0],  # goes to cell (2, 1)
            [5.0, 50.0],  # goes to cell (1, 0)
        ]
    )

    index = torch.tensor(
        [
            [0, 0],  # cell (0, 0)
            [0, 1],  # cell (0, 1)
            [0, 0],  # cell (0, 0)
            [2, 1],  # cell (2, 1)
            [1, 0],  # cell (1, 0)
        ]
    )

    shape = (3, 2)  # 3 rows, 2 columns

    aggregated, present = scatter_mean(tensor, index, shape)

    # Check shape
    assert aggregated.shape == (3, 2, 2)  # (3, 2) grid with 2 features
    assert present.shape == (3, 2)

    # Check aggregated values
    # Cell (0, 0): mean of [1.0, 10.0] and [3.0, 30.0] = [2.0, 20.0]
    assert torch.allclose(aggregated[0, 0], torch.tensor([2.0, 20.0]))

    # Cell (0, 1): [2.0, 20.0] (single value)
    assert torch.allclose(aggregated[0, 1], torch.tensor([2.0, 20.0]))

    # Cell (1, 0): [5.0, 50.0] (single value)
    assert torch.allclose(aggregated[1, 0], torch.tensor([5.0, 50.0]))

    # Cell (1, 1): empty, should be NaN
    assert torch.isnan(aggregated[1, 1]).all()

    # Cell (2, 0): empty, should be NaN
    assert torch.isnan(aggregated[2, 0]).all()

    # Cell (2, 1): [4.0, 40.0] (single value)
    assert torch.allclose(aggregated[2, 1], torch.tensor([4.0, 40.0]))

    # Check present mask
    expected_present = torch.tensor([[True, True], [True, False], [False, True]])
    assert torch.equal(present, expected_present)


def test_scatter_mean_custom_fill_value():
    """Test scatter_mean with a custom fill value"""
    tensor = torch.tensor([[1.0, 2.0]])
    index = torch.tensor([[0, 0]])
    shape = (2, 2)
    fill_value = -999.0

    aggregated, present = scatter_mean(tensor, index, shape, fill_value=fill_value)

    # Cell (0, 0) should have the value
    assert torch.allclose(aggregated[0, 0], torch.tensor([1.0, 2.0]))

    # Other cells should have the fill value
    assert (aggregated[0, 1] == fill_value).all()
    assert (aggregated[1, 0] == fill_value).all()
    assert (aggregated[1, 1] == fill_value).all()

    # Only (0, 0) should be present
    assert present[0, 0]
    assert not present[0, 1]
    assert not present[1, 0]
    assert not present[1, 1]


@requires_multi_gpu
def test_scatter_mean_distributed():
    """Test scatter_mean with distributed aggregation across ranks"""
    if not dist.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()

    # Create device mesh and get subgroup of size 2
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    rank = dist.get_rank()
    local_rank = rank % group_size  # Rank within the subgroup

    # Use CUDA for testing (required for NCCL backend)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Create a simple test case with 2 ranks
    # Each rank has different observations that should be aggregated
    # Grid shape: (4, 2) with 2 features
    # We'll shard along dim 0 (the first dimension with size 4)

    if local_rank == 0:
        # Rank 0: observations in cells (0,0), (0,1), (1,0)
        tensor = torch.tensor(
            [
                [1.0, 10.0],  # (0, 0)
                [2.0, 20.0],  # (0, 1)
                [3.0, 30.0],  # (1, 0)
            ],
            device=device,
        )
        index = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device)
    else:
        # Rank 1: observations in cells (2,0), (2,1), (3,1)
        tensor = torch.tensor(
            [
                [4.0, 40.0],  # (2, 0)
                [5.0, 50.0],  # (2, 1)
                [6.0, 60.0],  # (3, 1)
            ],
            device=device,
        )
        index = torch.tensor([[2, 0], [2, 1], [3, 1]], device=device)

    shape = (4, 2)  # Full shape before sharding
    shard_dim = 0  # Shard along first dimension

    # Call with distributed group
    aggregated, present = scatter_mean(
        tensor, index, shape, group=group, shard_dim=shard_dim
    )

    # After sharding, each rank should have shape (2, 2, 2)
    # Rank 0 gets rows 0-1, Rank 1 gets rows 2-3
    assert aggregated.shape == (2, 2, 2), f"Got shape {aggregated.shape}"
    assert present.shape == (2, 2), f"Got shape {present.shape}"

    if local_rank == 0:
        # Rank 0 should have:
        # Cell (0, 0): [1.0, 10.0]
        # Cell (0, 1): [2.0, 20.0]
        # Cell (1, 0): [3.0, 30.0]
        # Cell (1, 1): NaN
        assert torch.allclose(
            aggregated[0, 0], torch.tensor([1.0, 10.0], device=device)
        )
        assert torch.allclose(
            aggregated[0, 1], torch.tensor([2.0, 20.0], device=device)
        )
        assert torch.allclose(
            aggregated[1, 0], torch.tensor([3.0, 30.0], device=device)
        )
        assert torch.isnan(aggregated[1, 1]).all()

        expected_present = torch.tensor([[True, True], [True, False]], device=device)
        assert torch.equal(present, expected_present)
    elif local_rank == 1:
        # Rank 1 should have:
        # Cell (2, 0): [4.0, 40.0]
        # Cell (2, 1): [5.0, 50.0]
        # Cell (3, 0): NaN
        # Cell (3, 1): [6.0, 60.0]
        assert torch.allclose(
            aggregated[0, 0], torch.tensor([4.0, 40.0], device=device)
        )
        assert torch.allclose(
            aggregated[0, 1], torch.tensor([5.0, 50.0], device=device)
        )
        assert torch.isnan(aggregated[1, 0]).all()
        assert torch.allclose(
            aggregated[1, 1], torch.tensor([6.0, 60.0], device=device)
        )

        expected_present = torch.tensor([[True, True], [False, True]], device=device)
        assert torch.equal(present, expected_present)


@requires_multi_gpu
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_scatter_mean_distributed_overlap(use_torch_compile):
    """Test distributed scatter_mean where multiple ranks contribute to the same cells"""
    if not dist.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()

    # Create device mesh and get subgroup of size 2
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    rank = dist.get_rank()
    local_rank = rank % group_size  # Rank within the subgroup
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Both ranks contribute to overlapping cells - should average across ranks
    if local_rank == 0:
        tensor = torch.tensor([[1.0, 10.0], [3.0, 30.0]], device=device)
        index = torch.tensor([[0, 0], [1, 0]], device=device)
    else:
        tensor = torch.tensor([[5.0, 50.0], [7.0, 70.0]], device=device)
        index = torch.tensor([[0, 0], [1, 0]], device=device)  # Same cells as rank 0

    shape = (4, 2)
    shard_dim = 0

    def run_scatter_mean(local_tensor, local_index):
        return scatter_mean(
            local_tensor, local_index, shape, group=group, shard_dim=shard_dim
        )

    if use_torch_compile:
        run_scatter_mean = torch.compile(run_scatter_mean)

    aggregated, present = run_scatter_mean(tensor, index)

    assert aggregated.shape == (2, 2, 2)
    assert present.shape == (2, 2)

    if local_rank == 0:
        # Cell (0, 0): mean of [1.0, 10.0] and [5.0, 50.0] = [3.0, 30.0]
        assert torch.allclose(
            aggregated[0, 0], torch.tensor([3.0, 30.0], device=device)
        )
        # Cell (1, 0): mean of [3.0, 30.0] and [7.0, 70.0] = [5.0, 50.0]
        assert torch.allclose(
            aggregated[1, 0], torch.tensor([5.0, 50.0], device=device)
        )

        # Other cells should be NaN
        assert torch.isnan(aggregated[0, 1]).all()
        assert torch.isnan(aggregated[1, 1]).all()

        expected_present = torch.tensor([[True, False], [True, False]], device=device)
        assert torch.equal(present, expected_present)

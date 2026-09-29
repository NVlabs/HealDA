# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from healda import training_stats


@pytest.fixture(autouse=True)
def _reset_distributed_state():
    training_stats._sync_called = False
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
    training_stats._rank = 0
    training_stats._sync_device = None
    training_stats._sync_called = False
    training_stats._counters.clear()
    training_stats._cumulative.clear()


def test_dtensor_report_and_sync():
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized with world_size={world_size}")

    # Initialize training_stats for multiprocessing
    training_stats.init_multiprocessing(rank=rank, sync_device=device)

    # Create a device mesh for DTensor
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    # Create a DTensor with the correct dtype (float64 for _counter_dtype)
    local_tensor = torch.randn(3, device=device, dtype=torch.float64)
    dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])
    # The bug is that DTensors can end up in _counters through various operations
    # Let's directly inject one to simulate this scenario
    # This represents what happens when DTensor operations preserve the DTensor type
    metric_name = "test_metric"
    if metric_name not in training_stats._counters:
        training_stats._counters[metric_name] = dict()

    # Inject the DTensor into _counters
    # In the real bug, this happens through operations in report() that preserve DTensor type
    training_stats._counters[metric_name][device] = dtensor

    # Now trigger the sync which should cause the error
    # This calls _sync() which tries to do: delta.add_(counter.to(device))
    # where delta is a regular torch.Tensor but counter is a DTensor
    training_stats.default_collector.update()

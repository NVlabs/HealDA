#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifies that the pytorch dataloader returns data in deterministic order in multi worker scenarios

This requires in_order=True
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp
import pytest
from typing import Any
import random


class MockDataset(Dataset):
    """Mock dataset that uses a lambda function to generate data."""

    def __init__(self, size: int = 1000):
        """
        Initialize mock dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Any:
        if idx >= self.size:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )
        time.sleep(random.random() * 0.001)
        return idx


@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
@pytest.mark.parametrize("context", ["spawn", "fork", "forkserver"])
@pytest.mark.parametrize("dataloader_in_order", [True, False])
def test_data_loader_deterministic(num_workers, dataloader_in_order, context):
    """Main function to run the bootstrap test."""
    # Check available CPU cores
    max_workers = mp.cpu_count()

    # Create mock dataset with simple lambda function
    dataset_size = 32
    dataset = MockDataset(size=dataset_size)

    # Test different worker configurations
    batch_size = 1

    if num_workers > max_workers:
        # print(f"Skipping {num_workers} workers (exceeds available cores)")
        pytest.skip()

    if not dataloader_in_order:
        pytest.xfail()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,  # Set to True if using GPU
        drop_last=False,
        in_order=dataloader_in_order,
        multiprocessing_context=context if num_workers > 0 else None,
    )

    batches = list(dataloader)
    all_data = torch.cat(batches)
    in_order = torch.all(all_data == torch.arange(len(dataset))).item()
    assert in_order

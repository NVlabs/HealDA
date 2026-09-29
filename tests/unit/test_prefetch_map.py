# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from healda.datasets.prefetch_map import prefetch_map


def test_prefetch_map_basic_functionality():
    """Test basic async dataloader functionality with simple data."""
    # Create simple test data using range
    data = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Simple transform that doubles the data
    def transform(x):
        return 2 * x

    # Create async loader
    async_loader = prefetch_map(data, transform)
    assert list(async_loader) == list(range(0, 20, 2))


def test_prefetch_map_error_handling():
    """Test error handling when transform raises an exception."""
    data = list(range(4))  # [0, 1, 2, 3]

    def failing_transform(x):
        raise ValueError("Test error")

    async_loader = prefetch_map(data, failing_transform)

    # Should raise the exception from the background thread
    with pytest.raises(ValueError, match="Test error"):
        list(async_loader)


def test_prefetch_map_uses_stream_dependency_without_host_sync():
    data = [torch.ones(4, device="cuda")]
    loader = prefetch_map(data, lambda value: value * 3)

    torch.cuda.set_sync_debug_mode("error")
    try:
        output = next(iter(loader))
        consumed = output + 1
    finally:
        torch.cuda.set_sync_debug_mode("default")
        loader._stop()

    torch.testing.assert_close(consumed.cpu(), torch.full((4,), 4.0))

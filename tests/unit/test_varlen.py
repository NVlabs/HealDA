# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from healda.varlen import idx_to_lengths, lengths_to_idx


def test_lengths_to_idx_basic():
    lengths = torch.tensor([[[2, 3], [1, 0]]])  # S=1, B=2, T=2
    idx = lengths_to_idx(lengths)
    # flat indices: 0×2, 1×3, 2×1, 3×0
    expected = torch.tensor([0, 0, 1, 1, 1, 2])
    assert torch.equal(idx, expected)


def test_lengths_to_idx_multi_sensor():
    lengths = torch.tensor([[[2], [3]], [[1], [4]]])  # S=2, B=2, T=1
    idx = lengths_to_idx(lengths)
    # flat indices: 0×2, 1×3, 2×1, 3×4
    expected = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    assert torch.equal(idx, expected)


def test_lengths_to_idx_empty():
    lengths = torch.zeros((2, 3, 4), dtype=torch.long)
    idx = lengths_to_idx(lengths)
    assert idx.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lengths_to_idx_with_output_size_does_not_synchronize():
    lengths = torch.tensor([[[2, 3], [1, 0]]], device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        idx = lengths_to_idx(lengths, output_size=6)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(idx.cpu(), torch.tensor([0, 0, 1, 1, 1, 2]))


def test_lengths_to_idx_unravel():
    lengths = torch.tensor([[[2, 3], [1, 0]]])  # S=1, B=2, T=2
    idx = lengths_to_idx(lengths)
    s, b, t = torch.unravel_index(idx, lengths.shape)
    assert torch.equal(s, torch.tensor([0, 0, 0, 0, 0, 0]))
    assert torch.equal(b, torch.tensor([0, 0, 0, 0, 0, 1]))
    assert torch.equal(t, torch.tensor([0, 0, 1, 1, 1, 0]))


def test_idx_to_lengths_roundtrip():
    """idx_to_lengths is the inverse of lengths_to_idx."""
    lengths = torch.tensor([[[2, 3], [1, 0]], [[1, 0], [2, 1]]], dtype=torch.long)
    idx = lengths_to_idx(lengths)
    assert torch.equal(idx_to_lengths(idx, lengths.shape), lengths)


def test_idx_to_lengths_subsample():
    """After subsampling, window counts equal the number of selected obs per window."""
    lengths = torch.tensor([[[3, 2], [4, 1]]], dtype=torch.long)  # S=1, B=2, T=2
    idx = lengths_to_idx(lengths)  # 10 obs total, flat indices 0-3

    # Select first obs from each window (one per window, except window 3 which has 1)
    selected = torch.tensor([0, 3, 4, 9])  # from windows 0, 1, 1, 3
    new_lengths = idx_to_lengths(idx[selected], lengths.shape)
    expected = torch.tensor([[[1, 2], [0, 1]]], dtype=torch.long)
    assert torch.equal(new_lengths, expected)


def test_idx_to_lengths_empty():
    lengths = torch.zeros((2, 3, 4), dtype=torch.long)
    idx = lengths_to_idx(lengths)
    assert torch.equal(idx_to_lengths(idx, lengths.shape), lengths)


def test_batch_idx_via_mod():
    """Verify the % (B*T) pattern produces correct bt indices."""
    lengths = torch.tensor([[[2, 3], [1, 0]], [[1, 0], [2, 1]]])  # S=2, B=2, T=2
    _, B, T = lengths.shape
    idx = lengths_to_idx(lengths) % (B * T)
    # Each obs gets its bt index in [0, B*T)
    assert idx.max() < B * T
    assert idx.numel() == lengths.sum().item()

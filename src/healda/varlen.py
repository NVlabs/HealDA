# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for variable-length sequence indexing."""

import math

import torch


def lengths_to_idx(
    lengths: torch.Tensor, output_size: int | None = None
) -> torch.Tensor:
    """Return a flat index for each observation.

    For a lengths tensor of any shape, returns a (n_obs,) tensor where each
    observation gets the flat index of its window in lengths.

    Use torch.unravel_index(lengths_to_idx(lengths), lengths.shape) to get
    per-dimension indices.

    Pass ``output_size`` (= n_obs) when the caller already knows it. Without it
    repeat_interleave must sum ``lengths`` on the host to size its output, which
    is a device sync on the thread feeding the GPU.
    """
    ind = torch.arange(lengths.numel(), device=lengths.device)
    return ind.repeat_interleave(lengths.flatten(), output_size=output_size)


def idx_to_lengths(idx: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Reconstruct a lengths tensor from flat window indices.

    Inverse of lengths_to_idx: given a (n_obs,) tensor of flat window indices
    (as returned by lengths_to_idx), count how many observations fall in each
    window and return a lengths tensor of the requested shape.

    Typical use: recompute lengths after subsampling observations.
        new_lengths = idx_to_lengths(lengths_to_idx(obs.lengths)[selected], obs.lengths.shape)
    """
    counts = torch.bincount(idx.long(), minlength=math.prod(shape))
    return counts.reshape(shape).to(idx.dtype)

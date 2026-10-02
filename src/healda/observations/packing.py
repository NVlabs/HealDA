# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack observations into per-pixel contiguous groups for backbone obs attention.

Each observation carries a flat pixel index ``flat_idx = batch_time_idx * npix + pix``.
Sorting by that index groups a pixel's observations contiguously, which lets the ragged
pixel cross-attention kernel address each pixel's tokens by prefix sums (``cu_seqlens_k``).

For bounded integer keys a counting sort is O(N) with a single atomic-scatter pass, faster
than argsort's multi-pass radix sort. Within-bucket order is nondeterministic (warp
scheduling), which is fine: attention is permutation-invariant over a pixel's key/value
tokens.
"""

import dataclasses

import torch

from healda.observations import types
from healda.varlen import lengths_to_idx

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _counting_sort_scatter(
        flat_idx_ptr,
        sorted_order_ptr,
        bucket_offsets_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        pixel = tl.load(flat_idx_ptr + offs, mask=mask).to(tl.int64)
        pos = tl.atomic_add(bucket_offsets_ptr + pixel, 1, mask=mask)
        tl.store(sorted_order_ptr + pos.to(tl.int32), offs.to(tl.int32), mask=mask)


def _counts_by_pixel(flat_idx: torch.Tensor, total_pixels: int) -> torch.Tensor:
    """Per-pixel counts, without a host sync.

    torch.bincount sizes its output from max(input), which it must read on the
    host -- a device sync even when minlength is given. total_pixels is already
    the exact size, so scatter_add_ into a preallocated buffer avoids it.
    """
    idx = flat_idx.long()
    counts = torch.zeros(total_pixels, dtype=torch.int64, device=flat_idx.device)
    return counts.scatter_add_(0, idx, torch.ones_like(idx))


def counting_sort_and_pack(flat_idx: torch.Tensor, total_pixels: int):
    """Counting sort + atomic scatter. Returns (sorted_order int32, counts int64)."""
    assert HAS_TRITON, "counting_sort_and_pack requires Triton"
    N = flat_idx.shape[0]
    device = flat_idx.device

    counts = _counts_by_pixel(flat_idx, total_pixels)
    bucket_offsets = torch.zeros(total_pixels, dtype=torch.int64, device=device)
    bucket_offsets[1:] = counts[:-1].cumsum(0)

    sorted_order = torch.empty(N, dtype=torch.int32, device=device)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _counting_sort_scatter[grid](flat_idx, sorted_order, bucket_offsets, N, BLOCK=BLOCK)
    return sorted_order, counts


def sort_and_pack(flat_idx: torch.Tensor, total_pixels: int):
    """Sort observations by flat pixel index.

    Uses the Triton counting sort when available (see the module docstring for why), else
    argsort. Returns ``(sorted_order int32 permutation, counts int64 per-pixel counts)``.
    Callers apply ``sorted_order`` once to reorder the per-observation tensors and keep only
    ``counts``, as ``cu_seqlens_k`` prefix sums; the permutation is not retained.
    """
    if HAS_TRITON and flat_idx.is_cuda:
        return counting_sort_and_pack(flat_idx, total_pixels)
    counts = _counts_by_pixel(flat_idx, total_pixels)
    sorted_order = flat_idx.argsort().int()
    return sorted_order, counts


# Per-observation fields of UnifiedObservation that must be reordered together so
# each token's payload stays attached to its pixel. All are required and length
# ``n_obs``.
_PER_OBS_FIELDS = (
    "obs",
    "time",
    "float_metadata",
    "pix",
    "local_channel",
    "local_platform",
    "obs_type",
    "global_channel",
    "global_platform",
)


def build_pixel_group_map(
    cu_seqlens_k: torch.Tensor, thresh_mult: float = 2.0
) -> types.PixelGroupMap:
    """Pack consecutive small pixels into shared kernel programs for obs attention.

    The ragged attention runs one kernel program per pixel; for the many tiny
    pixels the fixed per-program cost (W_k/W_v load, prologue, launch latency)
    dominates the actual math. Pairing two small pixels into one program loads
    those weights once and cuts the program count -- the kernel's binding
    constraint.

    A pixel is "small" when its count < ``thresh_mult`` * median(nonzero counts);
    median-relative (not an absolute cap) so it keeps grouping when the typical
    pixel is large. Empty pixels are dropped (their output is already zero and
    the backward skips them). Pure function of ``cu_seqlens_k``, so it is built
    once per batch and reused by every layer and both passes.

    Returns:
        ``PixelGroupMap`` with two int32 tensors on the input device:
        ``program_ptr`` has shape ``[n_programs + 1]`` and
        ``program_pixels`` has shape ``[n_nonzero_pixels]``. Program ``p`` owns
        pixels ``program_pixels[program_ptr[p]:program_ptr[p + 1]]``.

    Example: counts ``[5, 0, 3, 4, 200]`` (nonzero median 4 via torch.median,
    threshold 8) -> large=[4], small=[0, 2, 3] -> programs ``[[4], [0, 2], [3]]``.
    Large pixels go first, each in its own program, then small pixels are paired
    (odd one left solo), giving ``program_ptr = [0, 1, 3, 4]`` and
    ``program_pixels = [4, 0, 2, 3]``. So program 0 owns pixel [4], program 1
    owns pixels [0, 2], program 2 owns pixel [3]. Pixel 1 is empty and is dropped.
    """
    device = cu_seqlens_k.device
    counts = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)
    nonzero_pixels = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if nonzero_pixels.numel() == 0:  # frame with no observations
        return types.PixelGroupMap(
            program_ptr=torch.zeros(1, dtype=torch.int32, device=device),
            program_pixels=torch.empty(0, dtype=torch.int32, device=device),
        )
    nonzero_counts = counts[nonzero_pixels].float()
    threshold = nonzero_counts.median() * thresh_mult
    is_small = nonzero_counts < threshold
    small_pixels = nonzero_pixels[is_small]
    large_pixels = nonzero_pixels[~is_small]

    # Large pixels stay solo; small pixels are taken two at a time, with a final
    # solo program if an odd one is left over.
    num_pairs = small_pixels.numel() // 2
    has_leftover = small_pixels.numel() % 2 == 1
    program_sizes = torch.cat(
        [
            torch.ones(large_pixels.numel(), dtype=torch.int64, device=device),
            torch.full((num_pairs,), 2, dtype=torch.int64, device=device),
            torch.ones(int(has_leftover), dtype=torch.int64, device=device),
        ]
    )
    program_ptr = torch.zeros(
        program_sizes.numel() + 1, dtype=torch.int32, device=device
    )
    program_ptr[1:] = torch.cumsum(program_sizes, 0).to(torch.int32)
    program_pixels = torch.cat(
        [large_pixels.to(torch.int32), small_pixels.to(torch.int32)]
    )
    return types.PixelGroupMap(
        program_ptr=program_ptr.contiguous(),
        program_pixels=program_pixels.contiguous(),
    )


def pack_observations_by_pixel(
    obs: "types.UnifiedObservation",
    pixel_order: str = "hpxpadxy",
    *,
    build_group_map: bool = True,
) -> "types.UnifiedObservation":
    """Sort a ``UnifiedObservation`` into per-pixel contiguous token groups.

    Returns a new ``UnifiedObservation`` whose per-observation fields are reordered by flat
    pixel index, with ``attention_packing`` describing the layout (``counts`` per pixel and
    ``cu_seqlens_k`` prefix sums over the full grid) that the backbone pixel cross-attention
    consumes; the scatter/embed path does not need it.

    ``pixel_order`` is recorded, not applied: ``pix`` already arrives in that order, and the
    model checks the two agree before attending.
    """
    assert obs.lengths is not None, "packing requires lengths"
    order, packing = pixel_packing(
        obs.pix,
        obs.lengths,
        obs.hpx_level,
        pixel_order,
        build_group_map=build_group_map,
    )
    packed = obs
    if order is not None:
        reordered = {name: getattr(obs, name)[order] for name in _PER_OBS_FIELDS}
        packed = dataclasses.replace(obs, **reordered)
    return dataclasses.replace(packed, attention_packing=packing)


def pixel_packing(
    pix: torch.Tensor,
    lengths: torch.Tensor,
    hpx_level: int,
    pixel_order: str,
    *,
    build_group_map: bool,
) -> tuple[torch.Tensor | None, "types.AttentionPacking"]:
    """The order that makes each pixel's observations contiguous, and the attention
    layout of the sorted observations.

    ``pix`` and ``lengths`` describe the observations in their incoming order; the order
    is None when there are none. Returned rather than applied so a caller can reorder
    raw per-row inputs before deriving per-row features: the result equals reordering
    the features, without holding both orders of them. ``TransformV2`` does this for
    ``float_metadata``, (n_obs, 50) float32.
    """
    npix = 12 * 4**hpx_level
    _, batch_size, time_size = lengths.shape
    total_pixels = batch_size * time_size * npix
    if pix.shape[0] == 0:
        order = None
        counts = torch.zeros(total_pixels, dtype=torch.int64, device=pix.device)
    else:
        batch_idx = lengths_to_idx(lengths, output_size=pix.shape[0]) % (
            batch_size * time_size
        )
        flat_idx = (batch_idx * npix + pix.long()).int()
        sorted_order, counts = sort_and_pack(flat_idx, total_pixels)
        order = sorted_order.long()
    cu_seqlens_k = torch.zeros(
        counts.shape[0] + 1, dtype=torch.int32, device=counts.device
    )
    cu_seqlens_k[1:] = counts.cumsum(0).to(torch.int32)
    packing = types.AttentionPacking(
        counts=counts,
        cu_seqlens_k=cu_seqlens_k,
        npix=npix,
        hpx_level=hpx_level,
        is_packed=True,
        group_map=build_pixel_group_map(cu_seqlens_k) if build_group_map else None,
        pixel_order=pixel_order,
    )
    return order, packing

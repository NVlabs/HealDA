# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pack_observations_by_pixel`` (the TransformV2 attention-prepack step)."""

import dataclasses

import pytest
import torch

from healda.observations.types import AttentionPacking, UnifiedObservation
from healda.observations.packing import (
    pack_observations_by_pixel,
    pixel_packing,
    sort_and_pack,
)
from healda.observations.preprocessing import features_v2

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_obs(window_of_obs, pix, hpx_level, batch, time, device):
    n = len(window_of_obs)
    lengths = torch.zeros(1, batch, time, dtype=torch.long)
    for w in window_of_obs:
        b, t = divmod(w, time)
        lengths[0, b, t] += 1
    # Each field stores the obs's original index, so the sort permutation is traceable.
    ident = torch.arange(n)
    return UnifiedObservation(
        obs=ident.float().to(device),
        time=ident.to(device),
        float_metadata=ident.float().unsqueeze(1).repeat(1, 3).to(device),
        pix=torch.tensor(pix, dtype=torch.long, device=device),
        local_channel=ident.to(device),
        local_platform=ident.to(device),
        obs_type=ident.to(device),
        global_channel=ident.to(device),
        global_platform=ident.to(device),
        hpx_level=hpx_level,
        lengths=lengths.to(device),
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_pack_single_window_alignment(device):
    hpx_level = 0  # npix = 12
    npix = 12 * 4**hpx_level
    pix = [7, 0, 7, 3, 7, 0, 11]
    window_of_obs = [0] * len(pix)  # single (batch=1, time=1) window
    obs = _make_obs(window_of_obs, pix, hpx_level, batch=1, time=1, device=device)

    packed = pack_observations_by_pixel(obs)
    ap = packed.attention_packing
    n = len(pix)

    assert ap.is_packed and ap.npix == npix
    assert ap.hpx_level == hpx_level
    assert ap.counts.sum().item() == n
    assert ap.cu_seqlens_k.shape == (npix + 1,)

    # pix is non-decreasing (tokens grouped by pixel).
    packed_pix = packed.pix.cpu()
    assert torch.all(packed_pix[1:] >= packed_pix[:-1])

    # All fields share one permutation: each row matches its stored identity index.
    idx = packed.obs.long().cpu()
    assert sorted(idx.tolist()) == list(range(n))
    assert torch.equal(packed.float_metadata[:, 0].long().cpu(), idx)
    assert torch.equal(packed.local_channel.cpu(), idx)
    assert torch.equal(packed.local_platform.cpu(), idx)
    assert torch.equal(packed.obs_type.cpu(), idx)
    assert torch.equal(packed.global_channel.cpu(), idx)
    assert torch.equal(packed.global_platform.cpu(), idx)
    assert torch.equal(packed_pix, torch.tensor(pix)[idx])

    # cu_seqlens_k slices recover exactly one pixel's tokens per segment.
    cu = ap.cu_seqlens_k.cpu()
    for p in range(npix):
        seg = packed_pix[cu[p] : cu[p + 1]]
        assert torch.all(seg == p)
        assert ap.counts.cpu()[p].item() == seg.numel()


@pytest.mark.parametrize("device", _DEVICES)
def test_pack_multi_window(device):
    hpx_level = 0
    npix = 12 * 4**hpx_level
    batch, time = 2, 1
    # First 4 obs in window 0, next 3 in window 1.
    window_of_obs = [0, 0, 0, 0, 1, 1, 1]
    pix = [5, 1, 5, 9, 2, 2, 8]
    obs = _make_obs(window_of_obs, pix, hpx_level, batch, time, device)

    packed = pack_observations_by_pixel(obs)
    ap = packed.attention_packing
    n = len(pix)

    assert ap.counts.numel() == batch * time * npix
    assert ap.counts.sum().item() == n

    # Flat key per token must be globally sorted, and counts must match bincount.
    idx = packed.obs.long().cpu()
    win = torch.tensor(window_of_obs)[idx]
    flat = win * npix + packed.pix.cpu()
    assert torch.all(flat[1:] >= flat[:-1])
    expected_flat = torch.tensor(window_of_obs) * npix + torch.tensor(pix)
    assert torch.equal(
        ap.counts.cpu(),
        torch.bincount(expected_flat, minlength=batch * time * npix),
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_pack_isolates_samples(device):
    # batch>1 isolation: every obs must land in its own sample's pixel bucket.
    # Packing recovers (b,t) from row order, so a wrong order leaks across samples.
    hpx_level = 0
    npix = 12 * 4**hpx_level
    batch, time = 2, 1
    # 5 obs in sample 0, 3 in sample 1; pixels overlap so a leak would collide.
    window_of_obs = [0, 0, 0, 0, 0, 1, 1, 1]
    pix = [3, 7, 3, 1, 7, 3, 7, 1]
    obs = _make_obs(window_of_obs, pix, hpx_level, batch, time, device)

    packed = pack_observations_by_pixel(obs)
    ap = packed.attention_packing
    counts = ap.counts.cpu()

    assert counts.numel() == batch * time * npix
    assert counts.sum().item() == len(pix)

    # Map each token to its bucket and to its origin sample (via stored index).
    bucket = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    bucket_sample = bucket // npix  # flat = (b*T+t)*npix + pix, here T=1 -> //npix
    origin_sample = torch.tensor(window_of_obs)[packed.obs.long().cpu()]
    assert torch.equal(bucket_sample, origin_sample)

    # Per-sample totals preserved.
    assert counts[:npix].sum().item() == window_of_obs.count(0)
    assert counts[npix:].sum().item() == window_of_obs.count(1)


@pytest.mark.parametrize("device", _DEVICES)
def test_pack_empty(device):
    hpx_level = 0
    npix = 12
    obs = _make_obs([], [], hpx_level, batch=1, time=1, device=device)
    packed = pack_observations_by_pixel(obs)
    ap = packed.attention_packing
    assert ap.is_packed
    assert ap.counts.shape == (npix,)
    assert ap.counts.sum().item() == 0
    assert ap.cu_seqlens_k.shape == (npix + 1,)
    assert torch.count_nonzero(ap.cu_seqlens_k) == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_sort_and_pack_invariants(device):
    # sort_and_pack returns a permutation + per-pixel counts.
    torch.manual_seed(0)
    total_pixels = 32
    flat_idx = torch.randint(0, total_pixels, (200,), dtype=torch.int32).to(device)
    sorted_order, counts = sort_and_pack(flat_idx, total_pixels)
    sorted_order = sorted_order.cpu()
    counts = counts.cpu()

    assert sorted(sorted_order.tolist()) == list(range(200))
    assert torch.equal(
        counts, torch.bincount(flat_idx.cpu().long(), minlength=total_pixels)
    )
    sorted_flat = flat_idx.cpu()[sorted_order.long()]
    assert torch.all(sorted_flat[1:] >= sorted_flat[:-1])


def test_attention_packing_no_sorted_order():
    # sorted_order is transient (applied in the packer) and must not be retained.
    fields = {f.name for f in dataclasses.fields(AttentionPacking)}
    assert "sorted_order" not in fields
    packing = AttentionPacking(
        counts=torch.tensor([0, 2, 1]),
        cu_seqlens_k=torch.tensor([0, 0, 2, 3], dtype=torch.int32),
        npix=3,
        hpx_level=0,
    )
    moved = packing.to(device="cpu")
    assert moved.is_packed and moved.npix == 3


def test_pack_can_skip_triton_group_map():
    obs = _make_obs([0, 0], [1, 2], hpx_level=0, batch=1, time=1, device="cpu")
    packed = pack_observations_by_pixel(obs, build_group_map=False)
    assert packed.attention_packing.group_map is None


@pytest.mark.parametrize("device", _DEVICES)
def test_featurizing_sorted_rows_matches_sorting_featurized_rows(device):
    """TransformV2 sorts raw rows before deriving float_metadata; per-row features make
    that equal to sorting the features."""
    g = torch.Generator().manual_seed(0)
    n, hpx_level = 257, 1
    pix = torch.randint(0, 12 * 4**hpx_level, (n,), generator=g)
    window_of_obs = torch.randint(0, 4, (n,), generator=g).sort().values.tolist()
    obs = _make_obs(
        window_of_obs, pix.tolist(), hpx_level, batch=2, time=2, device=device
    )

    satellite = torch.rand(n, generator=g) < 0.5
    nan = torch.tensor(float("nan"))
    raw = {
        "time": 1_735_776_000 * 10**9
        + torch.randint(-(3 * 3600) * 10**9, 3 * 3600 * 10**9, (n,), generator=g),
        "lon": torch.rand(n, generator=g) * 360,
        "lat": torch.rand(n, generator=g) * 180 - 90,
        "height": torch.where(satellite, nan, torch.rand(n, generator=g) * 1e4),
        "pressure": torch.where(satellite, nan, torch.rand(n, generator=g) * 1e3),
        "scan_angle": torch.where(satellite, torch.rand(n, generator=g) * 50, nan),
        "sat_zenith_angle": torch.where(
            satellite, torch.rand(n, generator=g) * 60, nan
        ),
        "sol_zenith_angle": torch.where(
            satellite, torch.rand(n, generator=g) * 180, nan
        ),
    }
    raw = {name: value.to(device) for name, value in raw.items()}
    target = torch.full((n,), 1_735_776_000, dtype=torch.int64, device=device)

    order, _ = pixel_packing(
        obs.pix, obs.lengths, hpx_level, "hpxpadxy", build_group_map=False
    )
    sorted_first = features_v2.compute_unified_metadata(
        target[order], **{name: value[order] for name, value in raw.items()}
    )
    featurized_first = features_v2.compute_unified_metadata(target, **raw)[order]
    assert torch.equal(sorted_first, featurized_first)

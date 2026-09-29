# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flex-disk spatial attention: order resolution, and the kernel against dense SDPA.

An order mismatch scrambles the geometry without changing a shape, so assert on physical
position rather than on two order strings matching.
"""

import math

import pytest
import torch

from healda.observations.types import healpix_pixel_order
from healda.models.attention import (
    DiskSpatialAttention,
    healpix_disk_block_mask,
)
from healda.models.dit import pipeline_token_order

earth2grid = pytest.importorskip("earth2grid")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention needs a GPU"
)


def _positions(order: str, level: int) -> torch.Tensor:
    grid = earth2grid.healpix.Grid(level=level, pixel_order=healpix_pixel_order(order))
    lat = torch.as_tensor(grid.lat, dtype=torch.float64).deg2rad()
    lon = torch.as_tensor(grid.lon, dtype=torch.float64).deg2rad()
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1)


def test_the_two_orders_are_genuinely_different():
    """Guards the guard: if these ever coincide, the order tests below are vacuous."""
    a, b = _positions("nest", 4), _positions("hpxpadxy", 4)
    offset = torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))
    assert float(offset.median()) > 10.0


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("te", "hpxpadxy"),
        ("diffusers", "hpxpadxy"),
        ("disk:21", "hpxpadxy"),
        ("disk-nest:21", "nest"),
    ],
)
def test_backend_string_resolves_pixel_order(backend, expected):
    assert pipeline_token_order(backend) == expected


@requires_gpu
def test_disk_mask_is_symmetric_and_keeps_the_diagonal():
    mask = healpix_disk_block_mask(12 * 4**3, 21.0, "cuda", "nest")
    dense = mask.to_dense().squeeze()[: 12 * 4**3, : 12 * 4**3]
    assert torch.equal(dense, dense.T)
    assert dense.diagonal().all()
    # Degenerate masks (all-on or all-off) would make the parity test below meaningless.
    assert 0.0 < dense.float().mean().item() < 1.0


@requires_gpu
@pytest.mark.parametrize("pixel_order", ["nest", "hpxpadxy"])
def test_disk_matches_dense_sdpa(pixel_order):
    """The kernel computes the same thing as dense attention under the same mask."""
    level, heads, dim_head, radius = 3, 4, 32, 21.0
    n = 12 * 4**level
    torch.manual_seed(0)

    module = (
        DiskSpatialAttention(
            query_dim=heads * dim_head,
            heads=heads,
            dim_head=dim_head,
            radius_deg=radius,
            pixel_order=pixel_order,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(1, 1, n, heads * dim_head, device="cuda", dtype=torch.bfloat16)

    qkv = module.qkv(x.reshape(1, n, -1))
    q, k, v = qkv.view(1, n, 3, heads, dim_head).permute(2, 0, 3, 1, 4)
    pos = _positions(pixel_order, level).cuda()
    allowed = (pos @ pos.T) >= math.cos(math.radians(radius))
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed
    )
    reference = module.proj(reference.transpose(1, 2).reshape(1, n, -1)).reshape(
        1, 1, n, -1
    )

    torch.testing.assert_close(module(x), reference, rtol=2e-2, atol=2e-2)


@requires_gpu
def test_dit_runs_end_to_end_on_the_disk_nest_backend():
    """The whole stack in nest: patch embed, disk attention, patch decode."""
    from healda.models.dit import DiT

    torch.manual_seed(0)
    net = DiT(
        num_layers=1,
        level_in=4,
        level_model=3,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        temporal_attention=True,
        attention_backend="disk-nest:21",
    ).cuda()

    assert net.spatial_token_order == "nest"
    assert net._disk_attentions, "the flex modules were not collected"

    n, t = 1, 1
    # bf16 as in production: the pinned flex tile is a 16-bit-only config.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(
            torch.randn(n, 3, t, net.domain.numel(), device="cuda"),
            torch.zeros([n], device="cuda"),
            class_labels=torch.empty([n, 0], device="cuda").int(),
            day_of_year=torch.ones([n, t], device="cuda"),
            second_of_day=torch.ones([n, t], device="cuda"),
        )
    assert out.out.shape == (n, 3, t, net.domain.numel())
    assert torch.isfinite(out.out).all()


@requires_gpu
def test_fp32_is_rejected_with_a_readable_message():
    module = DiskSpatialAttention(
        query_dim=128, heads=4, dim_head=32, radius_deg=21.0, pixel_order="nest"
    ).cuda()
    with pytest.raises(TypeError, match="bf16"):
        module(torch.randn(1, 1, 12 * 4**3, 128, device="cuda"))

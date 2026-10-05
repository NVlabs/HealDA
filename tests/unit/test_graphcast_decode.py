# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mesh2Grid decode: edge geometry, the pixel-order contract, and the edge-block factoring."""

import pytest
import torch

pytest.importorskip("earth2grid")
pytest.importorskip("scipy")

from healda.observations.types import healpix_pixel_order  # noqa: E402
from healda.models.graphcast_decode import (  # noqa: E402
    GraphCastDecoder,
    build_bilinear_weights,
    build_mesh_edges,
)

LEVEL_MESH, K, NLAT, NLON = 2, 4, 8, 16


def _decoder(**kw):
    args = dict(
        in_channels=4 * 8,
        aux_channels=2,
        out_channels=3,
        level_in=1,
        level_mesh=LEVEL_MESH,
        k=K,
        pixel_order="nest",
        nlat=NLAT,
        nlon=NLON,
        remat=False,
    )
    args.update(kw)
    return GraphCastDecoder(**args)


def test_edge_geometry_is_finite_and_orthonormal_at_the_poles():
    idx, efeat, tgt = build_mesh_edges(LEVEL_MESH, K, "nest", NLAT, NLON)
    assert idx.shape == (NLAT * NLON, K)
    assert int(idx.min()) >= 0 and int(idx.max()) < 12 * 4**LEVEL_MESH
    # Rows 0 and NLAT-1 are the exact poles, where the East axis degenerates.
    assert torch.isfinite(efeat).all()
    e, n, u, dist = efeat.unbind(-1)
    torch.testing.assert_close(e**2 + n**2 + u**2, dist**2, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        tgt.norm(dim=-1), torch.ones(NLAT * NLON), rtol=1e-5, atol=1e-5
    )


def test_pixel_order_changes_the_graph_and_nest_picks_the_real_nearest():
    import earth2grid

    nest, _, tgt = build_mesh_edges(LEVEL_MESH, K, "nest", NLAT, NLON)
    hpx, _, _ = build_mesh_edges(LEVEL_MESH, K, "hpxpadxy", NLAT, NLON)
    assert not torch.equal(nest, hpx), "the two orders must not index the same nodes"

    # Assert on physical position, not on the label: the nest neighbour really is nearest.
    grid = earth2grid.healpix.Grid(
        level=LEVEL_MESH, pixel_order=healpix_pixel_order("nest")
    )
    lat = torch.as_tensor(grid.lat, dtype=torch.float64).deg2rad()
    lon = torch.as_tensor(grid.lon, dtype=torch.float64).deg2rad()
    pts = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1)
    target = 37
    cos = pts @ tgt[target].double()
    assert int(cos.argmax()) == int(nest[target, 0])


def test_bilinear_weights_are_a_partition_of_unity():
    idx, weight = build_bilinear_weights(LEVEL_MESH, NLAT, NLON, "nest")
    assert idx.shape[0] == weight.shape[0] == NLAT * NLON
    assert int(idx.min()) >= 0 and int(idx.max()) < 12 * 4**LEVEL_MESH
    torch.testing.assert_close(
        weight.sum(-1), torch.ones(NLAT * NLON), rtol=1e-5, atol=1e-5
    )


def test_a_finer_mesh_requires_nest():
    with pytest.raises(ValueError, match="only contiguous in nest"):
        _decoder(pixel_order="hpxpadxy")
    # Equal levels need no child ordering, so any order is fine.
    _decoder(level_in=LEVEL_MESH, in_channels=8, pixel_order="hpxpadxy")


def test_unpack_must_consume_the_whole_latent():
    with pytest.raises(ValueError, match="divisible"):
        _decoder(in_channels=4 * 8 + 2)


def test_edge_block_matches_the_naive_per_edge_form():
    """The concat trick must be the per-edge MLP it factors, not an approximation of it."""
    torch.manual_seed(0)
    decoder = _decoder()
    bt = 2
    z = torch.randn(bt, 12 * 4**1, 32)
    aux = torch.randn(bt, NLAT * NLON, 2)

    out = decoder(z, aux)

    u = decoder._unpack_and_lift(z)
    # The baseline reads the reduced (level_in) nodes, not the mesh.
    qu = u.reshape(bt, -1, decoder.patch, decoder.h).mean(2)
    bu = qu[:, decoder.bidx.reshape(-1)].reshape(
        bt, NLAT * NLON, decoder.bidx.shape[-1], decoder.h
    )
    q_dyn = (decoder.bweight.unsqueeze(0).unsqueeze(-1) * bu).sum(2)
    q = decoder.target_norm(q_dyn + decoder.target(aux))

    uj = u[:, decoder.idx.reshape(-1)].reshape(bt, NLAT * NLON, K, decoder.h)
    per_edge = (
        decoder.lin_src(uj)
        + decoder.lin_dst(q).unsqueeze(2)
        + decoder.lin_e(decoder.efeat)
    )
    m = decoder.ln(decoder.w2(torch.nn.functional.silu(per_edge))).sum(2)
    reference = decoder.out(q + decoder.node(torch.cat([q, m], -1)))
    reference = reference.transpose(1, 2).reshape(bt, -1, NLAT, NLON)

    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)


def test_remat_does_not_change_values_or_gradients():
    torch.manual_seed(0)
    z = torch.randn(2, 12 * 4**1, 32, requires_grad=True)
    aux = torch.randn(2, NLAT * NLON, 2)

    grads = []
    outs = []
    for remat in (False, True):
        torch.manual_seed(0)
        decoder = _decoder(remat=remat)
        z.grad = None
        out = decoder(z, aux)
        out.sum().backward()
        outs.append(out.detach().clone())
        grads.append(z.grad.clone())

    torch.testing.assert_close(outs[0], outs[1])
    torch.testing.assert_close(grads[0], grads[1])


def test_mesh_children_carry_nothing_independent_of_their_parent():
    """Why the baseline reads at level_in: a parent's children hold no extra information."""
    torch.manual_seed(0)
    decoder = _decoder()
    n_tokens = 12 * 4**1
    z = torch.randn(1, n_tokens, 32)

    u = decoder._unpack_and_lift(z)
    group = u.shape[1] // n_tokens

    perturbed = z.clone()
    perturbed[0, 0] += 1.0
    changed = (decoder._unpack_and_lift(perturbed) - u).abs().sum(-1)[0] > 1e-6
    # Exactly token 0's own children move: no child sees any other token.
    assert changed[:group].all()
    assert not changed[group:].any()


def test_forward_shape_and_finiteness():
    torch.manual_seed(0)
    decoder = _decoder()
    z = torch.randn(2, 12 * 4**1, 32)
    aux = torch.randn(2, NLAT * NLON, 2)
    out = decoder(z, aux)
    assert out.shape == (2, 3, NLAT, NLON)
    assert torch.isfinite(out).all()


def test_fp32_output_leaves_the_head_out_of_bf16_autocast():
    torch.manual_seed(0)
    decoder = _decoder()
    z = torch.randn(1, 12 * 4, 4 * 8)
    aux = torch.randn(1, NLAT * NLON, 2)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        rounded = decoder(z, aux)
        decoder.fp32_output = True
        exact = decoder(z, aux)
    assert rounded.dtype == torch.bfloat16
    assert exact.dtype == torch.float32
    torch.testing.assert_close(exact, rounded.float(), rtol=2e-2, atol=2e-2)

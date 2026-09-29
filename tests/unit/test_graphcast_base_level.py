# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""base_level on GraphCastDecoder: the 99ch arm's baseline, without changing the checkpoint.

It interpolates the baseline on the mesh level where ours averages the children back to the
parent first. Must add no parameters, or the two settings are not one checkpoint.
"""

import pytest
import torch

from healda.models.graphcast_decode import GraphCastDecoder

NLAT, NLON = 33, 64
IN_CH, AUX_CH, OUT_CH = 64, 5, 7
LEVEL_IN, LEVEL_MESH = 2, 3


def build(base_level=None):
    return GraphCastDecoder(
        in_channels=IN_CH,
        aux_channels=AUX_CH,
        out_channels=OUT_CH,
        level_in=LEVEL_IN,
        level_mesh=LEVEL_MESH,
        k=4,
        pixel_order="nest",
        nlat=NLAT,
        nlon=NLON,
        base_level=base_level,
        remat=False,
    )


def inputs(bt=2):
    # aux is aux_channels wide and carries the per-target position.
    return (
        torch.randn(bt, 12 * 4**LEVEL_IN, IN_CH),
        torch.randn(bt, NLAT * NLON, AUX_CH),
    )


def test_same_parameters_either_way():
    """No new weights, so a checkpoint moves between the two settings."""
    a, b = build(), build(LEVEL_MESH)
    ka = {n: tuple(p.shape) for n, p in a.named_parameters()}
    kb = {n: tuple(p.shape) for n, p in b.named_parameters()}
    assert ka == kb
    b.load_state_dict(a.state_dict())  # the real check


def test_mesh_level_reads_the_children_not_their_mean():
    dec = build(LEVEL_MESH)
    assert dec.base_group == 1  # mean over 1 == identity
    assert int(dec.bidx.max()) >= 12 * 4**LEVEL_IN  # addresses the finer level
    dec_default = build()
    assert dec_default.base_group == 4
    assert int(dec_default.bidx.max()) < 12 * 4**LEVEL_IN


def test_the_two_settings_actually_differ():
    """Same weights, different baseline geometry -> different output.

    Without this the test suite would pass on a base_level that silently did nothing.
    """
    a, b = build(), build(LEVEL_MESH)
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    z, aux = inputs()
    with torch.no_grad():
        assert not torch.allclose(a(z, aux), b(z, aux))


def test_default_is_unchanged():
    """base_level=None must be exactly today's behaviour."""
    torch.manual_seed(0)
    a = build()
    torch.manual_seed(0)
    b = build(LEVEL_IN)
    z, aux = inputs()
    a.eval(), b.eval()
    with torch.no_grad():
        torch.testing.assert_close(a(z, aux), b(z, aux))


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        build(LEVEL_MESH + 1)
    with pytest.raises(ValueError):
        build(LEVEL_IN - 1)


def test_position_reaches_the_decoder_through_aux():
    """aux is the only position signal, so moving a point's aux position must change its
    output."""
    dec = build().eval()
    z, aux = inputs()
    aux2 = aux.clone()
    aux2[:, :, 0] += 1.0
    with torch.no_grad():
        assert not torch.allclose(dec(z, aux), dec(z, aux2))

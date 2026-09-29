# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

from healda.models.dit import AdaLayerNormZero

C = 64  # hidden channels
E = 128  # conditioning embedding channels


def _layernorm_ref(x):
    # Affine-free LayerNorm over the last dim with the module's eps.
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def _seed_linear(mod):
    # Default adaLN-Zero often zero-inits the projection; randomize it so the
    # reference can't pass trivially with all-zero shift/scale/gate.
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(0.1 * torch.randn_like(p))


def test_dit_block_setting_n_blocks_2():
    torch.manual_seed(0)
    batch, time, npix = 2, 3, 48
    ada = AdaLayerNormZero(embedding_dim=C, emb_channels=E, n_blocks=2)
    _seed_linear(ada)

    x = torch.randn(batch, time, npix, C)
    emb = torch.randn(batch, E)

    normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada(x, emb)

    # Caller-expected chunk order, each broadcast over (time, npix) -> [batch, 1, 1, C].
    params = (p[:, None, None, :] for p in ada.linear(emb).chunk(6, dim=1))
    (
        expected_shift_msa,
        expected_scale_msa,
        expected_gate_msa,
        expected_shift_mlp,
        expected_scale_mlp,
        expected_gate_mlp,
    ) = params
    expected_normed = _layernorm_ref(x) * (1 + expected_scale_msa) + expected_shift_msa

    for name, got, want in [
        ("normed", normed, expected_normed),
        ("gate_msa", gate_msa, expected_gate_msa),
        ("shift_mlp", shift_mlp, expected_shift_mlp),
        ("scale_mlp", scale_mlp, expected_scale_mlp),
        ("gate_mlp", gate_mlp, expected_gate_mlp),
    ]:
        assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
        torch.testing.assert_close(got, want, msg=lambda m, n=name: f"{n}: {m}")


def test_obs_attn_setting_n_blocks_1():
    torch.manual_seed(1)
    # The obs-attention path flattens batch and time into one leading dim.
    batch_time, npix = 6, 48
    ada = AdaLayerNormZero(embedding_dim=C, emb_channels=E, n_blocks=1)
    _seed_linear(ada)

    x = torch.randn(batch_time, npix, C)
    emb = torch.randn(batch_time, E)

    normed, gate = ada(x, emb)

    shift, scale, expected_gate = (
        p[:, None, :] for p in ada.linear(emb).chunk(3, dim=1)
    )
    expected_normed = _layernorm_ref(x) * (1 + scale) + shift

    assert normed.shape == (batch_time, npix, C)
    assert gate.shape == (batch_time, 1, C)
    torch.testing.assert_close(normed, expected_normed)
    torch.testing.assert_close(gate, expected_gate)


def test_modulation_is_per_sample_and_broadcast():
    # Modulation must depend only on the sample's emb: constant across the spatial
    # dim, and identical samples must yield identical modulation.
    torch.manual_seed(2)
    batch_time, npix = 4, 20
    ada = AdaLayerNormZero(embedding_dim=C, emb_channels=E, n_blocks=1)
    _seed_linear(ada)

    emb = torch.randn(batch_time, E)
    emb[1] = emb[0]  # duplicate sample 0 into sample 1
    x = torch.randn(batch_time, npix, C)

    _, gate = ada(x, emb)

    # Constant across npix (broadcast dim is size 1 by construction, but confirm
    # the values are genuinely shared, not accidentally spatially varying).
    assert gate.shape[1] == 1
    # Duplicated emb -> identical gate.
    torch.testing.assert_close(gate[0], gate[1])
    # Distinct emb -> distinct gate.
    assert not torch.allclose(gate[0], gate[2])


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_output_count_matches_n_blocks(n_blocks):
    ada = AdaLayerNormZero(embedding_dim=C, emb_channels=E, n_blocks=n_blocks)
    x = torch.randn(2, 5, C)
    emb = torch.randn(2, E)
    out = ada(x, emb)
    # normed + one gate + (3*n_blocks - 3) extra modulation tensors.
    assert len(out) == 3 * n_blocks - 1

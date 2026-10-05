# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-space transforms must round-trip exactly and read siblings correctly."""

import numpy as np
import pytest
import torch

from healda.datasets.da import state_transforms as st

CHANNELS = ["tcwv", "tas", "tcw"]
CONFIG = "era5_104ch"


def _state(backend):
    x = np.array([[[20.0]], [[288.0]], [[20.5]]], dtype=np.float64)  # (C, 1, 1)
    return torch.as_tensor(x) if backend == "torch" else x


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_tcw_becomes_the_residual_and_round_trips(backend):
    x = _state(backend)
    original = x.clone() if backend == "torch" else x.copy()
    model = st.to_model_space(x, CHANNELS, CONFIG, channel_axis=0)
    assert np.array_equal(np.asarray(x), np.asarray(original))
    # tcw -> tcw - tcwv; the sibling and unrelated channels are untouched
    assert float(np.asarray(model[2]).ravel()[0]) == pytest.approx(0.5)
    assert float(np.asarray(model[0]).ravel()[0]) == pytest.approx(20.0)
    assert float(np.asarray(model[1]).ravel()[0]) == pytest.approx(288.0)

    back = st.to_physical_space(model, CHANNELS, CONFIG, channel_axis=0)
    assert float(np.asarray(back[2]).ravel()[0]) == pytest.approx(20.5)
    assert np.allclose(np.asarray(back), np.asarray(x))


def test_inverse_reads_the_predicted_sibling_not_the_input():
    """tcwv is predicted directly, so the inverse must use the model's tcwv."""
    model = np.array([[[25.0]], [[288.0]], [[0.5]]])  # model predicted tcwv=25
    back = st.to_physical_space(model, CHANNELS, CONFIG, channel_axis=0)
    assert float(np.asarray(back[2]).ravel()[0]) == pytest.approx(25.5)


def test_config_without_transforms_is_a_passthrough():
    x = _state("numpy")
    assert np.array_equal(
        st.to_model_space(x, CHANNELS, "era5_99ch", channel_axis=0), x
    )


def test_missing_sibling_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="tcwv"):
        st.to_model_space(_state("numpy")[1:], ["tas", "tcw"], CONFIG, channel_axis=0)


def test_restore_fill_only_touches_outside_the_domain():
    from healda.datasets.da.state_masks import CHANNEL_DOMAIN_FILL, HPX64, state_mask

    channels = ["sst", "tas"]
    valid = state_mask("ocean", HPX64)
    x = np.zeros((2, valid.size))
    x[0] = 275.0  # sst everywhere, including land
    x[1] = 288.0  # tas is unmasked and must be untouched
    out = st.restore_fill(x, channels, HPX64, channel_axis=0)

    assert np.all(out[0][valid] == 275.0), "ocean sst must be untouched"
    assert np.all(
        out[0][~valid] == CHANNEL_DOMAIN_FILL["sst"]
    ), "land sst must be filled"
    assert np.all(out[1] == 288.0), "an unmasked channel must be untouched"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_clamp_physical_clips_bounded_channels_only(backend):
    channels = ["Q850", "sic", "tcc", "T850", "tcwv"]
    x = np.array([-1e-4, 1.12, -0.1, -5.0, np.nan], dtype=np.float32)[:, None, None]
    state = torch.as_tensor(x) if backend == "torch" else x
    out = np.asarray(st.clamp_physical(state, channels))
    np.testing.assert_array_equal(out[:4, 0, 0], [0.0, 1.0, 0.0, -5.0])
    assert np.isnan(out[4, 0, 0])
    assert np.asarray(state)[1, 0, 0] == np.float32(1.12)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from healda.observations import types


def test_build_empty_batch_uses_expected_shape():
    batch = types.empty_batch(
        batch_gpu=2,
        out_channels=3,
        condition_channels=3,
        time_length=4,
        x_size=5,
        device=torch.device("cpu"),
    )

    assert batch["target"].shape == (2, 3, 4, 5)
    assert batch["condition"].shape == (2, 3, 4, 5)
    assert batch["second_of_day"].shape == (2, 4)
    assert batch["day_of_year"].shape == (2, 4)
    assert batch["labels"].shape == (2, 0)
    assert batch["timestamp"].shape == (2, 4)
    assert "background" not in batch


def test_build_empty_batch_adds_background_when_requested():
    batch = types.empty_batch(
        batch_gpu=2,
        out_channels=3,
        condition_channels=3,
        time_length=4,
        x_size=5,
        device=torch.device("cpu"),
        background_channels=3,
    )

    assert batch["background"].shape == (2, 3, 4, 5)


def test_build_empty_batch_rejects_non_positive_x_size():
    with pytest.raises(ValueError, match="x_size must be positive"):
        types.empty_batch(
            batch_gpu=1,
            out_channels=1,
            condition_channels=1,
            time_length=1,
            x_size=0,
            device=torch.device("cpu"),
        )

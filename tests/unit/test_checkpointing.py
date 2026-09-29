# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from healda.training.checkpoint import Checkpoint
from healda import models
from healda.config.models import ModelConfigV1


def _assert_state_dict_equal(d1, d2):
    assert set(d1) == set(d2)
    for k in d1:
        assert d1[k].equal(d2[k])


def test_checkpointing(tmp_path):
    config = ModelConfigV1(architecture="dit-test")
    model = models.get_model(config)

    state_dict = model.state_dict()
    with Checkpoint(tmp_path / "test.checkpoint", "w") as checkpoint:
        checkpoint.write_model(model)
        checkpoint.write_model_config(config)

    with Checkpoint(tmp_path / "test.checkpoint", "r") as checkpoint:
        model = checkpoint.read_model()
        assert config == checkpoint.read_model_config()
        _assert_state_dict_equal(model.state_dict(), state_dict)

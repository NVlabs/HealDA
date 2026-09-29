# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import torch

from healda.datasets.da.transform import TransformV2
import healda.cli.train as train_mod


def test_training_loop_reuses_the_dataset_transform(monkeypatch):
    seen = {}
    dataset_transform = object.__new__(TransformV2)

    class FakeDataset:
        transform = dataset_transform

        def __len__(self):
            return 0

    def fake_build(name, **kwargs):
        seen["kwargs"] = kwargs
        return FakeDataset()

    monkeypatch.setattr(train_mod, "build_training_dataset", fake_build)
    loop = dataclasses.replace(train_mod.LOOPS["v2-ufs-hpx64"])
    assert loop.task is not None
    loop._device_mesh = SimpleNamespace(get_local_rank=lambda dim: 0, shape=(1, 1, 1))

    dataset = loop.get_dataset(train=True)
    assert seen["kwargs"]["transform_options"] == loop._transform_options
    stage = loop._device_stage_for_dataset(dataset)
    assert stage.keywords["transform"] is dataset_transform
    loop.dumps()


def test_mask_training_device_stage_only_moves_tensors():
    loop = train_mod.TrainingLoop()
    loop.device = torch.device("cpu")
    assert loop.mask_training

    batch = {
        "target": torch.arange(24).reshape(1, 2, 3, 4),
        "condition": torch.arange(24, 48).reshape(1, 2, 3, 4),
        "timestamp": torch.tensor([[1, 2, 3]]),
    }
    out = loop._device_stage_for_dataset(object())(batch)

    for key in batch:
        assert torch.equal(out[key], batch[key])

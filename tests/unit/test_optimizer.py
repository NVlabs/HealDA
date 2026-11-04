# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
from torch import nn

from healda.cli.train import LOOPS


def test_adamw_without_a_recipe():
    loop = LOOPS["v2-nnja-latlon-dense"]
    assert loop.recipe is None
    net = nn.ModuleDict(
        {
            "blocks": nn.Linear(16, 16),
            "norm": nn.LayerNorm(16),
            "embed_v2_patch": nn.Linear(4, 16),
        }
    )
    opt = loop.get_optimizer(net.named_parameters())
    assert type(opt) is torch.optim.AdamW
    names = {id(p): n for n, p in net.named_parameters()}
    groups = [
        ([names[id(p)] for p in g["params"]], g["lr"], g["weight_decay"], g["betas"])
        for g in opt.param_groups
    ]
    betas = (0.9, 0.95)
    assert groups == [
        (["blocks.weight"], 2e-4, 0.05, betas),
        (["blocks.bias", "norm.weight", "norm.bias"], 2e-4, 0.0, betas),
        (["embed_v2_patch.weight"], 1e-4, 0.05, betas),
        (["embed_v2_patch.bias"], 1e-4, 0.0, betas),
    ]
    assert all(g["eps"] == 1e-8 for g in opt.param_groups)


def test_default_schedule_is_warmup_flat_cosine():
    loop = LOOPS["v2-nnja-latlon-dense"]
    warmup, flat, decay = loop.lr_rampup_img, loop.flat_imgs, loop.decay_imgs
    floor = loop.lr_min / loop.lr
    assert loop.lr_scale(warmup / 2) == pytest.approx(0.5)
    assert loop.lr_scale(warmup + flat) == 1.0
    mid = 0.5 * (1 - floor) * (1 + math.cos(math.pi * 0.5)) + floor
    assert loop.lr_scale(warmup + flat + decay / 2) == pytest.approx(mid)
    assert loop.lr_scale(warmup + flat + decay) == floor


def test_default_schedule_values():
    # Values of the warmup/cosine schedule before it moved into lr_scale.
    loop = LOOPS["v2-nnja-latlon-dense"]
    expected = {
        0: 0.0,
        5000: 0.0001,
        10_000: 0.0002,
        1_010_000: 0.00015000000000000001,
        3_010_000: 0.0,
        3_011_000: 0.0,
    }
    assert {n: loop.lr_scale(n) * loop.lr for n in expected} == expected

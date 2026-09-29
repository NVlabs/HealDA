# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from healda.utils.regridding import ConservativeRegridder


def test_conservative_regridder_ignores_partial_nan_blocks_and_fills_empty_blocks():
    regridder = ConservativeRegridder.__new__(ConservativeRegridder)
    torch.nn.Module.__init__(regridder)
    regridder.regridder = torch.nn.Identity()
    regridder.coarsen_factor = 4
    values = torch.tensor([[1.0, float("nan"), 3.0, float("nan")] + [float("nan")] * 4])

    out = regridder(values, ignore_nan=True, fill_value=-1.0)

    torch.testing.assert_close(out, torch.tensor([[2.0, -1.0]]))

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

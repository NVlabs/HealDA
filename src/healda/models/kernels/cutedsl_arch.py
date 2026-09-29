# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute-capability gate shared by the CuTe DSL kernels. torch-only, so it imports
without nvidia-cutlass-dsl."""

import torch

# The kernels are generated and tuned for Blackwell only (B200, B300).
MIN_CAPABILITY = (10, 0)


def unsupported_arch_reasons() -> list[str]:
    if not torch.cuda.is_available():
        return []
    cap = torch.cuda.get_device_capability()
    if cap >= MIN_CAPABILITY:
        return []
    return [
        f"compute capability >= {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} "
        f"(got {cap[0]}.{cap[1]})"
    ]

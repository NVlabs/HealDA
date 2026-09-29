# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("healda")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]

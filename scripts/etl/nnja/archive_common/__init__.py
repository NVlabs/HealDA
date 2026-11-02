# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor-agnostic machinery shared by the NNJA archive packages.

Contract-specific behaviour (footer provenance, validation) is supplied by the
sensor package.
"""

from .archive_writer import ArchiveDayWriter, WriteSummary, writer_options

__all__ = ["ArchiveDayWriter", "WriteSummary", "writer_options"]

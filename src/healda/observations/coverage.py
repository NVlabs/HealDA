# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which obs archive a config reads, and how far it extends."""

from healda.datasets.base import DatasetMetadata
from healda.observations import sensors_nnja
from healda.observations.loaders.ufs import UFS_OBS_COVERAGE


def obs_coverage(obs_config) -> DatasetMetadata:
    """Coverage of the obs archive this config reads."""
    if getattr(obs_config, "use_nnja_sat", False):
        return sensors_nnja.OBS_COVERAGE
    return UFS_OBS_COVERAGE

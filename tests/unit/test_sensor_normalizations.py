# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every sensor loads its means and stds from a CSV, not the zeros/ones placeholder.

``SensorConfig`` uses ``means=0, stds=1`` only when ``stats_csv=None``.
"""

import numpy as np
import pytest

from healda.observations.sensors import SENSOR_CONFIGS as UFS_SENSORS
from healda.observations.sensors_nnja import SENSOR_CONFIGS as NNJA_SENSORS


def _is_placeholder(config) -> bool:
    means, stds = np.asarray(config.means), np.asarray(config.stds)
    return bool(np.all(means == 0) and np.all(stds == 1))


# conv-plevel declares stats_csv=None: its normalizations are per pressure level.
DELIBERATELY_UNNORMALIZED = {"conv-plevel"}


@pytest.mark.parametrize("name", sorted(set(UFS_SENSORS) - DELIBERATELY_UNNORMALIZED))
def test_ufs_sensors_have_real_normalizations(name):
    assert not _is_placeholder(
        UFS_SENSORS[name]
    ), f"{name} has zero means and unit stds"


@pytest.mark.parametrize("name", sorted(NNJA_SENSORS))
def test_nnja_sensors_have_real_normalizations(name):
    assert not _is_placeholder(NNJA_SENSORS[name])


def test_the_two_vocabularies_stay_separate():
    """ufs and nnja read different CSV sets; one must never silently serve the other."""
    assert "cris" in NNJA_SENSORS and "cris-fsr" in UFS_SENSORS


@pytest.mark.parametrize("name", sorted(DELIBERATELY_UNNORMALIZED))
def test_unnormalized_sensors_say_so_explicitly(name):
    assert UFS_SENSORS[name].stats_csv is None

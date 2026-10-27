# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from healda.cli.train import TrainingLoop
from healda.config.models import ModelConfigV1, ObsConfig


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_nnja_conv": True}, "use_nnja_conv requires use_nnja_sat=True"),
        ({"use_nnja_satwnd": True}, "use_nnja_satwnd requires use_nnja_conv=True"),
        ({"nnja_wind_dropout": -0.1}, "nnja_wind_dropout must be in"),
        ({"nnja_wind_dropout": 1.1}, "nnja_wind_dropout must be in"),
        ({"nnja_dropout_scope": "batch"}, "nnja_dropout_scope must be"),
        ({"nnja_gpsro_saids": "nonsense"}, "nnja_gpsro_saids must be"),
        ({"nnja_max_quality_mark": 16}, "nnja_max_quality_mark must be"),
    ],
)
def test_observation_config_rejects_invalid_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ObsConfig(**kwargs)


def test_observation_config_canonicalizes_serialized_sequences():
    config = ObsConfig(
        drop_sensors=["airs-pca"],
        drop_report_types=[240, 241],
        drop_obs_families=["amv"],
        nnja_sensors=["atms"],
    )

    assert config.drop_sensors == ("airs-pca",)
    assert config.drop_report_types == (240, 241)
    assert config.drop_obs_families == ("amv",)
    assert config.nnja_sensors == ("atms",)


def test_legacy_gpsro_said_values_migrate_rather_than_raise():
    """Old checkpoint configs carry the pre-rename spellings; they must still load."""
    assert ObsConfig(nnja_gpsro_saids="all").nnja_gpsro_saids == "full"
    assert ObsConfig(nnja_gpsro_saids="ufs2024").nnja_gpsro_saids == "legacy"


@pytest.mark.parametrize(("nside", "level"), [(128, 7), (64, 6), (None, None)])
def test_checkpoints_with_a_thinning_nside_load_at_its_level(nside, level):
    stored = json.dumps({"obs_config": {"nnja_thin_nside": nside}})
    assert ModelConfigV1.loads(stored).obs_config.nnja_sat_thin_hpx_level == level
    assert TrainingLoop.loads(stored).obs_config.nnja_sat_thin_hpx_level == level

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NNJA satellite plus conventional: one channel id space."""

import asyncio
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from healda.config.models import ObsConfig
from healda.datasets.base import NormalizationStats
from healda.observations.loaders import combined as combined
from healda.observations import sensors, sensors_nnja
from healda.observations.preprocessing.conv_plevel import expanded_local_channel
from healda.observations.loaders.ufs import LOCAL_CHANNEL_ID
from healda.datasets.da.transform import TransformV2
from healda.observations.schema import GLOBAL_CHANNEL_ID, SENSOR_ID


def test_conv_is_appended_without_moving_a_satellite_channel():
    for sensor in sensors_nnja.SENSOR_ORDER:
        assert combined.SENSOR_OFFSET[sensor] == sensors_nnja.SENSOR_OFFSET[sensor]
        assert (
            combined.SENSOR_NAME_TO_ID[sensor] == sensors_nnja.SENSOR_NAME_TO_ID[sensor]
        )
    assert combined.SENSOR_OFFSET[combined.CONV_SENSOR] == sensors_nnja.NCHANNEL


def test_channel_table_covers_every_id_exactly_once():
    table = combined.channel_table()
    ids = table[GLOBAL_CHANNEL_ID.name].to_numpy(zero_copy_only=False)
    assert np.array_equal(ids, np.arange(combined.NCHANNEL))

    sensor_id = table[SENSOR_ID.name].to_numpy(zero_copy_only=False)
    for name in combined.SENSOR_ORDER:
        block = ids[sensor_id == combined.SENSOR_NAME_TO_ID[name]]
        assert block.min() == combined.SENSOR_OFFSET[name]
        assert block.size == combined.channel_count(name)


def test_conv_stats_survive_the_rebase():
    # Placeholder 0/1 stats are what build_conv_plevel_channel_table returns when it is
    # given no normalizations, so their absence is the check that real ones were passed.
    conv = combined.channel_table().slice(combined.SENSOR_OFFSET[combined.CONV_SENSOR])
    mean = conv["mean"].to_numpy(zero_copy_only=False)
    stddev = conv["stddev"].to_numpy(zero_copy_only=False)
    assert not np.any((mean == 0.0) & (stddev == 1.0))
    assert conv["name"].to_pylist()[0] == "gps_angle_1000"
    assert conv["name"].to_pylist()[-1] == "ps"


def test_transform_numbers_sensors_the_way_the_loader_did():
    # The two vocabularies disagree for every sensor but atms, and GSI has no "cris" at
    # all, so a transform left on the GSI map mislabels rows the combined loader emitted.
    transform = TransformV2(
        sensors=list(combined.SENSOR_ORDER),
        use_nnja_sat=True,
        target_normalization=NormalizationStats(center=np.zeros(1), scales=np.ones(1)),
    )
    assert transform._ordered_sensor_ids.tolist() == [
        combined.SENSOR_NAME_TO_ID[name] for name in combined.SENSOR_ORDER
    ]
    assert sorted(transform._platform_luts) == sorted(
        combined.SENSOR_NAME_TO_ID.values()
    )
    for name in combined.SENSOR_ORDER:
        lut = transform._platform_luts[combined.SENSOR_NAME_TO_ID[name]]
        for local, global_id in enumerate(combined.platform_ids(name)):
            assert lut[global_id] == local


def test_sensor_identities_survive_json():
    # Checkpointing writes model.json with json.dumps(dataclasses.asdict(model_config)),
    # which rejects a numpy scalar. pandas hands back int64, so the ids are cast on read.
    identity = {
        name: {
            "sensor_id": combined.SENSOR_NAME_TO_ID[name],
            "nchannel": combined.channel_count(name),
            "platform_ids": combined.platform_ids(name),
        }
        for name in combined.SENSOR_ORDER
    }
    json.dumps(identity)
    for name, fields in identity.items():
        assert type(fields["sensor_id"]) is int, name
        assert type(fields["nchannel"]) is int, name
        assert all(type(p) is int for p in fields["platform_ids"]), name


def test_the_gsi_vocabulary_cannot_serve_the_combined_loader():
    # Guards the flag itself: without it the transform silently renumbers six sensors and
    # raises on the seventh, which is how this reached a training run.
    assert "cris" not in sensors.SENSOR_NAME_TO_ID
    disagree = [
        name
        for name in combined.SENSOR_ORDER
        if sensors.SENSOR_NAME_TO_ID.get(name) != combined.SENSOR_NAME_TO_ID[name]
    ]
    assert set(disagree) == set(combined.SENSOR_ORDER) - {"atms"}


def test_rebase_restates_conv_identity_from_its_local_channel():
    # A conv row arrives numbered in the GSI vocabulary; only the two identity columns
    # move, and they move to whatever local_channel_id already said.
    local = np.array([0, 45, 91], dtype=np.uint16)
    gsi = pa.table(
        {
            GLOBAL_CHANNEL_ID.name: pa.array(
                sensors.SENSOR_OFFSET["conv-plevel"] + local, GLOBAL_CHANNEL_ID.type
            ),
            SENSOR_ID.name: pa.array(
                np.full(local.size, sensors.SENSOR_NAME_TO_ID["conv-plevel"]),
                SENSOR_ID.type,
            ),
            LOCAL_CHANNEL_ID.name: pa.array(local, LOCAL_CHANNEL_ID.type),
            "Observation": pa.array([1.0, 2.0, 3.0], pa.float32()),
        }
    )

    rebased = combined._rebase(gsi, local)
    assert rebased[GLOBAL_CHANNEL_ID.name].to_pylist() == [
        combined.SENSOR_OFFSET[combined.CONV_SENSOR] + int(value) for value in local
    ]
    assert set(rebased[SENSOR_ID.name].to_pylist()) == {
        combined.SENSOR_NAME_TO_ID[combined.CONV_SENSOR]
    }
    # Values and the local id are untouched: conv keeps the normalization it arrived with.
    assert rebased["Observation"].to_pylist() == [1.0, 2.0, 3.0]
    assert rebased[LOCAL_CHANNEL_ID.name].to_pylist() == local.tolist()


def test_use_nnja_conv_requires_use_nnja_sat():
    with pytest.raises(ValueError, match="use_nnja_conv requires use_nnja_sat"):
        ObsConfig(use_nnja_conv=True, conv_level_channels=True)


def test_use_nnja_satwnd_requires_use_nnja_conv():
    with pytest.raises(ValueError, match="use_nnja_satwnd requires use_nnja_conv"):
        ObsConfig(
            use_nnja_sat=True,
            use_nnja_satwnd=True,
            conv_level_channels=True,
        )


def test_satwnd_thin_hpx_level_reaches_the_satwnd_loader():
    from healda.datasets.da.tasks import _get_conv_loader
    from healda.observations.system import ObsPipeline

    obs = ObsConfig(
        use_nnja_sat=True,
        use_nnja_conv=True,
        use_nnja_satwnd=True,
        conv_level_channels=True,
        nnja_satwnd_thin_hpx_level=6,
    )
    loader = _get_conv_loader(obs, ObsPipeline(obs, training=False))
    assert loader._loader.satwnd.thin_hpx_level == 6


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_nnja_wind_dropout_must_be_a_probability(value):
    with pytest.raises(ValueError, match="nnja_wind_dropout must be in"):
        ObsConfig(nnja_wind_dropout=value)


def test_nnja_wind_dropout_requires_nnja_conventional():
    with pytest.raises(ValueError, match="nnja_wind_dropout requires use_nnja_conv"):
        ObsConfig(nnja_wind_dropout=0.5)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan])
def test_nnja_platform_channel_dropout_must_be_a_probability(probability):
    with pytest.raises(
        ValueError, match="platform/channel dropout probability must be in"
    ):
        ObsConfig(
            use_nnja_sat=True,
            conv_level_channels=True,
            nnja_platform_channel_dropout=(("amsua", "metop-b", 3, probability),),
        )


def test_nnja_platform_channel_dropout_requires_nnja_satellite():
    with pytest.raises(
        ValueError, match="nnja_platform_channel_dropout requires use_nnja_sat"
    ):
        ObsConfig(nnja_platform_channel_dropout=(("amsua", "metop-b", 3, 0.2),))


def _write_base_channel_table(path) -> str:
    # ps/q/t/u/v — enough for the synthetic base-conv rows below.
    local_ids = np.arange(3, 8, dtype=np.uint16)
    pq.write_table(
        pa.table(
            {
                GLOBAL_CHANNEL_ID.name: pa.array(
                    sensors.SENSOR_OFFSET["conv"] + local_ids,
                    type=GLOBAL_CHANNEL_ID.type,
                ),
                "mean": pa.array([1000.0, 0.005, 280.0, 0.0, 0.0], type=pa.float32()),
                "stddev": pa.array([10.0, 0.001, 5.0, 10.0, 10.0], type=pa.float32()),
                "min_valid": pa.array(
                    [500.0, 0.0, 150.0, -100.0, -100.0], type=pa.float32()
                ),
                "max_valid": pa.array(
                    [1100.0, 1.0, 350.0, 100.0, 100.0], type=pa.float32()
                ),
                "is_conv": pa.array([True] * 5),
            }
        ),
        path,
    )
    return str(path)


def test_nnja_conventional_loader_emits_gsi_conv_plevel(tmp_path):
    # Physical base-conv rows → shared filter → pressure-level ids + by-level norm,
    # still numbered in the GSI vocabulary so CombinedObsLoader can rebase them.
    loader = combined.NNJAConventionalLoader(
        include_gpsro=False,
        conv_min_pressure_hpa=1.0,
    )
    # t at 850 hPa and surface ps; values inside the synthetic min/max bounds.
    base_local = np.array([5, 3], dtype=np.uint16)
    pressure = np.array([850.0, 1013.0], dtype=np.float32)
    values = np.array([280.0, 1000.0], dtype=np.float32)
    table = pa.table(
        {
            "Latitude": pa.array([10.0, 20.0], pa.float32()),
            "Longitude": pa.array([30.0, 40.0], pa.float32()),
            "Absolute_Obs_Time": pa.array(
                np.array(
                    ["2022-01-01T00:00:00", "2022-01-01T00:00:00"],
                    dtype="datetime64[ns]",
                )
            ),
            "DA_window": pa.array(
                np.array(
                    ["2022-01-01T00:00:00", "2022-01-01T00:00:00"],
                    dtype="datetime64[ns]",
                )
            ),
            "Platform_ID": pa.array([30, 28], pa.uint16()),
            "Observation": pa.array(values, pa.float32()),
            GLOBAL_CHANNEL_ID.name: pa.array(
                sensors.SENSOR_OFFSET["conv"] + base_local, GLOBAL_CHANNEL_ID.type
            ),
            "Sat_Zenith_Angle": pa.nulls(2, pa.float32()),
            "Sol_Zenith_Angle": pa.nulls(2, pa.float32()),
            "Scan_Angle": pa.nulls(2, pa.float32()),
            "Pressure": pa.array(pressure, pa.float32()),
            "Height": pa.array([1500.0, 10.0], pa.float32()),
            "Observation_Type": pa.array([120, 181], pa.uint16()),
            "QC_Flag": pa.nulls(2, pa.int32()),
            "Analysis_Use_Flag": pa.array([1, 1], pa.int8()),
            LOCAL_CHANNEL_ID.name: pa.array(base_local, LOCAL_CHANNEL_ID.type),
            SENSOR_ID.name: pa.array(
                np.full(2, sensors.SENSOR_NAME_TO_ID["conv"], dtype=np.uint16),
                SENSOR_ID.type,
            ),
        }
    )

    out = loader._to_conv_plevel(table)
    assert out.num_rows == 2
    expanded = expanded_local_channel(base_local, pressure).astype(np.uint16)
    assert out[LOCAL_CHANNEL_ID.name].to_pylist() == expanded.tolist()
    assert (
        out[GLOBAL_CHANNEL_ID.name].to_pylist()
        == (sensors.SENSOR_OFFSET["conv-plevel"] + expanded).tolist()
    )
    assert set(out[SENSOR_ID.name].to_pylist()) == {
        sensors.SENSOR_NAME_TO_ID["conv-plevel"]
    }

    mean, std = loader._plevel_norm_lookup
    expected = ((values - mean[expanded]) / std[expanded]).astype(np.float32)
    np.testing.assert_allclose(out["Observation"].to_numpy(), expected, rtol=1e-5)

    # After CombinedObsLoader rebase, ids land in the NNJA-appended conv block.
    rebased = combined._rebase(
        out, out[LOCAL_CHANNEL_ID.name].to_numpy(zero_copy_only=False)
    )
    assert rebased[GLOBAL_CHANNEL_ID.name].to_pylist() == [
        combined.SENSOR_OFFSET[combined.CONV_SENSOR] + int(value) for value in expanded
    ]


def test_nnja_conventional_loader_sel_time_empty_archive(tmp_path):
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    loader = combined.NNJAConventionalLoader(
        archive_root=str(cycles),
        include_gpsro=False,
        obs_context_hours=(-3, 0),
    )
    # No cycle files → empty tables, not an error.
    result = asyncio.run(loader.sel_time(pd.DatetimeIndex(["2022-01-01T00:00:00"])))
    assert result["obs_v2"][0].num_rows == 0


def test_get_conv_loader_switches_on_use_nnja_conv(tmp_path, monkeypatch):
    from healda.datasets.da import tasks

    _write_base_channel_table(tmp_path / "channel_table.parquet")
    monkeypatch.setattr(tasks.config, "UFS_OBS_PATH", str(tmp_path))
    nnja = tasks.build_obs_loader(
        ObsConfig(use_nnja_sat=True, use_nnja_conv=True, conv_level_channels=True),
        training=False,
    ).conventional
    assert isinstance(nnja, combined.NNJAConventionalLoader)
    ufs = tasks.build_obs_loader(
        ObsConfig(use_nnja_sat=True, use_nnja_conv=False, conv_level_channels=True),
        training=False,
    ).conventional
    assert not isinstance(ufs, combined.NNJAConventionalLoader)


def test_get_conv_loader_applies_wind_dropout_only_while_training(
    tmp_path, monkeypatch
):
    from healda.datasets.da import tasks

    _write_base_channel_table(tmp_path / "channel_table.parquet")
    monkeypatch.setattr(tasks.config, "UFS_OBS_PATH", str(tmp_path))
    config = ObsConfig(
        use_nnja_sat=True,
        use_nnja_conv=True,
        conv_level_channels=True,
        nnja_wind_dropout=0.5,
    )

    train_loader = tasks.build_obs_loader(config, training=True).conventional
    eval_loader = tasks.build_obs_loader(config, training=False).conventional
    assert train_loader._loader.wind_obs_dropout == 0.5
    assert eval_loader._loader.wind_obs_dropout == 0.0


def test_platform_channel_rules_below_one_are_training_only(tmp_path, monkeypatch):
    """Under 1 the rule is a regulariser; at 1 it is a denial and has to hold when scoring."""
    from healda.datasets.da import tasks

    _write_base_channel_table(tmp_path / "channel_table.parquet")
    monkeypatch.setattr(tasks.config, "UFS_OBS_PATH", str(tmp_path))
    config = ObsConfig(
        use_nnja_sat=True,
        use_nnja_conv=True,
        conv_level_channels=True,
        nnja_platform_channel_dropout=(
            ("amsua", "metop-b", 3, 0.2),
            ("amsua", "metop-b", 6, 1.0),
        ),
    )

    train_loader = tasks.build_obs_loader(config, training=True)
    eval_loader = tasks.build_obs_loader(config, training=False)

    assert len(train_loader.satellite._platform_channel_dropout["amsua"]) == 2
    kept = eval_loader.satellite._platform_channel_dropout["amsua"]
    assert [probability for _, _, probability in kept] == [1.0]

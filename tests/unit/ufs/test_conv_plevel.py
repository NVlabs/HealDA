# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa

from healda.observations.sensors import (
    CONV_GPS_CHANNELS,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    SENSOR_NAME_TO_ID,
    SENSOR_OFFSET,
)
from healda.observations.preprocessing.conv_plevel import (
    CONV_PLEVEL_BASE_LOCAL_CHANNELS,
    CONV_PLEVEL_GPS_EXPANDED_RANGE,
    CONV_PLEVEL_GPS_LEVEL2_EXPANDED_RANGE,
    CONV_PLEVEL_N_LEVELS,
    CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL,
    CONV_PLEVEL_UV_EXPANDED_RANGE,
    PRESSURE_LEVELS_HPA,
    build_conv_plevel_channel_table,
    conv_plevel_channel_map,
    expanded_local_channel,
    nearest_pressure_level_index,
)


def _expected_expanded_range(base_local_channels):
    base_to_group = {
        int(local_channel): group
        for group, local_channel in enumerate(CONV_PLEVEL_BASE_LOCAL_CHANNELS)
    }
    groups = [
        base_to_group[int(local_channel)] for local_channel in base_local_channels
    ]
    return min(groups) * CONV_PLEVEL_N_LEVELS, (max(groups) + 1) * CONV_PLEVEL_N_LEVELS


def test_nearest_pressure_level_index_matches_expected_bins():
    pressure = np.array(
        [
            0.1,
            49.0,
            50.0,
            75.0,
            75.1,
            200.0,
            224.9,
            225.0,
            225.1,
            962.5,
            962.6,
            1000.0,
            1100.0,
        ],
        dtype=np.float32,
    )

    idx = nearest_pressure_level_index(pressure)
    levels = PRESSURE_LEVELS_HPA[idx]

    np.testing.assert_array_equal(
        levels,
        np.array(
            [
                50,
                50,
                50,
                50,
                100,
                200,
                200,
                200,
                250,
                925,
                1000,
                1000,
                1000,
            ],
            dtype=np.int16,
        ),
    )


def test_expanded_ranges_follow_base_conv_sensor_groups():
    assert CONV_PLEVEL_GPS_EXPANDED_RANGE == _expected_expanded_range(CONV_GPS_CHANNELS)
    assert CONV_PLEVEL_GPS_LEVEL2_EXPANDED_RANGE == _expected_expanded_range(
        CONV_GPS_LEVEL2_CHANNELS
    )
    assert CONV_PLEVEL_UV_EXPANDED_RANGE == _expected_expanded_range(CONV_UV_CHANNELS)


def test_expanded_local_channel_maps_surface_and_vertical_channels():
    base_local = np.array([4, 4, 5, 6, 7, 3], dtype=np.int16)
    pressure = np.array([1000.0, 50.0, 700.0, 225.1, 1100.0, 250.0], dtype=np.float32)

    expanded = expanded_local_channel(base_local, pressure)

    np.testing.assert_array_equal(
        expanded,
        np.array(
            [
                39,  # q at 1000 hPa: fourth vertical base channel, first level
                51,  # q at 50 hPa: fourth vertical base channel, last level
                55,  # t at 700 hPa
                73,  # u at 250 hPa
                78,  # v at 1000 hPa
                CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL,  # ps is never level-expanded
            ],
            dtype=np.int16,
        ),
    )


def test_conv_plevel_channel_map_is_stable_and_complete():
    table = conv_plevel_channel_map()
    mapping = table.to_pydict()
    conv_plevel_offset = SENSOR_OFFSET["conv-plevel"]

    assert table.num_rows == 92
    assert mapping["Global_Channel_ID"][0] == conv_plevel_offset
    assert mapping["Global_Channel_ID"][-1] == conv_plevel_offset + 91
    assert mapping["local_channel_id"] == list(range(92))
    assert len(set(mapping["Global_Channel_ID"])) == 92
    assert mapping["name"][0] == "gps_angle_1000"
    assert mapping["name"][39] == "q_1000"
    assert mapping["name"][51] == "q_50"
    assert mapping["name"][-1] == "ps"
    assert mapping["Level_hPa"][-1] == -1


def test_build_conv_plevel_channel_table_prefers_level_stats_for_vertical_channels():
    conv_offset = SENSOR_OFFSET["conv"]
    conv_plevel_sensor_id = SENSOR_NAME_TO_ID["conv-plevel"]

    base_channel_table = pa.table(
        {
            "Global_Channel_ID": pa.array(
                [conv_offset + local for local in range(8)], type=pa.uint16()
            ),
            "mean": pa.array([10.0 + local for local in range(8)], type=pa.float32()),
            "stddev": pa.array([1.0 + local for local in range(8)], type=pa.float32()),
        }
    )
    level_stats = pa.table(
        {
            "Global_Channel_ID": pa.array([conv_offset + 4], type=pa.uint16()),
            "Level_hPa": pa.array([1000], type=pa.int16()),
            "level_mean": pa.array([0.25], type=pa.float32()),
            "level_stddev": pa.array([0.5], type=pa.float32()),
        }
    )

    table = build_conv_plevel_channel_table(
        level_stats=level_stats,
        base_channel_table=base_channel_table,
        include_local_channel_id=True,
    )
    rows = table.to_pandas().set_index("name")

    assert table.num_rows == 92
    assert rows.loc["q_1000", "sensor_id"] == conv_plevel_sensor_id
    assert rows.loc["q_1000", "local_channel_id"] == 39
    assert rows.loc["q_1000", "mean"] == 0.25
    assert rows.loc["q_1000", "stddev"] == 0.5
    assert rows.loc["q_925", "mean"] == 14.0
    assert rows.loc["q_925", "stddev"] == 5.0
    assert rows.loc["ps", "mean"] == 13.0
    assert rows.loc["ps", "stddev"] == 4.0

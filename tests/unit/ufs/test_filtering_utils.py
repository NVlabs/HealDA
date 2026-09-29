# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test script for the filtering.py implementation.
This tests the vectorized filtering approach for conventional observations.
"""

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import pytest
from healda.observations.preprocessing.filtering import filter_observations
from healda.observations.schema import (
    get_combined_observation_schema,
    GLOBAL_CHANNEL_ID,
)
from healda.observations.loaders.ufs import LOCAL_CHANNEL_ID
from healda.observations.sensors import build_channel_table as get_channel_table
from healda.observations.sensors import (
    CONV_GPS_CHANNELS,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    CONV_UV_IN_SITU_TYPES,
    QCLimits,
    SENSOR_OFFSET,
)


def create_test_data():
    """Create test data with different platform types."""
    # Create test data with different channel IDs
    conv_offset = SENSOR_OFFSET["conv"]

    # GPS channels: 0, 1, 2
    gps_channels = [conv_offset + 0, conv_offset + 1, conv_offset + 2]
    # PS channel: 3
    ps_channels = [conv_offset + 3]
    # Q channel: 4
    q_channels = [conv_offset + 4]
    # T channel: 5
    t_channels = [conv_offset + 5]
    # UV channels: 6, 7
    uv_channels = [conv_offset + 6, conv_offset + 7]

    # Combine all channels
    all_channels = gps_channels + ps_channels + q_channels + t_channels + uv_channels

    # Create test data
    n_rows = len(all_channels)
    data = {
        # Required common fields
        "Latitude": np.random.uniform(-90, 90, n_rows),
        "Longitude": np.random.uniform(-180, 180, n_rows),
        "Absolute_Obs_Time": np.array(
            [np.datetime64("2023-01-01T00:00:00")] * n_rows, dtype="datetime64[ns]"
        ),
        "DA_window": np.array(
            [np.datetime64("2023-01-01T00:00:00")] * n_rows, dtype="datetime64[ns]"
        ),
        "Platform_ID": np.random.randint(1, 100, n_rows),
        "Global_Channel_ID": all_channels,
        "Observation": np.random.uniform(0, 100, n_rows),
        # Satellite-specific fields (nullable)
        "Sat_Zenith_Angle": np.full(n_rows, None, dtype=object),
        "Sol_Zenith_Angle": np.full(n_rows, None, dtype=object),
        "Scan_Angle": np.full(n_rows, None, dtype=object),
        # Conventional-specific fields
        "Pressure": np.random.uniform(200, 1100, n_rows),
        "Height": np.random.uniform(0, 50000, n_rows),
        "Observation_Type": np.random.randint(1, 10, n_rows),
        # Analysis fields (nullable)
        "QC_Flag": np.random.choice([0, 1], n_rows),
        "Analysis_Use_Flag": np.random.choice([0, 1], n_rows),
        "Obs_Minus_Forecast_adjusted": np.random.uniform(-10, 10, n_rows),
        "Obs_Minus_Forecast_unadjusted": np.random.uniform(-10, 10, n_rows),
    }

    obs_table = pa.table(data, schema=get_combined_observation_schema())

    # Add channel metadata via join (mimics UFSUnifiedLoader._add_channel_metadata)
    def _add_channel_metadata(table):
        channel_table = get_channel_table()

        # Add local_channel_id (same as UFSUnifiedLoader.channel_table property)
        sensor_id = np.asarray(channel_table["sensor_id"])
        local_channel_ids = []
        offset = 0
        for i in range(len(sensor_id)):
            if sensor_id[i] != sensor_id[i - 1]:
                offset = i
            local_channel_ids.append(i - offset)
        channel_table = channel_table.append_column(
            LOCAL_CHANNEL_ID.name, pa.array(local_channel_ids, type=pa.uint16())
        )

        return table.join(
            channel_table.select(
                [
                    GLOBAL_CHANNEL_ID.name,
                    LOCAL_CHANNEL_ID.name,
                    "min_valid",
                    "max_valid",
                    "is_conv",
                ]
            ),
            GLOBAL_CHANNEL_ID.name,
        )

    return _add_channel_metadata(obs_table)


def test_vectorized_filtering():
    """Test that vectorized filtering works correctly."""
    table = create_test_data()

    # Test filtering using the unified filter_observations function
    filtered_table = filter_observations(table, qc_filter=False)

    # Test with QC filtering enabled
    qc_filtered_table = filter_observations(table, qc_filter=True)

    # Verify that filtering produces results
    assert filtered_table.num_rows >= 0
    assert qc_filtered_table.num_rows >= 0
    assert qc_filtered_table.num_rows <= filtered_table.num_rows


def _mask_expression(
    table,
    qc_filter=False,
    conv_uv_in_situ_only=False,
    conv_gps_level1_only=False,
    conv_min_pressure_hpa=None,
):
    """The `pc.field` form observation_mask replaced, kept here as the oracle."""
    height = pc.field("Height")
    pressure = pc.field("Pressure")
    obs = pc.field("Observation")
    analysis_use = pc.field("Analysis_Use_Flag")
    qc_flag = pc.field("QC_Flag")
    min_valid = pc.field("min_valid")
    max_valid = pc.field("max_valid")
    local_id = pc.field("local_channel_id")
    is_conv = pc.field("is_conv")
    obs_type = pc.field("Observation_Type")

    if conv_min_pressure_hpa is None:
        conv_min_pressure_hpa = QCLimits.PRESSURE_MIN_DEFAULT
    is_gps = pc.is_in(local_id, pa.array(CONV_GPS_CHANNELS))
    height_ok = pc.is_finite(height) & (
        (height >= QCLimits.HEIGHT_MIN) & (height <= QCLimits.HEIGHT_MAX)
    )
    min_pressure = pc.if_else(
        is_gps,
        pa.scalar(QCLimits.PRESSURE_MIN_GPS),
        pa.scalar(float(conv_min_pressure_hpa)),
    )
    pressure_ok = pc.is_finite(pressure)
    pressure_ok &= (pressure >= min_pressure) & (pressure <= QCLimits.PRESSURE_MAX)
    conv_ok = pressure_ok & height_ok
    if qc_filter:
        conv_ok &= analysis_use == pa.scalar(1)
    if conv_uv_in_situ_only:
        is_uv_channel = pc.is_in(local_id, pa.array(CONV_UV_CHANNELS))
        is_in_situ = pc.is_in(
            obs_type,
            pa.array(CONV_UV_IN_SITU_TYPES, type=table["Observation_Type"].type),
        )
        conv_ok &= ~is_uv_channel | is_in_situ
    if conv_gps_level1_only:
        conv_ok &= ~pc.is_in(local_id, pa.array(CONV_GPS_LEVEL2_CHANNELS))

    ok = pc.is_finite(obs)
    ok &= obs >= min_valid
    ok &= obs <= max_valid
    sat_ok = ok
    if qc_filter:
        sat_ok &= qc_flag == 0
    ok &= pc.if_else(is_conv, conv_ok, sat_ok)
    return ok


def _filter_by_expression(table, **flags):
    return table.filter(_mask_expression(table, **flags))


@pytest.mark.parametrize("qc_filter", [False, True])
@pytest.mark.parametrize("uv_in_situ_only", [False, True])
@pytest.mark.parametrize("gps_level1_only", [False, True])
def test_mask_matches_the_expression_it_replaced(
    qc_filter, uv_in_situ_only, gps_level1_only
):
    """Every flag combination, since only the defaults were checked against real data."""
    np.random.seed(0)
    table = create_test_data()
    kwargs = dict(
        qc_filter=qc_filter,
        conv_uv_in_situ_only=uv_in_situ_only,
        conv_gps_level1_only=gps_level1_only,
        conv_min_pressure_hpa=500.0,
    )
    got, want = (
        filter_observations(table, **kwargs),
        _filter_by_expression(table, **kwargs),
    )

    assert got.num_rows == want.num_rows
    for name in got.schema.names:
        # Not object dtype: the all-null angle columns arrive as NaN, which compares
        # unequal to itself unless numpy can see a float array.
        left = got[name].combine_chunks().to_numpy(zero_copy_only=False)
        right = want[name].combine_chunks().to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(left, right, err_msg=name)


def _wide_table(rows):
    rng = np.random.default_rng(0)
    observation = rng.uniform(0, 100, rows)
    # Values the range check must drop, one of each kind.
    observation[:3] = [np.nan, np.inf, -1.0][:rows]
    return pa.table(
        {
            "Observation": observation,
            "min_valid": np.zeros(rows),
            "max_valid": np.full(rows, 100.0),
            "QC_Flag": rng.integers(0, 2, rows),
            "Analysis_Use_Flag": rng.integers(0, 2, rows),
            "Height": rng.uniform(0, 50_000, rows),
            "Pressure": rng.uniform(200, 1100, rows),
            "Observation_Type": rng.integers(1, 10, rows),
            "local_channel_id": pa.array(rng.integers(0, 8, rows), type=pa.uint16()),
            "is_conv": np.zeros(rows, dtype=bool),
        }
    )


@pytest.mark.parametrize("rows", [1_000, 32_769, 100_000])
def test_filter_returns_one_chunk(rows):
    """`Table.filter(Expression)` fragments at 32,768 rows; a materialized mask does not.

    The fragmentation is invisible in the values and only shows up downstream, where
    the transform gathers each column chunk by chunk.
    """
    table = _wide_table(rows)
    assert table.column(0).num_chunks == 1

    got = filter_observations(table)
    assert got.column(0).num_chunks == 1
    assert got.num_rows == rows - 3
    want = _filter_by_expression(table)
    assert want.num_rows == got.num_rows
    # The oracle is the fragmented one, which is the whole point of the change.
    assert want.column(0).num_chunks == -(-rows // 32_768)


def test_filter_handles_an_empty_table():
    empty = _wide_table(0)
    assert filter_observations(empty).num_rows == 0

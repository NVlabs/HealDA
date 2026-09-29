# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration test for UFS Unified Loader with combined schema.
"""

import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime
import tempfile
import os

from healda.observations.loaders.ufs import LOCAL_CHANNEL_ID, UFSUnifiedLoader
from healda.observations.schema import (
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
    get_combined_observation_schema,
    get_channel_table_schema,
)
from healda.observations.sensors import build_channel_table as get_channel_table
from healda.observations.sensors import SENSOR_CONFIGS


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sensor directory
        sensor_dir = os.path.join(temp_dir, "atms")
        os.makedirs(sensor_dir, exist_ok=True)

        # Create date directory
        date_dir = os.path.join(sensor_dir, "20200101")
        os.makedirs(date_dir, exist_ok=True)

        # Create sample data with all required schema fields
        n_obs = 50
        data = {
            # Common fields
            "Latitude": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "Longitude": np.random.uniform(-180, 180, n_obs).astype(np.float32),
            "Absolute_Obs_Time": pd.date_range(
                "2020-01-01", periods=n_obs, freq="1h"
            ).astype("datetime64[ns]"),
            "DA_window": pd.date_range("2020-01-01", periods=n_obs, freq="3h").astype(
                "datetime64[ns]"
            ),
            "Platform_ID": np.random.randint(0, 32, n_obs).astype(np.uint16),
            "Observation": np.random.uniform(0, 400, n_obs).astype(np.float32),
            "Global_Channel_ID": np.random.randint(0, 100, n_obs).astype(np.uint16),
            # Satellite-specific fields
            "Sat_Zenith_Angle": np.random.uniform(0, 90, n_obs).astype(np.float32),
            "Sol_Zenith_Angle": np.random.uniform(0, 90, n_obs).astype(np.float32),
            "Scan_Angle": np.random.uniform(-45, 45, n_obs).astype(np.float32),
            # Conventional fields (nullable)
            "Pressure": np.full(n_obs, np.nan, dtype=np.float32),
            "Height": np.full(n_obs, np.nan, dtype=np.float32),
            "Observation_Type": np.full(n_obs, np.nan, dtype=np.uint16),
            # Analysis fields (nullable)
            "QC_Flag": np.random.randint(0, 2, n_obs).astype(np.int32),
            "Analysis_Use_Flag": np.full(n_obs, np.nan, dtype=np.int8),
            "Obs_Minus_Forecast_adjusted": np.random.uniform(-10, 10, n_obs).astype(
                np.float32
            ),
            "Obs_Minus_Forecast_unadjusted": np.random.uniform(-10, 10, n_obs).astype(
                np.float32
            ),
        }

        # Write parquet file, one row group per DA_window as etl_unified does, which is the
        # layout the loader plans row groups against.
        schema = get_combined_observation_schema()
        table = pa.table(data, schema=schema).sort_by("DA_window")
        parquet_path = os.path.join(date_dir, "0.parquet")
        window = np.asarray(table["DA_window"])
        starts = np.flatnonzero(np.r_[True, window[1:] != window[:-1]])
        with pa.parquet.ParquetWriter(parquet_path, schema) as writer:
            for start, end in zip(starts, [*starts[1:], len(window)]):
                writer.write_table(
                    table.slice(start, end - start), row_group_size=end - start
                )

        # Create channel table for normalization
        channel_table = get_channel_table()
        channel_table_path = os.path.join(temp_dir, "channel_table.parquet")
        pa.parquet.write_table(channel_table, channel_table_path)

        # No need for availability_df.pkl - using try/catch approach

        yield temp_dir


@pytest.mark.asyncio
async def test_ufs_unified_loader(temp_data_dir):
    """Test UFSUnifiedLoader basic functionality."""
    # Initialize loader
    loader = UFSUnifiedLoader(
        data_path=temp_data_dir,
        sensors=["atms"],
        filesystem_type="local",
    )

    # Test basic properties
    assert loader.sensors == ["atms"]

    # Test data loading
    times = pd.DatetimeIndex([datetime(2020, 1, 1, 12)])
    result = await loader.sel_time(times)

    for result in result["obs_v2"]:
        # Validate schema matches expected output schema
        expected_schema = loader.output_schema
        assert result.schema.equals(expected_schema)


def test_channel_metadata_matches_the_arrow_join(temp_data_dir):
    """The positional gather replaced a hash join and must be equivalent to it.

    Ids repeat and arrive out of order, so this covers a plain lookup table too. The
    join does not preserve row order, hence the sort; the gather does, which the
    following assertion pins separately.
    """
    loader = UFSUnifiedLoader(
        data_path=temp_data_dir, sensors=["atms"], filesystem_type="local"
    )
    ids = np.array([7, 3, 3, 0, 42, 7], dtype=np.uint16)
    # A unique second column, so sorting is a total order despite the repeated ids.
    obs = pa.table({"Global_Channel_ID": ids, "row": np.arange(len(ids))})

    gathered = loader._add_channel_metadata(obs)
    joined = obs.join(
        loader.channel_table.select(
            [
                GLOBAL_CHANNEL_ID.name,
                LOCAL_CHANNEL_ID.name,
                SENSOR_ID.name,
                *loader._extra_channel_fields,
            ]
        ),
        GLOBAL_CHANNEL_ID.name,
    )
    assert np.array_equal(np.asarray(gathered["Global_Channel_ID"]), ids)
    assert set(gathered.schema.names) == set(joined.schema.names)
    left, right = gathered.sort_by("row"), joined.sort_by("row")
    for name in gathered.schema.names:
        np.testing.assert_array_equal(
            np.asarray(left[name]), np.asarray(right[name]), err_msg=name
        )


def test_rewritten_channel_ids_stay_non_nullable():
    """conv-plevel is the only sensor that rewrites its ids.

    Naming the column rather than passing a field declares the replacement nullable,
    which leaves conv-plevel's window the one table in the sensor set whose flag
    disagrees, and the concatenation at the end of sel_time then fails.
    """
    schema = pa.schema([GLOBAL_CHANNEL_ID])
    assert not GLOBAL_CHANNEL_ID.nullable, "the guard below assumes a non-null field"
    table = pa.table([pa.array([1, 2], type=GLOBAL_CHANNEL_ID.type)], schema=schema)

    rewritten = UFSUnifiedLoader._set_typed_column(
        table, GLOBAL_CHANNEL_ID, np.array([3, 4])
    )
    assert not rewritten.schema.field(GLOBAL_CHANNEL_ID.name).nullable
    pa.concat_tables([table, rewritten])


def test_unknown_channel_id_raises(temp_data_dir):
    """Silently attaching another channel's stats would be worse than failing."""
    loader = UFSUnifiedLoader(
        data_path=temp_data_dir, sensors=["atms"], filesystem_type="local"
    )
    unknown = 10**6
    ids = pa.array([3, unknown], type=pa.uint32())
    with pytest.raises(KeyError, match=str(unknown)):
        loader._add_channel_metadata(pa.table({"Global_Channel_ID": ids}))


@pytest.mark.asyncio
async def test_ufs_unified_loader_empty_dataset():
    """Test UFSUnifiedLoader with empty dataset (no data files)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty directory structure
        sensor_dir = os.path.join(temp_dir, "atms")
        os.makedirs(sensor_dir, exist_ok=True)

        # Initialize loader with empty directory
        loader = UFSUnifiedLoader(
            data_path=temp_dir,
            sensors=["atms"],
            filesystem_type="local",
        )

        # Test basic properties
        assert loader.sensors == ["atms"]

        # Test data loading with empty dataset
        times = pd.DatetimeIndex([datetime(2020, 1, 1, 12)])
        result = await loader.sel_time(times)

        result = result["obs_v2"][0]

        # Check result structure - should return empty table with proper schema
        assert isinstance(result, pa.Table)
        assert result.num_rows == 0

        # Validate schema matches expected output schema
        expected_schema = loader.output_schema
        assert result.schema.equals(expected_schema)


def test_get_channel_table_structure():
    """Test that get_channel_table returns correct table structure and schema."""
    table = get_channel_table()

    # Check that it's a PyArrow table
    assert isinstance(table, pa.Table)

    # Check schema matches expected channel table schema
    expected_schema = get_channel_table_schema()
    assert table.schema.equals(expected_schema)


def test_get_channel_table_sensor_mapping():
    """Test that sensor IDs and channel IDs are correctly mapped."""
    table = get_channel_table()

    # Convert to pandas for easier analysis
    df = table.to_pandas()

    # Calculate expected total channels
    expected_total_channels = sum(cfg.channels for cfg in SENSOR_CONFIGS.values())
    assert len(df) == expected_total_channels

    # Check that Global_Channel_ID is sequential starting from 0
    assert df["Global_Channel_ID"].min() == 0
    assert df["Global_Channel_ID"].max() == expected_total_channels - 1
    assert df["Global_Channel_ID"].is_monotonic_increasing

    # Check sensor_id mapping
    sensor_names = list(SENSOR_CONFIGS.keys())
    for i, sensor_name in enumerate(sensor_names):
        sensor_mask = df["sensor_id"] == i
        expected_channels = SENSOR_CONFIGS[sensor_name].channels
        assert sensor_mask.sum() == expected_channels


def test_get_channel_table_conventional_handling():
    """Test that conventional sensors are handled correctly."""
    table = get_channel_table()
    df = table.to_pandas()

    # Find conventional sensor index
    conv_sensor_id = list(SENSOR_CONFIGS.keys()).index("conv")
    conv_mask = df["sensor_id"] == conv_sensor_id

    # Check is_conv flag
    assert df[conv_mask]["is_conv"].all()
    assert not df[~conv_mask]["is_conv"].any()

    # Check conventional sensor naming
    conv_names = df[conv_mask]["name"].tolist()
    expected_conv_names = ["gps_angle", "gps_t", "gps_q", "ps", "q", "t", "u", "v"]
    assert conv_names == expected_conv_names


def test_get_channel_table_consistency():
    """Test that channel table is internally consistent."""
    table = get_channel_table()
    df = table.to_pandas()

    # Check that all Global_Channel_IDs are unique
    assert df["Global_Channel_ID"].nunique() == len(df)

    # Check that sensor_id values are valid
    max_sensor_id = len(SENSOR_CONFIGS) - 1
    assert df["sensor_id"].min() >= 0
    assert df["sensor_id"].max() <= max_sensor_id

    # Check that min_valid <= max_valid for all channels
    assert (df["min_valid"] <= df["max_valid"]).all()

    # Check that all names are non-empty strings
    assert df["name"].str.len().min() > 0
    assert df["name"].notna().all()

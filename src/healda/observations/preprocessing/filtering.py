# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared quality control filtering utilities for observation data.

This module provides reusable filtering functions that can be used by both
the original UFSDataset and the new UFSUnifiedLoader to ensure consistent
quality control across different data loading approaches.
"""

import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.sensors import (
    CONV_GPS_CHANNELS,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    CONV_UV_IN_SITU_TYPES,
    QCLimits,
    SENSOR_CONFIGS,
    SENSOR_OFFSET,
)


def _get_index_range(sensor):
    start = SENSOR_OFFSET[sensor]
    end = start + SENSOR_CONFIGS[sensor].channels
    return start, end


def _all(*masks):
    result = masks[0]
    for mask in masks[1:]:
        result = pc.and_kleene(result, mask)
    return result


def _get_conv_filter_mask(
    table: pa.Table,
    qc_filter: bool = False,
    uv_in_situ_only: bool = False,
    gps_level1_only: bool = False,
    min_pressure_default_hpa: float | None = None,
):
    """Get filter mask for conventional observations.

    Filtering happens after joining base conv metadata and before any optional
    conv-plevel remap, so local_channel_id is still in the original 8-channel
    conv vocabulary here.
    """
    local_id = table["local_channel_id"]
    height, pressure = table["Height"], table["Pressure"]

    is_gps = pc.is_in(local_id, value_set=pa.array(CONV_GPS_CHANNELS))
    if min_pressure_default_hpa is None:
        min_pressure_default_hpa = QCLimits.PRESSURE_MIN_DEFAULT

    height_ok = _all(
        pc.is_finite(height),
        pc.greater_equal(height, QCLimits.HEIGHT_MIN),
        pc.less_equal(height, QCLimits.HEIGHT_MAX),
    )

    min_pressure = pc.if_else(
        is_gps,
        pa.scalar(QCLimits.PRESSURE_MIN_GPS),
        pa.scalar(float(min_pressure_default_hpa)),
    )
    pressure_ok = _all(
        pc.is_finite(pressure),
        pc.greater_equal(pressure, min_pressure),
        pc.less_equal(pressure, QCLimits.PRESSURE_MAX),
    )

    ok = _all(pressure_ok, height_ok)

    if qc_filter:
        ok = pc.and_kleene(ok, pc.equal(table["Analysis_Use_Flag"], pa.scalar(1)))

    if uv_in_situ_only:
        is_uv_channel = pc.is_in(local_id, value_set=pa.array(CONV_UV_CHANNELS))
        is_in_situ = pc.is_in(
            table["Observation_Type"],
            value_set=pa.array(
                CONV_UV_IN_SITU_TYPES, type=table["Observation_Type"].type
            ),
        )
        ok = pc.and_kleene(ok, pc.or_kleene(pc.invert(is_uv_channel), is_in_situ))

    if gps_level1_only:
        is_gps_level2 = pc.is_in(local_id, value_set=pa.array(CONV_GPS_LEVEL2_CHANNELS))
        ok = pc.and_kleene(ok, pc.invert(is_gps_level2))

    return ok


def observation_mask(
    table: pa.Table,
    qc_filter: bool = False,
    conv_uv_in_situ_only: bool = False,
    conv_gps_level1_only: bool = False,
    conv_min_pressure_hpa: float | None = None,
) -> pa.ChunkedArray:
    """Boolean keep-mask for `filter_observations`, one element per row."""
    obs = table["Observation"]
    ok = _all(
        pc.is_finite(obs),
        pc.greater_equal(obs, table["min_valid"]),
        pc.less_equal(obs, table["max_valid"]),
    )

    sat_ok = ok
    if qc_filter:
        sat_ok = pc.and_kleene(sat_ok, pc.equal(table["QC_Flag"], 0))

    conv_filter = _get_conv_filter_mask(
        table,
        qc_filter,
        conv_uv_in_situ_only,
        conv_gps_level1_only,
        conv_min_pressure_hpa,
    )
    return pc.and_kleene(ok, pc.if_else(table["is_conv"], conv_filter, sat_ok))


def filter_observations(
    table: pa.Table,
    qc_filter: bool = False,
    conv_uv_in_situ_only: bool = False,
    conv_gps_level1_only: bool = False,
    conv_min_pressure_hpa: float | None = None,
) -> pa.Table:
    """
    Unified filtering function for observation data.

    The mask is materialized with compute kernels, not `pc.field` expressions. Acero is a
    streaming engine for data larger than memory, and what that buys does not apply to an
    in-memory table this size. It emits a record batch per 32,768 rows, leading to large number of chunks
    (which also slows the to_numpy gather in transform).
    Acero does evaluate the predicate 2x faster, but loses the take that runs once per batch and column.

    Args:
        table: PyArrow table containing observation data
        qc_filter: Whether to apply QC flag filtering
        conv_uv_in_situ_only: Exclude satellite UV (keep in-situ only)
        conv_gps_level1_only: Exclude GPS T/Q retrievals (keep bending angle)
        conv_min_pressure_hpa: Minimum pressure for non-GPS conv obs in hPa.
            Defaults to QCLimits.PRESSURE_MIN_DEFAULT. GPS keeps its own 0.5 hPa
            floor.

    Returns:
        Filtered PyArrow table
    """
    mask = observation_mask(
        table,
        qc_filter,
        conv_uv_in_situ_only,
        conv_gps_level1_only,
        conv_min_pressure_hpa,
    )
    return table.filter(mask)

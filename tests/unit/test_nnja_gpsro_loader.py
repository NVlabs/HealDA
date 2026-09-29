# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the cycle-based NNJA GPS-RO Parquet loader."""

import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from healda.observations.loaders.nnja_conventional import NNJAConvLoader
from healda.observations.loaders.nnja_gpsro import (
    GPS_LEGACY_SAIDS,
    NNJAGpsroLoader,
    height_to_pressure_hpa,
    qfro_bit_set,
)
from healda.observations.sensors import (
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_OFFSET,
)
from healda.observations.schema import GLOBAL_CHANNEL_ID

TARGET = pd.Timestamp("2025-07-02T00:00:00")


def _conv_test_module():
    path = Path(__file__).with_name("test_nnja_conv_loader.py")
    spec = importlib.util.spec_from_file_location("test_nnja_conv_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _qfro_with_bit(bit: int) -> int:
    return 1 << (16 - bit)


def _gpsro_table(positive: bool, *, with_dry_pressure: bool = True) -> pa.Table:
    if positive:
        times = [
            TARGET + pd.Timedelta(hours=1),
            TARGET + pd.Timedelta(hours=2),
            TARGET + pd.Timedelta(hours=2.5),
        ]
        rows = {
            "occ_latitude": [10.0, 20.0, 30.0],
            "occ_longitude": [350.0, 40.0, 100.0],
            "tangent_latitude": [np.nan, 21.0, np.nan],
            "tangent_longitude": [np.nan, 41.0, np.nan],
            "obs_time": times,
            "bending_angle": [0.02, 0.03, 0.04],
            "qfro": [0, 0, _qfro_with_bit(5)],
            "impact_parameter": [
                6_378_000.0 + 5_000,
                6_378_000.0 + 10_000,
                6_378_000.0 + 2_000,
            ],
            "earth_radius_curvature": [6_378_000.0, 6_378_000.0, 6_378_000.0],
            "satellite_id": [755, 3, 755],
            "dry_pressure_hpa": [540.0, 265.0, 790.0],
        }
    else:
        times = [
            TARGET - pd.Timedelta(hours=2),
            TARGET - pd.Timedelta(hours=1),
            TARGET,
            TARGET - pd.Timedelta(hours=0.5),
        ]
        rows = {
            "occ_latitude": [40.0, -20.0, 5.0, 15.0],
            "occ_longitude": [10.0, 200.0, 300.0, 80.0],
            "tangent_latitude": [40.5, np.nan, 5.0, np.nan],
            "tangent_longitude": [10.5, np.nan, 300.0, np.nan],
            "obs_time": times,
            "bending_angle": [0.01, 0.015, 0.0, 0.025],
            "qfro": [0, 0, 0, 0],
            "impact_parameter": [
                6_378_000.0 + 3_000,
                6_378_000.0 + 8_000,
                6_378_000.0 + 4_000,
                6_378_000.0 + 6_000,
            ],
            "earth_radius_curvature": [6_378_000.0] * 4,
            # 66 is deliberately outside the UFS SAID allowlist.
            "satellite_id": [755, 804, 755, 66],
            "dry_pressure_hpa": [700.0, 360.0, 620.0, 480.0],
        }
    arrays = {
        "occ_latitude": pa.array(rows["occ_latitude"], type=pa.float64()),
        "occ_longitude": pa.array(rows["occ_longitude"], type=pa.float64()),
        "tangent_latitude": pa.array(rows["tangent_latitude"], type=pa.float64()),
        "tangent_longitude": pa.array(rows["tangent_longitude"], type=pa.float64()),
        "obs_time": pa.array(rows["obs_time"], type=pa.timestamp("ns", tz="UTC")),
        "bending_angle": pa.array(rows["bending_angle"], type=pa.float32()),
        "qfro": pa.array(rows["qfro"], type=pa.int64()),
        "impact_parameter": pa.array(rows["impact_parameter"], type=pa.float64()),
        "earth_radius_curvature": pa.array(
            rows["earth_radius_curvature"], type=pa.float64()
        ),
        "satellite_id": pa.array(rows["satellite_id"], type=pa.int64()),
    }
    if with_dry_pressure:
        arrays["dry_pressure_hpa"] = pa.array(
            rows["dry_pressure_hpa"], type=pa.float64()
        )
    return pa.table(arrays)


def _write_gpsro_archive(root, *, with_dry_pressure: bool = True) -> str:
    directory = root / "2025"
    directory.mkdir(parents=True)
    path = directory / "gdas.20250702.t00z.gpsro.tm00.bufr_d.parquet"
    schema = _gpsro_table(False, with_dry_pressure=with_dry_pressure).schema
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(_gpsro_table(False, with_dry_pressure=with_dry_pressure))
        writer.write_table(_gpsro_table(True, with_dry_pressure=with_dry_pressure))
    return str(root)


def _write_channel_table(
    root, *, with_gps: bool = True, with_conv: bool = False
) -> str:
    path = root / "channel_table.parquet"
    local_ids = []
    means, stds, mins, maxs = [], [], [], []
    if with_gps:
        local_ids.append(0)
        means.append(0.02)
        stds.append(0.01)
        mins.append(0.0)
        maxs.append(0.1)
    if with_conv:
        local_ids.extend([3, 4, 5, 6, 7])
        means.extend([1000.0, 0.005, 280.0, 0.0, 0.0])
        stds.extend([10.0, 0.001, 5.0, 10.0, 10.0])
        mins.extend([500.0, 0.0, 150.0, -100.0, -100.0])
        maxs.extend([1100.0, 1.0, 350.0, 100.0, 100.0])
    local_ids = np.asarray(local_ids, dtype=np.uint16)
    pq.write_table(
        pa.table(
            {
                GLOBAL_CHANNEL_ID.name: pa.array(
                    SENSOR_OFFSET["conv"] + local_ids, type=GLOBAL_CHANNEL_ID.type
                ),
                "mean": pa.array(means, type=pa.float32()),
                "stddev": pa.array(stds, type=pa.float32()),
                "min_valid": pa.array(mins, type=pa.float32()),
                "max_valid": pa.array(maxs, type=pa.float32()),
            }
        ),
        path,
    )
    return str(path)


def _gpsro_loader(tmp_path, **overrides) -> NNJAGpsroLoader:
    options = {
        "archive_root": _write_gpsro_archive(tmp_path / "gpsro"),
        "normalize": False,
    }
    options.update(overrides)
    return NNJAGpsroLoader(**options)


def test_qfro_bit_and_height_pressure_helpers():
    assert qfro_bit_set(np.array([_qfro_with_bit(5)]), 5).tolist() == [True]
    assert qfro_bit_set(np.array([0]), 5).tolist() == [False]
    pressure = height_to_pressure_hpa(np.array([0.0, 5000.0, 25_000.0, 40_000.0]))
    assert pressure[0] > pressure[1] > pressure[2] > pressure[3]
    assert pressure[0] == np.float32(1013.25)
    # Above the old 20 km clip, pressures must keep separating.
    assert pressure[2] < 54.75
    assert pressure[3] < pressure[2]


def test_sel_time_keeps_assimilable_bending_angles(tmp_path):
    loader = _gpsro_loader(tmp_path)
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]

    assert table.schema == loader.output_schema
    # Negative half: 0.01 (755), 0.015 (804). Positive half: 0.02 (755), 0.03 (3).
    # Dropped: zero bending, QFRO bit 5, SAID 66.
    assert table.num_rows == 4
    assert set(np.asarray(table["local_channel_id"])) == {0}
    assert set(np.asarray(table["Platform_ID"])) == {PLATFORM_NAME_TO_ID["gps"]}
    assert set(np.asarray(table["Observation_Type"])).issubset(GPS_LEGACY_SAIDS)
    np.testing.assert_allclose(
        np.sort(np.asarray(table["Observation"])), [0.01, 0.015, 0.02, 0.03], atol=1e-6
    )
    assert 200.0 in np.asarray(table["Longitude"])
    assert 350.0 in np.asarray(table["Longitude"])
    assert np.isfinite(np.asarray(table["Pressure"])).all()
    assert (np.asarray(table["Height"]) > 0).all()
    # With no blended column in this legacy-shaped fixture, v3 falls back to dry pressure.
    np.testing.assert_allclose(
        np.sort(np.asarray(table["Pressure"])),
        [265.0, 360.0, 540.0, 700.0],
        atol=1e-3,
    )


def test_v3_prefers_blended_pressure_and_corrected_height(tmp_path):
    loader = _gpsro_loader(tmp_path)
    source = _gpsro_table(False)
    source = source.append_column(
        "blended_pressure_5km_hpa",
        pa.array([710.0, 370.0, 630.0, 490.0], type=pa.float64()),
    )
    source = source.append_column(
        "impact_height_m",
        pa.array([3000.0, 8050.0, 4000.0, 6000.0], type=pa.float64()),
    )
    source = source.append_column(
        "heit_at_impact_m",
        pa.array([3100.0, np.nan, 4100.0, 6100.0], type=pa.float64()),
    )

    table = loader._to_long(source, TARGET, positive_half=False)

    # Rows 0/1 survive the existing QC. Blended pressure wins over dry
    # pressure; corrected height wins where present and falls back per row.
    np.testing.assert_allclose(np.asarray(table["Pressure"]), [710.0, 370.0])
    np.testing.assert_allclose(np.asarray(table["Height"]), [3100.0, 8050.0])


def test_pressure_falls_back_to_isa_without_dry_column(tmp_path):
    loader = _gpsro_loader(
        tmp_path,
        archive_root=_write_gpsro_archive(
            tmp_path / "gpsro_isa", with_dry_pressure=False
        ),
    )
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    pressure = np.asarray(table["Pressure"])
    height = np.asarray(table["Height"])
    expected = height_to_pressure_hpa(height)
    np.testing.assert_allclose(pressure, expected, rtol=1e-5)


def test_invalid_stored_isa_pressure_falls_back_per_row(tmp_path):
    loader = _gpsro_loader(tmp_path)
    source = _gpsro_table(False, with_dry_pressure=False).append_column(
        "standard_atmosphere_pressure_hpa",
        pa.array([650.0, np.nan, 0.0, -1.0], type=pa.float64()),
    )

    table = loader._to_long(source, TARGET, positive_half=False)

    assert table.num_rows == 2
    expected = np.array(
        [650.0, height_to_pressure_hpa(np.array([8000.0]))[0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(table["Pressure"]), expected, rtol=1e-5)


def test_normalization_uses_packaged_gps_angle_statistics(tmp_path):
    """z-score comes from the packaged conv vocabulary, not from any archive table."""
    conv = SENSOR_CONFIGS["conv"]

    def observations(root, normalize):
        # Fresh root per call: the helper writes its fixtures on construction.
        loader = _gpsro_loader(root, normalize=normalize)
        table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
        return np.sort(np.asarray(table["Observation"]))

    physical = observations(tmp_path / "physical", False)
    expected = (physical - conv.means[0]) / conv.stds[0]
    np.testing.assert_allclose(observations(tmp_path / "z", True), expected, rtol=1e-5)


def test_conv_loader_appends_gpsro_rows(tmp_path):
    conv_mod = _conv_test_module()
    cycles_root = conv_mod._write_archive(tmp_path / "cycles")
    gpsro_root = _write_gpsro_archive(tmp_path / "gpsro")

    loader = NNJAConvLoader(
        archive_root=cycles_root,
        gpsro_archive_root=gpsro_root,
        normalize=False,
        include_gpsro=True,
    )
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    local = set(np.asarray(table["local_channel_id"]))
    assert 0 in local
    assert {3, 4, 5, 6, 7}.issubset(local)
    assert table.num_rows == 12 + 4

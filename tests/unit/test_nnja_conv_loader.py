# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the cycle-based NNJA conventional Parquet loader."""

import asyncio

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from healda.observations.loaders.nnja_conventional import (
    SATWND_REPORT_TYPES,
    SURFACE_WIND_TYPES,
    NNJAConvLoader,
    is_withheld,
)
from healda.observations.loaders.nnja_gpsro import GPS_LEGACY_SAIDS
from healda.observations.preprocessing.filtering import _get_conv_filter_mask
from healda.observations.sensors import SENSOR_CONFIGS, SENSOR_OFFSET
from healda.observations.schema import GLOBAL_CHANNEL_ID

TARGET = pd.Timestamp("2025-07-02T00:00:00")


def _source_table(positive: bool) -> pa.Table:
    if positive:
        rows = {
            "XOB": [350.0, 20.0],
            "YOB": [40.0, -20.0],
            "DHR": [1.0, 2.5],
            "POB": [1010.0, 600.0],
            "ZOB": [100.0, np.nan],
            "CAT": [0.0, 4.0],
            "TYP": [187.0, 290.0],
            "TOB": [10.0, np.nan],
            "TQM": [2.0, 15.0],
            "QOB": [6000.0, np.nan],
            "QQM": [2.0, 15.0],
            "UOB": [np.nan, 20.0],
            "VOB": [np.nan, -10.0],
            "WQM": [15.0, 1.0],
            "PQM": [2.0, 2.0],
        }
    else:
        rows = {
            "XOB": [10.0, 200.0, 300.0, 40.0],
            "YOB": [30.0, 45.0, -10.0, 5.0],
            "DHR": [-1.0, -2.0, 0.0, -0.5],
            "POB": [1000.0, 500.0, 700.0, 850.0],
            "ZOB": [50.0, 5500.0, 3000.0, 1200.0],
            "CAT": [0.0, 1.0, 4.0, 2.0],
            "TYP": [181.0, 120.0, 220.0, 133.0],
            "TOB": [20.0, 0.0, np.nan, 50.0],
            "TQM": [2.0, 1.0, 15.0, 14.0],
            "QOB": [5000.0, 3000.0, np.nan, np.nan],
            "QQM": [2.0, 1.0, 15.0, 15.0],
            "UOB": [np.nan, np.nan, 10.0, np.nan],
            "VOB": [np.nan, np.nan, -5.0, np.nan],
            "WQM": [15.0, 15.0, 2.0, 15.0],
            "PQM": [2.0, 2.0, 2.0, 2.0],
        }
    return pa.table(
        {name: pa.array(values, type=pa.float32()) for name, values in rows.items()}
    )


def _write_archive(root) -> str:
    directory = root / "2025"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gdas.20250702.t00z.prepbufr.nr.parquet"
    schema = _source_table(False).schema
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(_source_table(False))
        writer.write_table(_source_table(True))
    return str(root)


def _write_channel_table(root) -> str:
    path = root / "channel_table.parquet"
    local_ids = np.arange(3, 8, dtype=np.uint16)
    pq.write_table(
        pa.table(
            {
                GLOBAL_CHANNEL_ID.name: pa.array(
                    SENSOR_OFFSET["conv"] + local_ids, type=GLOBAL_CHANNEL_ID.type
                ),
                "mean": pa.array([1000.0, 0.005, 280.0, 0.0, 0.0], type=pa.float32()),
                "stddev": pa.array([10.0, 0.001, 5.0, 10.0, 10.0], type=pa.float32()),
                "min_valid": pa.array(
                    [500.0, 0.0, 150.0, -100.0, -100.0], type=pa.float32()
                ),
                "max_valid": pa.array(
                    [1100.0, 1.0, 350.0, 100.0, 100.0], type=pa.float32()
                ),
            }
        ),
        path,
    )
    return str(path)


def _loader(tmp_path, **overrides) -> NNJAConvLoader:
    options = {
        "archive_root": _write_archive(tmp_path / "cycles"),
        "normalize": False,
        "include_gpsro": False,
    }
    options.update(overrides)
    return NNJAConvLoader(**options)


def test_sel_time_unpivots_units_qc_and_cycle_halves(tmp_path):
    loader = _loader(tmp_path)
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]

    assert table.schema == loader.output_schema
    assert table.num_rows == 12
    windows = pd.to_datetime(np.asarray(table["DA_window"]))
    assert set(windows) == {TARGET, TARGET + pd.Timedelta(hours=3)}
    assert set(np.asarray(table["local_channel_id"])) == {3, 4, 5, 6, 7}
    assert set(np.asarray(table["Analysis_Use_Flag"])) == {1}

    local = np.asarray(table["local_channel_id"])
    observation = np.asarray(table["Observation"])
    np.testing.assert_allclose(np.sort(observation[local == 4]), [0.003, 0.005, 0.006])
    np.testing.assert_allclose(
        np.sort(observation[local == 5]), [273.15, 283.15, 293.15], atol=2e-5
    )
    np.testing.assert_allclose(np.sort(observation[local == 6]), [10.0, 20.0])
    np.testing.assert_allclose(np.sort(observation[local == 7]), [-10.0, -5.0])
    ascat = np.asarray(table["Observation_Type"]) == 290
    np.testing.assert_allclose(np.unique(np.asarray(table["Height"])[ascat]), [10.0])

    # POB on the type-120 profile level is a coordinate, not a ps observation.
    np.testing.assert_allclose(np.sort(observation[local == 3]), [1000.0, 1010.0])
    assert 200.0 in np.asarray(table["Longitude"])
    assert 350.0 in np.asarray(table["Longitude"])

    times = set(np.asarray(table["Absolute_Obs_Time"]))
    assert (TARGET - pd.Timedelta(hours=2)).to_datetime64() in times
    assert (TARGET + pd.Timedelta(hours=2.5)).to_datetime64() in times


def test_normalization_uses_packaged_conv_channel_statistics(tmp_path):
    """z-score comes from the packaged conv vocabulary, not from any archive table.

    Asserted as a relation between the two modes rather than against literals, so
    regenerating the normalization CSVs cannot silently invalidate it.
    """
    conv = SENSOR_CONFIGS["conv"]

    def observations(root, normalize):
        # Fresh root per call: the helper writes its fixtures on construction.
        loader = _loader(root, normalize=normalize)
        table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
        order = np.lexsort(
            (np.asarray(table["Observation"]), np.asarray(table["local_channel_id"]))
        )
        return (
            np.asarray(table["local_channel_id"])[order],
            np.asarray(table["Observation"])[order],
        )

    local, physical = observations(tmp_path / "physical", False)
    local_z, z = observations(tmp_path / "z", True)

    np.testing.assert_array_equal(local, local_z)
    expected = (physical - conv.means[local]) / conv.stds[local]
    np.testing.assert_allclose(z, expected, rtol=1e-5)


def test_quality_filter_can_be_disabled_but_flag_is_preserved(tmp_path):
    loader = _loader(tmp_path, max_quality_mark=None)
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    local = np.asarray(table["local_channel_id"])
    flags = np.asarray(table["Analysis_Use_Flag"])

    assert table.num_rows == 13
    assert np.count_nonzero((local == 5) & (flags == 0)) == 1


def test_wind_dropout_targets_ascat_but_preserves_other_winds(tmp_path):
    loader = _loader(tmp_path, wind_obs_dropout=1.0)
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    report_type = np.asarray(table["Observation_Type"])

    assert 290 not in report_type
    assert set(report_type[np.isin(np.asarray(table["local_channel_id"]), [6, 7])]) == {
        220
    }


def test_withhold_stations_drops_only_held_out_sondes_and_land_stations(tmp_path):
    candidates = np.array([f"{i:05d}" for i in range(100)], dtype=object)
    held, kept = (
        candidates[is_withheld(candidates)][0],
        candidates[~is_withheld(candidates)][0],
    )
    # Negative-half rows are report types 181, 120, 220, 133 (aircraft, never held out).
    table = _source_table(False).append_column(
        "SID", pa.array([held, held, kept, held])
    )
    directory = tmp_path / "cycles" / "2025"
    directory.mkdir(parents=True)
    pq.write_table(table, directory / "gdas.20250702.t00z.prepbufr.nr.parquet")

    def report_types(withhold):
        loader = NNJAConvLoader(
            archive_root=str(tmp_path / "cycles"),
            normalize=False,
            include_gpsro=False,
            max_quality_mark=None,
            withhold_stations=withhold,
        )
        out = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
        return set(np.asarray(out["Observation_Type"]))

    assert report_types(False) == {181, 120, 220, 133}
    assert report_types(True) == {220, 133}


def test_missing_cycles_return_typed_empty_tables(tmp_path):
    loader = NNJAConvLoader(
        archive_root=str(tmp_path / "missing"),
        include_gpsro=False,
    )
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert table.num_rows == 0
    assert table.schema == loader.output_schema


def test_drop_restricted_aircraft_removes_acars_and_amdar(tmp_path):
    archive = tmp_path / "restricted" / "2025"
    archive.mkdir(parents=True)
    rows = {
        "XOB": [10.0, 20.0, 30.0],
        "YOB": [30.0, 40.0, 50.0],
        "DHR": [-1.0, -1.0, -1.0],
        "POB": [850.0, 400.0, 300.0],
        "ZOB": [1500.0, 7000.0, 9000.0],
        "CAT": [1.0, 1.0, 1.0],
        "TYP": [181.0, 133.0, 233.0],
        "TOB": [15.0, -20.0, np.nan],
        "TQM": [1.0, 1.0, 15.0],
        "QOB": [4000.0, 2000.0, np.nan],
        "QQM": [1.0, 1.0, 15.0],
        "UOB": [np.nan, np.nan, 12.0],
        "VOB": [np.nan, np.nan, -8.0],
        "WQM": [15.0, 15.0, 1.0],
        "PQM": [1.0, 15.0, 15.0],
    }
    pq.write_table(
        pa.table(
            {name: pa.array(values, type=pa.float32()) for name, values in rows.items()}
        ),
        archive / "gdas.20250702.t00z.prepbufr.nr.parquet",
    )
    options = {
        "archive_root": str(archive.parent),
        "normalize": False,
        "include_gpsro": False,
    }
    kept = NNJAConvLoader(**options, drop_restricted_aircraft=True)
    table = asyncio.run(kept.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert set(np.asarray(table["Observation_Type"])) == {181}

    included = NNJAConvLoader(**options, drop_restricted_aircraft=False)
    table = asyncio.run(included.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert set(np.asarray(table["Observation_Type"])) == {181, 133, 233}


def _write_prepbufr_amv_case(root) -> tuple[str, str]:
    root.mkdir()
    source = _source_table(False).slice(2, 1)
    source = source.set_column(
        source.schema.get_field_index("TYP"),
        "TYP",
        pa.array([245.0], type=pa.float32()),
    )
    archive = root / "cycles" / "2025"
    archive.mkdir(parents=True)
    pq.write_table(
        source,
        archive / "gdas.20250702.t00z.prepbufr.nr.parquet",
    )
    return str(archive.parent)


def test_missing_satwnd_file_yields_no_amvs_rather_than_prepbufr_ones(tmp_path):
    """A missing archive day may yield fewer AMVs. It must never substitute PrepBUFR's,
    which are thinned and QC'd differently."""
    cycles = _write_prepbufr_amv_case(tmp_path / "fallback")
    loader = NNJAConvLoader(
        archive_root=cycles,
        normalize=False,
        include_gpsro=False,
        include_satwnd=True,
        satwnd_archive_root=str(tmp_path / "missing_satwnd"),
        obs_context_hours=(-3, 0),
    )

    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]

    amv = np.isin(np.asarray(table["Observation_Type"]), tuple(SATWND_REPORT_TYPES))
    assert not amv.any()


def test_present_satwnd_window_suppresses_prepbufr_amv_duplicate(tmp_path):
    cycles = _write_prepbufr_amv_case(tmp_path / "dedupe")
    satwnd = tmp_path / "satwnd"
    satwnd.mkdir()
    pq.write_table(
        pa.table(
            {
                "time_utc": pa.array([TARGET.to_datetime64()]),
                "da_window": pa.array([TARGET.to_datetime64()]),
                "cycle": pa.array([TARGET.to_datetime64()]),
                "latitude": pa.array([10.0], type=pa.float64()),
                "longitude": pa.array([20.0], type=pa.float64()),
                "hpx4096_nest": pa.array([123 << 14], type=pa.uint32()),
                "wind_u_derived": pa.array([2.0], type=pa.float64()),
                "wind_v_derived": pa.array([-2.0], type=pa.float64()),
                "assigned_pressure": pa.array([50_000.0], type=pa.float64()),
                "gsi_observation_type": pa.array([245], type=pa.uint16()),
                "gsi_type_mapping_tier": pa.array(
                    ["1-direct-gsi-sattab"], type=pa.string()
                ),
                # Keys for read-time type resolution; the stored type is only a
                # fallback, so the loader requires them.
                "ncep_dump_subtype": pa.array(["NC005030"], type=pa.string()),
                "satellite_id": pa.array([270.0], type=pa.float64()),
                "wind_computation_method": pa.array([1.0], type=pa.float64()),
                "height": pa.array([None], type=pa.float64()),
                "sdmedit_wind_quality_mark": pa.array([None], type=pa.uint32()),
            }
        ),
        satwnd / "20250702.parquet",
    )
    loader = NNJAConvLoader(
        archive_root=cycles,
        normalize=False,
        include_gpsro=False,
        include_satwnd=True,
        satwnd_archive_root=str(satwnd),
        obs_context_hours=(-3, 0),
    )

    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]

    assert table.num_rows == 2
    np.testing.assert_allclose(np.sort(np.asarray(table["Observation"])), [-2.0, 2.0])


def test_obs_context_hours_must_be_3h_aligned(tmp_path):
    with pytest.raises(
        ValueError, match="obs_context_hours endpoints must be 3-hour aligned"
    ):
        NNJAConvLoader(
            include_gpsro=False,
            obs_context_hours=(-2, 3),
        )
    # Valid endpoints still construct.
    loader = NNJAConvLoader(
        include_gpsro=False,
        obs_context_hours=(-3, 3),
    )
    assert loader.obs_context_hours == (-3, 3)


def _surface_wind_table() -> pa.Table:
    """Surface winds as the archive holds them: a pressure and an ELV, no ZOB on any row.

    282 is the control. GSI does place it, at ELV+20 m rather than +10, but we exclude it
    because its POB is a 1013 hPa placeholder -- so it must stay unplaced either way.
    """
    rows = {
        "XOB": [10.0, 20.0, 30.0, 40.0, 50.0],
        "YOB": [10.0, 20.0, 30.0, 40.0, 50.0],
        "DHR": [0.0, 0.0, 0.0, 0.0, 0.0],
        "POB": [1010.0, 1005.0, 1000.0, 1013.0, 1008.0],
        "ZOB": [np.nan] * 5,
        "ELV": [0.0, 150.0, 300.0, 50.0, 25.0],
        "CAT": [0.0] * 5,
        "TYP": [280.0, 281.0, 287.0, 282.0, 280.0],
        "TOB": [np.nan] * 5,
        "TQM": [15.0] * 5,
        "QOB": [np.nan] * 5,
        "QQM": [15.0] * 5,
        "UOB": [5.0, 6.0, 7.0, 8.0, 9.0],
        "VOB": [-5.0, -6.0, -7.0, -8.0, -9.0],
        # The last row is marker 3: usable to NCEP, dropped by the default cut of 2.
        "WQM": [2.0, 2.0, 2.0, 2.0, 3.0],
        "PQM": [15.0] * 5,
    }
    return pa.table(
        {name: pa.array(values, type=pa.float32()) for name, values in rows.items()}
    )


def _surface_loader(tmp_path, **overrides) -> NNJAConvLoader:
    root = tmp_path / "sfc"
    directory = root / "2025"
    directory.mkdir(parents=True, exist_ok=True)
    table = _surface_wind_table()
    with pq.ParquetWriter(
        directory / "gdas.20250702.t00z.prepbufr.nr.parquet", table.schema
    ) as writer:
        writer.write_table(table)
    options = {
        "archive_root": str(root),
        "normalize": False,
        "include_gpsro": False,
    }
    options.update(overrides)
    return NNJAConvLoader(**options)


def _surface_table(loader) -> pa.Table:
    return asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]


def test_surface_winds_fills_the_height_the_archive_omits(tmp_path):
    """The flag's whole effect in the loader: it populates Height, it does not gate rows.

    The rows load either way; what changes is whether Height is finite, which is what
    the conventional filter screens on downstream.
    """
    off = _surface_table(_surface_loader(tmp_path / "off"))
    on = _surface_table(_surface_loader(tmp_path / "on", surface_winds=True))

    assert off.num_rows == on.num_rows
    assert not np.isfinite(np.asarray(off["Height"])).any()

    report_type, height = np.asarray(on["Observation_Type"]), np.asarray(on["Height"])
    for typ, elv in ((280, 0.0), (281, 150.0), (287, 300.0)):
        np.testing.assert_allclose(height[report_type == typ], elv + 10.0)
    # 282 is a surface wind the flag deliberately does not recover.
    assert not np.isfinite(height[report_type == 282]).any()


def test_surface_winds_is_what_carries_them_through_the_conventional_filter(tmp_path):
    """End to end: without a height the conv filter drops every surface wind."""
    off = _surface_table(_surface_loader(tmp_path / "off"))
    on = _surface_table(_surface_loader(tmp_path / "on", surface_winds=True))

    def kept(table):
        mask = (
            pc.fill_null(_get_conv_filter_mask(table), False)
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        return set(np.asarray(table["Observation_Type"])[mask].tolist())

    assert not kept(off) & set(SURFACE_WIND_TYPES)
    assert kept(on) & set(SURFACE_WIND_TYPES) == {280, 281, 287}
    assert 282 not in kept(on)


def test_surface_winds_does_not_relax_the_quality_mark(tmp_path):
    """The marker-3 row stays out until max_quality_mark is raised, flag or no flag."""
    strict = _surface_table(_surface_loader(tmp_path / "s", surface_winds=True))
    relaxed = _surface_table(
        _surface_loader(tmp_path / "r", surface_winds=True, max_quality_mark=3)
    )
    strict_types = np.asarray(strict["Observation_Type"])
    relaxed_types = np.asarray(relaxed["Observation_Type"])

    assert np.count_nonzero(strict_types == 280) < np.count_nonzero(
        relaxed_types == 280
    )


@pytest.mark.parametrize(
    "gpsro_saids, expected",
    [("legacy", GPS_LEGACY_SAIDS), ("full", None)],
)
def test_gpsro_saids_selects_the_said_allowlist(tmp_path, gpsro_saids, expected):
    loader = _surface_loader(tmp_path, include_gpsro=True, gpsro_saids=gpsro_saids)

    assert loader.gpsro.said_allowlist is expected


def test_unknown_gpsro_saids_raises_rather_than_falling_back_to_legacy(tmp_path):
    with pytest.raises(ValueError, match='gpsro_saids must be "legacy" or "full"'):
        _surface_loader(tmp_path, include_gpsro=True, gpsro_saids="all")


def test_checkpoints_written_before_the_rename_still_load():
    """ObsConfig validates the value, so without the alias a checkpoint trained before
    "ufs2024" became "legacy" cannot be read at all.
    """
    from healda.config.models import ObsConfig

    assert ObsConfig(nnja_gpsro_saids="ufs2024").nnja_gpsro_saids == "legacy"
    assert ObsConfig(nnja_gpsro_saids="all").nnja_gpsro_saids == "full"


def test_synthetic_tropical_cyclone_winds_are_not_loaded(tmp_path):
    rows = {
        "XOB": [10.0, 20.0],
        "YOB": [10.0, 20.0],
        "DHR": [0.0, 0.0],
        "POB": [500.0, 500.0],
        "ZOB": [5500.0, 5500.0],
        "CAT": [1.0, 1.0],
        "TYP": [210.0, 220.0],
        "TOB": [np.nan] * 2,
        "TQM": [15.0] * 2,
        "QOB": [np.nan] * 2,
        "QQM": [15.0] * 2,
        "UOB": [5.0, 6.0],
        "VOB": [-5.0, -6.0],
        "WQM": [2.0, 2.0],
        "PQM": [15.0] * 2,
    }
    table = pa.table({k: pa.array(v, type=pa.float32()) for k, v in rows.items()})
    directory = tmp_path / "2025"
    directory.mkdir(parents=True)
    pq.write_table(table, directory / "gdas.20250702.t00z.prepbufr.nr.parquet")
    loader = NNJAConvLoader(
        archive_root=str(tmp_path), normalize=False, include_gpsro=False
    )
    loaded = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert set(np.asarray(loaded["Observation_Type"]).tolist()) == {220}


@pytest.mark.parametrize("fill", [None, "sonde"])
def test_pressure_height_fill(tmp_path, fill):
    # 220 without ZOB, 220 with ZOB, 120 without ZOB, 282 without ZOB.
    rows = {
        "XOB": [10.0, 20.0, 30.0, 40.0],
        "YOB": [10.0, 20.0, 30.0, 40.0],
        "DHR": [0.0] * 4,
        "POB": [500.0, 300.0, 700.0, 1013.0],
        "ZOB": [np.nan, 9000.0, np.nan, np.nan],
        "CAT": [1.0, 4.0, 1.0, 0.0],
        "TYP": [220.0, 220.0, 120.0, 282.0],
        "TOB": [np.nan, np.nan, 5.0, np.nan],
        "TQM": [2.0] * 4,
        "QOB": [np.nan] * 4,
        "QQM": [15.0] * 4,
        "UOB": [5.0, 6.0, np.nan, 7.0],
        "VOB": [-5.0, -6.0, np.nan, -7.0],
        "WQM": [2.0] * 4,
        "PQM": [15.0] * 4,
    }
    directory = tmp_path / "2025"
    directory.mkdir()
    pq.write_table(
        pa.table({k: pa.array(v, type=pa.float32()) for k, v in rows.items()}),
        directory / "gdas.20250702.t00z.prepbufr.nr.parquet",
    )
    loader = NNJAConvLoader(
        archive_root=str(tmp_path),
        normalize=False,
        include_gpsro=False,
        pressure_height_fill=fill,
    )
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    pressure = np.asarray(table["Pressure"])
    height = dict(zip(pressure.tolist(), np.asarray(table["Height"]).tolist()))

    assert height[300.0] == pytest.approx(9000.0)
    assert np.isnan(height[1013.0])
    assert np.isfinite(height[500.0]) == (fill is not None)
    if fill is not None:
        assert height[500.0] == pytest.approx(5574, abs=2)
    assert np.isnan(height[700.0])


def test_holdout_covers_every_assimilated_land_station_type():
    from healda.observations.loaders import nnja_conventional as conv

    land_mass, land_wind = {181, 183, 187}, {281, 284, 287}
    assimilated = (conv.T_REPORT_TYPES | conv.Q_REPORT_TYPES | conv.UV_REPORT_TYPES) & (
        land_mass | land_wind
    )
    assert assimilated <= conv.HOLDOUT_REPORT_TYPES


def test_station_holdout_is_deterministic_and_about_ten_percent():
    ids = np.array([f"{i:05d}" for i in range(20000)], dtype=object)
    withheld = is_withheld(ids)
    assert 0.08 < withheld.mean() < 0.12
    np.testing.assert_array_equal(withheld, is_withheld([f" {i} " for i in ids]))
    assert not is_withheld(["", None, np.nan]).any()

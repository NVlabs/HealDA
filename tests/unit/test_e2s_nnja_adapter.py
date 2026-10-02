# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""earth2studio NNJA frames restated as archive tables and read by the archive loaders."""

import asyncio

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from healda.observations.adapters import e2s_nnja
from healda.observations.loaders.nnja_conventional import NNJAConvLoader
from healda.observations.loaders.nnja_gpsro import NNJAGpsroLoader
from healda.observations.loaders.nnja_satwnd import NNJASatwndLoader
from healda.observations.loaders.nnja_wide import NNJAWideLoader
from healda.observations.preprocessing import ir_spectral, satwnd_gsi_types
from healda.observations.preprocessing.gpsro_pressure import derive_level_coordinates
from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels

CYCLE = pd.Timestamp("2022-01-01T00")
ELRC = 6_371_000.0


def _gps_frame(time: pd.Timestamp) -> pd.DataFrame:
    heights = np.arange(0.0, 40_001.0, 500.0)
    refractivity = 300.0 * np.exp(-heights / 7_000.0)
    impact_heights = np.array([2_500.0, 10_000.0, 25_000.0])
    common = {
        "time": time,
        "type": 750,
        "station": "07500027",
        "quality": 0,
        "radius_curvature": ELRC,
        "geoid_undulation": 10.0,
        "class": "GPSRO",
        "cycle_time": CYCLE,
    }
    levels = pd.DataFrame(
        {
            **common,
            "lat": [-9.0, -9.5, -10.0],
            "lon": [290.0, 290.5, 291.0],
            "elev": impact_heights,
            "observation": [0.02, 0.005, 0.001],
            "variable": "gps",
        }
    )
    profile = pd.DataFrame(
        {
            **common,
            "lat": -10.5,
            "lon": 289.75,
            "elev": heights,
            "observation": refractivity,
            "variable": "gps_refractivity",
        }
    )
    return pd.concat([levels, profile], ignore_index=True)


def test_gpsro_tables_derive_with_the_etl_function():
    frame = _gps_frame(CYCLE - pd.Timedelta(minutes=30))
    tables = e2s_nnja.gpsro_tables(frame)
    assert list(tables) == [CYCLE]
    table = tables[CYCLE].to_pandas()
    assert len(table) == 3
    profile = frame[frame["variable"] == "gps_refractivity"]
    expected = derive_level_coordinates(
        np.array([2_500.0, 10_000.0, 25_000.0]) + ELRC,
        profile["elev"].to_numpy(),
        profile["observation"].to_numpy(),
        latitude_degrees=-10.5,
        earth_radius_of_curvature_m=ELRC,
        geoid_undulation_m=10.0,
    )
    for name, values in expected.items():
        np.testing.assert_allclose(table[name], values)
    assert (table["occ_latitude"] == -10.5).all()
    assert table["transmitter_id"].tolist() == [27, 27, 27]
    np.testing.assert_allclose(
        table["impact_parameter"], table["impact_height_m"] + ELRC
    )


@pytest.mark.parametrize("file_cycle", [CYCLE, CYCLE + pd.Timedelta(hours=6)])
def test_rows_are_keyed_by_the_file_they_were_decoded_from(file_cycle):
    # An observation at CYCLE + 3h sits at the edge of both files' windows.
    frame = _gps_frame(CYCLE + pd.Timedelta(hours=3)).assign(cycle_time=file_cycle)
    assert list(e2s_nnja.gpsro_tables(frame)) == [file_cycle]


def test_tz_aware_times_key_like_naive_utc():
    frame = _gps_frame(CYCLE)

    def aware(column):
        return frame[column].dt.tz_localize("UTC").dt.tz_convert("US/Pacific")

    frame = frame.assign(time=aware("time"), cycle_time=aware("cycle_time"))
    assert list(e2s_nnja.gpsro_tables(frame)) == [CYCLE]


def test_gpsro_loader_reads_tables_like_an_archive_file():
    tables = e2s_nnja.gpsro_tables(_gps_frame(CYCLE - pd.Timedelta(minutes=30)))
    loader = NNJAGpsroLoader(said_allowlist=None, normalize=False, table_source=tables)
    ((window, table),) = loader._load_cycle(CYCLE, (False, True))
    assert window == CYCLE
    rows = table.to_pandas()
    assert len(rows) == 3
    # Height is HEIT at the impact parameter, pressure the blended candidate.
    derived = tables[CYCLE].to_pandas()
    np.testing.assert_allclose(rows["Height"], derived["heit_at_impact_m"], rtol=1e-6)
    np.testing.assert_allclose(
        rows["Pressure"], derived["blended_pressure_5km_hpa"], rtol=1e-6
    )
    assert loader._load_cycle(CYCLE + pd.Timedelta(hours=6), (False,)) == []


def _wind_frame() -> pd.DataFrame:
    winds = pd.DataFrame(
        {
            "time": [CYCLE - pd.Timedelta(hours=3), CYCLE + pd.Timedelta(hours=1)],
            "lat": [10.0, 12.0],
            "lon": [70.0, 250.0],
            "pres": [30_000.0, 50_000.0],
            "satellite_id": [473, 270],
            "subset": ["NC005024", "NC005030"],
            "wind_method": [1.0, 1.0],
            "wind_method_local": [np.nan, np.nan],
            "height_method": [np.nan, np.nan],
            "satellite_za": [20.0, 30.0],
            "quality": [np.nan, np.nan],
            "class": "SATWND",
            "cycle_time": CYCLE,
        }
    )
    u = winds.assign(observation=[5.0, -3.0], variable="u")
    v = winds.assign(observation=[1.0, 2.0], variable="v")
    return pd.concat([u, v], ignore_index=True)


def test_satwnd_tables_rejoin_components_and_type_every_wind():
    table = e2s_nnja.satwnd_tables(_wind_frame())[CYCLE].to_pandas()
    assert table["wind_u_derived"].tolist() == [5.0, -3.0]
    assert table["wind_v_derived"].tolist() == [1.0, 2.0]
    # INSAT-3DR is typed through the PREPDATA route; GOES-R through GSI's table.
    assert table["gsi_observation_type"].tolist() == [241, 245]
    assert table["gsi_type_mapping_tier"].tolist() == [
        "3-prepdata",
        "1-direct-gsi-sattab",
    ]
    assert pd.isna(table["gsi_internal_subtype"].iloc[0])
    assert table["gsi_internal_subtype"].iloc[1] == 15
    # A wind at cycle - 3h belongs to the cycle's own window, as the ETL writes it.
    assert table["da_window"].tolist() == [CYCLE, CYCLE + pd.Timedelta(hours=3)]
    assert (table["cycle"] == CYCLE).all()


def test_satwnd_loader_reads_tables_per_window():
    tables = e2s_nnja.satwnd_tables(_wind_frame())
    loader = NNJASatwndLoader(normalize=False, table_source=tables)
    loaded = dict(loader._load_day(CYCLE, [CYCLE, CYCLE + pd.Timedelta(hours=3)]))
    assert set(loaded) == {CYCLE, CYCLE + pd.Timedelta(hours=3)}
    first = loaded[CYCLE].to_pandas()
    assert first["Observation_Type"].unique().tolist() == [241]
    assert sorted(first["Observation"].tolist()) == [1.0, 5.0]


def test_cmcm_types_only_subsets_without_swcm():
    types = satwnd_gsi_types.resolve(
        np.array(["NC005010", "NC005010", "NC005030"], dtype=object),
        np.array([257.0, 257.0, 270.0]),
        np.array([np.nan, np.nan, np.nan]),
        np.array([1.0, 1.0, 1.0]),
    )
    # Both subsets lack SWCM entirely, so CMCM stands in for each.
    assert types.report_type.tolist() == [245, 245, 245]
    partial = satwnd_gsi_types.resolve(
        np.array(["NC005010", "NC005010"], dtype=object),
        np.array([257.0, 257.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, 1.0]),
    )
    # SWCM present in the subset: a row missing it stays untyped.
    assert partial.report_type.tolist() == [245, -1]


def test_null_subsets_stay_untyped():
    types = satwnd_gsi_types.resolve(
        np.array(["NC005010", None, np.nan, pd.NA], dtype=object),
        np.array([257.0] * 4),
        np.array([1.0] * 4),
    )
    assert types.report_type.tolist() == [245, -1, -1, -1]


def test_satwnd_rejects_v_rows_out_of_wind_order():
    frame = _wind_frame()
    with pytest.raises(ValueError, match="wind order"):
        e2s_nnja.satwnd_tables(frame.iloc[[3, 0, 2, 1]])


def test_derivation_screens_unusable_refractivity_levels_itself():
    heights = np.arange(0.0, 40_001.0, 500.0)
    refractivity = 300.0 * np.exp(-heights / 7_000.0)
    impacts = np.array([2_500.0, 10_000.0, 25_000.0]) + ELRC
    kwargs = dict(
        latitude_degrees=-10.5,
        earth_radius_of_curvature_m=ELRC,
        geoid_undulation_m=10.0,
    )
    screened = derive_level_coordinates(impacts, heights, refractivity, **kwargs)
    # The same profile as decoded: a missing HEIT, a missing ARFR and a non-positive ARFR.
    raw_heights = np.concatenate([heights, [np.nan, 1_250.0, 2_750.0]])
    raw_refractivity = np.concatenate([refractivity, [250.0, np.nan, -5.0]])
    raw = derive_level_coordinates(impacts, raw_heights, raw_refractivity, **kwargs)
    for name in screened:
        np.testing.assert_array_equal(raw[name], screened[name])


def _conv_frame() -> pd.DataFrame:
    # A radiosonde's surface and drifted 850 hPa levels and a wind report, as
    # NNJAObsConv returns them: one row per variable, in Pa, K and kg/kg.
    sonde = {
        "time": CYCLE - pd.Timedelta(hours=1),
        "report_time": CYCLE - pd.Timedelta(hours=1),
        "station": "72403",
        "type": 120,
        "lat": 38.98,
        "lon": 282.53,
        "report_lat": 38.98,
        "report_lon": 282.53,
        "station_elev": 88.0,
        "pressure_quality": 2,
        "class": "ADPUPA",
        "cycle_time": CYCLE,
    }
    surface = dict(sonde, level_cat=0, pres=100_000.0, elev=88.0, quality=2)
    upper = dict(sonde, level_cat=1, pres=85_000.0, elev=1_500.0, quality=1)
    drifted = dict(upper, time=CYCLE - pd.Timedelta(minutes=30), lat=39.0, lon=282.75)
    wind = dict(sonde, time=CYCLE + pd.Timedelta(hours=1), type=220, quality=2)
    wind.update(report_time=wind["time"], level_cat=1, pres=85_000.0, elev=1_500.0)
    rows = [
        dict(surface, variable="pres", observation=100_000.0),
        dict(surface, variable="t", observation=np.float32(15.3) + 273.15),
        dict(surface, variable="q", observation=np.float32(8_000 * 1e-6)),
        dict(drifted, variable="t", observation=np.float32(4.1) + 273.15),
        dict(wind, variable="u", observation=7.5),
        dict(wind, variable="v", observation=-2.0),
    ]
    frame = pd.DataFrame(rows)
    frame["observation"] = frame["observation"].astype(np.float32)
    return frame


def test_prepbufr_tables_restate_one_row_per_level_in_archive_units():
    ((cycle, table),) = e2s_nnja.prepbufr_tables(_conv_frame()).items()
    assert cycle == CYCLE
    surface, upper, wind = table.to_pylist()
    assert (surface["POB"], surface["PQM"]) == (1000.0, 2.0)
    assert surface["TOB"] == np.float32(15.3)
    assert surface["QOB"] == 8000.0
    assert (upper["TOB"], upper["TQM"]) == (np.float32(4.1), 1.0)
    assert upper["QOB"] is None
    assert (wind["UOB"], wind["VOB"], wind["WQM"]) == (7.5, -2.0, 2.0)
    assert [row["DHR"] for row in (surface, upper, wind)] == [-1.0, -1.0, 1.0]
    assert (upper["XOB"], upper["YOB"]) == (np.float32(282.53), np.float32(38.98))
    assert (upper["XDR"], upper["YDR"]) == (282.75, 39.0)
    assert [row["HRDR"] for row in (surface, upper, wind)] == [-1.0, -0.5, 1.0]


@pytest.mark.parametrize(
    "file_cycle, dhr", [(CYCLE, 3.0), (CYCLE + pd.Timedelta(hours=6), -3.0)]
)
def test_prepbufr_hours_are_relative_to_the_file_cycle(file_cycle, dhr):
    edge = CYCLE + pd.Timedelta(hours=3)
    frame = _conv_frame().assign(time=edge, report_time=edge, cycle_time=file_cycle)
    ((cycle, table),) = e2s_nnja.prepbufr_tables(frame).items()
    assert cycle == file_cycle
    assert set(table["DHR"].to_pylist()) == set(table["HRDR"].to_pylist()) == {dhr}


def test_conv_loader_reads_prepbufr_tables_like_an_archive_file(tmp_path):
    tables = e2s_nnja.prepbufr_tables(_conv_frame())
    name = f"gdas.{CYCLE:%Y%m%d}.t{CYCLE:%H}z.prepbufr.nr.parquet"
    (tmp_path / f"{CYCLE:%Y}").mkdir()
    pq.write_table(tables[CYCLE], tmp_path / f"{CYCLE:%Y}" / name)

    def load(**kw):
        loader = NNJAConvLoader(include_gpsro=False, normalize=False, **kw)
        return loader._load_cycle(CYCLE, (False, True))

    lats = {}
    for drift in (False, True):
        from_tables = load(prepbufr_table_source=tables, balloon_drift=drift)
        lats[drift] = {v for _, t in from_tables for v in t["Latitude"].to_pylist()}
        from_archive = load(archive_root=str(tmp_path), balloon_drift=drift)
        assert [w for w, _ in from_tables] == [CYCLE, CYCLE + pd.Timedelta(hours=3)]
        assert [w for w, _ in from_archive] == [w for w, _ in from_tables]
        for (_, a), (_, b) in zip(from_tables, from_archive):
            assert a.equals(b)
    assert lats == {False: {np.float32(38.98)}, True: {np.float32(38.98), 39.0}}
    channels = [c for _, t in from_tables for c in t["local_channel_id"].to_pylist()]
    # ps, q, t at the surface; t at 850 hPa; u, v.
    assert sorted(channels) == [3, 4, 5, 5, 6, 7]


def _sat_frame(sensor: str, channels, detector=None) -> pd.DataFrame:
    # Three footprints of NNJAObsSat rows, footprint-major; the second lacks its
    # first channel, as NNJAObsSat emits no row for a missing observation.
    rows = []
    times = [CYCLE - pd.Timedelta(hours=1), CYCLE, CYCLE + pd.Timedelta(hours=2)]
    for footprint, time in enumerate(times):
        for position, channel in enumerate(channels):
            if footprint == 1 and position == 0:
                continue
            rows.append(
                {
                    "time": time,
                    "lat": 10.0 + footprint,
                    "lon": -20.0 + footprint,
                    "satellite": ["n20", "npp", "n20"][footprint],
                    "scan_position": 5 + footprint,
                    "scan_line": 1,
                    "sensor_index": channel,
                    "satellite_za": 30.0,
                    "solza": 60.0,
                    "observation": 200.0 + position + footprint,
                    "variable": e2s_nnja.SAT_VARIABLE.get(sensor, sensor),
                    "cycle_time": CYCLE,
                }
            )
    frame = pd.DataFrame(rows)
    if detector is not None:
        frame["detector"] = detector
    return frame


def test_sat_tables_pivot_footprints_into_channel_columns():
    frame = _sat_frame("atms", np.arange(1, 23))
    # Brightness temperature is not the archive's ATMS column.
    frame = pd.concat([frame, frame.head(1).assign(variable="atms")])
    (table,) = e2s_nnja.sat_tables(frame, "atms").values()
    rows = table.to_pandas()
    assert rows["platform_id"].tolist() == [225, 224, 225]
    assert rows["field_of_view"].tolist() == [5, 6, 7]
    assert rows["antenna_temperature__ch_00001"].isna().tolist() == [False, True, False]
    assert rows["antenna_temperature__ch_00022"].tolist() == [221.0, 222.0, 223.0]
    windows = pd.to_datetime(rows["da_window"]).dt.tz_localize(None).tolist()
    assert windows == [CYCLE, CYCLE, CYCLE + pd.Timedelta(hours=3)]
    pixels, _ = hpx_nest_pixels(rows["latitude"], rows["longitude"], 11)
    assert rows["hpx2048_nest"].tolist() == pixels.tolist()


def test_cris_tables_store_the_archive_code_and_detector():
    channel = max(ir_spectral.ir_channel_preset("ir32")["cris"])
    frame = _sat_frame("cris", sorted(ir_spectral.ir_channel_preset("ir32")["cris"]), 4)
    (table,) = e2s_nnja.sat_tables(frame, "cris").values()
    rows = table.to_pandas()
    assert rows["field_of_view"].tolist() == [4, 4, 4]
    assert rows["field_of_regard"].tolist() == [5, 6, 7]
    field = table.schema.field(f"spectral_radiance_code__ch_{channel:05d}")
    assert field.type == pa.int32()
    encoding = ir_spectral.archive_value_encoding(field)
    assert encoding == {"reference": -100000, "scale": 7}
    radiance = ir_spectral.radiance_mw("cris", rows[field.name], **encoding)
    kelvin = ir_spectral.brightness_temperature(
        radiance, ir_spectral.wavenumber_cm_inverse("cris", [channel])
    )
    np.testing.assert_allclose(kelvin, [231.0, 232.0, 233.0], atol=0.05)
    with pytest.raises(ValueError, match="detector"):
        e2s_nnja.sat_tables(frame.drop(columns="detector"), "cris")


def test_wide_loader_reads_tables_like_an_archive_file(tmp_path):
    tables = e2s_nnja.sat_tables(_sat_frame("atms", np.arange(1, 23)), "atms")
    # The archive layout: a day file per sensor, one row group per DA window.
    (table,) = tables.values()
    window = table["da_window"].to_numpy()
    (tmp_path / "atms").mkdir()
    path = tmp_path / "atms" / f"{CYCLE:%Y%m%d}.parquet"
    with pq.ParquetWriter(path, table.schema) as out:
        for value in np.unique(window):
            out.write_table(table.filter(pa.array(window == value)))

    def load(**source):
        loader = NNJAWideLoader(["atms"], thin_nside=1, normalize=False, **source)
        return asyncio.run(loader.sel_time(pd.DatetimeIndex([CYCLE])))["obs_v2"][0]

    from_tables = load(table_source={"atms": tables})
    assert from_tables.num_rows > 0
    assert from_tables.equals(load(archive_root=str(tmp_path)))
    assert load(table_source={}).num_rows == 0


def test_analysis_tables_equal_the_per_stream_adapters():
    gps = _gps_frame(CYCLE - pd.Timedelta(minutes=30))
    atms = _sat_frame("atms", np.arange(1, 23))
    pairs = [
        (
            e2s_nnja.analysis_tables(
                conv=_conv_frame(), sensors=(), ir_channels="ir32"
            ),
            "prepbufr_tables",
            e2s_nnja.prepbufr_tables(_conv_frame()),
        ),
        (
            e2s_nnja.analysis_tables(conv=gps, sensors=(), ir_channels="ir32"),
            "gpsro_tables",
            e2s_nnja.gpsro_tables(gps),
        ),
        (
            e2s_nnja.analysis_tables(
                satwnd=_wind_frame(), sensors=(), ir_channels="ir32"
            ),
            "satwnd_tables",
            e2s_nnja.satwnd_tables(_wind_frame()),
        ),
    ]
    sat = e2s_nnja.analysis_tables(sat=atms, sensors=["atms"], ir_channels="ir32")
    pairs.append((sat["satellite_tables"], "atms", e2s_nnja.sat_tables(atms, "atms")))
    for tables, argument, expected in pairs:
        got = tables[argument]
        assert got.keys() == expected.keys() and expected
        for cycle in expected:
            assert got[cycle].equals(expected[cycle])


def test_gpsro_tables_of_a_frame_without_gps_rows_are_empty():
    assert e2s_nnja.gpsro_tables(_gps_frame(CYCLE).assign(variable="t")) == {}

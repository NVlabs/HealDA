# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""earth2studio NNJA frames restated as archive tables and read by the archive loaders."""

import numpy as np
import pandas as pd
import pytest

from healda.observations.adapters import e2s_nnja
from healda.observations.loaders.nnja_gpsro import NNJAGpsroLoader
from healda.observations.loaders.nnja_satwnd import NNJASatwndLoader
from healda.observations.preprocessing import satwnd_gsi_types
from healda.observations.preprocessing.gpsro_pressure import derive_level_coordinates

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


@pytest.mark.parametrize(
    "offset, cycle",
    [
        # NCEP dumps cover [cycle - 3h, cycle + 3h).
        (pd.Timedelta(hours=-3), CYCLE),
        (pd.Timedelta(hours=-3, seconds=-1), CYCLE - pd.Timedelta(hours=6)),
        (pd.Timedelta(hours=3), CYCLE + pd.Timedelta(hours=6)),
    ],
)
def test_rows_are_keyed_by_the_cycle_file_that_holds_them(offset, cycle):
    assert list(e2s_nnja.gpsro_tables(_gps_frame(CYCLE + offset))) == [cycle]


def test_tz_aware_times_key_like_naive_utc():
    frame = _gps_frame(CYCLE)
    aware = frame.assign(
        time=frame["time"].dt.tz_localize("UTC").dt.tz_convert("US/Pacific")
    )
    assert list(e2s_nnja.gpsro_tables(aware)) == [CYCLE]


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


def test_written_archive_reads_back_like_the_tables(tmp_path):
    gps = e2s_nnja.gpsro_tables(_gps_frame(CYCLE - pd.Timedelta(minutes=30)))
    winds = e2s_nnja.satwnd_tables(_wind_frame())
    e2s_nnja.write_gpsro_archive(gps, str(tmp_path / "gpsro"))
    e2s_nnja.write_satwnd_archive(winds, str(tmp_path / "satwnd"))
    windows = [CYCLE, CYCLE + pd.Timedelta(hours=3)]

    def gpsro(**kw):
        loader = NNJAGpsroLoader(said_allowlist=None, normalize=False, **kw)
        return loader._load_cycle(CYCLE, (False, True))

    def satwnd(**kw):
        return NNJASatwndLoader(normalize=False, **kw)._load_day(CYCLE, windows)

    for from_tables, from_archive in (
        (gpsro(table_source=gps), gpsro(archive_root=str(tmp_path / "gpsro"))),
        (satwnd(table_source=winds), satwnd(archive_root=str(tmp_path / "satwnd"))),
    ):
        assert [w for w, _ in from_tables] == [w for w, _ in from_archive]
        for (_, a), (_, b) in zip(from_tables, from_archive):
            assert a.equals(b)


def test_satwnd_rejoin_does_not_depend_on_row_order():
    frame = _wind_frame()
    in_order = e2s_nnja.satwnd_tables(frame)[CYCLE].to_pandas()
    shuffled = e2s_nnja.satwnd_tables(frame.iloc[[3, 0, 2, 1]])[CYCLE].to_pandas()
    columns = ["time_utc", "wind_u_derived", "wind_v_derived", "gsi_observation_type"]
    pd.testing.assert_frame_equal(
        in_order[columns].sort_values("time_utc", ignore_index=True),
        shuffled[columns].sort_values("time_utc", ignore_index=True),
    )


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

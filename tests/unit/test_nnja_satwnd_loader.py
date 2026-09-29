# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NNJA SATWND/AMV conventional-wind loader."""

import asyncio
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from healda.observations.loaders.nnja_satwnd import NNJASatwndLoader
from healda.observations.sensors import SENSOR_OFFSET
from healda.observations.schema import GLOBAL_CHANNEL_ID

TARGET = pd.Timestamp("2025-07-20T00:00:00")


def _write_channel_table(path) -> str:
    local_ids = np.array([6, 7], dtype=np.uint16)
    pq.write_table(
        pa.table(
            {
                GLOBAL_CHANNEL_ID.name: pa.array(
                    SENSOR_OFFSET["conv"] + local_ids, type=GLOBAL_CHANNEL_ID.type
                ),
                "mean": pa.array([0.0, 0.0], type=pa.float32()),
                "stddev": pa.array([10.0, 10.0], type=pa.float32()),
                "min_valid": pa.array([-100.0, -100.0], type=pa.float32()),
                "max_valid": pa.array([100.0, 100.0], type=pa.float32()),
            }
        ),
        path,
    )
    return str(path)


def _write_archive(root) -> str:
    root.mkdir()
    # Rows 0 and 1 compete in one HPX5/pressure/time/report-type cell; row 1
    # is closer to the DA window and must win. Row 3 has a failing quality mark.
    table = pa.table(
        {
            "time_utc": pa.array(
                np.array(
                    [
                        "2025-07-19T23:15:00",
                        "2025-07-19T23:30:00",
                        "2025-07-19T23:00:00",
                        "2025-07-19T23:15:00",
                    ],
                    dtype="datetime64[ns]",
                )
            ),
            "da_window": pa.array(
                np.full(4, TARGET.to_datetime64(), dtype="datetime64[ns]")
            ),
            # Source-file cycle: the QC quarantine keys on it and refuses to
            # run blind rather than passing everything.
            "cycle": pa.array(
                np.full(4, TARGET.to_datetime64(), dtype="datetime64[ns]")
            ),
            "latitude": pa.array([10.0, 10.1, -20.0, 30.0], type=pa.float64()),
            "longitude": pa.array([-5.0, -5.1, 140.0, 20.0], type=pa.float64()),
            "hpx4096_nest": pa.array(
                [(123 << 14) + 1, (123 << 14) + 2, 124 << 14, 125 << 14],
                type=pa.uint32(),
            ),
            "wind_u_derived": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
            "wind_v_derived": pa.array([-1.0, -2.0, -3.0, -4.0], type=pa.float64()),
            "assigned_pressure": pa.array(
                [50_000.0, 50_000.0, 30_000.0, 70_000.0], type=pa.float64()
            ),
            "gsi_observation_type": pa.array([245, 245, 253, 250], type=pa.uint16()),
            "height": pa.array([None, None, None, None], type=pa.float64()),
            "sdmedit_wind_quality_mark": pa.array([None, None, 2, 3], type=pa.uint32()),
            "gsi_type_mapping_tier": pa.array(
                ["1-direct-gsi-sattab"] * 4, type=pa.string()
            ),
            "ncep_dump_subtype": pa.array(["NC005099"] * 4, type=pa.string()),
            "wind_computation_method": pa.array([1.0] * 4, type=pa.float64()),
        }
    )
    pq.write_table(table, root / "20250720.parquet")
    return str(root)


def test_sel_time_thins_amvs_and_emits_conv_uv(tmp_path):
    loader = NNJASatwndLoader(
        archive_root=_write_archive(tmp_path / "satwnd"),
        obs_context_hours=(-3, 0),
        normalize=False,
    )

    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]

    assert table.schema == loader.output_schema
    # 6, not 4: the fixture's quality_mark=3 row is no longer dropped. GSI
    # rejects only marks 12 and 14, and this archive only ever carries 2, so
    # `max_quality_mark` was removed rather than left as inert protection.
    assert table.num_rows == 6
    local = np.asarray(table["local_channel_id"])
    observation = np.asarray(table["Observation"])
    np.testing.assert_allclose(sorted(observation[local == 6]), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(sorted(observation[local == 7]), [-4.0, -3.0, -2.0])
    np.testing.assert_allclose(np.unique(table["Pressure"]), [300.0, 500.0, 700.0])
    assert np.isfinite(np.asarray(table["Height"])).all()
    assert set(np.asarray(table["Observation_Type"])) == {245, 250, 253}
    # The quality_mark=3 row is KEPT but flagged non-assimilable, rather than
    # being silently dropped as it was when max_quality_mark existed.
    assert set(np.asarray(table["Analysis_Use_Flag"])) == {0, 1}


def test_qi_screen_runs_through_qc_config_not_the_promoted_scalar(tmp_path):
    # The old min_per_cent_confidence screened the PROMOTED scalar, which is
    # qify for EUMETSAT/JMA. The replacement resolves by GNAP via satwnd_qc.
    from healda.observations.preprocessing.satwnd_qc import SatwndQCConfig

    loader = NNJASatwndLoader(
        archive_root=_write_archive(tmp_path / "satwnd"),
        obs_context_hours=(-3, 0),
        normalize=False,
        qc_config=SatwndQCConfig(apply_qifn=True),
    )
    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    # No quality_diagnostics column, so no qifn GNAP resolves: the screen must
    # record that it could not be evaluated and keep the rows rather than guess.
    assert table.num_rows == 6


def test_missing_required_column_is_an_error_not_a_silent_drop(tmp_path):
    root = tmp_path / "satwnd"
    root.mkdir()
    pq.write_table(
        pa.table({"time_utc": pa.array([], type=pa.timestamp("ns"))}),
        root / "20250720.parquet",
    )
    loader = NNJASatwndLoader(
        archive_root=str(root),
        obs_context_hours=(-3, 0),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))


def test_missing_daily_file_returns_typed_empty_table(tmp_path):
    loader = NNJASatwndLoader(
        archive_root=str(tmp_path / "missing"),
        obs_context_hours=(-3, 0),
    )

    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert table.num_rows == 0
    assert table.schema == loader.output_schema


def test_wind_dropout_removes_satwnd_pairs(tmp_path):
    loader = NNJASatwndLoader(
        archive_root=_write_archive(tmp_path / "satwnd"),
        obs_context_hours=(-3, 0),
        normalize=False,
        wind_obs_dropout=1.0,
    )

    table = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert table.num_rows == 0


def test_wind_dropout_is_applied_after_thinning(tmp_path):
    archive = _write_archive(tmp_path / "satwnd")
    loader = NNJASatwndLoader(
        archive_root=archive,
        obs_context_hours=(-3, 0),
        normalize=False,
        wind_obs_dropout=0.5,
    )
    source = pq.read_table(f"{archive}/{TARGET:%Y%m%d}.parquet")

    table = loader._to_long(source, TARGET, np.random.default_rng(3), 0.5)

    # Four valid source rows thin to three model-facing observations first.
    # Seed 3 then retains one of those three; dropping before thinning would
    # retain two cells because a redundant source row can replace its winner.
    assert table.num_rows == 2  # one retained observation, emitted as u/v


def test_thinning_keeps_opposite_sides_of_window_in_separate_time_bins(tmp_path):
    archive = _write_archive(tmp_path / "satwnd")
    source = pq.read_table(f"{archive}/{TARGET:%Y%m%d}.parquet")
    times = source["time_utc"].to_numpy(zero_copy_only=False).copy()
    times[0] = (TARGET - pd.Timedelta(hours=2)).to_datetime64()
    times[1] = (TARGET + pd.Timedelta(hours=2)).to_datetime64()
    source = source.set_column(
        source.schema.get_field_index("time_utc"),
        "time_utc",
        pa.array(times, type=pa.timestamp("ns")),
    )
    loader = NNJASatwndLoader(
        archive_root=archive,
        obs_context_hours=(-3, 3),
        normalize=False,
    )

    table = loader._to_long(source, TARGET)

    # Rows 0/1 share HPX, pressure, and report type. Their signed offsets put
    # them in distinct bins; absolute-distance binning would discard one.
    assert table.num_rows == 8


@pytest.mark.parametrize("field", ["pressure_bin_hpa", "time_bin_hours"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_thinning_bin_sizes_must_be_positive(tmp_path, field, value):
    with pytest.raises(ValueError, match=rf"{field} must be positive"):
        NNJASatwndLoader(
            **{field: value},
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_wind_dropout_must_be_a_probability(tmp_path, value):
    with pytest.raises(ValueError, match="wind_obs_dropout must be in"):
        NNJASatwndLoader(
            wind_obs_dropout=value,
        )


def test_null_healpix_pixel_is_dropped_not_cast_to_garbage(tmp_path):
    """A null pixel must not reach the uint64 thinning key.

    NaN -> uint64 is undefined, so such a row lands in an arbitrary cell where
    it can displace a real observation, and numpy reports it only as a warning
    naming a shared helper line. Measured at 1-2 rows on scattered 2001-2002
    cycles -- rare enough that a 238-file sample called it zero.
    """
    root = _write_archive(tmp_path / "satwnd")
    path = f"{root}/{TARGET:%Y%m%d}.parquet"
    table = pq.read_table(path)
    field = table.schema.field("hpx4096_nest")
    pixels = table["hpx4096_nest"].to_pylist()
    pixels[0] = None
    table = table.set_column(
        table.schema.get_field_index("hpx4096_nest"),
        field,
        pa.array(pixels, type=field.type),
    )
    pq.write_table(table, path)

    loader = NNJASatwndLoader(
        archive_root=root,
        normalize=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))

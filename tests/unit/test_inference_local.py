# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses

import numpy as np
import pandas as pd
import pytest
import zarr

from healda.cli.inference_local import (
    enumerate_to_dict,
    find_matching_indices,
    inference_obs_config,
    scoring_times,
)
from healda.config.models import ObsConfig
from healda.observations.system import ChannelDenial, ObsFilters


def test_find_matching_indices_filters_missing():
    available = pd.date_range("2022-01-01", "2022-01-10", freq="12h")
    targets = pd.DatetimeIndex(
        ["2022-01-02 00:00", "2022-01-05 00:00", "2022-01-15 00:00"]
    )

    indices, matched = find_matching_indices(targets, available)

    assert len(indices) == 2, f"Expected 2 matches, got {len(indices)}"
    assert len(matched) == 2, f"Expected 2 matched times, got {len(matched)}"


def test_find_matching_indices_empty_targets():
    available = pd.date_range("2022-01-01", "2022-01-10", freq="12h")
    targets = pd.DatetimeIndex([])

    indices, matched = find_matching_indices(targets, available)

    assert len(indices) == 0
    assert len(matched) == 0


def test_enumerate_to_dict():
    array = np.array([10, 20, 30])
    result = enumerate_to_dict(array)

    assert result == {10: 0, 20: 1, 30: 2}


def test_enumerate_to_dict_handles_duplicates():
    array = np.array([10, 20, 10])
    result = enumerate_to_dict(array)

    # Last occurrence wins
    assert result == {10: 2, 20: 1}


# What a live NNJA arm trained with.
TRAINED = ObsConfig(
    use_obs=True,
    context_start=-3,
    context_end=3,
    use_conv=True,
    use_infrared_pca=True,
    use_airs_pca=True,
    conv_gps_level1_only=True,
    conv_min_pressure_hpa=1.0,
    use_conv_level_stats=True,
    conv_level_channels=True,
    use_nnja_sat=True,
    use_nnja_conv=True,
    use_nnja_satwnd=True,
    nnja_wind_dropout=0.5,
    drop_sensors=("airs-pca",),
)


# The scoring parser's defaults: unset means "keep what the checkpoint has".
_FLAG_DEFAULTS = {
    "conv_uv_in_situ_only": None,
    "conv_gps_level1_only": None,
    "drop_restricted_aircraft": None,
    "balloon_drift": None,
    "withhold_stations": None,
    "conv_min_pressure_hpa": None,
    "ascat_only_scatterometer": None,
    "drop_sensors": [],
    "drop_report_types": [],
    "drop_obs_families": [],
    "drop_obs_channel_ids": [],
    "drop_platform_channels": [],
    "no_denials": False,
    "denials": None,
    "keep_source_flagged": False,
}


def _args(**kwargs):
    return argparse.Namespace(**{**_FLAG_DEFAULTS, **kwargs})


def test_scoring_times_still_defaults_to_the_split_year():
    assert scoring_times(False, "12h", "test")[0].year == 2022
    assert scoring_times(False, "12h", "train")[0].year == 2021


def test_the_checkpoints_observing_system_survives_inference():
    """Rebuilding an ObsConfig from CLI flags scored NNJA arms with the GSI vocabulary,
    because use_nnja_sat defaulted to False and nothing said so."""
    got = inference_obs_config(TRAINED, _args())
    assert got.use_nnja_sat and got.use_nnja_conv and got.use_nnja_satwnd
    assert got.conv_level_channels and got.use_conv_level_stats
    assert (
        got.context_start == -3
    ), "no flag overrides the window; it rides the checkpoint"
    assert got.nnja_ir_channels == TRAINED.nnja_ir_channels


def test_dropped_sensors_union_with_the_checkpoints_own():
    """A scoring run may drop more sensors than training did, never fewer."""
    got = inference_obs_config(TRAINED, _args(drop_sensors=("atms",)))
    assert set(got.drop_sensors) == {"airs-pca", "atms"}


def test_run_removals_are_extra_filters_not_recipe_changes():
    from healda.cli.inference_local import inference_filters_from_args

    args = _args(
        drop_platform_channels=["atms:npp:5"],
        drop_report_types=[188],
        drop_obs_families=["amv"],
    )
    assert inference_obs_config(TRAINED, args) == TRAINED
    extra = inference_filters_from_args(args)
    assert ChannelDenial("atms", "npp", 5) in extra.channel_denials
    assert 188 in extra.report_types and 240 in extra.report_types


def test_an_explicit_flag_overrides_the_checkpoint():
    got = inference_obs_config(TRAINED, _args(conv_uv_in_situ_only=True))
    assert got.conv_uv_in_situ_only is True
    assert got.conv_gps_level1_only == TRAINED.conv_gps_level1_only


def test_inference_never_inherits_training_dropout():
    """nnja_wind_dropout rides in the checkpoint; scoring must not resample observations.
    build_obs_loader(training=False) is what zeroes it, so assert the seam is still there."""
    from healda.observations.system import ObsPipeline

    got = inference_obs_config(TRAINED, _args())
    assert got.nnja_wind_dropout == 0.5, "the config still carries it"
    assert ObsPipeline(got, training=False).random_drop.wind == 0.0
    assert ObsPipeline(got, training=True).random_drop.wind == 0.5


def test_the_output_records_the_observing_system_that_produced_it():
    """Two scoring runs can now differ only in their denials, so the resolved ObsConfig has
    to travel with the zarr or the outputs are indistinguishable afterwards."""
    import json

    from healda.inference_output import provenance_attrs

    denied = dataclasses.replace(TRAINED, drop_obs_families=("amv",))
    attrs = provenance_attrs(denied, ObsFilters())

    assert set(attrs) >= {
        "history",
        "host",
        "date_created",
        "git_revision",
        "git_worktree",
        "git_branch",
        "obs_config",
    }
    recorded = json.loads(attrs["obs_config"])
    assert recorded["drop_obs_families"] == ["amv"]
    assert recorded["use_nnja_sat"] is True


def test_the_extra_filters_default_to_packaged_denials_and_provider_flags():
    from healda.cli.inference_local import inference_filters_from_args

    extra = inference_filters_from_args(_args())
    assert extra.drop_source_flagged
    assert (
        ChannelDenial("amsua", "metop-c", 4, "2026-03-16T03:00")
        in extra.channel_denials
    )
    opted_out = inference_filters_from_args(
        _args(no_denials=True, keep_source_flagged=True)
    )
    assert opted_out.channel_denials == ()
    assert not opted_out.drop_source_flagged


def test_a_store_extends_by_time_and_records_who_wrote_each_slot(tmp_path):
    from healda.inference_output import open_output_store, record_run, store_slots

    path = str(tmp_path / "analysis.zarr")
    step = pd.Timedelta("6h")
    identity = {"checkpoint": "ckpt", "obs_config": "{}"}
    grid = (("lat", "lon"), (2, 3))

    first = pd.date_range("2026-01-01T00", periods=3, freq=step)
    group = open_output_store(
        path, ["Z500"], first, step, extend=False, identity=identity, grid=grid
    )
    record_run(group, store_slots(group, first, step), {"history": "first"})

    later = pd.date_range("2026-01-01T18", periods=3, freq=step)
    group = open_output_store(
        path, ["Z500"], later, step, extend=True, identity=identity, grid=grid
    )
    group = zarr.open_group(path, mode="r+")
    slots = store_slots(group, later, step)
    assert slots.tolist() == [3, 4, 5]
    record_run(group, slots, {"history": "second"})

    assert group["Z500"].shape[0] == 6
    assert group["run_id"][:].tolist() == [0, 0, 0, 1, 1, 1]
    assert [r["history"] for r in group.attrs["runs"]] == ["first", "second"]

    with pytest.raises(ValueError, match="obs_config"):
        open_output_store(
            path,
            ["Z500"],
            later,
            step,
            extend=True,
            identity={**identity, "obs_config": "other"},
            grid=grid,
        )
    with pytest.raises(ValueError, match="new store"):
        open_output_store(
            path,
            ["Z500"],
            first - step,
            step,
            extend=True,
            identity=identity,
            grid=grid,
        )


def test_a_store_with_frames_holds_chunked_metrics_that_grow_with_it(tmp_path):
    from healda.inference_output import open_output_store

    path = str(tmp_path / "analysis.zarr")
    step = pd.Timedelta("6h")
    kwargs = dict(
        identity={"checkpoint": "c"},
        grid=(("lat", "lon"), (2, 3)),
        frames=8,
        time_chunk=4,
    )
    first = pd.date_range("2026-01-01T00", periods=6, freq=step)
    group = open_output_store(
        path, ["Z500", "T850"], first, step, extend=False, **kwargs
    )
    assert group["rmse"].shape == (6, 2, 8)
    assert group["rmse"].chunks == (4, 2, 8)
    assert group["mae"].attrs["fields"] == ["Z500", "T850"]

    later = pd.date_range("2026-01-02T12", periods=4, freq=step)
    open_output_store(path, ["Z500", "T850"], later, step, extend=True, **kwargs)
    assert zarr.open_group(path, mode="r")["rmse"].shape == (10, 2, 8)

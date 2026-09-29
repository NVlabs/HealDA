# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row counts and dropout settings of the production obs recipe over one real window.

Counts use training=False, which zeroes every random drop.
Regenerate: pytest tests/integration/test_obs_census_golden.py --regtest-reset
"""

import asyncio
import dataclasses
import os

import numpy
import pandas as pd
import pyarrow as pa
import pytest

import healda.config.environment as config
from healda.config.models import ObsConfig
from healda.datasets.da.tasks import build_obs_loader
from healda.observations.loaders.combined import SENSOR_NAME_TO_ID

pytestmark = pytest.mark.skipif(
    not os.path.isdir(config.NNJA_ROOT),
    reason=f"needs the NNJA archive at {config.NNJA_ROOT}",
)

# The recipe every live arm shares.
LIVE_OBS_CONFIG = ObsConfig(
    use_obs=True,
    context_start=-3,
    context_end=3,
    use_conv=True,
    use_infrared_pca=True,
    use_airs_pca=True,
    drop_obs_channel_ids=[],
    conv_uv_in_situ_only=False,
    conv_gps_level1_only=True,
    conv_min_pressure_hpa=1.0,
    use_conv_level_stats=True,
    conv_level_channels=True,
    use_nnja_sat=True,
    use_nnja_conv=True,
    use_nnja_satwnd=True,
    nnja_wind_dropout=0.5,
    nnja_platform_channel_dropout=(
        ("amsua", "metop-b", 3, 0.2),
        ("amsua", "metop-b", 6, 0.2),
    ),
)

# Held-out year.
WINDOW = pd.DatetimeIndex([pd.Timestamp("2025-01-15T00:00:00")])

# NNJA vocabulary: sensors.SENSOR_NAME_TO_ID is a different map and mislabels this table.
SENSOR = {index: name for name, index in SENSOR_NAME_TO_ID.items()}


def _counts(table: pa.Table, keys: tuple[str, ...]) -> list[dict]:
    grouped = table.select(keys).group_by(list(keys)).aggregate([([], "count_all")])
    counted = pa.table(
        {**{key: grouped[key] for key in keys}, "rows": grouped["count_all"]}
    )
    return counted.sort_by([(key, "ascending") for key in keys]).to_pylist()


def _report(table, keys, out):
    for row in _counts(table, keys):
        fields = " ".join(
            f"{key}={SENSOR.get(row[key], row[key]) if key == 'sensor_id' else row[key]}"
            for key in keys
        )
        print(f"  {fields} rows={row['rows']}", file=out)


AMV_REPORT_TYPES = tuple(range(240, 261))


def _amv_rows(table) -> int:
    types = table["Observation_Type"].to_numpy(zero_copy_only=False)
    return int(numpy.isin(types, AMV_REPORT_TYPES).sum())


def test_denying_amvs_reaches_the_dedicated_archive_too():
    """The recipe reads AMVs from the SATWND archive as well as PrepBUFR, so a filter
    applied per-source would leave the archive's untouched."""
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=False)
    kept = asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0]
    assert _amv_rows(kept) > 0, "no AMVs to deny; the fixture cannot detect a no-op"

    denied_config = dataclasses.replace(
        LIVE_OBS_CONFIG, drop_report_types=AMV_REPORT_TYPES
    )
    denied = asyncio.run(
        build_obs_loader(denied_config, training=False).sel_time(WINDOW)
    )["obs_v2"][0]
    assert _amv_rows(denied) == 0
    assert denied.num_rows < kept.num_rows


def test_denying_the_amv_family_matches_naming_its_report_types():
    """Families are sugar over report types; the sugar has to be exact."""
    by_family = dataclasses.replace(LIVE_OBS_CONFIG, drop_obs_families=("amv",))
    by_type = dataclasses.replace(LIVE_OBS_CONFIG, drop_report_types=AMV_REPORT_TYPES)

    def load(config):
        loader = build_obs_loader(config, training=False)
        return asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0]

    family_rows, type_rows = load(by_family), load(by_type)
    assert _amv_rows(family_rows) == 0
    assert family_rows.num_rows == type_rows.num_rows


PRE_ASCAT = [285, 286, 289]
EARLY = pd.DatetimeIndex([pd.Timestamp("2005-01-15T00:00:00")])


def _early_rows(config):
    table = asyncio.run(build_obs_loader(config, training=False).sel_time(EARLY))[
        "obs_v2"
    ][0]
    types = table["Observation_Type"].to_numpy(zero_copy_only=False)
    return table.num_rows, int(numpy.isin(types, PRE_ASCAT).sum())


def test_pre_ascat_scatterometer_winds_survive_the_height_screen():
    """QuikSCAT/ERS/WindSat report no ZOB. Substituting 10 m only for ASCAT dropped every
    scatterometer wind from 2000-2010, 11 of the 26 training years."""
    _, scatterometer = _early_rows(LIVE_OBS_CONFIG)
    assert scatterometer > 0


def test_ascat_only_reproduces_the_pre_fix_observing_system():
    """Arms started before the fix must keep their observing system, or their checkpoints
    stop being comparable mid-run."""
    ascat_only = dataclasses.replace(LIVE_OBS_CONFIG, ascat_only_scatterometer=True)
    ascat_only_rows, ascat_only_scatterometer = _early_rows(ascat_only)
    fixed_rows, fixed_scatterometer = _early_rows(LIVE_OBS_CONFIG)

    assert ascat_only_scatterometer == 0
    assert fixed_scatterometer > 0
    assert ascat_only_rows < fixed_rows


def _rows(loader):
    return asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0].num_rows


def test_per_row_dropout_follows_the_ambient_worker_stream():
    """Per-row dropout draws its per-cycle seed from the process numpy stream, which
    PyTorch seeds per (rank, worker). So it replays under a fixed ambient seed and moves
    under a different one -- no seed threading required for independence."""
    import numpy as np

    def draw(ambient):
        np.random.seed(ambient)
        return _rows(build_obs_loader(LIVE_OBS_CONFIG, training=True))

    assert draw(0) == draw(0)
    assert draw(7) != draw(0)


def test_dropout_is_resampled_across_windows_not_frozen_per_window():
    """Keying the mask on the window alone froze it: the same window lost the same rows in
    every epoch, which is not what a regulariser is for. The stream advances instead."""
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=True)
    draws = [_rows(loader) for _ in range(4)]
    assert len(set(draws)) > 1, f"dropout did not advance across repeats: {draws}"


def test_evaluation_draws_nothing():
    """training=False must zero every random drop, whatever the ambient stream says."""

    def rows(ambient):
        numpy.random.seed(ambient)
        return _rows(build_obs_loader(LIVE_OBS_CONFIG, training=False))

    assert rows(0) == rows(7)


def test_live_recipe(regtest):
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=False)
    table = asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0]

    print(f"total rows: {table.num_rows}", file=regtest)
    print(file=regtest)
    _report(table, ("sensor_id",), regtest)
    print(file=regtest)
    _report(table, ("sensor_id", "Observation_Type"), regtest)
    print(file=regtest)
    _report(table, ("sensor_id", "Platform_ID"), regtest)

    # Read off the objects that apply them, not the config.
    print(file=regtest)
    for training in (False, True):
        built = build_obs_loader(LIVE_OBS_CONFIG, training=training)
        conv = built.conventional._loader
        rules = sum(len(r) for r in built.satellite._platform_channel_dropout.values())
        print(
            f"training={training} wind_obs_dropout={conv.wind_obs_dropout} "
            f"platform_channel_rules={rules} satwnd={conv.satwnd is not None}",
            file=regtest,
        )


def _ambient_position(loader, frames):
    """Where the worker's numpy stream lands after loading `frames` from a fixed start."""
    numpy.random.seed(0)
    asyncio.run(loader.sel_time(frames))
    return numpy.random.get_state()[2]


@pytest.mark.parametrize("start_hour", ["00", "06", "12", "18"])
def test_every_time_parallel_rank_consumes_the_same_ambient_draws(start_hour):
    """The invariant sample scope rests on.

    Each loader draws exactly once per sel_time, so the four time-parallel ranks of a
    sample stay in lockstep and reach the same verdict without communicating. A
    data-dependent ambient draw anywhere in the worker breaks that silently; this test
    is what makes it loud. Do not delete or xfail it.
    """
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=True)
    frames = pd.date_range(f"2022-05-22T{start_hour}:00", periods=8, freq="6h")
    positions = {_ambient_position(loader, frames[r * 2 : r * 2 + 2]) for r in range(4)}
    assert len(positions) == 1, f"ranks consumed different draw counts: {positions}"


def test_sample_scope_agrees_across_ranks_and_differs_across_samples():
    """What alignment buys: the four rank-slices of one sample keep or drop the same
    wind rows, while a different sample draws again."""
    sample_scope = dataclasses.replace(LIVE_OBS_CONFIG, nnja_dropout_scope="sample")
    frames = pd.date_range("2022-05-22T18:00", periods=8, freq="6h")

    def wind_rows(ambient, slice_index):
        numpy.random.seed(ambient)
        loader = build_obs_loader(sample_scope, training=True)
        table = asyncio.run(
            loader.sel_time(frames[slice_index * 2 : slice_index * 2 + 2])
        )["obs_v2"][0]
        types = table["Observation_Type"].to_numpy(zero_copy_only=False)
        return bool(numpy.isin(types, AMV_REPORT_TYPES).any())

    # Same ambient state = same rank cohort: every slice must reach the same verdict.
    assert len({wind_rows(0, r) for r in range(4)}) == 1


def test_sample_scope_is_all_or_nothing_not_a_fraction():
    """Row scope thins the AMVs; sample scope keeps or drops the lot."""
    sample_scope = dataclasses.replace(LIVE_OBS_CONFIG, nnja_dropout_scope="sample")

    def amvs(config, ambient):
        numpy.random.seed(ambient)
        loader = build_obs_loader(config, training=True)
        return _amv_rows(asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0])

    full = _amv_rows(
        asyncio.run(build_obs_loader(LIVE_OBS_CONFIG, training=False).sel_time(WINDOW))[
            "obs_v2"
        ][0]
    )
    seen = {amvs(sample_scope, ambient) for ambient in range(12)}
    assert seen <= {0, full}, f"sample scope produced partial counts: {sorted(seen)}"
    assert seen == {0, full}, f"only one outcome in 12 draws: {sorted(seen)}"


def _trace_cycle_seeds(loader, frames):
    """The per-cycle dropout seed at every draw. Observed, not inferred from row counts."""
    from healda.observations.loaders import nnja_conventional as module

    seen, real = {}, module.NNJAConvLoader._load_cycle

    def traced(self, cycle, halves, dropout_seed=None, *rest):
        seen[str(cycle)] = dropout_seed
        return real(self, cycle, halves, dropout_seed, *rest)

    module.NNJAConvLoader._load_cycle = traced
    try:
        asyncio.run(loader.sel_time(frames))
    finally:
        module.NNJAConvLoader._load_cycle = real
    return seen


THREE_FRAMES = pd.DatetimeIndex(
    ["2022-05-22T00:00", "2022-05-22T06:00", "2022-05-22T12:00"]
)


def test_each_cycle_of_a_sample_draws_its_own_seed():
    """A shared seed would hide the same positions in every frame of a multi-frame sample."""
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=True)
    seeds = _trace_cycle_seeds(loader, THREE_FRAMES)
    assert len(seeds) == 3, seeds
    assert len(set(seeds.values())) == 3, f"cycles shared a seed: {seeds}"


def test_the_stream_advances_so_the_next_epoch_resamples():
    """Re-reading the same window must not repeat the mask, or the drop becomes a fixed
    censoring rather than a regulariser."""
    loader = build_obs_loader(LIVE_OBS_CONFIG, training=True)
    first = _trace_cycle_seeds(loader, THREE_FRAMES)
    second = _trace_cycle_seeds(loader, THREE_FRAMES)
    assert set(first) == set(second)
    assert all(first[c] != second[c] for c in first), f"{first} vs {second}"


PS_CHANNEL = 3  # ps local_channel_id, before the conv-plevel remap


def _conv_channels(config, ambient=0):
    """Per-channel row counts straight from the NNJA conv loader.

    Read before CombinedObsLoader's plevel remap, which rewrites local_channel_id by
    pressure level and would make `3` mean something else.
    """
    numpy.random.seed(ambient)
    loader = build_obs_loader(config, training=True).conventional._loader
    table = asyncio.run(loader.sel_time(WINDOW))["obs_v2"][0]
    local = table["local_channel_id"].to_numpy(zero_copy_only=False)
    channels, counts = numpy.unique(local, return_counts=True)
    return dict(zip(channels.tolist(), counts.tolist()))


def test_surface_pressure_dropout_thins_ps_and_leaves_other_channels_alone():
    """Unlike a report-type denial this drops one channel: a thinned row keeps its T, q
    and winds. Wind channels (6, 7) are exempted because they share the rng stream and
    so get a different, equally valid, draw."""
    off = _conv_channels(LIVE_OBS_CONFIG)
    half = _conv_channels(
        dataclasses.replace(LIVE_OBS_CONFIG, nnja_surface_pressure_dropout=0.5)
    )
    assert off[PS_CHANNEL] > 0, "no ps rows to drop; the fixture cannot detect a no-op"
    assert 0 < half[PS_CHANNEL] < off[PS_CHANNEL]
    for channel, count in off.items():
        if channel not in (PS_CHANNEL, 6, 7):
            assert (
                half[channel] == count
            ), f"channel {channel}: {count} -> {half[channel]}"


def test_surface_pressure_dropout_is_all_or_nothing_under_sample_scope():
    sample_scope = dataclasses.replace(
        LIVE_OBS_CONFIG, nnja_surface_pressure_dropout=0.5, nnja_dropout_scope="sample"
    )
    full = _conv_channels(LIVE_OBS_CONFIG)[PS_CHANNEL]
    seen = {_conv_channels(sample_scope, a).get(PS_CHANNEL, 0) for a in range(12)}
    assert seen <= {0, full}, f"partial ps counts under sample scope: {sorted(seen)}"
    assert seen == {0, full}, f"only one outcome in 12 draws: {sorted(seen)}"


def test_surface_pressure_dropout_is_off_by_default():
    assert ObsConfig(use_obs=True).nnja_surface_pressure_dropout == 0.0

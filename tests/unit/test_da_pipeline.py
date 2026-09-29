# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core DA data-path tests that need no zarr stores.

Covers the pieces the recipe/stats refactor touched: channel encoding, the
state-normalization CSV schema + aliasing, recipe<->stats consistency, the
dropout-aware windowing, and the transform's batch contract. Uses the committed
state-normalization CSVs and synthetic arrays only.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import inspect

import healda.utils.datetime
from healda.datasets.base import NormalizationStats, VariableConfig
from healda.datasets.da import state_stats
from healda.datasets.da.state_stats import (
    get_batch_info,
    load_channel_stats,
)
from healda.datasets.da.tasks import TASK_CONFIGS
from healda.datasets.da.hourly_latlon_dataset import HourlyLatlonDataset
from healda.datasets.da.packed_da_dataset import PackedDADataset
from healda.datasets.da.transform import (
    TransformV2,
    _frames_are_consecutive_slices,
    reorder_from_nest,
)
from healda.config.variables import VARIABLE_CONFIGS, encode_channels
from healda.datasets.merged_dataset import _FrameIndexGenerator


@pytest.mark.parametrize("backend", [HourlyLatlonDataset, PackedDADataset])
def test_backend_takes_training_and_has_no_kwargs_catch_all(backend):
    parameters = inspect.signature(backend.__init__).parameters
    assert "training" in parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_encode_channels_order():
    vc = VARIABLE_CONFIGS["era5_74ch"]
    ch = encode_channels(vc)
    # 3D vars x levels first (in order), then the 2D vars.
    assert ch[0] == f"{vc.variables_3d[0]}{vc.levels[0]}"
    assert ch[-len(vc.variables_2d) :] == list(vc.variables_2d)
    assert len(ch) == len(vc.variables_3d) * len(vc.levels) + len(vc.variables_2d)


def test_get_batch_info_channels_match_encode():
    vc = VARIABLE_CONFIGS["era5_74ch"]
    bi = get_batch_info(vc)
    assert bi.channels == encode_channels(vc)
    assert len(bi.center) == len(bi.channels) == len(bi.scales)


def test_load_channel_stats_folds_case():
    # The CSVs disagree on capitalisation the same way the stores do: era5_13_levels
    # spells 10 hPa U10, era5_104ch_hpx spells it u10. Either must resolve.
    channels = ["U10", "V10", "Z10", "T10", "tcw"]
    for stats_file in ("era5_13_levels_stats.csv", "era5_104ch_hpx_stats.csv"):
        stats = load_channel_stats(stats_file, channels)
        assert len(stats.center) == len(channels)
        assert np.all(np.isfinite(stats.center))
        assert np.all(stats.scales > 0)
    # Exact match still wins over the fold.
    assert load_channel_stats(
        "era5_104ch_hpx_stats.csv", ["u10"]
    ).center == pytest.approx(
        load_channel_stats("era5_104ch_hpx_stats.csv", ["U10"]).center
    )


def _write_stats_csv(path, rows):
    """rows: list of (variable, level, mean, std)."""
    pd.DataFrame(rows, columns=["variable", "level", "mean", "std"]).to_csv(
        path, index=False
    )


def test_load_channel_stats_overlay_replaces_only_named_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(state_stats, "NORMALIZATIONS_DIR", tmp_path)
    _write_stats_csv(
        tmp_path / "base.csv",
        [("sst", -1, 291.5, 10.3), ("tas", -1, 288.0, 5.0), ("u", 1000, 0.0, 6.0)],
    )
    _write_stats_csv(tmp_path / "overlay.csv", [("sst", -1, 275.0, 2.0)])

    channels = ["u1000", "sst", "tas"]
    stats = load_channel_stats("base.csv", channels, overlay_file="overlay.csv")

    by_channel = dict(zip(channels, zip(stats.center, stats.scales)))
    assert by_channel["sst"] == pytest.approx((275.0, 2.0)), "overlay replaces sst"
    assert by_channel["tas"] == pytest.approx((288.0, 5.0)), "tas untouched"
    assert by_channel["u1000"] == pytest.approx((0.0, 6.0)), "u1000 untouched"


def test_load_channel_stats_overlay_rejects_channel_absent_from_base(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(state_stats, "NORMALIZATIONS_DIR", tmp_path)
    _write_stats_csv(tmp_path / "base.csv", [("sst", -1, 291.5, 10.3)])
    _write_stats_csv(tmp_path / "overlay.csv", [("sic", -1, 0.05, 0.2)])

    with pytest.raises(KeyError, match="sic"):
        load_channel_stats("base.csv", ["sst"], overlay_file="overlay.csv")


def test_load_channel_stats_without_overlay_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr(state_stats, "NORMALIZATIONS_DIR", tmp_path)
    _write_stats_csv(tmp_path / "base.csv", [("sst", -1, 291.5, 10.3)])

    stats = load_channel_stats("base.csv", ["sst"])
    assert stats.center == pytest.approx([291.5])
    assert stats.scales == pytest.approx([10.3])


@pytest.mark.parametrize(
    "name", [n for n, c in TASK_CONFIGS.items() if c.dataset_backend == "packed"]
)
def test_recipe_stats_consistency(name):
    """Every recipe's stats CSVs must cover exactly its channel set."""
    cfg = TASK_CONFIGS[name]
    target_channels = encode_channels(VARIABLE_CONFIGS[cfg.target.variable_config])
    target_stats = load_channel_stats(
        cfg.target.stats_file, target_channels, cfg.target.source
    )
    assert len(target_stats.center) == len(target_channels)

    for state in cfg.inputs:
        channels = encode_channels(VARIABLE_CONFIGS[state.variable_config])
        stats = load_channel_stats(state.stats_file, channels, state.source)
        assert len(stats.center) == len(channels), f"{name}: {state.name}"

    residual_input = cfg.residual_input
    if cfg.residual_stats_file is not None:
        assert residual_input is not None
        residual_stats = load_channel_stats(
            cfg.residual_stats_file, target_channels, residual_input.source
        )
        assert len(residual_stats.center) == len(target_channels)


def test_indexer_no_window_spans_dropout():
    """Windows must stay within a contiguous time segment (no straddling gaps)."""
    times = pd.date_range("2020-01-01", periods=20, freq="6h").values
    times = np.delete(times, range(8, 12))  # 16 times with a 30h hole
    gen = _FrameIndexGenerator(
        times, time_length=4, frame_step=1, model_rank=0, model_world_size=1
    )
    step = np.timedelta64(6, "h")
    n = gen.get_valid_length()
    assert n > 0
    for k in range(n):
        frames = gen.generate_frame_indices([k])[0]
        window = times[frames]
        assert np.all(np.diff(window) == step), f"window {k} spans the gap: {window}"


def test_indexer_time_length_one_keeps_all_frames():
    times = pd.date_range("2020-01-01", periods=10, freq="6h").values
    times = np.delete(times, [4, 5])  # gaps do not matter for single frames
    gen = _FrameIndexGenerator(
        times, time_length=1, frame_step=1, model_rank=0, model_world_size=1
    )
    assert gen.get_valid_length() == len(times)


def test_consecutive_slice_reuse_requires_identical_child_shapes():
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 4).copy()
    assert _frames_are_consecutive_slices([base[0], base[1]])
    assert not _frames_are_consecutive_slices([base[0], base[1, :2]])


def test_transform_preserves_multibatch_multiframe_layout():
    vc = VariableConfig(
        name="layout_test",
        variables_2d=["state"],
        variables_3d=[],
        levels=[],
        variables_static=["orog", "lfrac"],
    )
    batch_size, time_size, npix = 2, 2, 12
    transform = TransformV2(
        variable_config=vc,
        hpx_level=0,
        sensors=[],
        target_normalization=NormalizationStats(center=[0.0], scales=[1.0]),
        static_condition=torch.arange(2 * npix, dtype=torch.float32).reshape(
            1, 2, 1, npix
        ),
    )

    states = np.arange(batch_size * time_size * npix, dtype=np.float32).reshape(
        batch_size, time_size, 1, npix
    )
    frames = [
        [{"state": states[b, t]} for t in range(time_size)] for b in range(batch_size)
    ]
    time = healda.utils.datetime.as_cftime(pd.Timestamp("2020-06-01"))
    times = [[time] * time_size for _ in range(batch_size)]

    batch = transform.transform(times, frames)
    source_target = batch["target"].clone()
    on_device = transform.device_transform(batch, "cpu")
    repeated = transform.device_transform(batch, "cpu")

    assert batch["target"].shape == (batch_size, time_size, 1, npix)
    assert torch.equal(batch["target"], source_target)
    assert on_device["target"].shape == (batch_size, 1, time_size, npix)
    assert on_device["condition"].shape == (batch_size, 2, time_size, npix)
    assert on_device["condition"].is_contiguous()
    assert on_device["condition"].data_ptr() == repeated["condition"].data_ptr()
    expected_condition = batch["condition"].expand(batch_size, -1, time_size, -1)
    torch.testing.assert_close(on_device["condition"], expected_condition)
    expected = reorder_from_nest(source_target.permute(0, 2, 1, 3), "hpxpadxy")
    torch.testing.assert_close(on_device["target"], expected)


def test_year_holdout_split_accepts_a_year_list():
    """Inference sets the split to the years a --start-date run spans, not a name."""
    from healda.datasets.splits import YearHoldoutSplit

    times = pd.date_range("2024-06-01", "2026-06-01", freq="30D")
    split = YearHoldoutSplit()
    assert split.select(times, [2025]).year.unique().tolist() == [2025]
    assert sorted(split.select(times, [2024, 2026]).year.unique()) == [2024, 2026]
    # The named splits are unchanged.
    assert split.select(times, "test").year.unique().tolist() == [2025]
    assert 2025 not in split.select(times, "train").year.unique()
    with pytest.raises(ValueError, match="Unknown split"):
        split.mask(times, "nonsense")


def test_scoring_does_not_get_training_obs_dropout():
    """`train` used to carry both "which span" and "augment like training", so every dated
    inference discarded half the wind obs. get_dataset(years=) is what separates them."""
    from healda.observations.system import ObsPipeline
    from healda.cli.train import LOOPS

    obs = LOOPS["v2-videoDA-nnja-nnjaConv-104ch-windDrop50"].obs_config
    assert obs.nnja_wind_dropout == 0.5
    assert ObsPipeline(obs, training=True).random_drop.wind == 0.5
    assert ObsPipeline(obs, training=False).random_drop.wind == 0.0


def test_transform_assembles_target_background_condition():
    vc = VARIABLE_CONFIGS["era5_74ch"]
    channels = encode_channels(vc)
    n_chan = len(channels)
    npix = 49152
    norm = NormalizationStats(center=np.zeros(n_chan), scales=np.ones(n_chan))
    static = torch.zeros(1, len(vc.variables_static), 1, npix)
    transform = TransformV2(
        variable_config=vc,
        hpx_level=6,
        sensors=[],
        target_normalization=norm,
        background_normalization=norm,
        static_condition=static,
    )

    state = np.random.randn(n_chan, npix).astype("float32")
    background = np.random.randn(n_chan, npix).astype("float32")
    t = healda.utils.datetime.as_cftime(pd.Timestamp("2020-06-01"))
    out = transform.transform([[t]], [[{"state": state, "background": background}]])

    # transform packs the frames as (b, t, c, x) and leaves normalizing and the reorder to
    # device_transform, which returns (b, c, t, x).
    assert out["target"].shape == (1, 1, n_chan, npix)
    assert out["background"].shape == (1, 1, n_chan, npix)
    assert out["condition"].shape[0] == 1 and out["condition"].shape[2] == 1

    target_before = out["target"].clone()
    background_before = out["background"].clone()
    on_device = transform.device_transform(out, "cpu")
    assert torch.equal(out["target"], target_before)
    assert torch.equal(out["background"], background_before)
    assert on_device["target"].shape == (1, n_chan, 1, npix)
    assert on_device["background"].shape == (1, n_chan, 1, npix)
    # Unit statistics, so normalizing is a no-op and the values must survive the round trip.
    assert torch.allclose(
        on_device["target"],
        reorder_from_nest(
            torch.from_numpy(state)[None, :, None],
            "hpxpadxy",
        ),
    )

    # No "background" key -> the transform must omit it (obs-only path).
    out_no_bg = transform.transform([[t]], [[{"state": state}]])
    assert "background" not in out_no_bg
    assert out_no_bg["target"].shape == (1, 1, n_chan, npix)


def test_obs_coverage_follows_the_archive_in_use():
    """The clamp follows the archive the config reads."""
    from healda.config.models import ObsConfig
    from healda.observations.coverage import obs_coverage

    ufs = obs_coverage(ObsConfig(use_obs=True))
    nnja = obs_coverage(
        ObsConfig(
            use_obs=True,
            use_nnja_sat=True,
            conv_level_channels=True,
            use_conv_level_stats=True,
        )
    )
    assert ufs.name == "ufs_obs"
    assert nnja.name == "nnja_obs"
    assert pd.Timestamp(nnja.end) > pd.Timestamp(ufs.end)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

pytest.importorskip("torch")

from healda.cli.train import LOOPS, TrainingLoop  # noqa: E402
from healda.training import presets  # noqa: E402

ALIASES = {
    "v2-ufs-hpx64": "v2-videoDA-convMinP1hPa-plevelNorm-plevelConvChannels",
    "v2-nnja-hpx64": "v2-videoDA-nnja-nnjaConv-104ch-windDrop50",
    "v2-nnja-latlon-beta": (
        "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll"
    ),
    "v2-nnja-latlon-final": (
        "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll"
    ),
    "v2-nnja-latlon-dense": (
        "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll"
        "-r14-thin128-satwnd64-qm3-sondeFill-drift-posfeat"
    ),
}


def test_shipped_presets_load_and_aliases_resolve():
    loaded = presets.load_preset_dir(presets.DEFAULT_DIR, TrainingLoop)
    files = sorted(p.stem for p in presets.DEFAULT_DIR.glob("*.yaml"))
    assert "debug-single-gpu" in files
    for alias, canonical in ALIASES.items():
        assert canonical in files
        assert loaded[alias] is loaded[canonical]
        assert LOOPS[alias] is LOOPS[canonical]


def test_shipped_presets_are_differences_over_a_parent():
    # Every shipped preset but the base extends another; the released v2 differs from the
    # beta in the fields its header names and nothing else.
    dense = LOOPS["v2-nnja-latlon-dense"]
    beta = LOOPS["v2-nnja-latlon-beta"]
    assert dense == dataclasses.replace(
        beta,
        attention_backend="disk-nest:14",
        latlon_decode=dataclasses.replace(beta.latlon_decode, fp32_output=True),
        sensor_embedder_config=dataclasses.replace(
            beta.sensor_embedder_config, features_position=True
        ),
        obs_config=dataclasses.replace(
            beta.obs_config,
            nnja_sat_thin_hpx_level=7,
            nnja_satwnd_thin_hpx_level=6,
            nnja_max_quality_mark=3,
            nnja_pressure_height_fill="sonde",
            nnja_balloon_drift=True,
        ),
    )


def test_yaml_round_trip(tmp_path):
    loop = LOOPS["debug-single-gpu"]
    path = tmp_path / "copy.yaml"
    path.write_text(presets.to_yaml(loop, aliases=["copy-alias"], header="a copy"))
    reloaded, aliases = presets.load_preset(path, TrainingLoop)
    assert aliases == ["copy-alias"]
    assert reloaded == loop


def _write(tmp_path, body: str):
    path = tmp_path / "p.yaml"
    path.write_text(body)
    return path


def test_extends_merges_nested_fields_and_replaces_lists(tmp_path):
    path = _write(
        tmp_path,
        "extends: v2-nnja-hpx64\n"
        "loop:\n"
        "  lr: 1.0e-4\n"
        "  obs_config:\n"
        "    nnja_wind_dropout: 0.0\n"
        "    nnja_platform_channel_dropout: []\n",
    )
    loop, aliases = presets.load_preset(path, TrainingLoop, LOOPS)
    parent = LOOPS["v2-nnja-hpx64"]
    assert aliases == []
    assert loop.lr == 1e-4
    assert loop.obs_config.nnja_wind_dropout == 0.0
    assert loop.obs_config.nnja_platform_channel_dropout == ()
    assert loop.obs_config.use_nnja_sat is parent.obs_config.use_nnja_sat is True
    assert loop.architecture == parent.architecture

    with pytest.raises(ValueError, match="not a preset"):
        presets.load_preset(path, TrainingLoop)
    path.write_text("extends: no-such-preset\nloop: {}\n")
    with pytest.raises(ValueError, match="'no-such-preset', which is not a preset"):
        presets.load_preset(path, TrainingLoop, LOOPS)


def test_preset_dir_resolves_extends_in_any_order_and_rejects_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b-alias\nloop:\n  lr: 1.0\n")
    (tmp_path / "b.yaml").write_text("aliases: [b-alias]\nloop:\n  seed: 7\n")
    loaded = presets.load_preset_dir(tmp_path, TrainingLoop)
    assert loaded["a"].seed == 7 and loaded["a"].lr == 1.0
    assert loaded["b"].lr == TrainingLoop().lr and loaded["b-alias"] is loaded["b"]

    (tmp_path / "b.yaml").write_text("aliases: [b-alias]\nextends: a\nloop: {}\n")
    with pytest.raises(ValueError, match="extends cycle"):
        presets.load_preset_dir(tmp_path, TrainingLoop)


def test_unknown_field_names_file_and_key(tmp_path):
    path = _write(tmp_path, "aliases: []\nloop:\n  obs_config:\n    use_obz: true\n")
    with pytest.raises(ValueError, match=r"p\.yaml.*obs_config\.use_obz"):
        presets.load_preset(path, TrainingLoop)

    path.write_text("inherits: other\nloop: {}\n")
    with pytest.raises(ValueError, match="unknown top-level keys"):
        presets.load_preset(path, TrainingLoop)


def test_float_fields_accept_ints_and_reject_exponent_strings(tmp_path):
    loop, _ = presets.load_preset(_write(tmp_path, "loop:\n  lr: 1\n"), TrainingLoop)
    assert loop.lr == 1.0 and type(loop.lr) is float
    for literal, hint in (
        ("1e-4", "0.0001"),
        ("1.0e5", "100000.0"),
        ("1e-8", "1.0e-08"),
    ):
        with pytest.raises(ValueError, match=f"signed exponent, e.g. {hint}"):
            presets.load_preset(
                _write(tmp_path, f"loop:\n  lr: {literal}\n"), TrainingLoop
            )

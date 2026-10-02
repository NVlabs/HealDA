# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""latlon_decode wiring: what the loop hands the backbone and the 0.25 degree tail."""

import dataclasses

import pytest
import torch

import healda.models
from healda.config.models import ARCHITECTURES
from healda.datasets.da.tasks import TASK_CONFIGS
from healda.models.hpx256_latlon import Hpx256LatlonModel
from healda.cli.train import LOOPS, TrainingLoop

LEV6_ARMS = (
    "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll",
)


class _Backbone(torch.nn.Module):
    def __init__(self, order):
        super().__init__()
        self.spatial_token_order = order
        self.compile_dit = False
        self.noise_embed = None
        self.inner_dim = 32
        self.level_model = 5
        self._level_in = 8


def _build(config):
    # On meta: this asserts on wiring, not weights, and dit-10B costs 0.07s instead of ~20s.
    with torch.device("meta"):
        return healda.models.get_model(config)


def test_wrapper_refuses_a_backbone_in_a_different_order():
    with pytest.raises(ValueError, match="pixel order disagreement"):
        Hpx256LatlonModel(
            _Backbone("nest"),
            out_channels=3,
            pixel_order="hpxpadxy",
        )


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_every_architecture_runs_in_the_order_the_transform_packs(architecture):
    """A get_model branch that drops the order reorders the data but not the model: no shape
    changes and nothing raises, so only a built network can catch it."""
    loop = TrainingLoop(
        architecture=architecture,
        attention_backend="disk-nest:10.46",
        dit_temporal_attention=True,
        time_length=2,
    )
    net = _build(loop.model_config)
    assert net.spatial_token_order == loop._pipeline_pixel_order == "nest"
    assert net.pos_embed.token_order == "nest"
    assert net._level_in == loop.resolved_hpx_in_level


def test_disk_attention_needs_temporal_attention():
    """Otherwise the pipeline reorders to nest and the disk is never used."""
    loop = TrainingLoop(
        attention_backend="disk-nest:10.46", dit_temporal_attention=False
    )
    with pytest.raises(NotImplementedError, match="requires temporal_attention"):
        _build(loop.model_config)


@pytest.mark.parametrize("name", LEV6_ARMS)
def test_lev6_arms_resolve_the_geometry_they_claim(name):
    """The registry's level_in never reaches the DiT on a latlon_decode arm, so assert on the
    built network rather than on the architecture tables."""
    loop = LOOPS[name]
    net = _build(loop.model_config)
    assert net.level_model == 6
    assert net._level_in == 8 == loop.resolved_hpx_in_level
    assert net.spatial_token_order == "nest"
    # The wrapper supplies the fine calendar; adding it at the coarse tokens would double it.
    assert net.pos_embed.calendar_embed is None
    assert net.patch_decode is None


@pytest.mark.parametrize("name", LEV6_ARMS)
def test_latlon_arms_read_the_latlon_task_not_the_bare_one(name):
    """A latlon_decode arm must resolve ``dataset`` to the ``_latlon`` variant.

    Falling back to the bare name is silent: shapes match, but the target is z-scored and
    denormalised with two different grids' statistics.
    """
    loop = LOOPS[name]
    assert loop.task_name == f"{loop.dataset}_latlon"
    assert loop.task.dataset_backend == "latlon"


@pytest.mark.parametrize("name", LEV6_ARMS)
def test_the_two_grids_do_not_share_a_stats_file(name):
    """HPX64 and 0.25 degree need separate normalisation, so the pair must not converge.

    Nothing in the task table itself enforces that the two entries stay distinct.
    """
    loop = LOOPS[name]
    hpx = TASK_CONFIGS[loop.dataset]
    latlon = loop.task
    assert latlon.target.stats_file != hpx.target.stats_file
    # Same channels either way: only the grid and its statistics differ.
    assert latlon.target.variable_config == hpx.target.variable_config


def test_a_pre_latlondecode_checkpoint_still_loads():
    """Checkpoints store the loop, and loads() does not filter unknown names."""
    import json

    from healda.cli.train import LatlonDecode

    old = json.dumps(
        {
            "dataset": "era5_104ch",
            "latlon_decode": True,
            "latlon_refine": "graphcast",
            "latlon_refine_k": 4,
            "latlon_refine_base_level": 7,
            "latlon_refine_duplicate_xyz": False,
            "latlon_refine_hidden": 384,
            "latlon_refine_rounds": 1,
            "decode_channels": 128,
            "geo_statics": True,
            "lat_harmonics": 0,
        }
    )
    assert TrainingLoop.loads(old).latlon_decode == LatlonDecode(
        base_level=7, k=4, decode_channels=128
    )
    assert (
        TrainingLoop.loads(json.dumps({"latlon_decode": False})).latlon_decode is None
    )


def test_a_checkpoint_using_a_dropped_tail_refuses_to_load():
    """Silently decoding through graphcast instead would score a different model."""
    import json

    with pytest.raises(ValueError, match="latlon_refine='mlp'"):
        TrainingLoop.loads(json.dumps({"latlon_decode": True, "latlon_refine": "mlp"}))


def test_the_old_flex_disk_spelling_still_resolves_to_nest():
    """The 0.25 degree checkpoints store "flex-disk-nest" in both loop.json and
    model.json. Normalising only one of them leaves the pipeline in hpxpadxy while the
    backbone runs in nest, which mis-decodes rather than raising.
    """
    import json

    from healda.models import dit

    assert dit.pipeline_token_order("flex-disk-nest:10.46") == "nest"
    assert dit.normalize_attention_backend("flex-disk-nest:10.46") == "disk-nest:10.46"
    assert dit.normalize_attention_backend("disk-nest:10.46") == "disk-nest:10.46"
    assert dit.normalize_attention_backend("te") == "te"

    loop = TrainingLoop.loads(
        json.dumps(
            {
                "dataset": "era5_104ch",
                "latlon_decode": True,
                "attention_backend": "flex-disk-nest:10.46",
                # use_obs defaults False, which reads as mask training; the arms set it.
                "obs_config": {"use_obs": True},
            }
        )
    )
    assert loop._pipeline_pixel_order == "nest"
    assert loop.model_config.attention_backend in (
        "disk-nest:10.46",
        "flex-disk-nest:10.46",
    )


def test_latlon_decode_without_a_latlon_task_raises():
    """Falling back to the HPX task would feed HPX64 targets to a 721x1440 decoder."""
    # __post_init__ resolves the task, so construction is what raises.
    with pytest.raises(ValueError, match="era5_99ch_latlon"):
        dataclasses.replace(
            LOOPS["v2-nnja-latlon-final"],
            dataset="era5_99ch",
        )


def test_latlon_decode_refuses_a_condition_it_would_discard():
    """Hpx256LatlonModel.forward drops the assembled condition, so a config that puts
    anything in it would size pos_embed for a tensor the backbone never sees."""
    latlon = LOOPS["v2-nnja-latlon-final"]
    # mask_training is use_obs=False with no background, which is how it is reached.
    no_obs = dataclasses.replace(
        latlon, obs_config=dataclasses.replace(latlon.obs_config, use_obs=False)
    )
    assert no_obs.mask_training
    with pytest.raises(NotImplementedError, match="builds its own condition"):
        no_obs.get_network()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""healda.inference: the observation and denormalization paths, which need no GPU."""

import asyncio
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from healda.config.models import ObsConfig
from healda.config.variables import VARIABLE_CONFIGS
from healda.datasets.base import BatchInfo
from healda.datasets.da import state_masks
from healda.datasets.da.tasks import build_conventional_loader
from healda.inference import (
    AnalysisModel,
    RebasedConventionalLoader,
    read_training_loop,
)
from healda.observations import sensors
from healda.observations.adapters import e2s_nnja
from healda.observations.loaders import combined
from healda.observations.schema import GLOBAL_CHANNEL_ID, SENSOR_ID
from tests.unit.test_e2s_nnja_adapter import CYCLE, _gps_frame, _wind_frame

# The NNJA lat/lon recipe's observation settings, spelled out so this file does not
# import the CLI (and with it the GPU kernels).
LATLON_OBS = ObsConfig(
    use_obs=True,
    use_conv=True,
    context_start=-3,
    context_end=3,
    conv_gps_level1_only=True,
    conv_min_pressure_hpa=1.0,
    use_conv_level_stats=True,
    conv_level_channels=True,
    use_nnja_sat=True,
    use_nnja_conv=True,
    use_nnja_satwnd=True,
    nnja_gpsro_saids="full",
    nnja_surface_winds=True,
)


def _sel(loader, times):
    return asyncio.run(loader.sel_time(pd.DatetimeIndex(times)))["obs_v2"]


def test_conventional_loader_from_tables_emits_the_combined_vocabulary():
    gps = e2s_nnja.gpsro_tables(_gps_frame(CYCLE - pd.Timedelta(minutes=30)))
    winds = e2s_nnja.satwnd_tables(_wind_frame())
    conventional = build_conventional_loader(
        LATLON_OBS, training=False, gpsro_table_source=gps, satwnd_table_source=winds
    )
    (raw,) = _sel(conventional, [CYCLE])
    (rebased,) = _sel(RebasedConventionalLoader(conventional), [CYCLE])

    assert raw.num_rows > 0
    assert rebased.num_rows == raw.num_rows
    # Alone, the conventional loader speaks GSI; the model's transform buckets by the
    # combined NNJA ids, where conv-plevel has a different sensor id and channel offset.
    assert set(raw[SENSOR_ID.name].to_pylist()) == {
        sensors.SENSOR_NAME_TO_ID[combined.CONV_SENSOR]
    }
    assert set(rebased[SENSOR_ID.name].to_pylist()) == {
        combined.SENSOR_NAME_TO_ID[combined.CONV_SENSOR]
    }
    offset = combined.SENSOR_OFFSET[combined.CONV_SENSOR]
    global_ids = rebased[GLOBAL_CHANNEL_ID.name].to_numpy()
    assert (global_ids >= offset).all()
    assert (global_ids < offset + combined.CONV_CHANNELS).all()


def test_table_sources_need_the_nnja_conventional_path():
    ufs_conv = ObsConfig(use_obs=True, use_conv=True, conv_level_channels=True)
    gps = e2s_nnja.gpsro_tables(_gps_frame(CYCLE))
    with pytest.raises(ValueError, match="use_nnja_conv"):
        build_conventional_loader(ufs_conv, training=False, gpsro_table_source=gps)


def test_rebased_loader_refuses_anything_but_conv():
    with pytest.raises(ValueError, match=combined.CONV_SENSOR):
        RebasedConventionalLoader(SimpleNamespace(sensors=["atms"]))


def _model(**loop_fields) -> AnalysisModel:
    defaults = dict(
        time_length=8,
        time_step=6,
        obs_config=LATLON_OBS,
        latlon_decode=True,
        use_masked_domain_loss=False,
        variable_config=VARIABLE_CONFIGS["era5_104ch"],
    )
    loop = SimpleNamespace(**{**defaults, **loop_fields})
    return AnalysisModel(
        net=torch.nn.Identity(), loop=loop, transform=None, device=torch.device("cpu")
    )


def test_frames_end_at_the_analysis_time_and_the_window_spans_them():
    model = _model()
    t = pd.Timestamp("2024-01-05T00")
    frames = model.frame_times(t)
    assert len(frames) == 8
    assert frames[-1] == t
    assert frames[0] == t - pd.Timedelta(hours=42)
    assert (frames[1:] - frames[:-1] == pd.Timedelta(hours=6)).all()
    start, end = model.observation_window(t)
    assert start == t - pd.Timedelta(hours=45)
    assert end == t + pd.Timedelta(hours=3)


@pytest.mark.parametrize("masked", [False, True])
def test_to_physical_denormalizes_inverts_tcw_and_restores_the_sst_fill(masked):
    channels = ["tcwv", "tcw", "sst"]
    batch_info = BatchInfo(
        channels=channels,
        center=np.array([10.0, 0.0, 280.0]),
        scales=np.array([5.0, 2.0, 10.0]),
    )
    model = _model(batch_info=batch_info, use_masked_domain_loss=masked)
    prediction = torch.zeros(1, 3, 721, 1440)
    prediction[:, 1] = 1.0  # tcw residual of one scale unit

    physical = model.to_physical(prediction)

    assert physical.shape == prediction.shape
    assert torch.all(physical[:, 0] == 10.0)
    # tcw is predicted as tcw - tcwv: 1 * 2 + 0, plus tcwv back.
    assert torch.all(physical[:, 1] == 12.0)
    ocean = torch.as_tensor(state_masks.state_mask("ocean"))
    assert torch.all(physical[:, 2][..., ocean] == 280.0)
    land_value = state_masks.domain_fill("sst") if masked else 280.0
    assert torch.all(physical[:, 2][..., ~ocean] == land_value)


def test_single_process_loop_unshards_and_drops_the_fused_tokenizer():
    from dataclasses import dataclass

    from healda.inference import single_process_loop

    @dataclass
    class Embedder:
        use_fused_mlp: bool = True
        channel_embed_dim: int = 16

    @dataclass
    class Loop:
        time_parallel: int = 8
        fsdp: bool = True
        compile_dit: bool = False
        sensor_embedder_config: Embedder | None = None

    loop = single_process_loop(Loop(sensor_embedder_config=Embedder()))
    assert (loop.time_parallel, loop.fsdp, loop.compile_dit) == (1, False, True)
    assert loop.sensor_embedder_config.use_fused_mlp is False
    assert loop.sensor_embedder_config.channel_embed_dim == 16
    assert single_process_loop(Loop(), compile_dit=False).compile_dit is False


def test_read_training_loop_requires_loop_json(tmp_path):
    path = tmp_path / "run.checkpoint"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("net_state.pth", io.BytesIO().getvalue())
    with pytest.raises(ValueError, match="loop.json"):
        read_training_loop(path)

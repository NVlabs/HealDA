# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""healda.inference: the observation and denormalization paths."""

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
from healda.datasets.da.tasks import build_obs_loader
from healda.cli.train import LatlonDecode
from healda.inference import DAModel, read_training_loop, set_inference_recipe
from healda.observations.adapters import e2s_nnja
from healda.observations import sensors, sensors_nnja
from healda.observations.system import ChannelDenial, load_denials
from healda.observations.loaders import combined
from healda.observations.schema import GLOBAL_CHANNEL_ID, SENSOR_ID
from tests.unit.test_e2s_nnja_adapter import CYCLE, _gps_frame, _wind_frame

# The NNJA lat/lon recipe's observation settings.
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


def test_tables_load_in_the_combined_vocabulary():
    gps = e2s_nnja.gpsro_tables(_gps_frame(CYCLE - pd.Timedelta(minutes=30)))
    winds = e2s_nnja.satwnd_tables(_wind_frame())
    loader = build_obs_loader(
        LATLON_OBS,
        training=False,
        satellite_table_source={},
        gpsro_table_source=gps,
        satwnd_table_source=winds,
        prepbufr_table_source={},
    )
    (rows,) = _sel(loader, [CYCLE])

    assert rows.num_rows > 0
    assert set(rows[SENSOR_ID.name].to_pylist()) == {
        combined.SENSOR_NAME_TO_ID[combined.CONV_SENSOR]
    }
    offset = combined.SENSOR_OFFSET[combined.CONV_SENSOR]
    global_ids = rows[GLOBAL_CHANNEL_ID.name].to_numpy()
    assert (global_ids >= offset).all()
    assert (global_ids < offset + combined.CONV_CHANNELS).all()


def test_table_sources_need_the_nnja_path():
    ufs = ObsConfig(use_obs=True, use_conv=True, conv_level_channels=True)
    with pytest.raises(ValueError, match="use_nnja_sat"):
        build_obs_loader(ufs, training=False, gpsro_table_source={})


def _model(**loop_fields) -> DAModel:
    defaults = dict(
        time_length=8,
        time_step=6,
        obs_config=LATLON_OBS,
        latlon_decode=True,
        use_masked_domain_loss=False,
        variable_config=VARIABLE_CONFIGS["era5_104ch"],
    )
    loop = SimpleNamespace(**{**defaults, **loop_fields})
    return DAModel(
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


def test_packaged_denials_name_real_channels():
    rows = load_denials()
    assert rows
    for denial in rows:
        config = sensors_nnja.SENSOR_CONFIGS[denial.sensor]
        assert denial.channel in config.channels
        assert sensors.PLATFORM_NAME_TO_ID[denial.platform] in config.platform_ids


def test_a_denial_rejects_an_empty_or_zoned_window():
    with pytest.raises(ValueError, match="before end"):
        ChannelDenial("amsua", "metop-b", 9, "2026-03-24T00:00", "2026-03-18T21:00")
    with pytest.raises(ValueError, match="zone"):
        ChannelDenial("amsua", "metop-b", 9, "2026-03-24T00:00Z")


def test_read_training_loop_requires_loop_json(tmp_path):
    path = tmp_path / "run.checkpoint"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("net_state.pth", io.BytesIO().getvalue())
    with pytest.raises(ValueError, match="loop.json"):
        read_training_loop(path)


def test_inference_recipe_sets_the_height_fill_and_the_output_head_precision():
    loop = SimpleNamespace(
        obs_config=LATLON_OBS, latlon_decode=LatlonDecode(fp32_output=False)
    )
    set_inference_recipe(loop)
    assert loop.obs_config.nnja_pressure_height_fill == "sonde"
    assert loop.latlon_decode.fp32_output is True

    set_inference_recipe(loop, fill_uv_sonde_heights=False, fp32_output_head=False)
    assert loop.obs_config.nnja_pressure_height_fill is None
    assert loop.latlon_decode.fp32_output is False

    hpx = SimpleNamespace(obs_config=LATLON_OBS, latlon_decode=None)
    set_inference_recipe(hpx)
    assert hpx.latlon_decode is None

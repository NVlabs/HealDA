# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""healda.inference: the observation and denormalization paths."""

import asyncio
import dataclasses
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from healda.config.models import ObsConfig
from healda.config.variables import VARIABLE_CONFIGS
from healda.datasets.base import BatchInfo, NormalizationStats
from healda.datasets.da import state_masks
from healda.datasets.da.tasks import build_obs_loader, get_sensors_for_config
from healda.datasets.da.transform import TransformV2
from healda.inference import DAModel, read_training_loop
from healda.observations.adapters import e2s_nnja
from healda.observations import sensors, sensors_nnja
from healda.observations.system import ChannelDenial, load_denials
from healda.observations.loaders import combined
from healda.observations.protocols import as_table
from healda.observations.schema import GLOBAL_CHANNEL_ID, SENSOR_ID
from tests.unit.test_e2s_nnja_adapter import (
    CYCLE,
    _gps_frame,
    _sat_frame,
    _wind_frame,
)

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
    return [
        as_table(obs)
        for obs in asyncio.run(loader.sel_time(pd.DatetimeIndex(times)))["obs_v2"]
    ]


def test_tables_load_in_the_combined_vocabulary():
    tables = e2s_nnja.analysis_tables_long(
        conv=_gps_frame(CYCLE - pd.Timedelta(minutes=30)),
        satwnd=_wind_frame(),
        ir_channels="ir32",
    )
    gps, winds = tables["gpsro_tables"], tables["satwnd_tables"]
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


def test_to_physical_clips_bounded_channels():
    # era5_104ch predicts tcw as a residual on tcwv, so both are present
    channels = ["tcwv", "tcw", "Q500", "tcc", "T500"]
    batch_info = BatchInfo(channels=channels, center=np.zeros(5), scales=np.ones(5))
    model = _model(batch_info=batch_info)
    values = torch.tensor([1.0, 0.0, -1e-4, 1.2, -5.0])
    prediction = values.view(1, 5, 1, 1).expand(1, 5, 721, 1440)

    physical = model.to_physical(prediction)

    assert torch.all(physical[:, 2] == 0.0)
    assert torch.all(physical[:, 3] == 1.0)
    assert torch.all(physical[:, 4] == -5.0)


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


class _InputRecorder(torch.nn.Module):
    """Stands in for the network: records the observations it is given."""

    def __init__(self, channel_rows: int):
        super().__init__()
        self.channel_rows = channel_rows
        self.seen = []

    def observation_id_bounds(self) -> dict[str, int]:
        return {"Global_Channel_ID": self.channel_rows}

    def tokenize_observations(self, unified_obs):
        self.seen.append(unified_obs)
        return unified_obs

    def forward(self, condition, **kwargs):
        b, _, t, _ = condition.shape
        return SimpleNamespace(out=torch.zeros(b, 3, t, 2, 2))


def _analysis_model(channel_rows: int = 2048) -> DAModel:
    model = _model(
        time_length=2,
        bf16=False,
        dit_temporal_attention_causal=False,
        batch_info=BatchInfo(
            channels=["tcwv", "tcw", "sst"],
            center=np.zeros(3),
            scales=np.ones(3),
        ),
    )
    model.net = _InputRecorder(channel_rows)
    model.transform = TransformV2(
        sensors=get_sensors_for_config(LATLON_OBS),
        use_nnja_sat=True,
        hpx_level=6,
        attention_prepack=True,
        target_normalization=NormalizationStats(center=np.zeros(1), scales=np.ones(1)),
    )
    return model


def _family_tables() -> dict:
    return e2s_nnja.analysis_tables_long(
        conv=_gps_frame(CYCLE - pd.Timedelta(minutes=30)),
        satwnd=_wind_frame(),
        sat=_sat_frame("atms", np.arange(1, 23)),
        sensors=["atms"],
        ir_channels="ir32",
    )


def _loaded_unified_table(model: DAModel, tables: dict) -> pa.Table:
    loader = build_obs_loader(
        LATLON_OBS,
        training=False,
        extra_filters=model.extra_filters,
        satellite_table_source=tables["satellite_tables"],
        gpsro_table_source=tables["gpsro_tables"],
        satwnd_table_source=tables["satwnd_tables"],
        prepbufr_table_source={},
    )
    return pa.concat_tables(_sel(loader, model.frame_times(CYCLE)))


def _assert_same(a, b):
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if dataclasses.is_dataclass(x):
            _assert_same(x, y)
        elif isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and torch.equal(x, y), field.name
        else:
            assert x == y, field.name


def test_the_unified_table_gives_the_family_tables_model_input():
    model = _analysis_model()
    tables = _family_tables()
    model.run_analysis([CYCLE], **tables)
    unified = _loaded_unified_table(model, tables)
    assert set(unified["sensor_id"].to_pylist()) == {
        combined.SENSOR_NAME_TO_ID["atms"],
        combined.SENSOR_NAME_TO_ID[combined.CONV_SENSOR],
    }
    model.run_analysis_unified([CYCLE], unified)

    from_tables, from_unified = model.net.seen
    assert from_tables.obs.numel() == unified.num_rows
    _assert_same(from_tables, from_unified)


def _with_channel(table: pa.Table, channel: int) -> pa.Table:
    index = table.schema.get_field_index("Global_Channel_ID")
    ids = pa.array(np.full(table.num_rows, channel, dtype=np.uint16))
    return table.set_column(index, table.schema.field(index), ids)


def test_ids_are_bounded_by_the_network_not_the_packaged_vocabulary():
    model = _analysis_model(channel_rows=combined.NCHANNEL + 8)
    unified = _loaded_unified_table(model, _family_tables())
    unpackaged = _with_channel(unified, combined.NCHANNEL + 7)
    model.run_analysis_unified([CYCLE], unpackaged)
    assert (model.net.seen[-1].global_channel == combined.NCHANNEL + 7).all()
    with pytest.raises(ValueError, match="Global_Channel_ID .* is outside the network"):
        model.run_analysis_unified(
            [CYCLE], _with_channel(unified, combined.NCHANNEL + 8)
        )


def test_rows_outside_the_analysis_windows_are_not_checked():
    model = _analysis_model(channel_rows=combined.NCHANNEL + 8)
    unified = _loaded_unified_table(model, _family_tables())
    outside = _with_channel(unified, combined.NCHANNEL + 8).set_column(
        unified.schema.get_field_index("DA_window"),
        "DA_window",
        pc.add(unified["DA_window"], pa.scalar(pd.Timedelta(days=30))),
    )
    model.run_analysis_unified([CYCLE], pa.concat_tables([unified, outside]))
    assert model.net.seen[-1].obs.numel() == unified.num_rows


def test_analyses_run_one_at_a_time_and_stack():
    model = _analysis_model()
    unified = _loaded_unified_table(model, _family_tables())
    times = [CYCLE, CYCLE + pd.Timedelta(hours=6)]
    analyses = model.run_analysis_unified(times, unified)
    assert analyses.shape[0] == 2
    assert [obs.lengths.shape[1] for obs in model.net.seen] == [1, 1]

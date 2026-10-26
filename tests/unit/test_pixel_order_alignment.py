# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from earth2grid import healpix
import threading
import time

import pytest
import torch
import torch.fx._symbolic_trace as fx_trace

from healda.datasets import static_data
from healda.datasets.base import NormalizationStats, VariableConfig
from healda.datasets.da.transform import (
    TransformV2,
    fixed_static_condition,
    zscore_static,
)
from healda.cli.train import TrainingLoop


_PIXEL_ORDERS = {
    "hpxpadxy": healpix.HEALPIX_PAD_XY,
    "nest": healpix.NEST,
}
_CONFIG = VariableConfig(
    name="pixel_order_test",
    variables_2d=["state"],
    variables_3d=[],
    levels=[],
    variables_static=["orog", "lfrac"],
)
_NORMALIZATION = NormalizationStats(center=[0.0], scales=[1.0])


def _from_nest(values, pixel_order):
    target = _PIXEL_ORDERS[pixel_order]
    if target is healpix.NEST:
        return values
    return healpix.reorder(values, healpix.NEST, target)


@pytest.mark.parametrize("attention_backend", ["diffusers", "te"])
def test_training_transform_order_follows_attention_backend(attention_backend):
    loop = TrainingLoop(attention_backend=attention_backend)

    assert loop._transform_options.pixel_order == "hpxpadxy"


@pytest.mark.parametrize(
    "pixel_order",
    ["hpxpadxy", "nest"],
)
def test_state_and_static_condition_share_pixel_order(monkeypatch, pixel_order):
    npix = 12
    orography = torch.arange(npix, dtype=torch.float32)
    land_fraction = torch.arange(npix - 1, -1, -1, dtype=torch.float32)
    monkeypatch.setattr(
        static_data, "load_orography", lambda *args, **kwargs: orography
    )
    monkeypatch.setattr(
        static_data, "load_lfrac", lambda *args, **kwargs: land_fraction
    )

    condition = fixed_static_condition(
        _CONFIG, hpx_level=0, source="ufs", pixel_order=pixel_order
    )
    expected_condition = (
        torch.stack([zscore_static(orography), zscore_static(land_fraction)])
        .unsqueeze(0)
        .unsqueeze(2)
    )
    expected_condition = _from_nest(expected_condition, pixel_order)

    transform = TransformV2(
        variable_config=_CONFIG,
        hpx_level=0,
        hpx_level_condition=0,
        target_normalization=_NORMALIZATION,
        pixel_order=pixel_order,
    )
    packed = torch.arange(npix, dtype=torch.float32).reshape(1, 1, 1, npix)
    state = transform._device_field(
        packed, "target", transform.mean, transform.std, "cpu"
    )
    expected_state = _from_nest(packed.permute(0, 2, 1, 3), pixel_order)

    assert torch.equal(condition, expected_condition)
    assert torch.equal(state, expected_state)


@pytest.mark.parametrize(
    "pixel_order",
    ["hpxpadxy", "nest"],
)
def test_ang2pix_and_packing_share_pixel_order(pixel_order):
    lon = torch.tensor([12.0, 140.0, 275.0, 80.0])
    lat = torch.tensor([-65.0, -10.0, 35.0, 70.0])
    count = lon.numel()
    ident = torch.arange(count, dtype=torch.float32)
    zeros = torch.zeros(count)
    obs_tensors = {
        "absolute_obs_time": torch.zeros(count, dtype=torch.int64),
        "latitude": lat,
        "longitude": lon,
        "height": zeros,
        "pressure": zeros,
        "scan_angle": zeros,
        "sat_zenith_angle": zeros,
        "sol_zenith_angle": zeros,
        "platform_id": torch.zeros(count, dtype=torch.int32),
        "observation_type": torch.zeros(count, dtype=torch.int32),
        "local_channel_id": torch.zeros(count, dtype=torch.int32),
        "global_channel_id": torch.zeros(count, dtype=torch.int32),
        "observation": ident,
        "target_time_sec": torch.zeros(count, dtype=torch.int64),
    }
    transform = TransformV2(
        variable_config=_CONFIG,
        hpx_level=1,
        sensors=["atms"],
        target_normalization=_NORMALIZATION,
        attention_prepack=True,
        pixel_order=pixel_order,
    )
    lengths = torch.tensor([[[count]]], dtype=torch.int32)
    obs = transform.device_transform(
        {"unified_obs": (obs_tensors, lengths)}, device="cpu"
    )["unified_obs"]

    grid = healpix.Grid(level=1, pixel_order=_PIXEL_ORDERS[pixel_order])
    expected_pix = grid.ang2pix(lon, lat).int()
    packing = obs.attention_packing

    assert torch.equal(obs.pix, expected_pix[obs.obs.long()])
    assert torch.all(obs.pix[1:] >= obs.pix[:-1])
    assert torch.equal(
        packing.counts,
        torch.bincount(expected_pix.long(), minlength=12 * 4**1),
    )


def _obs_batch(lon, lat):
    count = lon.numel()
    zeros = torch.zeros(count)
    obs_tensors = {
        "absolute_obs_time": torch.zeros(count, dtype=torch.int64),
        "latitude": lat,
        "longitude": lon,
        "height": zeros,
        "pressure": zeros,
        "scan_angle": zeros,
        "sat_zenith_angle": zeros,
        "sol_zenith_angle": zeros,
        "platform_id": torch.zeros(count, dtype=torch.uint16),
        "observation_type": torch.zeros(count, dtype=torch.uint16),
        "local_channel_id": torch.zeros(count, dtype=torch.uint16),
        "global_channel_id": torch.zeros(count, dtype=torch.uint16),
        "observation": torch.arange(count, dtype=torch.float32),
        "target_time_sec": torch.zeros(count, dtype=torch.int64),
    }
    return {"unified_obs": (obs_tensors, torch.tensor([[[count]]], dtype=torch.int32))}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled pixel ops are CUDA only"
)
def test_compiled_pixel_ops_in_a_prefetch_thread():
    g = torch.Generator().manual_seed(0)
    lon = torch.cat(
        [torch.rand(5000, generator=g) * 360, torch.tensor([0.0, 90.0, 180.0, 270.0])]
    )
    lat = torch.cat(
        [
            torch.rand(5000, generator=g) * 180 - 90,
            torch.tensor([70.0, 43.5, 90.0, -60.5]),
        ]
    )
    transform = TransformV2(
        variable_config=_CONFIG,
        hpx_level=6,
        sensors=["atms"],
        target_normalization=_NORMALIZATION,
        attention_prepack=True,
        features_position=True,
        pixel_order="nest",
    )
    transform.compile_pixel_ops("cuda")
    expected = transform.device_transform(_obs_batch(lon, lat), device="cuda")[
        "unified_obs"
    ]
    other = torch.compile(lambda x: x * 2)
    other(torch.ones(4, device="cuda"))
    results, errors = [], []

    def prefetch():
        try:
            for _ in range(20):
                results.append(
                    transform.device_transform(_obs_batch(lon, lat), device="cuda")[
                        "unified_obs"
                    ]
                )
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=prefetch)
    # Setting torch's private FX tracing flag stands in for the main thread compiling a
    # model: dynamo refuses compiled calls meanwhile
    fx_trace._is_fx_tracing_flag = True
    try:
        thread.start()
        time.sleep(0.5)
    finally:
        fx_trace._is_fx_tracing_flag = False
    while thread.is_alive():
        other(torch.ones(4, device="cuda"))
    thread.join()

    assert not errors
    for obs in results:
        assert torch.equal(obs.pix, expected.pix)
        assert torch.equal(
            obs.float_metadata.view(torch.int16),
            expected.float_metadata.view(torch.int16),
        )

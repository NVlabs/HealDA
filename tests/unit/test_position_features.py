# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from earth2grid import healpix

from healda.config.models import SensorEmbedderConfig
from healda.observations.preprocessing.features import (
    N_FEATURES,
    position_features,
)

LEVEL = 6
PIXEL_DEG = math.degrees(math.sqrt(4 * math.pi / (12 * 4**LEVEL)))


def offsets(lon, lat):
    grid = healpix.Grid(LEVEL, pixel_order=healpix.NEST)
    lon, lat = (
        torch.tensor(lon, dtype=torch.float64),
        torch.tensor(lat, dtype=torch.float64),
    )
    center_lon, center_lat = grid.pix2ang(grid.ang2pix(lon, lat))
    return (
        position_features(lon, lat, center_lon, center_lat, LEVEL),
        center_lon,
        center_lat,
    )


def test_offset_is_zero_at_the_pixel_centre_and_scales_with_pixel_size():
    grid = healpix.Grid(LEVEL, pixel_order=healpix.NEST)
    center_lon, center_lat = grid.pix2ang(torch.tensor([1000, 30000]))
    at_centre, _, _ = offsets(center_lon.tolist(), center_lat.tolist())
    assert torch.allclose(at_centre[:, :2], torch.zeros(2, 2), atol=1e-6)
    north = (center_lat + 0.3 * PIXEL_DEG).tolist()
    shifted, lon_c, lat_c = offsets(center_lon.tolist(), north)
    moved = lat_c != center_lat
    assert torch.allclose(
        shifted[~moved, 1], torch.full((int((~moved).sum()),), 0.3), atol=1e-5
    )


def test_offset_wraps_across_the_dateline_and_stays_within_a_pixel():
    lon = torch.linspace(-180, 359.9, 2000).tolist()
    lat = torch.linspace(-89, 89, 2000).tolist()
    out, _, _ = offsets(lon, lat)
    assert out.dtype == torch.float32 and out.shape == (2000, 4)
    assert out[:, :2].abs().max() < 1.0
    assert torch.allclose(
        out[:, 2],
        torch.sin(torch.deg2rad(torch.tensor(lon, dtype=torch.float64))).float(),
        atol=1e-6,
    )


def test_meta_dim_counts_the_offset_columns():
    assert SensorEmbedderConfig().meta_dim == N_FEATURES
    assert SensorEmbedderConfig(features_position=True).meta_dim == N_FEATURES + 4


def test_position_features_are_packed_only_when_the_embedder_asks():
    import dataclasses

    import healda.models
    from healda.cli.train import LOOPS, NNJA_LATLON

    released = LOOPS[NNJA_LATLON]
    posfeat = dataclasses.replace(
        released,
        sensor_embedder_config=dataclasses.replace(
            released.sensor_embedder_config, features_position=True
        ),
    )
    options = healda.models.transform_options_for
    assert not options(released.model_config).features_position
    assert options(posfeat.model_config).features_position


def test_dense_preset_is_the_released_one_plus_its_listed_changes():
    import dataclasses

    from healda.cli.train import LOOPS, NNJA_LATLON

    released, dense = LOOPS[NNJA_LATLON], LOOPS["v2-nnja-latlon-dense"]
    changed = {
        f.name
        for f in dataclasses.fields(released)
        if getattr(released, f.name) != getattr(dense, f.name)
    }
    assert changed == {
        "attention_backend",
        "latlon_decode",
        "sensor_embedder_config",
        "obs_config",
    }
    assert dense.sensor_embedder_config.features_position
    assert dense.latlon_decode.fp32_output
    obs = dense.obs_config
    assert (obs.nnja_thin_nside, obs.nnja_satwnd_thin_hpx_level) == (128, 6)
    assert obs.nnja_max_quality_mark == 3
    assert obs.nnja_pressure_height_fill == "sonde" and obs.nnja_balloon_drift

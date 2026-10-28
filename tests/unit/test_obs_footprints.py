# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Footprint blocks expand to the same long rows on every device and through every path."""

import types
import weakref

import cftime
import numpy as np
import pyarrow as pa
import pytest
import torch

from healda.datasets.base import NormalizationStats
from healda.datasets.da.transform import TransformV2
from healda.inference import DAModel
from healda.observations.footprints import (
    FootprintBlock,
    expand,
    expand_column,
    valid_measurements,
)
from healda.observations.loaders import combined
from healda.observations.loaders.nnja_wide import NNJAWideLoader
from healda.observations.protocols import Observations, as_table

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SENSORS = ["amsua", "atms"]
TIME = cftime.DatetimeGregorian(2025, 1, 3, 0)


def make_block(sensor, channels, footprints, seed, valid="random"):
    rng = np.random.default_rng(seed)
    shape = (channels, footprints)
    mask = {
        "random": rng.random(shape) < 0.6,
        "all": np.ones(shape, dtype=bool),
        "none": np.zeros(shape, dtype=bool),
    }[valid]
    start = np.datetime64("2025-01-02T21:00", "ns")
    offset = combined.SENSOR_OFFSET[sensor]
    return FootprintBlock(
        sensor_id=combined.SENSOR_NAME_TO_ID[sensor],
        values=rng.standard_normal(shape).astype(np.float32),
        valid=mask,
        footprint_columns={
            "Latitude": rng.uniform(-90, 90, footprints).astype(np.float32),
            "Longitude": rng.uniform(0, 360, footprints).astype(np.float32),
            "Absolute_Obs_Time": start + rng.integers(0, 10**12, footprints),
            "DA_window": np.full(footprints, start),
            "Platform_ID": rng.integers(0, 40, footprints).astype(np.uint16),
            "Sat_Zenith_Angle": rng.uniform(-60, 60, footprints).astype(np.float32),
            "Sol_Zenith_Angle": rng.uniform(0, 180, footprints).astype(np.float32),
            "Scan_Angle": rng.uniform(-50, 50, footprints).astype(np.float32),
        },
        global_channel_id=np.arange(offset, offset + channels, dtype=np.uint16),
        local_channel_id=np.arange(channels, dtype=np.uint16),
    )


def reference_rows(block, name):
    """Long rows as per-channel flatnonzero, the order expand_column promises."""
    channel, footprint = np.nonzero(block.valid)
    if name == "Observation":
        return block.values[channel, footprint]
    if name == "Global_Channel_ID":
        return block.global_channel_id[channel]
    if name == "local_channel_id":
        return block.local_channel_id[channel]
    return block.footprint_columns[name][footprint]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("valid", ["random", "all", "none"])
def test_expand_column_matches_the_channel_major_reference(device, valid):
    block = make_block("atms", 5, 33, seed=1, valid=valid)
    measurements = valid_measurements(block, device)
    names = ["Observation", "Global_Channel_ID", "local_channel_id"]
    for name in names + list(block.footprint_columns):
        expected = reference_rows(block, name)
        out = np.empty(block.num_rows, dtype=expected.dtype)
        tensor = torch.from_numpy(out.view(np.int64) if out.dtype.kind == "M" else out)
        moved = tensor.to(device)
        expand_column(block, name, measurements, moved)
        tensor.copy_(moved.cpu())
        np.testing.assert_array_equal(out, expected, err_msg=name)


def test_a_cleared_cell_drops_exactly_its_row():
    schema = NNJAWideLoader(sensors=["amsua"]).output_schema
    block = make_block("amsua", 3, 10, seed=2, valid="all")
    before = expand(block, schema)
    block.valid[1, 4] = False
    after = expand(block, schema)
    keep = np.ones(30, dtype=bool)
    keep[1 * 10 + 4] = False
    assert after.equals(before.filter(pa.array(keep)))


def test_cell_dropout_draws_like_row_dropout():
    loader = NNJAWideLoader(sensors=["amsua"])
    block = make_block("amsua", 6, 200, seed=10)
    platform = np.random.default_rng(11).choice([3, 7], 200).astype(np.uint16)
    block.footprint_columns["Platform_ID"] = platform
    ids = block.global_channel_id
    rules = ((3, int(ids[2]), 0.5), (7, int(ids[4]), 0.3), (7, int(ids[1]), 0.0))
    long = expand(block, loader.output_schema)

    # Row-level reference: one draw per targeted long row, rule after rule.
    row_platform = long["Platform_ID"].to_numpy()
    row_channel = long["Global_Channel_ID"].to_numpy()
    keep = np.ones(long.num_rows, dtype=bool)
    rng = np.random.default_rng(5)
    for platform_id, channel_id, probability in rules:
        targeted = (row_platform == platform_id) & (row_channel == channel_id)
        if probability > 0.0 and targeted.any():
            keep[targeted] = rng.random(int(targeted.sum())) >= probability

    loader._apply_platform_channel_dropout(block, np.random.default_rng(5), rules)
    assert not keep.all()
    assert expand(block, loader.output_schema).equals(long.filter(pa.array(keep)))


def frames_of(blocks_by_frame, as_tables):
    empty = NNJAWideLoader(sensors=SENSORS)._empty()
    frames = []
    for blocks in blocks_by_frame:
        obs = Observations(blocks=blocks, table=empty)
        frames.append({"obs_v2": obs.to_table() if as_tables else obs})
    return [frames]


def test_observations_to_table_is_its_blocks_expanded_then_its_table():
    schema = NNJAWideLoader(sensors=SENSORS).output_schema
    blocks = [make_block("atms", 4, 50, seed=3), make_block("amsua", 6, 20, seed=4)]
    rows = expand(make_block("amsua", 6, 10, seed=12), schema)
    obs = Observations(blocks=blocks, table=rows)
    expected = pa.concat_tables([*(expand(block, schema) for block in blocks), rows])
    assert obs.num_rows == expected.num_rows
    assert as_table(obs).equals(expected)
    assert as_table(expected) is expected


def transform(**kwargs):
    return TransformV2(
        sensors=SENSORS,
        use_nnja_sat=True,
        target_normalization=NormalizationStats(center=np.zeros(1), scales=np.ones(1)),
        **kwargs,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_blocks_pack_bitwise_like_their_long_tables(device):
    # Two frames; atms listed before amsua per frame, so the sensor sort reorders them.
    blocks_by_frame = [
        [make_block("atms", 4, 50, seed=3), make_block("amsua", 6, 20, seed=4)],
        [
            make_block("atms", 4, 0, seed=5),
            make_block("amsua", 6, 30, seed=6, valid="all"),
            make_block("amsua", 6, 25, seed=7, valid="none"),
        ],
    ]
    times = [[TIME, TIME + np.timedelta64(6, "h").item()]]
    expected, expected_lengths = transform()._process_obs(
        times, frames_of(blocks_by_frame, as_tables=True)
    )
    packed, lengths = transform()._process_obs(
        times, frames_of(blocks_by_frame, as_tables=False), device
    )
    assert torch.equal(lengths, expected_lengths)
    assert list(packed) == list(expected)
    for key, want in expected.items():
        got = packed[key].cpu()
        assert got.dtype == want.dtype, key
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_device_tensors_are_freed_before_the_forward():
    tr = transform(hpx_level=6, attention_prepack=True)
    blocks = [make_block("atms", 4, 5000, seed=8), make_block("amsua", 6, 3000, seed=9)]
    packed = weakref.WeakValueDictionary()
    process_obs = tr._process_obs

    def recording_process_obs(*args, **kwargs):
        tensors, lengths = process_obs(*args, **kwargs)
        packed.update(tensors)
        return tensors, lengths

    tr._process_obs = recording_process_obs
    alive_at_forward = []

    class Net:
        def tokenize_observations(self, unified_obs):
            alive_at_forward.append(sorted(packed))
            return unified_obs

        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(out=torch.zeros(1, 1, 1, 1, device="cuda"))

    model = types.SimpleNamespace(
        device=torch.device("cuda"),
        transform=tr,
        net=Net(),
        loop=types.SimpleNamespace(bf16=False, dit_temporal_attention_causal=False),
    )
    DAModel._forward(model, [[TIME]], frames_of([blocks], as_tables=False))
    assert alive_at_forward == [[]]


def test_training_featurizes_observations_as_inference_does():
    tr = transform(hpx_level=6, attention_prepack=True)
    blocks = [make_block("atms", 4, 500, seed=8), make_block("amsua", 6, 300, seed=9)]
    frames = frames_of([blocks], as_tables=False)
    inference, _ = tr.transform_observations([[TIME]], frames, "cpu")
    # The obs part of the batch `transform` builds for training.
    batch = {"unified_obs": tr._process_obs([[TIME]], frames)}
    keys = list(batch["unified_obs"][0])
    training = tr.device_transform(batch, "cpu")["unified_obs"]
    assert list(batch["unified_obs"][0]) == keys
    for got, want in [
        (training.obs, inference.obs),
        (training.float_metadata, inference.float_metadata),
        (training.pix, inference.pix),
        (training.local_channel, inference.local_channel),
        (training.global_channel, inference.global_channel),
        (training.global_platform, inference.global_platform),
        (training.lengths, inference.lengths),
    ]:
        assert torch.equal(got, want)


def test_no_observations_give_empty_features_in_training_and_inference():
    tr = transform(hpx_level=6, attention_prepack=True)
    frames = frames_of([[make_block("atms", 4, 0, seed=8)]], as_tables=False)
    inference, _ = tr.transform_observations([[TIME]], frames, "cpu")
    batch = {"unified_obs": tr._process_obs([[TIME]], frames)}
    training = tr.device_transform(batch, "cpu")["unified_obs"]
    for obs in (inference, training):
        assert obs.obs.shape[0] == obs.pix.shape[0] == obs.float_metadata.shape[0] == 0
        assert not obs.attention_packing.counts.any()
        assert not obs.attention_packing.cu_seqlens_k.any()

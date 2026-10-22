# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: obs rows must follow model sensor order, not ascending id.

Packing recovers each obs's (batch, time) window from its position in lengths,
so a wrong row order misassigns obs once B*T > 1 (invisible at B*T == 1).
"""

import pyarrow as pa
import pytest
import torch

from healda.datasets.da.transform import TransformV2, _map_platform_to_local
from healda.varlen import lengths_to_idx


def _sensor_batch(sensor_id, payloads):
    # One record batch per (sensor, b, t); unique payloads stay traceable.
    n = len(payloads)
    return pa.record_batch(
        {
            "sensor_id": pa.array([sensor_id] * n, type=pa.int16()),
            "payload": pa.array(payloads, type=pa.int64()),
        }
    )


def _pack(frames, sensor_rank, ordered_sensor_ids, B, T):
    # Mirror _process_obs's sort + lengths build; return (window_of_obs, payloads)
    # with window = batch_idx*T + time_idx.
    parts = TransformV2._sorted_parts(frames, sensor_rank)
    model_rank = {int(s): r for r, s in enumerate(ordered_sensor_ids.tolist())}
    lengths = torch.zeros((len(model_rank), B, T), dtype=torch.int32)
    for _, b, t, part in parts:
        lengths[model_rank[part["sensor_id"][0].as_py()], b, t] += part.num_rows
    payloads = torch.cat(
        [torch.from_numpy(part["payload"].to_numpy().copy()) for *_, part in parts]
    )
    window_of_obs = lengths_to_idx(lengths) % (B * T)
    return window_of_obs, payloads


def _assemble_bt(obs_by_window, ordered_sensor_ids, B, T, split=None):
    # obs_by_window: (b,t) -> {sensor_id: [..]}; split cuts each group into batches of
    # that many rows.
    frames = []
    for b in range(B):
        frames.append([])
        for t in range(T):
            batches = []
            for sensor_id, payloads in obs_by_window[(b, t)].items():
                step = split or len(payloads)
                for start in range(0, len(payloads), step):
                    batches.append(
                        _sensor_batch(sensor_id, payloads[start : start + step])
                    )
            frames[b].append({"obs_v2": pa.Table.from_batches(batches)})
    sensor_rank = {int(s): r for r, s in enumerate(ordered_sensor_ids.tolist())}
    return _pack(frames, sensor_rank, ordered_sensor_ids, B, T)


def _assemble(per_sample, ordered_sensor_ids):
    # T==1 wrapper: window index == sample index. per_sample[b] = {sensor_id: [..]}.
    B = len(per_sample)
    obs_by_window = {(b, 0): per_sample[b] for b in range(B)}
    return _assemble_bt(obs_by_window, ordered_sensor_ids, B=B, T=1)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_platform_mapping_uses_sensor_local_luts(device):
    lengths = torch.tensor([[[2], [1]], [[1], [0]]], device=device)
    platform = torch.tensor([1, 3, 1, 3], device=device)
    lut_matrix = torch.tensor([[0, 10, 0, 30], [0, 100, 0, 300]], device=device)

    actual = _map_platform_to_local(platform, lengths, lut_matrix)

    torch.testing.assert_close(actual.cpu(), torch.tensor([10, 30, 10, 300]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_platform_mapping_does_not_synchronize():
    lengths = torch.tensor([[[2]], [[1]]], device="cuda")
    platform = torch.tensor([1, 1, 1], device="cuda")
    lut_matrix = torch.tensor([[0, 10], [0, 100]], device="cuda")

    torch.cuda.set_sync_debug_mode("error")
    try:
        _map_platform_to_local(platform, lengths, lut_matrix)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_obs_partitioned_by_sample_with_non_ascending_sensor_order():
    # Sensor 9 sits mid-list despite having the largest id (the conv-plevel case).
    ordered_sensor_ids = torch.tensor([0, 9, 1], dtype=torch.int32)
    per_sample = [
        {0: [100, 101], 9: [200, 201, 202], 1: [300]},  # sample 0
        {0: [110], 9: [210], 1: [310, 311]},  # sample 1
    ]
    expected = {
        0: {100, 101, 200, 201, 202, 300},
        1: {110, 210, 310, 311},
    }

    window_of_obs, payloads = _assemble(per_sample, ordered_sensor_ids)

    for sample, want in expected.items():
        got = set(payloads[window_of_obs == sample].tolist())
        assert got == want, (
            f"sample {sample} observations leaked across the batch: "
            f"got {sorted(got)}, expected {sorted(want)}"
        )


def test_obs_partitioned_by_time_frame_multiframe_batch1():
    # B=1, T=2: B*T > 1, so the bug scatters obs across frames even at batch 1.
    ordered_sensor_ids = torch.tensor([0, 9, 1], dtype=torch.int32)
    obs_by_window = {
        (0, 0): {0: [100, 101], 9: [200, 201, 202], 1: [300]},  # frame 0
        (0, 1): {0: [110], 9: [210], 1: [310, 311]},  # frame 1
    }
    expected = {
        0: {100, 101, 200, 201, 202, 300},
        1: {110, 210, 310, 311},
    }

    window_of_obs, payloads = _assemble_bt(obs_by_window, ordered_sensor_ids, B=1, T=2)

    for frame, want in expected.items():
        got = set(payloads[window_of_obs == frame].tolist())
        assert got == want, (
            f"frame {frame} observations leaked across time: "
            f"got {sorted(got)}, expected {sorted(want)}"
        )


def test_a_window_split_across_record_batches_lands_in_one_frame():
    # lengths_3d reads (sensor, b, t) off row 0 of each batch and accumulates, so a group
    # spread over several batches must total the same as one. Filtering with an expression
    # used to split every group at 32,768 rows; the materialized mask leaves one batch.
    ordered_sensor_ids = torch.tensor([0, 9, 1], dtype=torch.int32)
    obs_by_window = {
        (0, 0): {0: [100, 101, 102], 9: [200, 201, 202, 203], 1: [300]},
        (0, 1): {0: [110, 111], 9: [210], 1: [310, 311, 312]},
    }

    whole = _assemble_bt(obs_by_window, ordered_sensor_ids, B=1, T=2)
    split = _assemble_bt(obs_by_window, ordered_sensor_ids, B=1, T=2, split=2)

    torch.testing.assert_close(whole[0], split[0])
    torch.testing.assert_close(whole[1], split[1])


def test_a_record_batch_mixing_sensors_raises():
    mixed = pa.record_batch(
        {
            "sensor_id": pa.array([0, 1], type=pa.int16()),
            "payload": pa.array([100, 300], type=pa.int64()),
        }
    )
    frames = [[{"obs_v2": pa.Table.from_batches([mixed])}]]
    with pytest.raises(ValueError, match="more than one sensor_id"):
        TransformV2._sorted_parts(frames, {0: 0, 1: 1})


def test_raw_ascending_sort_would_leak_across_batch():
    # Negative control: sorting by ascending sensor_id against lengths in model order
    # must still leak.
    ordered_sensor_ids = torch.tensor([0, 9, 1], dtype=torch.int32)
    per_sample = [
        {0: [100, 101], 9: [200, 201, 202], 1: [300]},
        {0: [110], 9: [210], 1: [310, 311]},
    ]
    frames = [
        [
            {
                "obs_v2": pa.Table.from_batches(
                    [_sensor_batch(s, p) for s, p in sample.items()]
                )
            }
        ]
        for sample in per_sample
    ]
    ascending = {s: r for r, s in enumerate(sorted(ordered_sensor_ids.tolist()))}
    window_of_obs, payloads = _pack(frames, ascending, ordered_sensor_ids, B=2, T=1)
    got0 = set(payloads[window_of_obs == 0].tolist())
    assert got0 != {100, 101, 200, 201, 202, 300}

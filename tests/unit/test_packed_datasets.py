# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed reader + dataset tests over synthetic in-memory stores (no zarr).

Covers the multi-lag alignment core of ``PackedDADataset`` / ``PackedStateReader``:
offset store-row reads, per-frame validity masking, and background assembly.
"""

import numpy as np
import pandas as pd
import xarray as xr

from healda.datasets.base import NormalizationStats
from healda.datasets.da.packed_da_dataset import (
    PackedDADataset,
    _AlignedSource,
    _concat_normalizations,
)
from healda.datasets.da.packed_state import PackedStateReader


def _make_reader(
    times, channel_names, *, values=None, is_valid=None, source_status=None, x=3
):
    # Real PackedStateReader over a synthetic in-memory store (no zarr). Default
    # values encode row identity as value[t, c, :] = t * 10 + c, so a read can be
    # traced back to the frame and channel it came from.
    times = pd.DatetimeIndex(times)
    nt, nc = len(times), len(channel_names)
    if values is None:
        rows = np.arange(nt)[:, None, None] * 10
        cols = np.arange(nc)[None, :, None]
        values = (rows + cols + np.zeros((nt, nc, x))).astype("float32")
    variables = {
        "data": xr.DataArray(
            values,
            dims=("time", "variable", "x"),
            coords={"time": times, "variable": list(channel_names)},
        )
    }
    if is_valid is not None:
        variables["is_valid"] = xr.DataArray(
            np.asarray(is_valid, dtype=bool), dims=("time", "variable")
        )
    if source_status is not None:
        variables["source_status"] = xr.DataArray(
            np.asarray(source_status, dtype=np.uint8), dims=("time", "variable")
        )
    dataset = xr.Dataset(variables)

    class _Entry:
        def to_xarray(self, chunks=None):
            return dataset

    return PackedStateReader(_Entry(), list(channel_names), source="")


def test_aligned_source_reads_offset_store_row():
    """A -6h source must read the store frame 6h before each query time."""
    store_times = pd.date_range("2020-01-01 00:00", periods=4, freq="6h")
    reader = _make_reader(store_times, ["c0", "c1"])
    source = _AlignedSource(reader, offset_hours=-6)

    # Query at 06:00 and 12:00; the -6h source should land on 00:00 and 06:00,
    # i.e. store rows 0 and 1.
    query_times = pd.to_datetime(["2020-01-01 06:00", "2020-01-01 12:00"])
    source.bind(query_times)

    frames = source.read_at({"time": np.array([0, 1])})

    assert frames.shape == (2, 2, 3)
    # value == row * 10 + channel: row 0 -> [0, 1], row 1 -> [10, 11].
    assert np.array_equal(frames[0, :, 0], [0, 1])
    assert np.array_equal(frames[1, :, 0], [10, 11])


def test_aligned_source_valid_mask_applies_offset():
    """valid_mask shifts the query times by the offset before checking validity."""
    store_times = pd.to_datetime(
        ["2019-12-31 18:00", "2020-01-01 00:00", "2020-01-01 12:00"]
    )
    is_valid = np.array([[True], [False], [True]])  # the 00:00 frame is invalid
    reader = _make_reader(store_times, ["c0"], is_valid=is_valid)
    source = _AlignedSource(reader, offset_hours=-6)

    target_times = pd.date_range("2020-01-01 00:00", periods=4, freq="6h")
    # Shifted to [-6h, 0h, 6h, 12h]: 6h is absent and 0h is present-but-invalid,
    # so only the target's 00:00 and 18:00 survive the offset source's mask.
    kept = target_times[source.valid_mask(target_times)]

    assert list(kept) == [target_times[0], target_times[3]]


def test_valid_store_mask_marks_invalid_and_absent_times():
    store_times = pd.date_range("2020-01-01 00:00", periods=4, freq="6h")
    is_valid = np.ones((4, 2), dtype=bool)
    is_valid[1, 1] = False  # frame 1: one channel unwritten -> invalid
    source_status = np.ones((4, 2), dtype=np.uint8)
    source_status[2, 0] = 0  # frame 2: one channel not WRITTEN_VALID -> invalid
    reader = _make_reader(
        store_times, ["c0", "c1"], is_valid=is_valid, source_status=source_status
    )

    query_times = pd.to_datetime(
        [
            "2020-01-01 00:00",  # present, valid
            "2020-01-01 06:00",  # present, invalid via is_valid
            "2020-01-01 12:00",  # present, invalid via source_status
            "2020-01-01 18:00",  # present, valid
            "2020-01-02 06:00",  # absent from the store
        ]
    )

    mask = reader.valid_store_mask(query_times)

    assert list(mask) == [True, False, False, True, False]


def test_concat_normalizations_preserves_input_order():
    a = NormalizationStats(center=[1.0, 2.0], scales=[3.0, 4.0])
    b = NormalizationStats(center=[5.0], scales=[6.0])

    out = _concat_normalizations([a, b])

    assert np.array_equal(out.center, [1.0, 2.0, 5.0])
    assert np.array_equal(out.scales, [3.0, 4.0, 6.0])


def test_get_concatenates_inputs_in_channel_order():
    """get() must stack input blocks into background in input/channel order."""
    times = pd.date_range("2020-01-01 00:00", periods=2, freq="6h")
    target = _AlignedSource(_make_reader(times, ["s0"]))
    # input A channels [a0, a1] -> values t*10 + c; input B channel [b0] offset
    # by 100 so the two blocks are distinguishable in the concatenation.
    input_a = _AlignedSource(_make_reader(times, ["a0", "a1"]))
    b_values = (np.arange(2)[:, None, None] * 10 + 100 + np.zeros((2, 1, 3))).astype(
        "float32"
    )
    input_b = _AlignedSource(_make_reader(times, ["b0"], values=b_values))
    for source in (target, input_a, input_b):
        source.bind(times)

    ds = PackedDADataset.__new__(PackedDADataset)
    ds._target = target
    ds._inputs = (input_a, input_b)
    ds._obs_loader = None
    ds._query_times = times
    ds._indexer = [{"time": np.array([0, 1])}]

    frame_times, objs = ds.get(0)

    assert len(frame_times) == 2
    assert len(objs) == 2
    background = np.stack([obj["background"] for obj in objs])  # (T, C, X)
    assert background.shape == (2, 3, 3)
    # Channel order is [a0, a1, b0]; row 0 -> [0, 1, 100], row 1 -> [10, 11, 110].
    assert np.array_equal(background[0, :, 0], [0, 1, 100])
    assert np.array_equal(background[1, :, 0], [10, 11, 110])

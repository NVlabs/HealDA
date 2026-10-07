# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from healda.inference_output import METRICS, AsyncZarrWriter, open_output_store

CHANNELS = ["Z500", "T850"]
CELLS = 32
SAMPLES = 10


def _build_group(tmp_path, time_chunk=1):
    group = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    for name in CHANNELS:
        group.create_array(
            name, shape=(SAMPLES, CELLS), chunks=(time_chunk, CELLS), dtype="f4"
        )
    return group


def _predictions():
    generator = torch.Generator().manual_seed(0)
    batches = []
    for start in range(0, SAMPLES, 2):
        data = torch.rand(2, len(CHANNELS), 3, CELLS, generator=generator)
        batches.append((np.arange(start, start + 2), data))
    return batches


def _expected_analysis(batches):
    return {
        name: np.concatenate([data[:, c, -1].numpy() for _, data in batches])
        for c, name in enumerate(CHANNELS)
    }


def test_writer_stores_the_last_frame_of_each_analysis(tmp_path):
    batches = _predictions()
    group = _build_group(tmp_path, time_chunk=4)
    writer = AsyncZarrWriter(group, CHANNELS, workers=2)
    for index, data in batches:
        writer.submit(index, data)
    writer.close()

    for name, want in _expected_analysis(batches).items():
        np.testing.assert_array_equal(group[name][:], want)


def test_writer_rounds_mantissa(tmp_path):
    group = _build_group(tmp_path, time_chunk=4)
    writer = AsyncZarrWriter(group, CHANNELS, workers=2, mantissa_bits=15)
    batches = _predictions()
    special = torch.tensor([float("nan"), float("inf"), -float("inf")])
    batches[0][1][0, 0, -1, :3] = special
    for index, data in batches:
        writer.submit(index, data)
    writer.close()

    written = group["Z500"][:]
    exact = _expected_analysis(batches)["Z500"]
    np.testing.assert_array_equal(written[0, :3], special.numpy())
    finite = np.isfinite(exact)
    assert np.all(written.view(np.uint32)[finite] & 0xFF == 0)
    np.testing.assert_allclose(written[finite], exact[finite], rtol=2.0**-16)
    assert not np.array_equal(written[finite], exact[finite])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_writer_copies_cuda_predictions_through_the_side_stream(tmp_path):
    batches = _predictions()
    group = _build_group(tmp_path, time_chunk=4)
    writer = AsyncZarrWriter(group, CHANNELS, workers=2)
    for index, data in batches:
        writer.submit(index, data.cuda())
    writer.close()

    for name, want in _expected_analysis(batches).items():
        np.testing.assert_array_equal(group[name][:], want)


def test_metrics_are_written_with_their_analysis_chunk(tmp_path):
    group = _build_group(tmp_path, time_chunk=4)
    frames = 3
    for name in METRICS:
        group.create_array(
            name,
            shape=(SAMPLES, len(CHANNELS), frames),
            chunks=(4, len(CHANNELS), frames),
            dtype="f4",
            fill_value=float("nan"),
        )
    writer = AsyncZarrWriter(group, CHANNELS, workers=2)
    expected = np.full((SAMPLES, len(CHANNELS), frames), np.nan, np.float32)
    # 8 of 10 times: the last chunk stays partial
    for index, data in _predictions()[:4]:
        rmse = np.random.default_rng(int(index[0])).random(
            (len(index), len(CHANNELS), frames), dtype=np.float32
        )
        expected[index] = rmse
        writer.submit(index, data, metrics=(rmse, rmse + 1))
    writer.close()

    np.testing.assert_array_equal(group["rmse"][:], expected)
    np.testing.assert_array_equal(group["mae"][:], expected + 1)
    assert group["rmse"].nchunks == 3


def test_a_store_labels_its_fields_when_created_and_when_extended(tmp_path):
    import xarray as xr

    path = str(tmp_path / "analysis.zarr")
    step = pd.Timedelta("6h")
    identity = {"checkpoint": "ckpt", "obs_config": "{}"}
    grid = (("lat", "lon"), (2, 3))
    times = pd.date_range("2026-01-01T00", periods=2, freq=step)
    channels = ["Z500", "tas"]
    open_output_store(
        path, channels, times, step, extend=False, identity=identity, grid=grid
    )
    ds = xr.open_zarr(path)
    assert ds["Z500"].attrs["units"] == "m2 s-2"
    assert ds["tas"].attrs["long_name"] == "Temperature at 2 m"

    zarr.open_group(path, mode="r+")["tas"].attrs.clear()
    zarr.consolidate_metadata(path)
    open_output_store(
        path, channels, times, step, extend=True, identity=identity, grid=grid
    )
    assert xr.open_zarr(path)["tas"].attrs["units"] == "K"

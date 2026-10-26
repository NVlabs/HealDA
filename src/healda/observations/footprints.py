# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Satellite observations at footprint level, and their expansion to long rows.

A satellite row group is a channel x footprint matrix. Its long rows are the valid
cells in channel-major order: every kept footprint of the first channel, ascending,
then of the next. The expansion is written in torch and runs on the device its output
lives on: the CPU for the long-table path and dataloader workers, the GPU at
inference, where the matrix crosses to the device instead of its long rows.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pyarrow as pa
import torch

from healda.observations.schema import SENSOR_ID, long_table
from healda.utils.profiling import cpu_timing_range

# Sources are read, never written; Arrow hands them over read-only. The filter also
# covers the transform, which imports this module.
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
)

OBSERVATION = "Observation"
# Unified columns with one value per channel
CHANNEL_COLUMNS = ("Global_Channel_ID", "local_channel_id")

# Same-width signed dtypes: CUDA has no uint16 gather or copy kernels, and the bits are
# what is moved.
_SIGNED = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


@dataclasses.dataclass
class FootprintBlock:
    """One sensor's selected footprints of one row group.

    ``values`` is (channels, footprints) float32; ``valid`` marks the cells that become
    long rows, None meaning every cell. ``footprint_columns`` holds one value per
    footprint under its unified column name; the channel ids one value per channel.
    """

    sensor_id: int
    values: np.ndarray
    valid: np.ndarray | None
    footprint_columns: dict[str, np.ndarray]
    global_channel_id: np.ndarray
    local_channel_id: np.ndarray

    @property
    def footprint_count(self) -> int:
        return self.values.shape[1]

    @property
    def num_rows(self) -> int:
        if self.valid is None:
            return self.values.size
        return int(np.count_nonzero(self.valid))

    def cell_mask(self) -> np.ndarray:
        """``valid``, materialized first when every cell is valid, for clearing cells."""
        if self.valid is None:
            self.valid = np.ones(self.values.shape, dtype=bool)
        return self.valid

    def channel_columns(self) -> dict[str, np.ndarray]:
        return {
            "Global_Channel_ID": self.global_channel_id,
            "local_channel_id": self.local_channel_id,
        }

    def has_column(self, name: str) -> bool:
        return (
            name == OBSERVATION
            or name in CHANNEL_COLUMNS
            or name in self.footprint_columns
        )


def as_tensor(array: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """``array`` on ``device``; datetime64 as its int64 ticks, unsigned as signed bits."""
    if array.dtype.kind == "M":
        array = array.view(np.int64)
    return bits(torch.from_numpy(np.ascontiguousarray(array))).to(device)


def bits(tensor: torch.Tensor) -> torch.Tensor:
    signed = _SIGNED.get(tensor.dtype)
    return tensor if signed is None else tensor.view(signed)


@dataclasses.dataclass(frozen=True)
class LongCells:
    """The valid cells of a block: how many each channel keeps, and their footprints,
    channel after channel."""

    counts: list[int]
    footprint: torch.Tensor


def long_cells(block: FootprintBlock, device: torch.device | str) -> LongCells | None:
    """The block's valid cells on ``device``; None when every cell is valid.

    One nonzero per channel row rather than one over the matrix: the per-row index is
    the footprint itself, where a flat index needs a division per cell.
    """
    if block.valid is None:
        return None
    counts = np.count_nonzero(block.valid, axis=1).tolist()
    valid = as_tensor(block.valid, device)
    footprint = torch.empty(sum(counts), dtype=torch.int64, device=device)
    offset = 0
    for channel, count in enumerate(counts):
        footprint[offset : offset + count] = valid[channel].nonzero().squeeze(1)
        offset += count
    return LongCells(counts, footprint)


def expand_column(
    block: FootprintBlock, name: str, cells: LongCells | None, out: torch.Tensor
) -> None:
    """Write column ``name`` of the block's long rows into ``out``, 1-D of ``num_rows``.

    ``cells`` is ``long_cells`` on ``out``'s device.
    """
    out = bits(out)
    if name in CHANNEL_COLUMNS or name == OBSERVATION:
        source = block.values if name == OBSERVATION else block.channel_columns()[name]
        source = as_tensor(source, out.device)
        if cells is None:
            grid = out.view(block.values.shape)
            grid.copy_(
                source if name == OBSERVATION else source[:, None].expand_as(grid)
            )
            return
        offset = 0
        for channel, count in enumerate(cells.counts):
            rows = out[offset : offset + count]
            if name == OBSERVATION:
                footprint = cells.footprint[offset : offset + count]
                torch.index_select(source[channel], 0, footprint, out=rows)
            else:
                rows.copy_(source[channel].expand(count))
            offset += count
        return
    column = as_tensor(block.footprint_columns[name], out.device)
    if cells is None:
        out.view(block.values.shape).copy_(column.expand(block.values.shape))
    else:
        torch.index_select(column, 0, cells.footprint, out=out)


def expand(block: FootprintBlock, schema: pa.Schema) -> pa.Table:
    """The block's long rows under ``schema``, in its channel-major order.

    Each column the block carries is `expand_column` run on the CPU into the array the
    table wraps; ``sensor_id`` is the block's; every other column of ``schema`` is null.
    """
    long_rows = block.num_rows
    arrays: dict[str, pa.Array] = {}
    sources = {
        OBSERVATION: block.values,
        **block.channel_columns(),
        **block.footprint_columns,
    }
    with cpu_timing_range("nnja.expand"):
        cells = long_cells(block, "cpu")
        for name, source in sources.items():
            expanded = np.empty(long_rows, dtype=source.dtype)
            expand_column(block, name, cells, as_tensor(expanded, "cpu"))
            arrays[name] = pa.array(expanded, type=schema.field(name).type)

    with cpu_timing_range("nnja.expand.table"):
        arrays[SENSOR_ID.name] = pa.array(
            np.full(long_rows, block.sensor_id, dtype=np.uint16), type=SENSOR_ID.type
        )
        return long_table(arrays, schema)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public contracts implemented by observation loaders."""

import dataclasses
from typing import Protocol, TypedDict

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from healda.observations.footprints import FootprintBlock, expand


@dataclasses.dataclass
class Observations:
    """One analysis time's observations: satellite footprint blocks plus long rows.

    ``table`` holds the long rows not in ``blocks`` and may be empty; its schema is the
    schema of the blocks' long rows.
    """

    blocks: list[FootprintBlock]
    table: pa.Table

    @property
    def num_rows(self) -> int:
        """Long rows after expansion, counted without expanding."""
        return sum(block.num_rows for block in self.blocks) + self.table.num_rows

    def to_table(self) -> pa.Table:
        """The long table: each block expanded on the CPU, in order, then ``table``."""
        schema = self.table.schema
        return pa.concat_tables(
            [*(expand(block, schema) for block in self.blocks), self.table]
        )


class ObservationBatch(TypedDict):
    obs_v2: list[Observations | pa.Table]


class ObservationLoader(Protocol):
    async def sel_time(self, times: pd.DatetimeIndex) -> ObservationBatch:
        """Load the observations of each requested analysis time, as a long table or
        as `Observations`."""
        ...


def as_table(obs: Observations | pa.Table) -> pa.Table:
    return obs if isinstance(obs, pa.Table) else obs.to_table()


OBS_TIME = "Absolute_Obs_Time"


def observed_before(
    obs: Observations | pa.Table, cutoff: pd.Timestamp
) -> Observations | pa.Table:
    """``obs`` without what was observed at or after ``cutoff``."""
    cutoff = cutoff.as_unit("ns").to_datetime64()
    stamp = pa.scalar(cutoff, type=pa.timestamp("ns"))
    # A loader returns one of two forms per analysis time (ObservationBatch): a long table
    # from a single-source loader (UFS, NNJA conventional, GPS-RO, satellite winds), or
    # Observations from NNJAWideLoader and CombinedObsLoader.
    if isinstance(obs, pa.Table):
        return obs.filter(pc.less(obs[OBS_TIME], stamp))
    # Observations: satellite footprints stay in their blocks with late ones marked
    # invalid, as channel denials do; the long rows beside them are filtered.
    blocks = [
        dataclasses.replace(
            block,
            valid=block.valid & (block.footprint_columns[OBS_TIME] < cutoff),
        )
        for block in obs.blocks
    ]
    return Observations(
        blocks=blocks, table=obs.table.filter(pc.less(obs.table[OBS_TIME], stamp))
    )

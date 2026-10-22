# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public contracts implemented by observation loaders."""

import dataclasses
from typing import Protocol, TypedDict

import pandas as pd
import pyarrow as pa

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

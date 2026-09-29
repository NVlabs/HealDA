# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import get_type_hints

import pandas as pd
import pyarrow as pa

from healda.observations import ObservationLoader
from healda.observations.protocols import ObservationBatch


class ExampleObservationLoader:
    async def sel_time(self, times: pd.DatetimeIndex) -> ObservationBatch:
        return {
            "obs_v2": [
                pa.table({"analysis_time": [time.to_datetime64()]}) for time in times
            ]
        }


def test_observation_loader_output_contract():
    times = pd.date_range("2026-01-01", periods=2, freq="6h")
    loader: ObservationLoader = ExampleObservationLoader()

    batch = asyncio.run(loader.sel_time(times))

    assert set(batch) == {"obs_v2"}
    assert len(batch["obs_v2"]) == len(times)
    assert all(isinstance(table, pa.Table) for table in batch["obs_v2"])


def test_observation_loader_protocol_declares_batch_return_type():
    assert get_type_hints(ObservationLoader.sel_time)["return"] is ObservationBatch
    assert ObservationBatch.__required_keys__ == frozenset({"obs_v2"})

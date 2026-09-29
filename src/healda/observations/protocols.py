# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public contracts implemented by observation loaders."""

from typing import Protocol, TypedDict

import pandas as pd
import pyarrow as pa


class ObservationBatch(TypedDict):
    obs_v2: list[pa.Table]


class ObservationLoader(Protocol):
    async def sel_time(self, times: pd.DatetimeIndex) -> ObservationBatch:
        """Load one observation table for each requested analysis time."""
        ...

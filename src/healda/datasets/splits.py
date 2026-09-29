# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple year-holdout split helper."""

import dataclasses
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["", "all", "train", "val", "test"]
# Inference sets the split to the calendar years a --start-date run spans, which may lie
# outside the fixed holdouts.
Split = SplitName | Sequence[int]


@dataclasses.dataclass(frozen=True)
class YearHoldoutSplit:
    """Train on all years except one validation year and one test year."""

    validation_year: int = 2022
    test_year: int = 2025

    def mask(self, times: pd.DatetimeIndex, split: Split) -> np.ndarray:
        years = times.year
        if not isinstance(split, str):
            return np.isin(years, list(split))

        if split in ("", "all"):
            return np.ones(len(times), dtype=bool)

        if split == "train":
            return np.asarray(
                (years != self.validation_year) & (years != self.test_year),
                dtype=bool,
            )

        if split == "val":
            return np.asarray(years == self.validation_year, dtype=bool)

        if split == "test":
            return np.asarray(years == self.test_year, dtype=bool)

        raise ValueError(f"Unknown split: {split!r}")

    def select(self, times: pd.DatetimeIndex, split: Split) -> pd.DatetimeIndex:
        return times[self.mask(times, split)]

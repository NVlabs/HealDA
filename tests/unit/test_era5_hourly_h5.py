# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import h5py
import numpy as np
import pandas as pd

from healda.datasets.era5_hourly_h5 import SIDECAR, Era5HourlyH5


def write_pack(root, channels, values):
    (root / "metadata").mkdir(parents=True)
    metadata = {
        "h5_path": "fields",
        "dhours": 6,
        "coords": {"lat": [10.0, 0.0], "lon": [0.0, 1.0, 2.0], "channel": channels},
    }
    (root / SIDECAR).write_text(json.dumps(metadata))
    with h5py.File(root / "2025.h5", "w") as f:
        f["fields"] = values


def test_a_channel_subset_reads_the_same_values_as_the_whole_pack(tmp_path):
    channels = ["u50", "v50", "t50", "z50"]
    rng = np.random.default_rng(0)
    values = rng.standard_normal((8, len(channels), 2, 3)).astype(np.float32)
    write_pack(tmp_path, channels, values)
    times = pd.DatetimeIndex(["2025-01-01T18", "2025-01-01T06", "2025-01-01T18"])
    steps = [3, 1, 3]

    whole = Era5HourlyH5(str(tmp_path), channels).read(times)
    np.testing.assert_array_equal(whole, values[steps])

    subset = Era5HourlyH5(str(tmp_path), ["z50", "u50"]).read(times)
    np.testing.assert_array_equal(subset, values[steps][:, [3, 0]])

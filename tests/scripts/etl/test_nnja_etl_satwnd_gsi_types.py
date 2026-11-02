# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ETL's GSI type table and the loader-side rules must type every wind the same."""

import numpy as np

from healda.observations.preprocessing import satwnd_gsi_types
from scripts.etl.nnja.satwnd.archive_etl import GSI_TYPE_TABLE


def test_type_table_matches_public_rules():
    keys = list(GSI_TYPE_TABLE)
    subset, said, swcm = (np.array(column) for column in zip(*keys))
    got = satwnd_gsi_types.resolve(
        subset.astype(object), said.astype(np.float64), swcm.astype(np.float64)
    )
    report_type, subtype, _, _, tier = zip(*GSI_TYPE_TABLE.values())
    np.testing.assert_array_equal(got.report_type, report_type)
    np.testing.assert_array_equal(got.internal_subtype, subtype)
    np.testing.assert_array_equal(got.tier, np.array(tier, dtype=object))

    rows, table = satwnd_gsi_types._lookup()
    public_keys = {
        (name, key >> 3, key & 7)
        for name, row in rows.items()
        for key in np.flatnonzero(table[row, 0] >= 0)
    }
    assert public_keys == set(keys)

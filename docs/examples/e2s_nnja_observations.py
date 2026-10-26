# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feed earth2studio NNJA observations through the HealDA NNJA loaders.

earth2studio returns raw decoded rows; ``e2s_nnja`` restates them in the NNJA archive
layout; the loaders then apply the same QC, thinning and normalization as training.

usage: python docs/examples/e2s_nnja_observations.py 2022-01-01T00
"""

import asyncio
import sys
from datetime import timedelta

import pandas as pd
from earth2studio.data import NNJAObsConv, NNJAObsSatwnd

from healda.observations.adapters import e2s_nnja
from healda.observations.loaders.combined import NNJAConventionalLoader


def main(target: pd.Timestamp) -> None:
    # The loader's (-3, 3) h context is one NCEP cycle file.
    tolerance = (timedelta(hours=-3), timedelta(hours=3))
    # Refractivity rows carry what the adapter derives height and pressure from.
    gps = NNJAObsConv(time_tolerance=tolerance)(
        target.to_pydatetime(), ["gps", "gps_refractivity"]
    )
    winds = NNJAObsSatwnd(time_tolerance=tolerance)(target.to_pydatetime(), ["u", "v"])

    tables = e2s_nnja.analysis_tables_long(conv=gps, satwnd=winds, ir_channels="ir32")
    loader = NNJAConventionalLoader(
        obs_context_hours=(-3, 3),
        gpsro_saids="full",
        include_satwnd=True,
        gpsro_table_source=tables["gpsro_tables"],
        satwnd_table_source=tables["satwnd_tables"],
    )
    obs = asyncio.run(loader.sel_time(pd.DatetimeIndex([target])))
    for name, tables in obs.items():
        print(name, [table.num_rows for table in tables])


if __name__ == "__main__":
    main(pd.Timestamp(sys.argv[1]))

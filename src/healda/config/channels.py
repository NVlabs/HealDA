# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical channel names and their spelling in the zarr stores, shared by the reader and the ETL."""

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

PRESSURE_VARIABLES = ["U", "V", "T", "Z", "Q"]
EXTRA_PRESSURE_VARIABLES = ["W"]

SURFACE_VARIABLES = [
    "tcwv",
    "tas",
    "uas",
    "vas",
    "100u",
    "100v",
    "pres_msl",
    "sst",
    "sic",
    "sp",
]
EXTRA_SURFACE_VARIABLES = ["sh2"]
ERA5_EXTRA_SURFACE_VARIABLES = ["d2m"]


def channel_name(variable: str, level: int | None = None) -> str:
    if level is None:
        return variable
    return f"{variable}{level}"


# Pressure channels are lowercased; surface channels are aliased.
STANDARD_TO_QUERY = {
    **{
        channel_name(variable, level): f"{variable.lower()}{level}"
        for variable in PRESSURE_VARIABLES
        for level in LEVELS
    },
    "tcwv": "tcwv",
    "tas": "t2m",
    "uas": "u10m",
    "vas": "v10m",
    "100u": "u100m",
    "100v": "v100m",
    "pres_msl": "msl",
    "sst": "sst",
    "sic": "sic",
    "sp": "sp",
}

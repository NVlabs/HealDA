# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from healda.datasets.base import VariableConfig

# Sentinel level for 2D / surface variables (no vertical level).
NO_LEVEL = -1

VARIABLE_CONFIGS = {}
VARIABLE_CONFIGS["era5_74ch"] = VariableConfig(
    name="era5_74ch",
    levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    variables_3d=["U", "V", "T", "Z", "Q"],
    variables_2d=[
        "tcwv",
        "tas",
        "uas",
        "vas",
        "100u",
        "100v",
        "pres_msl",
        "sst",
        "sic",
    ],
    variables_static=["orog", "land_surface"],
)
# --- packed 99-channel era5 target sets ---
# Order is the model's channel order only: the reader gathers by channel name, so it
# need not match the store's axis, and a name the store lacks errors out.
ERA5_SURFACE_SYNOPTIC = [
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
    "d2m",
    "skt",
]
ERA5_SURFACE_CLOUD = ["tcc", "lcc", "mcc", "hcc"]
ERA5_SURFACE_LAND = ["stl1", "stl2", "swvl1", "swvl2", "sd"]
_ERA5_99CH_SURFACE = [
    *ERA5_SURFACE_SYNOPTIC,
    *ERA5_SURFACE_CLOUD,
    *ERA5_SURFACE_LAND,
]
_ERA5_99CH_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
# 86 channels: no vertical velocity
VARIABLE_CONFIGS["era5_86ch"] = VariableConfig(
    name="era5_86ch",
    levels=_ERA5_99CH_LEVELS,
    variables_3d=["U", "V", "T", "Z", "Q"],
    variables_2d=_ERA5_99CH_SURFACE,
    variables_static=["orog", "land_surface"],
)
# add vertical velocity W
VARIABLE_CONFIGS["era5_99ch"] = VariableConfig(
    name="era5_99ch",
    levels=_ERA5_99CH_LEVELS,
    variables_3d=["U", "V", "T", "Z", "Q", "W"],
    variables_2d=_ERA5_99CH_SURFACE,
    variables_static=["orog", "land_surface"],
)
VARIABLE_CONFIGS["era5_104ch"] = VariableConfig(
    name="era5_104ch",
    levels=_ERA5_99CH_LEVELS,
    variables_3d=["U", "V", "T", "Z", "Q", "W"],
    variables_2d=_ERA5_99CH_SURFACE + ["tcw"],
    variables_static=["orog", "land_surface"],
    # exclude Q10 and W10
    extra_levels_by_var={"U": [10], "V": [10], "T": [10], "Z": [10]},
)

# Canonical channel name -> store channel name, per data source.
CHANNEL_ALIASES: dict[str, dict[str, str]] = {}


def resolve_store_channels(channels: list[str], source: str) -> list[str]:
    """Map canonical channel names to a source's stored channel names."""
    aliases = CHANNEL_ALIASES.get(source, {})
    return [aliases.get(channel, channel) for channel in channels]


def _encode_channel(channel: tuple[str, int]) -> str:
    name, level = channel
    return name if level == NO_LEVEL else f"{name}{level}"


def variable_levels(config: VariableConfig, var: str) -> list[int]:
    """Levels carried for ``var`` -- ``levels`` plus any ``extra_levels_by_var``."""
    return config.levels + config.extra_levels_by_var.get(var, [])


def channel_index(config: VariableConfig) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            (v, level)
            for v in config.variables_3d
            for level in variable_levels(config, v)
        ]
        + [(v, NO_LEVEL) for v in config.variables_2d],
        names=["variable", "level"],
    )


def encode_channels(config: VariableConfig) -> list[str]:
    """Channel names for a variable config: 3D vars x levels, then 2D vars."""
    return [_encode_channel(t) for t in channel_index(config)]


def validate_variable_config(config: VariableConfig) -> None:
    channels = encode_channels(config)
    duplicates = sorted({name for name in channels if channels.count(name) > 1})
    if duplicates:
        raise ValueError(f"{config.name}: duplicate channels: {duplicates}")
    unknown_extra_variables = set(config.extra_levels_by_var) - set(config.variables_3d)
    if unknown_extra_variables:
        raise ValueError(
            f"{config.name}: extra levels for unknown variables: "
            f"{sorted(unknown_extra_variables)}"
        )
    if set(config.variables_2d) & set(config.variables_3d):
        raise ValueError(f"{config.name}: variables cannot be both 2D and 3D")
    if not channels:
        raise ValueError(f"{config.name}: at least one prognostic channel is required")
    if not config.variables_static:
        raise ValueError(f"{config.name}: at least one static field is required")
    if len(config.variables_static) != len(set(config.variables_static)):
        raise ValueError(f"{config.name}: duplicate static fields")
    prognostic_variables = set(config.variables_2d) | set(config.variables_3d)
    if prognostic_variables & set(config.variables_static):
        raise ValueError(
            f"{config.name}: variables cannot be both prognostic and static"
        )


def validate_variable_configs() -> None:
    for key, config in VARIABLE_CONFIGS.items():
        if key != config.name:
            raise ValueError(f"{key}: registry key does not match {config.name!r}")
        validate_variable_config(config)


validate_variable_configs()

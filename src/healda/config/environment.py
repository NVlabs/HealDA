# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


non_config = dir()

CACHE_ROOT = os.getenv("HEALDA_CACHE_ROOT", os.path.expanduser("~/.cache"))
CACHE_DIR = os.path.join(CACHE_ROOT, "healda")

# Every store is a paired `<NAME>` path and `<NAME>_PROFILE`. A profile is an
# rclone remote name, resolved against ~/.config/rclone/rclone.conf; empty means
# the path needs no credentials. Each store carries its own, so two stores can
# live on different backends.

#######
# UFS #
#######
UFS_HPX6_ZARR = os.getenv("UFS_HPX6_ZARR", "")
UFS_LAND_DATA_ZARR = os.getenv("UFS_LAND_DATA_ZARR", "")
UFS_LAND_DATA_PROFILE = os.getenv("UFS_LAND_DATA_PROFILE", "")
UFS_ZARR_PROFILE = os.getenv("UFS_ZARR_PROFILE", "")
UFS_OBS_PATH = os.getenv("UFS_OBS_PATH", "")
UFS_OBS_PROFILE = os.getenv("UFS_OBS_PROFILE", "")

#########################
# NNJA Parquet archives #
#########################
# The ONE nnja constant. Every archive path is derived from it at the point of
# use via nnja_archive(), so moving the collection is a single override and no
# module keeps its own copy under another name.
NNJA_ROOT = os.getenv("NNJA_ROOT", "")


def nnja_archive(*parts: str) -> str:
    """Path under NNJA_ROOT, e.g. nnja_archive("parquet", "satwnd")."""
    return os.path.join(NNJA_ROOT, *parts)


# project file
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT", os.path.join(PROJECT_ROOT, "training-runs")
)

###########################################
# Hourly 0.25 degree ERA5 HDF5 year files #
###########################################
ERA5_HOURLY_ROOT = os.getenv("ERA5_HOURLY_ROOT", "")
# Defaults to ERA5_HOURLY_ROOT; set separately only to point elsewhere.
ERA5_HOURLY_73CH = os.getenv("ERA5_HOURLY_73CH") or ERA5_HOURLY_ROOT
# One directory per variable: ERA5_HOURLY_EXT_ROOT/<variable>/<year>.h5.
ERA5_HOURLY_EXT_ROOT = os.getenv("ERA5_HOURLY_EXT_ROOT", "")


def path_list(value: str) -> list[str]:
    # One dataset's coverage is often split across directories, so every setting
    # below takes a ``:``-separated list.
    return [part.strip() for part in value.split(":") if part.strip()]


####################################
# ERA5 state, HPX64 104ch (packed) #
####################################
ERA5_HPX64_104CH_ZARR = os.getenv(
    "ERA5_HPX64_104CH_ZARR", os.path.join(DATA_ROOT, "era5_hpx64_104ch_packed.zarr")
)
ERA5_HPX64_104CH_ZARR_PROFILE = os.getenv("ERA5_HPX64_104CH_ZARR_PROFILE", "")

_config_vars = dict(vars())


def print_config():
    # TODO remove
    print("Environment settings:")
    print("-" * 80)
    for v in _config_vars:
        if v == "non_config":
            continue

        if v in non_config:
            continue

        value = _config_vars[v]
        print(f"{v}={value}")
    print("-" * 80)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import earth2grid
import healda.config.environment as config
from healda.utils.cache import AtomicFileCache
from healda.utils.regridding import ConservativeRegridder
from healda.utils.storage import get_storage_options
from healda.datasets import catalog
import xarray as xr
import zarr
import torch
import numpy as np


GRAVITY = 9.80665

# The ERA5 invariants are the 1979-01-01 NCAR RDA fields on the 0.25 degree grid.
# Geopotential is converted to metres so orography matches the UFS field's units.
_NCAR_INVARIANTS = "s3://nsf-ncar-era5/e5.oper.invariant/197901"
_STEM = "ll025sc.1979010100_1979010100.nc"
ERA5_INVARIANT_FILES = {
    "orog": f"e5.oper.invariant.128_129_z.{_STEM}",
    "lfrac": f"e5.oper.invariant.128_172_lsm.{_STEM}",
}
ERA5_INVARIANT_VARIABLES = {"orog": "Z", "lfrac": "LSM"}


@functools.cache
def _load_era5_invariant(name: str, hpx_level: int) -> torch.Tensor:
    source = f"{_NCAR_INVARIANTS}/{ERA5_INVARIANT_FILES[name]}"
    cache = AtomicFileCache(config.CACHE_DIR)
    local_path = cache.ensure(source, {"anon": True})
    with xr.open_dataset(local_path) as ds:
        values = ds[ERA5_INVARIANT_VARIABLES[name]].squeeze().values.astype(np.float64)
    if name == "orog":
        values = values / GRAVITY

    src_grid = earth2grid.latlon.equiangular_lat_lon_grid(
        nlat=values.shape[0], nlon=values.shape[1]
    )
    regridder = ConservativeRegridder(
        latlon_grid=src_grid, out_level=hpx_level, dtype=torch.float64
    )
    return regridder(torch.from_numpy(np.ascontiguousarray(values)))


# Loads the land data from the UFS dataset.
@functools.cache
def load_ufs_lfrac(hpx_level) -> torch.Tensor:
    src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat=768, nlon=1536)
    hpx_grid = earth2grid.healpix.Grid(
        level=hpx_level, pixel_order=earth2grid.healpix.NEST
    )
    regridder = earth2grid.get_regridder(src_grid, hpx_grid)

    # get static iputs
    land_data = zarr.open_group(
        config.UFS_LAND_DATA_ZARR,
        storage_options=get_storage_options(config.UFS_LAND_DATA_PROFILE),
    )
    land_fraction = land_data["lfrac"][:]
    land_fraction = regridder(torch.from_numpy(land_fraction).to(torch.float64))
    return land_fraction


@functools.cache
def load_ufs_orography() -> np.ndarray:
    entry = catalog.ufs()
    group = entry.to_zarr()
    return group["orog"][:]


def _check_source(source: str):
    if source not in ("ufs", "era5"):
        raise ValueError(f"static source must be 'ufs' or 'era5', got {source!r}")


def load_lfrac(hpx_level, source: str = "ufs"):
    _check_source(source)
    if source == "ufs":
        return load_ufs_lfrac(hpx_level)
    return _load_era5_invariant("lfrac", hpx_level)


def load_orography(hpx_level: int = 6, source: str = "ufs"):
    _check_source(source)
    if source == "ufs":
        # The UFS store holds orography on the HPX6 grid, so there is nothing to regrid.
        return load_ufs_orography()
    return _load_era5_invariant("orog", hpx_level)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ERA5/IFS static maps used as a geographic prior: orog, LSM, and IFS envelope orography.

sdor, slor, isor, sdfor, anor are the standard IFS subgrid-orography fields
(gravity-wave / TOFD), not a homemade roughness proxy.
"""

from __future__ import annotations

import functools

import earth2grid
import numpy as np
import torch
import xarray as xr
from earth2grid import healpix

import healda.config.environment as config
from healda.datasets.static_data import (
    GRAVITY,
)
from healda.utils.cache import AtomicFileCache

NLAT, NLON = 721, 1440

# orog + lsm from the ERA5 invariants pack; the rest are NCAR e5.oper.invariant.
ERA5_STATIC_NAMES = (
    "orog",
    "lsm",
    "sdor",
    "slor",
    "isor",
    "sdfor",
    "anor_sin",
    "anor_cos",
)
N_ERA5_STATICS = len(ERA5_STATIC_NAMES)

# Public NSF NCAR ERA5 mirror. Every static comes from here, anonymously, so a fresh
# install needs no credentials and no site-specific bucket.
ERA5_INVARIANTS_URI = "s3://nsf-ncar-era5/e5.oper.invariant/197901"

# Order must match ERA5_STATIC_NAMES.
_ERA5_NC = (
    ("orog", "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", "Z"),
    ("lsm", "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc", "LSM"),
    ("sdor", "e5.oper.invariant.128_160_sdor.ll025sc.1979010100_1979010100.nc", "SDOR"),
    ("slor", "e5.oper.invariant.128_163_slor.ll025sc.1979010100_1979010100.nc", "SLOR"),
    ("isor", "e5.oper.invariant.128_161_isor.ll025sc.1979010100_1979010100.nc", "ISOR"),
    (
        "sdfor",
        "e5.oper.invariant.128_074_sdfor.ll025sc.1979010100_1979010100.nc",
        "SDFOR",
    ),
    ("anor", "e5.oper.invariant.128_162_anor.ll025sc.1979010100_1979010100.nc", "ANOR"),
)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std <= 0.0:
        raise ValueError("constant static field has no scale to normalise by")
    return (x - x.mean()) / np.float32(std)


def _load_nc(path: str, var: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        return np.ascontiguousarray(ds[var].squeeze().values.astype(np.float32))


@functools.cache
def load_era5_statics_latlon() -> torch.Tensor:
    """Z-scored (orog, lsm, sdor, slor, isor, sdfor, anor_sin, anor_cos) at 0.25°.

    Shape ``(8, 721, 1440)``. anor is encoded as ``(sin, cos)`` of the angle.
    """
    cache = AtomicFileCache(config.CACHE_DIR)
    fields: list[np.ndarray] = []
    for _short, fname, var in _ERA5_NC:
        # Public bucket, so anonymous: a configured AWS profile is resolved even on the anon
        # path and fails with ProfileNotFound.
        path = cache.ensure(f"{ERA5_INVARIANTS_URI}/{fname}", {"anon": True})
        raw = _load_nc(path, var)
        if _short == "orog":
            # Z is geopotential; the static is height.
            raw = raw / np.float32(GRAVITY)
        if _short == "anor":
            fields.append(np.sin(raw).astype(np.float32))
            fields.append(np.cos(raw).astype(np.float32))
        else:
            fields.append(_zscore(raw))
    stacked = np.stack(fields, axis=0)
    if stacked.shape != (N_ERA5_STATICS, NLAT, NLON):
        raise ValueError(f"IFS statics shape {stacked.shape}")
    return torch.from_numpy(stacked)


def load_era5_statics_hpx(level: int, pixel_order) -> torch.Tensor:
    """Bilinear regrid of ``load_era5_statics_latlon`` to HEALPix ``level``.

    ``pixel_order`` is required: defaulting it is how a static field ends up scrambled
    relative to the tokens it conditions.
    """
    latlon = load_era5_statics_latlon()
    src = earth2grid.latlon.equiangular_lat_lon_grid(nlat=NLAT, nlon=NLON)
    dst = healpix.Grid(level=level, pixel_order=pixel_order)
    regrid = earth2grid.get_regridder(src, dst).to(dtype=torch.float32)
    return regrid(latlon.float())

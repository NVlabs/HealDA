# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import earth2grid
import torch
from earth2grid import healpix


def add_south_pole_mean(x: torch.Tensor) -> torch.Tensor:
    """Add south pole using zonal mean of southernmost latitude.

    Aurora outputs 720 lat (patch_size=4 constraint) → adds 721st via zonal mean.
    Other models output 721 lat → pass-through unchanged.
    """
    if x.shape[-2] == 721:
        return x

    pole_values = x[..., -1:, :].mean(dim=-1, keepdim=True)
    pole_values = pole_values.expand(*x.shape[:-2], 1, x.shape[-1])

    return torch.cat([x, pole_values], dim=-2)


def get_latlon_bilinear_regridder(latlon_grid, regrid_level, dtype):
    hpx_grid = healpix.Grid(level=regrid_level, pixel_order=healpix.PixelOrder.NEST)
    return earth2grid.get_regridder(latlon_grid, hpx_grid).to(dtype)


class ConservativeRegridder(torch.nn.Module):
    """Bilinear to a finer HEALPix level, then block average down to ``out_level``.

    earth2grid offers only point interpolation, and a target cell here spans about 16
    source cells, so sampling would discard most of them. HEALPix cells are equal area,
    which makes the block mean over the 4**(regrid_level - out_level) children an area
    average. This is how every ERA5 channel reaches HPX64.

    ``ignore_nan=True`` uses nanmean over HPX8 children so land-NaN SST/SIC do not
    poison coastal cells. ``fill_value`` replaces remaining all-NaN (pure land) cells.
    """

    def __init__(
        self, latlon_grid=None, regrid_level=8, out_level=6, dtype=torch.float32
    ):
        super().__init__()

        if latlon_grid is None:
            latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                nlat=721, nlon=1440
            )
        self.regridder = get_latlon_bilinear_regridder(latlon_grid, regrid_level, dtype)
        self.coarsen_factor = 4 ** (regrid_level - out_level)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ignore_nan: bool = False,
        fill_value: float | None = None,
    ) -> torch.Tensor:
        high = self.regridder(x)
        grouped = high.reshape(high.shape[:-1] + (-1, self.coarsen_factor))
        out = torch.nanmean(grouped, dim=-1) if ignore_nan else grouped.mean(-1)
        if fill_value is not None:
            out = torch.where(
                torch.isfinite(out), out, out.new_full((), float(fill_value))
            )
        return out

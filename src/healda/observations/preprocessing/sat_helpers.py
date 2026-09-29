# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the satellite observation writers and readers."""

from __future__ import annotations

import numpy as np


def hpx_nest_pixels(
    lat: np.ndarray, lon: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray]:
    """NESTED HEALPix pixel at ``level`` per position, and where the position is valid.

    Archives store pixels at a fine level; thinning cells are coarser parents obtained
    by shifting (``pixel >> 2 * (level - thin_level)``).

    Longitude may be given on either [0, 360) or [-180, 180). Invalid positions get
    pixel 0 and ``valid`` False, since the grid maps NaN to an arbitrary pixel.
    """
    import torch
    from earth2grid import healpix

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
    pixels = np.zeros(lat.size, dtype=np.uint32)
    if valid.any():
        grid = healpix.Grid(level=level, pixel_order=healpix.NEST)
        lon_signed = np.mod(lon[valid] + 180.0, 360.0) - 180.0
        with torch.inference_mode():
            found = grid.ang2pix(
                torch.from_numpy(np.ascontiguousarray(lon_signed)),
                torch.from_numpy(np.ascontiguousarray(lat[valid])),
            )
        pixels[valid] = found.cpu().numpy().astype(np.uint32)
    return pixels, valid

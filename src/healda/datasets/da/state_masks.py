# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-channel validity masks for channels the loader fills where the quantity doesn't exist
(sst/sic filled 290 K/0 over land, swvl1/swvl2 filled 0 at sea, sd also carries snow on sea
ice). Scoring a channel over its fill dilutes its normalisation. Land/ocean uses ERA5's own
`sst == 290.0`, not a thresholded land-fraction mask.

Masks are derived per grid and never regridded from one to the other: bilinear on a binary
mask blurs the coastline represented by the source grid.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np
import torch

MASKS_DIR = pathlib.Path(__file__).parent / "state_masks"
LATLON_025 = "latlon_025_masks.npz"
HPX64 = "hpx64_masks.npz"
NLAT, NLON = 721, 1440
NPIX_HPX64 = 49152  # 12 * 64^2

#: Cell count per mask file, so a file and a grid cannot be mismatched.
_GRID_SHAPE = {LATLON_025: (NLAT, NLON), HPX64: (NPIX_HPX64,)}

# Channel -> the mask it's valid on. `stl1` is absent: valid over land and ocean alike,
# not yet characterised.
CHANNEL_DOMAIN = {
    "sst": "ocean",
    "sic": "ocean",
    "swvl1": "land",
    "swvl2": "land",
    "sd": "land_or_ice",
}

#: What the loader writes OUTSIDE a channel's domain (mirrors CHANNEL_DOMAIN). The ETL, the
#: mask builder, and restore_fill must all agree on this value -- import it, don't restate it.
CHANNEL_DOMAIN_FILL: dict[str, float] = {"sst": 290.0, "sic": 0.0}


def domain_fill(channel: str) -> float:
    """What the loader writes where ``channel`` does not exist; 0.0 for unmasked channels."""
    return CHANNEL_DOMAIN_FILL.get(channel, 0.0)


@functools.cache
def _load(file_name: str = LATLON_025) -> dict[str, np.ndarray]:
    shape = _GRID_SHAPE.get(file_name)
    if shape is None:
        raise KeyError(
            f"{file_name!r} has no registered grid shape; add it to _GRID_SHAPE"
        )
    size = int(np.prod(shape))
    with np.load(MASKS_DIR / file_name) as data:
        stored = {
            # packbits pads to a byte boundary; trim before reshaping.
            key: np.unpackbits(data[key]).astype(bool)[:size].reshape(shape)
            for key in ("ocean", "land_or_ice")
        }
        stored["provenance"] = str(data["provenance"])
        stored["pixel_order"] = (
            str(data["pixel_order"]) if "pixel_order" in data else None
        )
    stored["land"] = ~stored["ocean"]
    return stored


def mask_pixel_order(file_name: str = LATLON_025) -> str | None:
    """Pixel order the mask is indexed in, or None for a non-HEALPix grid."""
    return _load(file_name)["pixel_order"]


def state_mask(
    name: str, file_name: str = LATLON_025, pixel_order: str | None = None
) -> np.ndarray:
    """Boolean mask ``name`` (``ocean``, ``land``, ``land_or_ice``).

    HEALPix masks are stored in NEST. Pass ``pixel_order`` when applying the mask to
    data the pipeline already reordered -- NEST and HEALPIX_PAD_XY place different
    points of the sphere at the same flat index, silently.
    """
    masks = _load(file_name)
    if name not in masks:
        raise KeyError(
            f"{name!r} not in "
            f"{sorted(k for k in masks if k not in ('provenance', 'pixel_order'))}"
        )
    mask = masks[name]
    stored = (masks["pixel_order"] or "").lower()
    if pixel_order is None or not stored or stored == pixel_order.lower():
        return mask
    if stored != "nest":
        raise ValueError(
            f"{file_name} is stored in {stored!r}; only nest can be reordered from"
        )
    from healda.datasets.da.transform import reorder_from_nest

    return np.asarray(reorder_from_nest(mask.astype(np.float32), pixel_order)) > 0.5


def channel_mask(
    channel: str, file_name: str = LATLON_025, pixel_order: str | None = None
) -> np.ndarray | None:
    """The validity mask for ``channel``, or None if it is valid over the whole grid.

    ``pixel_order``: see ``state_mask`` -- required whenever this is applied to data the
    pipeline has already reordered out of the store's NEST.
    """
    domain = CHANNEL_DOMAIN.get(channel)
    return None if domain is None else state_mask(domain, file_name, pixel_order)


def loss_weight_mask(
    channels: list[str],
    device=None,
    dtype=torch.float32,
    file_name: str = LATLON_025,
    pixel_order: str | None = None,
) -> torch.Tensor:
    """``(C, *grid_shape)`` multiplier: 1 where a channel is valid, 0 on its fill.

    0/1 only -- see ``loss_rescale_mask``/``hpx_loss_rescale_mask`` for the renormalising
    version the loss actually wants. Do NOT pass this to ``spatial_loss_weight`` (it
    collapses longitude first, so a mask varying in both lat and lon can't go through
    it); it's applied inside ``run_model_step``'s ``channel_mask`` instead.

    ``pixel_order``: pass the run's order (e.g. ``"hpxpadxy"``) if this multiplies data
    the pipeline reordered out of the store's NEST -- see ``state_mask``.

    Cached: this is a full (C, *grid_shape) allocation, e.g. 104 x 721 x 1440 floats,
    that's mostly-ones since almost no channel has a CHANNEL_DOMAIN entry, so it must
    not be rebuilt (or re-copied to device) on every training step.
    """
    return _cached_loss_weight_mask(
        tuple(channels), device or "cpu", dtype, file_name, pixel_order
    )


@functools.cache
def _cached_loss_weight_mask(
    channels: tuple[str, ...],
    device,
    dtype: torch.dtype,
    file_name: str,
    pixel_order: str | None,
) -> torch.Tensor:
    shape = _GRID_SHAPE[file_name]
    out = torch.ones(len(channels), *shape, dtype=dtype)
    for i, name in enumerate(channels):
        mask = channel_mask(name, file_name, pixel_order)
        if mask is not None:
            out[i] = torch.from_numpy(mask).to(dtype)
    return out.to(device)


def _rescale_by_valid_fraction(
    base: torch.Tensor, valid_fraction: torch.Tensor
) -> torch.Tensor:
    """base * (0 where valid_fraction is 0, else 1 / valid_fraction)."""
    rescale = torch.where(
        valid_fraction > 0,
        1.0 / valid_fraction.clamp_min(1e-12),
        torch.zeros_like(valid_fraction),
    )
    return base * rescale


def loss_rescale_mask(
    channels: list[str],
    area_weight: torch.Tensor,
    device=None,
    dtype=torch.float32,
    file_name: str = LATLON_025,
) -> torch.Tensor:
    """``(1, C, 1, NLAT, NLON)`` multiplier that masks AND renormalises in one pass, so
    ``reduce_weighted(error * this, spatial)`` is the area-weighted mean over just the
    valid domain -- plain masking + averaging would dilute by the whole grid instead.

    ``area_weight`` must be ``spatial_loss_weight``'s own per-latitude cos(lat) weight,
    so the valid fraction uses identical weighting to the loss.

    HEALPix is equal-area and needs a different computation -- see
    ``hpx_loss_rescale_mask`` -- so this raises rather than silently mis-broadcasting.
    """
    if _GRID_SHAPE[file_name] != (NLAT, NLON):
        raise ValueError(
            f"loss_rescale_mask is only defined for the (NLAT, NLON) grid, got "
            f"{file_name!r} with shape {_GRID_SHAPE[file_name]}"
        )
    # area_weight may already be on the loss's device (e.g. spatial_loss_weight's
    # cos(lat) tensor); build base there too, or `base * area_weight` below crashes
    # on a device mismatch regardless of what `device` the caller passed.
    target_device = device if device is not None else area_weight.device
    base = loss_weight_mask(
        channels, device=target_device, dtype=dtype, file_name=file_name
    )
    area_weight = area_weight.to(device=target_device, dtype=dtype).reshape(1, NLAT, 1)
    valid_fraction = (base * area_weight).mean(dim=(1, 2), keepdim=True)
    return _rescale_by_valid_fraction(base, valid_fraction).unsqueeze(0).unsqueeze(2)


def hpx_loss_rescale_mask(
    channels: list[str],
    device=None,
    dtype=torch.float32,
    file_name: str = HPX64,
    pixel_order: str | None = None,
) -> torch.Tensor:
    """``(1, C, 1, NPIX)`` equal-area counterpart to ``loss_rescale_mask``: HEALPix
    pixels are already equal-area, so the valid fraction is a plain mean over the pixel
    axis, no ``area_weight`` needed.

    ``pixel_order``: see ``loss_weight_mask``.
    """
    if _GRID_SHAPE[file_name] != (NPIX_HPX64,):
        raise ValueError(
            f"hpx_loss_rescale_mask is only defined for a flat HEALPix grid, got "
            f"{file_name!r} with shape {_GRID_SHAPE[file_name]}"
        )
    base = loss_weight_mask(
        channels, dtype=dtype, file_name=file_name, pixel_order=pixel_order
    )
    valid_fraction = base.mean(dim=-1, keepdim=True)
    out = _rescale_by_valid_fraction(base, valid_fraction).unsqueeze(0).unsqueeze(2)
    return out.to(device) if device is not None else out


def provenance(file_name: str = LATLON_025) -> str:
    return _load(file_name)["provenance"]

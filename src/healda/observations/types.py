# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import NotRequired, TypedDict, Optional
import dataclasses

import torch


PIXEL_ORDER_NAMES = ("hpxpadxy", "nest")


def healpix_pixel_order(name: str):
    """earth2grid pixel order for a name carried by ``AttentionPacking.pixel_order``.

    The import is deferred because earth2grid costs ~3 s and this module is on 11 import
    chains that do not otherwise need it.
    """
    import earth2grid.healpix

    orders = {
        "hpxpadxy": earth2grid.healpix.HEALPIX_PAD_XY,
        "nest": earth2grid.healpix.NEST,
    }
    if name not in orders:
        raise ValueError(
            f"Unknown pixel order {name!r}; expected one of {PIXEL_ORDER_NAMES}"
        )
    return orders[name]


@dataclasses.dataclass
class PixelGroupMap:
    """CSR map for grouping non-empty pixels into shared attention programs."""

    program_ptr: torch.Tensor
    program_pixels: torch.Tensor

    def to(self, device=None, dtype=None, non_blocking=True):
        # dtype is intentionally ignored: group-map tensors are integer indices.
        del dtype
        return PixelGroupMap(
            program_ptr=self.program_ptr.to(device=device, non_blocking=non_blocking),
            program_pixels=self.program_pixels.to(
                device=device, non_blocking=non_blocking
            ),
        )


@dataclasses.dataclass
class AttentionPacking:
    """Precomputed packing metadata for backbone observation cross-attention.

    Observations are sorted by flat pixel index so each pixel's key/value tokens
    are contiguous; ``cu_seqlens_k`` holds the prefix sums into the packed token
    array. ``counts`` is the per-pixel observation count. The sort permutation is
    applied to the observation tensors in the data transform and is not retained
    here: this struct only describes the resulting packed layout.
    """

    counts: torch.Tensor
    cu_seqlens_k: torch.Tensor
    npix: int
    hpx_level: int
    is_packed: bool = True
    # Optional CSR map pairing small pixels into shared kernel programs for the
    # ragged obs cross-attention. ``None`` -> one program per pixel.
    group_map: Optional[PixelGroupMap] = None
    # Pixel order the sort was done in. ``cu_seqlens_k`` addresses state tokens by position,
    # so this must match the order the blocks run in, or every observation cross-attends to
    # the wrong pixel.
    pixel_order: str = "hpxpadxy"

    def to(self, device=None, dtype=None, non_blocking=True):
        # dtype is intentionally ignored: packing tensors are integer indices.
        del dtype

        def _move(x):
            if x is None:
                return None
            return x.to(device=device, non_blocking=non_blocking)

        return AttentionPacking(
            counts=_move(self.counts),
            cu_seqlens_k=_move(self.cu_seqlens_k),
            npix=self.npix,
            hpx_level=self.hpx_level,
            is_packed=self.is_packed,
            group_map=(
                None
                if self.group_map is None
                else self.group_map.to(device=device, non_blocking=non_blocking)
            ),
            pixel_order=self.pixel_order,
        )


@dataclasses.dataclass
class UnifiedObservation:
    """Unified observation structure for both satellite and conventional observations."""

    obs: torch.Tensor  # (n_obs,) observation values
    time: (
        torch.Tensor
    )  # (n_obs,) observation timestamps (ns since epoch). Not used by the model.
    float_metadata: torch.Tensor  # (n_obs, n_features) pre-computed float features

    # Integer metadata fields, each shaped (n_obs,).
    pix: torch.Tensor  # HEALPix pixel index in the model's expected pixel order
    local_channel: torch.Tensor
    local_platform: torch.Tensor
    obs_type: torch.Tensor
    global_channel: torch.Tensor

    hpx_level: int  # HEALPix level that pix is defined at
    global_platform: torch.Tensor

    lengths: torch.Tensor | None = (
        None  # 3D: (n_active_sensors, batch, time) per-window obs counts
    )

    # Backbone obs cross-attention: packing metadata + (optionally) pre-tokenized
    # observation tokens, both produced by the data transform.
    attention_packing: AttentionPacking | None = None
    obs_tokens: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        device: str = "cpu",
        hpx_level: int = 8,
        batch_dims: tuple[int, int] = (1, 1),
    ) -> "UnifiedObservation":
        B, T = batch_dims
        return cls(
            obs=torch.empty(0, device=device),
            time=torch.empty(0, dtype=torch.long, device=device),
            float_metadata=torch.empty((0, 28), device=device),
            pix=torch.empty(0, dtype=torch.long, device=device),
            local_channel=torch.empty(0, dtype=torch.long, device=device),
            local_platform=torch.empty(0, dtype=torch.long, device=device),
            obs_type=torch.empty(0, dtype=torch.long, device=device),
            global_channel=torch.empty(0, dtype=torch.long, device=device),
            global_platform=torch.empty(0, dtype=torch.long, device=device),
            hpx_level=hpx_level,
            lengths=torch.zeros(1, B, T, dtype=torch.long, device=device),
        )

    @property
    def batch_dims(self):
        """Return (batch, time) shape from 3D offsets (S, B, T)."""
        if self.lengths is not None:
            return self.lengths.shape[-2:]
        else:
            return ()

    def __repr__(self):
        nobs = self.obs.shape[0]
        return f"UnifiedObservation({nobs=}, batch_dims={self.batch_dims})"

    def to(self, device=None, dtype=None, non_blocking=True):
        """Move all tensors to device and/or convert dtype."""

        def _move_tensor(x):
            if x is None:
                return
            return x.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return UnifiedObservation(
            obs=_move_tensor(self.obs),
            time=_move_tensor(self.time),
            float_metadata=_move_tensor(self.float_metadata),
            pix=_move_tensor(self.pix),
            local_channel=_move_tensor(self.local_channel),
            local_platform=_move_tensor(self.local_platform),
            obs_type=_move_tensor(self.obs_type),
            global_channel=_move_tensor(self.global_channel),
            global_platform=_move_tensor(self.global_platform),
            hpx_level=self.hpx_level,
            lengths=_move_tensor(self.lengths),
            attention_packing=(
                None
                if self.attention_packing is None
                else self.attention_packing.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
            ),
            obs_tokens=_move_tensor(self.obs_tokens),
        )


class Batch(TypedDict):
    target: torch.Tensor  # (b, c, t, x) - main atmospheric variables
    condition: torch.Tensor  # (b, c_cond, t, x) - conditioning variables
    second_of_day: torch.Tensor  # (b, t) - seconds of day
    day_of_year: torch.Tensor  # (b, t) - day of year
    labels: torch.Tensor  # (b, num_classes) - one-hot encoded labels
    timestamp: torch.Tensor  # (b, t) - timestamps as seconds since epoch
    unified_obs: Optional[UnifiedObservation]  # Unified observation data (v2)
    # Optional normalized background state in the same packed channel order as target.
    # It can also be concatenated into condition so the model receives it as input.
    background: NotRequired[torch.Tensor]  # (b, c, t, x)


def empty_batch(
    *,
    batch_gpu: int,
    out_channels: int,
    condition_channels: int,
    time_length: int,
    x_size: int,
    device: torch.device | str,
    background_channels: int | None = None,
) -> Batch:
    if x_size <= 0:
        raise ValueError(f"x_size must be positive, got {x_size}")

    batch: Batch = {
        "target": torch.empty(
            [batch_gpu, out_channels, time_length, x_size], device=device
        ),
        "condition": torch.empty(
            [batch_gpu, condition_channels, time_length, x_size], device=device
        ),
        "second_of_day": torch.empty([batch_gpu, time_length], device=device),
        "day_of_year": torch.empty([batch_gpu, time_length], device=device),
        "labels": torch.empty([batch_gpu, 0], device=device),
        "timestamp": torch.empty(
            [batch_gpu, time_length], dtype=torch.long, device=device
        ),
        "unified_obs": UnifiedObservation.empty(
            device=device, batch_dims=(batch_gpu, time_length)
        ),
    }
    if background_channels is not None:
        batch["background"] = torch.empty(
            [batch_gpu, background_channels, time_length, x_size], device=device
        )
    return batch


@torch.compiler.disable
def split_by_sensor(
    obs: UnifiedObservation, target_sensor_ids: list[int]
) -> dict[int, UnifiedObservation]:
    """
    Slice a UnifiedObservation into per-sensor sub-objects using its precomputed offsets.

    Args:
        obs: UnifiedObservation
        target_sensor_ids: list of int sensor IDs to extract

    Returns:
        dict[int, UnifiedObservation]: mapping from sensor_id -> sliced UnifiedObservation.
                                       If a sensor_id has no data, returns an empty slice
                                       (same structure, 0 rows).
    """
    if obs.lengths is None:
        raise ValueError("lengths is required for split_by_sensor")

    if obs.attention_packing is not None and obs.attention_packing.is_packed:
        raise ValueError(
            "split_by_sensor does not support attention-prepacked UnifiedObservation."
        )

    lengths = obs.lengths  # [S, B, T]
    device = obs.obs.device
    B, T = obs.batch_dims

    # Pre-split all obs-indexed fields along dim 0 using per-sensor total counts
    sizes = lengths.sum(dim=(1, 2)).tolist()
    obs_fields = [
        obs.obs,
        obs.time,
        obs.float_metadata,
        obs.pix,
        obs.local_channel,
        obs.local_platform,
        obs.obs_type,
        obs.global_channel,
        obs.global_platform,
    ]
    splits = [torch.split(f, sizes) for f in obs_fields]

    if len(target_sensor_ids) < len(sizes):
        raise ValueError(
            "target_sensor_ids must include the configured sensor order for split_by_sensor"
        )

    out = {}
    for s_local, sensor_id in enumerate(target_sensor_ids):
        if s_local >= len(sizes):
            sensor_lengths = torch.zeros((1, B, T), dtype=lengths.dtype, device=device)
            out[sensor_id] = UnifiedObservation(
                obs=obs.obs[:0],
                time=obs.time[:0],
                float_metadata=obs.float_metadata[:0],
                pix=obs.pix[:0],
                local_channel=obs.local_channel[:0],
                local_platform=obs.local_platform[:0],
                obs_type=obs.obs_type[:0],
                global_channel=obs.global_channel[:0],
                global_platform=obs.global_platform[:0],
                hpx_level=obs.hpx_level,
                lengths=sensor_lengths,
            )
        else:
            out[sensor_id] = UnifiedObservation(
                obs=splits[0][s_local],
                time=splits[1][s_local],
                float_metadata=splits[2][s_local],
                pix=splits[3][s_local],
                local_channel=splits[4][s_local],
                local_platform=splits[5][s_local],
                obs_type=splits[6][s_local],
                global_channel=splits[7][s_local],
                global_platform=splits[8][s_local],
                hpx_level=obs.hpx_level,
                lengths=lengths[s_local : s_local + 1],
            )

    return out

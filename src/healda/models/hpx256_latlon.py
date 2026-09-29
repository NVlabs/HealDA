# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiT wrapper: IFS statics + fine calendar on HPX256 → 0.25° prediction.

Same gridded inputs as obs-only DA (statics, not the target state). Weather is
the loss target at 0.25°; observations enter the backbone as cross-attention.
"""

from __future__ import annotations

import einops
import earth2grid
import torch
import torch.nn as nn
from earth2grid import healpix

import healda.utils.profiling
from healda.observations.types import healpix_pixel_order
from healda.datasets.era5_statics import (
    N_ERA5_STATICS,
    NLAT,
    NLON,
    load_era5_statics_hpx,
    load_era5_statics_latlon,
)
from healda.models.dit import Output
from healda.models.graphcast_decode import GraphCastDecoder
from healda.models.embedding import (
    CalendarEmbedding,
    solar_zenith_cosine,
    sphere_position,
)

# xyz (3, buffered) plus the solar zenith (1, per step). Latitude reaches the model through
# xyz and nowhere else: CalendarEmbedding carries only longitude.
GEO_CHANNELS = 4

# CalendarEmbedding(lon, n) emits 4n channels: n harmonics × (sin,cos) of
# local solar time, then the same of year fraction. n=16 → 32 tod + 32 doy.
CALENDAR_FREQS = 16
CALENDAR_CHANNELS = CALENDAR_FREQS * 4


class Hpx256LatlonModel(nn.Module):
    """Concat statics + Fourier calendar at HPX, run the DiT, decode to 0.25 degrees.

    Batch ``hidden_states`` is ignored: the DA condition is statics, not coarsened ERA5.
    The learned positional embedding stays on the coarse tokens; the calendar is
    concatenated at the fine grid so the patch mix sees intra-token diurnal variation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        out_channels: int,
        pixel_order: str,
        hpx_level: int = 8,
        decode_k: int = 4,
        decode_base_level: int | None = None,
        fine_calendar: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.hpx_level = hpx_level
        self.fine_calendar = fine_calendar

        # Every geometric field built here is indexed by flat pixel, so it must be built in
        # the order the backbone's tokens are in.
        self.pixel_order = pixel_order
        backbone_order = getattr(backbone, "spatial_token_order", None)
        if backbone_order is not None and backbone_order != pixel_order:
            raise ValueError(
                f"pixel order disagreement: the backbone's tokens are in {backbone_order!r} "
                f"but this wrapper builds its statics, geo fields and output regridder in "
                f"{pixel_order!r}."
            )

        aux_channels = (
            N_ERA5_STATICS + (CALENDAR_CHANNELS if fine_calendar else 0) + GEO_CHANNELS
        )

        level_in = backbone.level_model
        # One level up when the backbone is coarser than its ingest level: the decoder's
        # unpack manufactures that level to reach useful mesh density.
        level_mesh = level_in if level_in >= backbone._level_in else level_in + 1
        self.graphcast_decode = GraphCastDecoder(
            in_channels=backbone.inner_dim,
            aux_channels=aux_channels,
            out_channels=out_channels,
            level_in=level_in,
            level_mesh=level_mesh,
            k=decode_k,
            pixel_order=pixel_order,
            nlat=NLAT,
            nlon=NLON,
            base_level=decode_base_level,
        )

        statics_ll = load_era5_statics_latlon()
        statics_hpx = load_era5_statics_hpx(
            hpx_level, pixel_order=healpix_pixel_order(pixel_order)
        )

        hpx = healpix.Grid(
            level=hpx_level, pixel_order=healpix_pixel_order(pixel_order)
        )
        latlon = earth2grid.latlon.equiangular_lat_lon_grid(nlat=NLAT, nlon=NLON)
        lat_hpx = torch.as_tensor(hpx.lat, dtype=torch.float32)
        lon_hpx = torch.as_tensor(hpx.lon, dtype=torch.float32)
        lat_grid = torch.as_tensor(latlon.lat, dtype=torch.float32)
        lon_grid = torch.as_tensor(latlon.lon, dtype=torch.float32)
        lat_flat = lat_grid.view(-1, 1).expand(NLAT, NLON).reshape(-1)
        lon_flat = lon_grid.view(1, -1).expand(NLAT, NLON).reshape(-1)
        self.register_buffer("geo_lat_hpx", lat_hpx, persistent=False)
        self.register_buffer("geo_lon_hpx", lon_hpx, persistent=False)
        self.register_buffer("geo_lat_ll", lat_flat, persistent=False)
        self.register_buffer("geo_lon_ll", lon_flat, persistent=False)
        self.register_buffer("statics_hpx", statics_hpx.float(), persistent=False)
        self.register_buffer("statics_ll", statics_ll.float(), persistent=False)
        # Concatenated rather than folded into statics_*: the condition channel order is
        # part of the checkpoint format, so new channels append at the end.
        self.register_buffer(
            "geo_xyz_hpx", sphere_position(lat_hpx, lon_hpx).float(), persistent=False
        )
        self.register_buffer(
            "geo_xyz_ll",
            sphere_position(lat_flat, lon_flat).view(3, NLAT, NLON).float(),
            persistent=False,
        )
        if fine_calendar:
            lon_ll = lon_flat
            self.calendar_hpx = CalendarEmbedding(
                torch.as_tensor(hpx.lon), CALENDAR_FREQS
            ).float()
            self.calendar_ll = CalendarEmbedding(lon_ll, CALENDAR_FREQS).float()
        else:
            self.calendar_hpx = None
            self.calendar_ll = None
        self._compile()

    def _geo(self, timestamp, *, on_hpx: bool):
        """(B, 1, T, ...) cos(solar zenith), the only time-varying geo channel."""
        b, t = timestamp.shape
        lat = self.geo_lat_hpx if on_hpx else self.geo_lat_ll
        lon = self.geo_lon_hpx if on_hpx else self.geo_lon_ll
        zenith = solar_zenith_cosine(timestamp, lat, lon)
        if not on_hpx:
            zenith = zenith.reshape(b, 1, t, NLAT, NLON)
        return zenith.to(self.statics_ll.dtype)

    def _conditioning(
        self,
        second: torch.Tensor,
        day: torch.Tensor,
        timestamp: torch.Tensor,
        *,
        on_hpx: bool,
    ) -> torch.Tensor:
        """``(b, c, t, *spatial)`` conditioning for one grid: HPX if ``on_hpx``, else 0.25 deg.

        Shared so the backbone's condition and the decoder's aux cannot drift apart.
        """
        b, t = second.shape
        statics, xyz, calendar = (
            (self.statics_hpx, "geo_xyz_hpx", self.calendar_hpx)
            if on_hpx
            else (self.statics_ll, "geo_xyz_ll", self.calendar_ll)
        )

        def over_bt(x: torch.Tensor) -> torch.Tensor:
            # (c, *spatial) -> (b, c, t, *spatial), a view: statics do not vary in b or t.
            return x.unsqueeze(0).unsqueeze(2).expand(b, -1, t, *(-1,) * (x.ndim - 1))

        pieces = [over_bt(statics), over_bt(getattr(self, xyz))]
        if calendar is not None:
            pieces.append(
                self._calendar(calendar, second, day, None if on_hpx else (NLAT, NLON))
            )
        # Hoisted out of the append: _geo breaks the graph, and dynamo would otherwise
        # guard the resume frame on a bound method rebuilt every call, recompiling until it
        # falls back to eager.
        zenith = self._geo(timestamp, on_hpx=on_hpx)
        pieces.append(zenith)
        return torch.cat(pieces, dim=1)

    def _cond(
        self, second: torch.Tensor, day: torch.Tensor, timestamp: torch.Tensor
    ) -> torch.Tensor:
        """The backbone's input conditioning, on HPX, ``(b, c, t, npix)``."""
        return self._conditioning(second, day, timestamp, on_hpx=True)

    def _ll_aux(
        self, second: torch.Tensor, day: torch.Tensor, timestamp: torch.Tensor
    ) -> torch.Tensor:
        """The decoder's 0.25 deg aux, time folded into batch: ``((b t), c, nlat, nlon)``."""
        aux = self._conditioning(second, day, timestamp, on_hpx=False)
        return einops.rearrange(aux, "b c t h w -> (b t) c h w")

    def _latlon_tail(self, hidden_states, second, day, timestamp):
        """hidden_states: (b, t, npix, c), the raw latent at the decoder's level_in.

        Noise and label conditioning reach the output only through the backbone.
        """
        b, t, npix, c = hidden_states.shape
        aux = self._ll_aux(second, day, timestamp)
        aux = aux.reshape(b * t, aux.shape[1], NLAT * NLON).transpose(1, 2)
        out = self.graphcast_decode(hidden_states.reshape(b * t, npix, c), aux)
        return einops.rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)

    def _compile(self):
        if getattr(self.backbone, "compile_dit", False):
            self._call_tail = torch.compile(self._latlon_tail)
            self._call_cond = torch.compile(self._cond)
        else:
            self._call_tail = self._latlon_tail
            self._call_cond = self._cond

    @property
    def transformer_blocks(self):
        return self.backbone.transformer_blocks

    @property
    def pos_embed(self):
        return self.backbone.pos_embed

    @property
    def patch_decode(self):
        return self.backbone.patch_decode

    @property
    def gradient_checkpointing(self):
        return self.backbone.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.backbone.gradient_checkpointing = value

    @property
    def gradient_checkpointing_last_n(self):
        return self.backbone.gradient_checkpointing_last_n

    @gradient_checkpointing_last_n.setter
    def gradient_checkpointing_last_n(self, value):
        self.backbone.gradient_checkpointing_last_n = value

    def set_time_parallel_group(self, group):
        self.backbone.set_time_parallel_group(group)

    def set_domain_parallel_group(self, group):
        self.backbone.set_domain_parallel_group(group)

    def fully_shard(self, mesh=None, mp_policy=None):
        self.backbone.fully_shard(mesh, mp_policy=mp_policy)
        kwargs = {} if mp_policy is None else {"mp_policy": mp_policy}
        torch.distributed.fsdp.fully_shard(self.graphcast_decode, mesh=mesh, **kwargs)

    def _calendar(self, embed, second_of_day, day_of_year, spatial):
        maps = embed(second_of_day=second_of_day, day_of_year=day_of_year)
        if spatial is None:
            return maps
        b, c, t, _ = maps.shape
        return maps.reshape(b, c, t, *spatial)

    def forward(self, hidden_states, **kwargs):
        # Deliberately dropped: the condition is rebuilt below at this model's grid, so
        # whatever run_model_step assembled never reaches the backbone. get_network refuses
        # the configs whose condition would have carried more (background, mask training).
        del hidden_states
        second = kwargs["second_of_day"]
        day = kwargs["day_of_year"]
        timestamp = kwargs["timestamp"]
        with healda.utils.profiling.nvtx_range("latlon:cond"):
            with healda.utils.profiling.cuda_timing_range("latlon:cond"):
                x = self._call_cond(second, day, timestamp)
        with healda.utils.profiling.nvtx_range("latlon:backbone"):
            decoded = self.backbone(x, **kwargs)
        with healda.utils.profiling.nvtx_range("latlon:tail"):
            with healda.utils.profiling.cuda_timing_range("latlon:tail"):
                pred = self._call_tail(decoded.out, second, day, timestamp)
        return Output(out=pred, obs=decoded.obs)

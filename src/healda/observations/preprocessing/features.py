# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-observation metadata features: the torch definition (``compute_unified_metadata``,
``position_features``) and ``featurize``, the CUDA kernel the data transform runs.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_cosf as fcos
from triton.language.extra.libdevice import fast_sinf as fsin
from triton.language.extra.libdevice import isnan

import healda.utils.profiling

N_FEATURES = 50
# Conv/sat-split layout written directly (no separate split step):
#   [0:10)  SHARED: LST fourier(2)[0:4) + dt [dt,dt²][4:6) + dt fourier(1)[6:8)
#                   + lat [sin,cos][8:10)
#   [10:30) SAT-private: scan fourier(3)[10:16) + sat_zen fourier(4)[16:24)
#                        + sol_zen fourier(3)[24:30)   -- zero for conv rows
#   [30:50) CONV-private: height fourier(5)[30:40) + pressure fourier(5)[40:50)
#                         -- zero for sat rows
# A row is conv iff height is not NaN; exactly one private block is non-zero.
#
# Normalization conventions (same as v1 height/pressure: normalize to [0,1] by physical max):
#   height:  h / 60000  -> [0, 1]
#   pressure: p / 1100  -> [0, 1]
#   scan:    ξ / 50     -> ~[-1, 1]   (data range: ~[-50, +50] deg)
#   sat_zen: θ / 90     -> ~[-0.67, +0.67]  (data range: ~[-60, +60] deg, signed)
#   sol_zen: θ / 180    -> ~[0.06, 0.94]    (data range: ~[10, 170] deg)
FILM_SPLIT_SHARED_META_DIM = 10
FILM_SPLIT_PRIVATE_META_DIM = 20  # width of each (sat / conv) private block


@healda.utils.profiling.nvtx
def compute_unified_metadata(
    target_time_sec: torch.Tensor,
    lon: torch.Tensor,
    lat: torch.Tensor,
    time: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    """(N, N_FEATURES) float32 v2 metadata; layout above. Degrees, time in ns, target in s.

    ``featurize`` computes the same features on CUDA.
    """
    device = lon.device
    n_obs = lon.shape[0]
    out = torch.zeros(n_obs, N_FEATURES, dtype=torch.float32, device=device)

    if n_obs == 0:
        return out

    is_conv = ~torch.isnan(height)

    TWO_PI = 2 * math.pi

    # --- shared: LST fourier(2) -> 4 ---
    lst = local_solar_time(lon, time)
    out[:, 0:4] = fourier_features(lst / 24.0 * TWO_PI, 2)

    # --- shared: relative time polynomial -> 2 ---
    target_time_ns = target_time_sec * 1_000_000_000
    dt_days = (time - target_time_ns).float() * 1e-9 / 86400.0
    out[:, 4] = dt_days
    out[:, 5] = dt_days**2

    # --- shared: relative time fourier(1) -> 2 ---
    out[:, 6:8] = fourier_features(dt_days, 1)

    # --- shared: latitude -> 2 ---
    lat_rad = torch.deg2rad(lat)
    out[:, 8] = torch.sin(lat_rad)
    out[:, 9] = torch.cos(lat_rad)

    # --- sat-private [10:30): scan fourier(3) + sat_zen fourier(4) + sol_zen fourier(3) ---
    is_sat = ~is_conv
    if is_sat.any():
        s = is_sat
        out[s, 10:16] = fourier_features(scan_angle[s] / 50.0 * TWO_PI, 3)
        out[s, 16:24] = fourier_features(sat_zenith_angle[s] / 90.0 * TWO_PI, 4)
        out[s, 24:30] = fourier_features(sol_zenith_angle[s] / 180.0 * TWO_PI, 3)

    # --- conv-private [30:50): height fourier(5) + pressure fourier(5) ---
    if is_conv.any():
        c = is_conv
        h_norm = torch.clamp(height[c] / 60000.0, 0.0, 1.0)
        out[c, 30:40] = fourier_features(h_norm * TWO_PI, 5)
        p_norm = torch.clamp(pressure[c] / 1100.0, 0.0, 1.0)
        out[c, 40:50] = fourier_features(p_norm * TWO_PI, 5)

    return out


def fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
    device = x_norm.device
    freqs = torch.arange(1, num_freqs + 1, device=device, dtype=x_norm.dtype)
    x_expanded = x_norm.unsqueeze(-1) * freqs
    sin_features = torch.sin(x_expanded)
    cos_features = torch.cos(x_expanded)
    return torch.cat([sin_features, cos_features], dim=-1)


def local_solar_time(
    lon_deg: torch.Tensor,
    abs_time_ns: torch.Tensor,
) -> torch.Tensor:
    sec_of_day = (abs_time_ns // 1_000_000_000) % 86400
    utc_hours = sec_of_day.float() / 3600.0
    lst = (utc_hours + lon_deg / 15.0) % 24.0
    return lst


# Columns appended after the N_FEATURES layout when the model asks for position features.
N_POSITION_FEATURES = 4


def position_features(
    lon: torch.Tensor,
    lat: torch.Tensor,
    center_lon: torch.Tensor,
    center_lat: torch.Tensor,
    hpx_level: int,
) -> torch.Tensor:
    """Per observation, shape (N, 4) float32: east and north offset from its HEALPix pixel
    centre in pixel sizes (about -0.7..0.7), then sin and cos of longitude. Degrees in.

    Computed in float32, as ``_position_features`` does, so both give the same bits
    on CUDA. The longitude offset is wrapped to [-180, 180] by adding or removing 360, so a
    point and a nearby centre subtract exactly.
    """
    deg2rad, inv_pixel = position_constants(hpx_level, lon.device)
    lon, lat = lon.float(), lat.float()
    center_lon, center_lat = center_lon.float(), center_lat.float()
    dlon = lon - center_lon
    dlon = torch.where(
        dlon > 180, dlon - 360, torch.where(dlon < -180, dlon + 360, dlon)
    )
    dx = dlon * torch.cos(center_lat * deg2rad) * inv_pixel
    dy = (lat - center_lat) * inv_pixel
    lon_rad = lon * deg2rad
    return torch.stack([dx, dy, torch.sin(lon_rad), torch.cos(lon_rad)], dim=1)


def position_constants(hpx_level: int, device) -> torch.Tensor:
    """float32 [deg2rad factor, 1 / pixel size in degrees] for both position-feature paths."""
    pixel_deg = math.degrees(math.sqrt(4 * math.pi / (12 * 4**hpx_level)))
    return torch.tensor(
        [math.pi / 180, 1 / pixel_deg], dtype=torch.float32, device=device
    )


# Columns featurize reads, in _featurize_kernel's argument order
INPUT_COLUMNS = {
    "latitude": torch.float32,
    "longitude": torch.float32,
    "absolute_obs_time": torch.int64,
    "target_time_sec": torch.int64,
    "height": torch.float32,
    "pressure": torch.float32,
    "scan_angle": torch.float32,
    "sat_zenith_angle": torch.float32,
    "sol_zenith_angle": torch.float32,
    "observation": torch.float32,
    "global_channel_id": torch.uint16,
    "local_channel_id": torch.uint16,
    "platform_id": torch.uint16,
    "observation_type": torch.uint16,
}

# Columns featurize returns, in _featurize_kernel's argument order; ids widen to int32
OUTPUT_COLUMNS = {
    "observation": torch.float32,
    "global_channel_id": torch.int32,
    "local_channel_id": torch.int32,
    "platform_id": torch.int32,
    "observation_type": torch.int32,
}


# fmt: off
@triton.jit
def _fourier(x, K: tl.constexpr):
    """sin(x * k) for k = 1..K, then cos(x * k), as ``fourier_features``."""
    sin = ()
    cos = ()
    for k in tl.static_range(1, K + 1):
        sin = sin + (fsin(x * k),)
        cos = cos + (fcos(x * k),)
    return sin + cos


@triton.jit
def _where(keep, cols):
    out = ()
    for i in tl.static_range(len(cols)):
        out = out + (tl.where(keep, cols[i], 0.0),)
    return out


@triton.jit
def _join_pairs(cols):
    out = ()
    for i in tl.static_range(0, len(cols), 2):
        out = out + (tl.join(cols[i], cols[i + 1]),)
    return out


@triton.jit
def _feature_tile(lat, lon, time_ns, target_s, height, pressure, scan, sat_zen, sol_zen,
                  p0, p1, p2, p3):
    """[rows, 64] float32 features of [rows] inputs; columns N_FEATURES.. hold p0..p3
    (position features, or zeros), then zeros.

    Every column is computed, the off-family private block as 0, so a store of the tile
    writes each output byte once and needs no zero-filled buffer. Each column is the
    expression of ``compute_unified_metadata`` in float32.
    """
    TWO_PI: tl.constexpr = 6.283185307179586
    DEG2RAD: tl.constexpr = 0.017453292519943295
    is_conv = ~isnan(height)
    is_sat = ~is_conv

    sod = (time_ns // 1000000000) % 86400
    utc_hr = sod.to(tl.float32) / 3600.0
    lst = (utc_hr + lon / 15.0) % 24.0
    lst_x = lst / 24.0 * TWO_PI
    dt_days = (time_ns - target_s * 1000000000).to(tl.float32) * 1e-9 / 86400.0
    scan_x = scan / 50.0 * TWO_PI
    sat_x = sat_zen / 90.0 * TWO_PI
    sol_x = sol_zen / 180.0 * TWO_PI
    h_x = tl.minimum(tl.maximum(height / 60000.0, 0.0), 1.0) * TWO_PI
    p_x = tl.minimum(tl.maximum(pressure / 1100.0, 0.0), 1.0) * TWO_PI
    zero = tl.zeros_like(lat)

    # Shared [0, 10): LST fourier(2), dt, dt², dt fourier(1), sin/cos latitude
    shared = _fourier(lst_x, 2) + (dt_days, dt_days * dt_days) + _fourier(dt_days, 1)
    shared = shared + _fourier(lat * DEG2RAD, 1)
    # Satellite [10, 30): scan fourier(3), satellite zenith fourier(4), solar zenith fourier(3)
    sat = _where(is_sat, _fourier(scan_x, 3) + _fourier(sat_x, 4) + _fourier(sol_x, 3))
    # Conventional [30, 50): height fourier(5), pressure fourier(5)
    conv = _where(is_conv, _fourier(h_x, 5) + _fourier(p_x, 5))
    cols = shared + sat + conv + (p0, p1, p2, p3) + (zero,) * 10

    # Six rounds of pairwise joins give [rows, 2, 2, 2, 2, 2, 2]; each join adds its axis
    # last, so column bit 0 is on the last axis
    tile = cols
    for _ in tl.static_range(6):
        tile = _join_pairs(tile)
    return tl.reshape(tl.permute(tile[0], (0, 6, 5, 4, 3, 2, 1)), (lat.shape[0], 64))


@triton.jit
def _store_feature_tile(out_ptr, off, m, tile, out_stride, N_FEAT: tl.constexpr):
    """Store rows ``off`` of a feature tile; a block's rows are one contiguous region."""
    f = tl.arange(0, 64)
    ptrs = out_ptr + off.to(tl.int64)[:, None] * out_stride + f[None, :]
    tl.store(ptrs, tile, mask=m[:, None] & (f[None, :] < N_FEAT))



@triton.jit
def _position_features(lon, lat, center_lon, center_lat, consts_ptr):
    """``position_features`` for [rows] inputs, bit for bit: the same float32 operations,
    a multiply by the reciprocal (Triton's ``/`` is approximate) and libdevice's float32
    sin and cos, which torch uses on CUDA."""
    deg2rad = tl.load(consts_ptr)
    inv_pixel = tl.load(consts_ptr + 1)
    center_lon = center_lon.to(tl.float32)
    center_lat = center_lat.to(tl.float32)
    dlon = lon - center_lon
    dlon = tl.where(dlon > 180.0, dlon - 360.0, tl.where(dlon < -180.0, dlon + 360.0, dlon))
    dx = dlon * libdevice.cos(center_lat * deg2rad) * inv_pixel
    dy = (lat - center_lat) * inv_pixel
    lon_rad = lon * deg2rad
    return dx, dy, libdevice.sin(lon_rad), libdevice.cos(lon_rad)


@triton.jit
def _featurize_kernel(lat_ptr, lon_ptr, time_ptr, target_ptr, height_ptr, press_ptr, scan_ptr,
                      sat_ptr, sol_ptr, obs_ptr, gch_ptr, lch_ptr, plat_ptr, otype_ptr, order_ptr,
                      out_ptr, o_obs, o_gch, o_lch, o_plat, o_otype,
                      center_lon_ptr, center_lat_ptr, consts_ptr, N, out_stride,
                      BLOCK: tl.constexpr, N_FEAT: tl.constexpr, SORTED: tl.constexpr,
                      POSITION: tl.constexpr):
    """Output row i from input row order[i] (row i unless SORTED): the output columns, and the
    feature tile with the position features under POSITION."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    if SORTED:
        row = tl.load(order_ptr + off, mask=m, other=0).to(tl.int64)
    else:
        row = off.to(tl.int64)
    lat = tl.load(lat_ptr + row, mask=m, other=0.0)
    lon = tl.load(lon_ptr + row, mask=m, other=0.0)
    time_ns = tl.load(time_ptr + row, mask=m, other=0)
    target_s = tl.load(target_ptr + row, mask=m, other=0)
    tl.store(o_obs + off, tl.load(obs_ptr + row, mask=m, other=0.0), mask=m)
    tl.store(o_gch + off, tl.load(gch_ptr + row, mask=m, other=0).to(tl.int32), mask=m)
    tl.store(o_lch + off, tl.load(lch_ptr + row, mask=m, other=0).to(tl.int32), mask=m)
    tl.store(o_plat + off, tl.load(plat_ptr + row, mask=m, other=0).to(tl.int32), mask=m)
    tl.store(o_otype + off, tl.load(otype_ptr + row, mask=m, other=0).to(tl.int32), mask=m)

    if POSITION:
        # Pixel centres are in output order
        p0, p1, p2, p3 = _position_features(
            lon, lat,
            tl.load(center_lon_ptr + off, mask=m, other=0.0),
            tl.load(center_lat_ptr + off, mask=m, other=0.0),
            consts_ptr,
        )
    else:
        p0 = tl.zeros_like(lat)
        p1 = p0
        p2 = p0
        p3 = p0
    tile = _feature_tile(
        lat, lon, time_ns, target_s,
        tl.load(height_ptr + row, mask=m, other=0.0), tl.load(press_ptr + row, mask=m, other=0.0),
        tl.load(scan_ptr + row, mask=m, other=0.0), tl.load(sat_ptr + row, mask=m, other=0.0),
        tl.load(sol_ptr + row, mask=m, other=0.0),
        p0, p1, p2, p3,
    )
    _store_feature_tile(out_ptr, off, m, tile, out_stride, N_FEAT)
# fmt: on


@healda.utils.profiling.nvtx
def featurize(
    columns: dict[str, torch.Tensor],
    order: torch.Tensor | None,
    centers: tuple[torch.Tensor, torch.Tensor] | None = None,
    hpx_level: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """``OUTPUT_COLUMNS`` of the ``INPUT_COLUMNS`` columns (all on one CUDA device) and their
    bf16 metadata, in one pass: output row i is input row ``order[i]``, or row i without
    ``order``. With ``order``, there are ``len(order)`` output rows.

    With ``centers``, the (lon, lat) HEALPix pixel centres of the output rows at
    ``hpx_level``, the metadata is (N, N_FEATURES + N_POSITION_FEATURES):
    ``compute_unified_metadata`` followed by ``position_features``. Otherwise
    (N, N_FEATURES). Features are computed in float32 and rounded to bf16 on store.
    """
    for name, dtype in INPUT_COLUMNS.items():
        if columns[name].dtype != dtype:
            raise TypeError(f"{name} is {columns[name].dtype}, expected {dtype}")
    n = columns["latitude"].shape[0] if order is None else order.shape[0]
    device = columns["latitude"].device
    out = {
        name: torch.empty(n, dtype=dtype, device=device)
        for name, dtype in OUTPUT_COLUMNS.items()
    }
    position = centers is not None
    width = N_FEATURES + (N_POSITION_FEATURES if position else 0)
    meta = torch.empty(n, width, dtype=torch.bfloat16, device=device)
    if n:
        center_lon, center_lat = centers if position else (meta, meta)
        consts = position_constants(hpx_level, device) if position else meta
        BLOCK = 64
        _featurize_kernel[(triton.cdiv(n, BLOCK),)](
            *(columns[name].contiguous() for name in INPUT_COLUMNS),
            meta if order is None else order,
            meta,
            *out.values(),
            center_lon,
            center_lat,
            consts,
            n,
            width,
            BLOCK=BLOCK,
            N_FEAT=width,
            SORTED=order is not None,
            POSITION=position,
            num_warps=1,
        )
    return out, meta

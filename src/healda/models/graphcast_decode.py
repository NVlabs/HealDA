# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphCast-style Mesh2Grid decode from HEALPix mesh nodes to the 0.25 degree lat/lon grid.

Three stages per target point: a bilinear dynamic baseline read off the backbone latent, a
conditioning term from statics and position, and a learned K-nearest edge-message residual on
top. The mesh nodes are the backbone's own tokens.

Requires NEST when the mesh is finer than the backbone (see ``_unpack_and_lift``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from earth2grid import healpix

from healda.observations.types import healpix_pixel_order

# Targets per chunk; bounds the (chunk, k, hidden) gather.
CHUNK = 131072


def _unit_sphere(lat, lon):
    la, lo = torch.deg2rad(lat.reshape(-1)), torch.deg2rad(lon.reshape(-1))
    return torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()], -1)


def _target_grid(nlat: int, nlon: int):
    tlat = torch.linspace(90.0, -90.0, nlat, dtype=torch.float64)
    tlon = torch.linspace(0.0, 360.0, nlon + 1, dtype=torch.float64)[:-1]
    return torch.meshgrid(tlat, tlon, indexing="ij")


def _local_frame(pts: torch.Tensor):
    """Orthonormal East/North at each point, with the poles handled explicitly.

    East = world_z x up degenerates exactly at the poles, which the target grid really
    contains, so fall back to world_x there rather than dividing by ~0 and producing NaN at
    two genuine rows.
    """
    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(pts)
    east = torch.cross(world_z, pts, dim=-1)
    norm = east.norm(dim=-1, keepdim=True)
    degenerate = norm.squeeze(-1) < 1e-8
    if degenerate.any():
        world_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).expand_as(pts)
        east = torch.where(
            degenerate.unsqueeze(-1), torch.cross(world_x, pts, dim=-1), east
        )
        norm = east.norm(dim=-1, keepdim=True)
    east = east / norm.clamp_min(1e-12)
    return east, torch.cross(pts, east, dim=-1)


def build_mesh_edges(level_mesh: int, k: int, pixel_order: str, nlat: int, nlon: int):
    """K nearest HPX(level_mesh) nodes for every target, plus per-edge geometry.

    ``pixel_order`` must be the pipeline's resolved token order: HPX pixel i is a different
    point on the sphere in NEST than in HEALPIX_PAD_XY.

    Returns idx (NTARGET, k), efeat (NTARGET, k, 4) as each neighbour's offset in the
    target's own East/North/Up frame plus chord distance, and tgt_xyz (NTARGET, 3).
    """
    from scipy.spatial import cKDTree

    grid = healpix.Grid(level=level_mesh, pixel_order=healpix_pixel_order(pixel_order))
    src = _unit_sphere(
        torch.as_tensor(grid.lat, dtype=torch.float64),
        torch.as_tensor(grid.lon, dtype=torch.float64),
    )
    tgt = _unit_sphere(*_target_grid(nlat, nlon))

    dist, idx = cKDTree(src.numpy()).query(tgt.numpy(), k=k, workers=-1)
    idx = torch.as_tensor(idx, dtype=torch.int64).reshape(nlat * nlon, k)
    dist = torch.as_tensor(dist, dtype=torch.float64).reshape(nlat * nlon, k)

    east, north = _local_frame(tgt)
    rel = src[idx.reshape(-1)].reshape(nlat * nlon, k, 3) - tgt.unsqueeze(1)
    efeat = torch.stack(
        [
            (rel * east.unsqueeze(1)).sum(-1),
            (rel * north.unsqueeze(1)).sum(-1),
            (rel * tgt.unsqueeze(1)).sum(-1),
            dist,
        ],
        -1,
    ).float()
    return idx, efeat, tgt.float()


def build_bilinear_weights(level: int, nlat: int, nlon: int, pixel_order: str):
    """HEALPix bilinear weights for the dynamic baseline.

    Bilinear rather than inverse-distance: it selects the enclosing face-local cell, so weights
    vary continuously as the target moves, with none of IDW's neighbour-swap discontinuity.
    """
    grid = healpix.Grid(level=level, pixel_order=healpix_pixel_order(pixel_order))
    tlat, tlon = _target_grid(nlat, nlon)
    regrid = grid.get_bilinear_regridder_to(tlat.numpy(), tlon.numpy())
    return (
        regrid.index.reshape(nlat * nlon, -1).long(),
        regrid.weight.reshape(nlat * nlon, -1).float(),
    )


class GraphCastDecoder(nn.Module):
    """Mesh2Grid decode: bilinear baseline + conditioning, then a learned edge residual."""

    def __init__(
        self,
        *,
        in_channels: int,
        aux_channels: int,
        out_channels: int,
        level_in: int,
        level_mesh: int,
        k: int,
        pixel_order: str,
        nlat: int,
        nlon: int,
        base_level: int | None = None,
        remat: bool = True,
    ):
        super().__init__()
        if level_mesh < level_in:
            raise ValueError(f"level_mesh={level_mesh} must be >= level_in={level_in}")
        patch = 4 ** (level_mesh - level_in)
        if patch > 1 and pixel_order != "nest":
            raise ValueError(
                f"level_mesh={level_mesh} > level_in={level_in} splits each token into "
                f"{patch} children, which is only contiguous in nest; got {pixel_order!r}."
            )
        # The unpack spends the whole latent on the children, so the width is determined.
        if in_channels % patch:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by patch={patch}"
            )
        hidden = in_channels // patch
        # Which level the bilinear baseline interpolates on. Adds no parameters either way,
        # so a checkpoint moves between level_in and level_mesh.
        base_level = level_in if base_level is None else base_level
        if not level_in <= base_level <= level_mesh:
            raise ValueError(
                f"base_level={base_level} must lie in [{level_in}, {level_mesh}]"
            )
        self.base_level = base_level
        #: children averaged per baseline node; 1 when reading the mesh level, so the mean is
        #: the identity and q_dyn reads `u` unchanged.
        self.base_group = 4 ** (level_mesh - base_level)
        self.h, self.k, self.patch, self.remat = hidden, k, patch, remat
        self.nlat, self.nlon = nlat, nlon

        self.unpack = nn.Linear(in_channels, patch * hidden)
        # aux already carries the unit-sphere target position, so it is not repeated here.
        self.target = nn.Linear(aux_channels, hidden)
        self.target_norm = nn.LayerNorm(hidden)
        # No bias, so it commutes with the gather: project the mesh nodes once here rather
        # than the k-times-larger gathered tensor per chunk.
        self.lin_src = nn.Linear(hidden, hidden, bias=False)
        self.lin_dst = nn.Linear(hidden, hidden)
        self.lin_e = nn.Linear(4, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.node = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        # Zero-init so the model starts at the bilinear baseline and learns the edge
        # residual, rather than mixing in an untrained edge aggregate from step 0.
        nn.init.zeros_(self.node[-1].weight)
        nn.init.zeros_(self.node[-1].bias)
        self.out = nn.Linear(hidden, out_channels)

        idx, efeat, _ = build_mesh_edges(level_mesh, k, pixel_order, nlat, nlon)
        bidx, bweight = build_bilinear_weights(base_level, nlat, nlon, pixel_order)
        buffers = [("idx", idx), ("efeat", efeat), ("bidx", bidx), ("bweight", bweight)]
        for name, buf in buffers:
            self.register_buffer(name, buf, persistent=False)
        self.call_chunk = self.one_chunk

    def compile_chunks(self) -> None:
        # One chunk is compiled, not forward: a compiled forward unrolls the chunk loop
        # into one graph, which keeps every chunk's intermediates alive at once.
        self.call_chunk = torch.compile(self.one_chunk, dynamic=False)

    def _unpack_and_lift(self, z: torch.Tensor) -> torch.Tensor:
        # Each token's `patch` output slots are its geometric children, which holds only
        # because NEST puts them at consecutive indices; __init__ enforces it.
        return self.unpack(z).reshape(z.shape[0], -1, self.h)

    def one_chunk(self, v, qu, aux, idx, efeat, bidx, bweight):
        bt, rows = aux.shape[:2]
        vj = v[:, idx.reshape(-1)].reshape(bt, rows, self.k, self.h)
        bu = qu[:, bidx.reshape(-1)].reshape(bt, rows, bidx.shape[-1], self.h)
        q_dyn = (bweight.unsqueeze(0).unsqueeze(-1) * bu).sum(2)

        q = self.target_norm(q_dyn + self.target(aux))

        x = vj + self.lin_dst(q).unsqueeze(2) + self.lin_e(efeat)
        m = self.ln(self.w2(F.silu(x))).sum(2)
        return self.out(q + self.node(torch.cat([q, m], -1)))

    def forward(self, z: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """z: (bt, npix_in, in_channels); aux: (bt, nlat*nlon, aux_channels)."""
        bt = z.shape[0]
        u = self._unpack_and_lift(z)
        v = self.lin_src(u)
        # base_group > 1 averages each parent's children back, so the stencil reads four
        # distinct tokens rather than four linear maps of one parent.
        qu = (
            u.reshape(u.shape[0], -1, self.base_group, self.h).mean(2)
            if self.base_group > 1
            else u
        )

        pieces = []
        for lo in range(0, self.nlat * self.nlon, CHUNK):
            rows = slice(lo, lo + CHUNK)
            args = (v, qu, aux[:, rows], self.idx[rows], self.efeat[rows])
            args += (self.bidx[rows], self.bweight[rows])
            if self.remat and torch.is_grad_enabled():
                pieces.append(
                    torch.utils.checkpoint.checkpoint(
                        self.call_chunk, *args, use_reentrant=False
                    )
                )
            else:
                pieces.append(self.call_chunk(*args))
        out = torch.cat(pieces, 1)
        return out.transpose(1, 2).reshape(bt, -1, self.nlat, self.nlon)

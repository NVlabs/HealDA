# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ERA5 year-file store: contiguous ``fields`` arrays and their sidecar.

Uncompressed float32, ``(time, channel, lat, lon)``. Contiguous layout plus early
allocation fixes every offset before the first write, so writers pwrite disjoint
ranges and readers pull a timestep without HDF5 on the read path. The cost: the
dataset cannot be resized, so later channels go in a companion file.

``metadata/data.json`` beside the years records channel order, cadence and grid;
channel order differs between sets, so readers select by name.
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from scripts.etl.era5.channels import CHAN_BYTES, NLAT, NLON
from healda.utils.storage import StorageConfigError, get_obstore, get_storage_options


# Relative to a dataset root.
from healda.datasets.era5_hourly_h5 import SIDECAR as _SIDECAR

SIDECAR = Path(_SIDECAR)


def sidecar_path(dataset_root: str | os.PathLike) -> Path:
    return Path(dataset_root).resolve() / SIDECAR


def sidecar_uri(dataset_root: str | os.PathLike) -> str:
    root = str(dataset_root)
    if root.startswith("s3://"):  # Path.resolve would collapse the // in s3://.
        return f"{root.rstrip('/')}/{SIDECAR.as_posix()}"
    return str(sidecar_path(root))


def read_sidecar(
    dataset_root: str | os.PathLike, profile: str | None = None
) -> dict[str, Any]:
    """Load the sidecar for a local or ``s3://`` dataset root, or raise if missing."""
    uri = sidecar_uri(dataset_root)

    if uri.startswith("s3://"):
        import fsspec

        from healda.utils.storage import get_storage_options

        fs = fsspec.filesystem("s3", **get_storage_options(profile))
        key = uri[len("s3://") :]
        if not fs.exists(key):
            raise SystemExit(
                f"{uri} is missing; the store cannot be interpreted without it"
            )
        return json.loads(fs.cat(key))

    path = Path(uri)
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; the store cannot be interpreted without it"
        )
    return json.loads(path.read_text())


def cadence_name(dhours: int) -> str:
    return "hourly" if dhours == 1 else f"{dhours}hourly"


def equiangular_coords(channels: list[str]) -> dict[str, Any]:
    return {
        "grid_type": "equiangular",
        "lat": [90.0 - 0.25 * i for i in range(NLAT)],
        "lon": [0.25 * i for i in range(NLON)],
        "channel": list(channels),
    }


def sidecar_document(
    *,
    name: str,
    channels: list[str],
    dhours: int,
    description: str,
    sources: str,
    coords: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset_name": name,
        "attrs": {"description": description, "sources": sources},
        "h5_path": "fields",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": dhours,
        "coords": coords if coords is not None else equiangular_coords(channels),
    }


def write_h5_attrs(path: str, document: dict[str, Any]) -> None:
    """Copy the sidecar onto the file, so a year is readable without its root."""
    coords = document["coords"]
    with h5py.File(path, "r+") as f:
        f.attrs.update(
            {
                "dataset_name": document["dataset_name"],
                "description": document["attrs"]["description"],
                "sources": document["attrs"]["sources"],
                "h5_path": document["h5_path"],
                "dims": document["dims"],
                "dhours": document["dhours"],
                "grid_type": coords["grid_type"],
                "lat": np.asarray(coords["lat"], dtype="f8"),
                "lon": np.asarray(coords["lon"], dtype="f8"),
                "channel": list(coords["channel"]),
            }
        )


def write_sidecar(
    dataset_root: str | os.PathLike,
    *,
    name: str,
    channels: list[str],
    dhours: int,
    description: str,
    sources: str,
    coords: dict[str, Any] | None = None,
) -> str:
    """Emit ``<dataset_root>/metadata/data.json``, or validate the one already there.

    A channel/cadence disagreement means this build belongs under a different root,
    so it raises rather than overwrites. ``coords`` defaults to the equiangular grid.
    """
    path = sidecar_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = json.loads(path.read_text())
        found = existing["coords"]["channel"]
        if found != list(channels):
            raise SystemExit(
                f"{path} describes {len(found)} channels {found[:3]}...{found[-1]}"
                f" but this build has {len(channels)} {channels[:3]}...{channels[-1]}."
                " Pass a different dataset root; one sidecar serves all years under it."
            )
        if existing.get("dhours") != dhours:
            raise SystemExit(
                f"{path} describes dhours={existing.get('dhours')} but this build is"
                f" dhours={dhours}. Pass a different dataset root."
            )
        return str(path)

    document = sidecar_document(
        name=name,
        channels=channels,
        dhours=dhours,
        description=description,
        sources=sources,
        coords=coords,
    )
    # Ranks racing to create it must not collide, and PIDs repeat across nodes.
    scratch = path.with_name(f".data.json.{uuid.uuid4().hex}")
    scratch.write_text(json.dumps(document, indent=2))
    os.replace(scratch, path)
    return str(path)


def create_output(path: str, nsteps: int, nchan: int, create: bool) -> int:
    """Return the byte offset of ``fields``, creating the file only if ``create``.

    Every rank calls this; concurrent creates would each truncate the same file.
    Shape/dtype are rechecked because pwrite offsets assume them.
    """
    if not create and not os.path.exists(path):
        raise SystemExit(f"{path} does not exist; run the setup stage first")

    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            dataset = f["fields"]
            if dataset.shape != (nsteps, nchan, NLAT, NLON):
                raise SystemExit(
                    f"{path} has shape {dataset.shape}, expected "
                    f"{(nsteps, nchan, NLAT, NLON)}"
                )
            if dataset.dtype != np.dtype("<f4"):
                raise SystemExit(f"{path} has dtype {dataset.dtype}, expected float32")
            offset = dataset.id.get_offset()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "w") as f:
            space = h5py.h5s.create_simple((nsteps, nchan, NLAT, NLON))
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_layout(h5py.h5d.CONTIGUOUS)
            dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            h5py.h5d.create(f.id, b"fields", h5py.h5t.IEEE_F32LE, space, dcpl=dcpl)
        with h5py.File(path, "r") as f:
            offset = f["fields"].id.get_offset()

    if offset is None:
        raise RuntimeError(f"{path} dataset is not contiguous")
    return offset


def pwrite_all(fd: int, data, offset: int) -> None:
    """``pwrite`` every byte of ``data``; a single call may write fewer than asked."""
    view = memoryview(data).cast("B")
    written = 0
    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:
            raise OSError("pwrite made no progress")
        written += n


def split_s3_url(url: str) -> tuple[str, str]:
    rest = url[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"{url!r} is not s3://bucket/key")
    return bucket, key


class RangeReader:
    """Reads exactly ``length`` bytes at ``offset`` from a local path or ``s3://`` URL.

    Handles are opened lazily so an instance survives a fork, and guarded by a lock
    so the threads sharing one reader cannot each build their own. obstore, not
    boto3: boto3 is not a healda dependency.
    """

    def __init__(self, source: str, profile: str | None = None):
        self.source = source
        self.is_s3 = source.startswith("s3://")
        if self.is_s3:
            if not profile:
                raise StorageConfigError(f"{source} needs an rclone profile to read")
            self.bucket, self.key = split_s3_url(source)
        self.profile = profile
        self._handle_ = None
        self._lock = threading.Lock()

    def _handle(self):
        with self._lock:
            if self._handle_ is None:
                if self.is_s3:
                    self._handle_ = get_obstore(self.profile, bucket=self.bucket)
                    if self._handle_ is None:
                        raise StorageConfigError(f"no rclone remote {self.profile!r}")
                else:
                    self._handle_ = os.open(self.source, os.O_RDONLY)
        return self._handle_

    def read(self, offset: int, length: int) -> bytes:
        handle = self._handle()
        if self.is_s3:
            import obstore

            for attempt in range(5):
                try:
                    data = bytes(
                        memoryview(
                            obstore.get_range(
                                handle, self.key, start=offset, length=length
                            )
                        )
                    )
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        else:
            chunks, got = [], 0
            while got < length:
                block = os.pread(handle, length - got, offset + got)
                if not block:
                    break
                chunks.append(block)
                got += len(block)
            data = b"".join(chunks)

        if len(data) != length:
            raise OSError(
                f"{self.source}: asked {length} bytes at {offset}, got {len(data)}"
            )
        return data

    def close(self) -> None:
        with self._lock:
            if self._handle_ is not None and not self.is_s3:
                os.close(self._handle_)
            self._handle_ = None

    def __enter__(self) -> "RangeReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def probe_source(source: str, profile: str | None) -> dict[str, Any]:
    """Read a store's header once: shape, dtype and the contiguous data offset.

    Refuses anything the offset arithmetic would silently corrupt. A chunked or
    compressed source has no single byte run per timestep, and a non-float32 or
    big-endian one would be copied verbatim into a dataset declaring otherwise --
    both produce plausible-looking garbage rather than an error, so they are
    checked here instead.
    """
    if source.startswith("s3://"):
        import fsspec

        bucket, key = split_s3_url(source)
        fs = fsspec.filesystem("s3", **get_storage_options(profile))
        handle = fs.open(f"{bucket}/{key}", "rb")
    else:
        handle = open(source, "rb")  # noqa: SIM115

    try:
        with h5py.File(handle, "r") as f:
            if "fields" not in f:
                raise SystemExit(f"{source} has no 'fields' dataset (keys={list(f)})")
            dataset = f["fields"]
            info = {
                "shape": tuple(int(n) for n in dataset.shape),
                "dtype": dataset.dtype,
                "offset": dataset.id.get_offset(),
                "chunks": dataset.chunks,
                "compression": dataset.compression,
            }
    finally:
        handle.close()

    if info["chunks"] is not None or info["compression"] is not None:
        raise SystemExit(
            f"{source} is chunked/compressed (chunks={info['chunks']},"
            f" compression={info['compression']}); ranged copying needs a"
            " contiguous, uncompressed source"
        )
    if info["offset"] is None:
        raise SystemExit(f"{source} dataset is not contiguous")
    if info["dtype"] != np.dtype("<f4"):
        raise SystemExit(
            f"{source} has dtype {info['dtype']}, expected little-endian float32"
        )
    if len(info["shape"]) != 4 or info["shape"][2:] != (NLAT, NLON):
        raise SystemExit(
            f"{source} has shape {info['shape']}, expected (time, channel,"
            f" {NLAT}, {NLON})"
        )
    return info


def index_runs(indices: list[int]) -> list[tuple[int, int, int]]:
    """Collapse indices into ``(out_channel, src_channel, count)`` runs, so a full
    copy is one read/write per timestep rather than one per channel."""
    runs: list[list[int]] = []
    for out_channel, src_channel in enumerate(indices):
        if runs and src_channel == runs[-1][1] + runs[-1][2]:
            runs[-1][2] += 1
        else:
            runs.append([out_channel, src_channel, 1])
    return [(a, b, c) for a, b, c in runs]


def copy_runs(
    reader: RangeReader,
    fd: int,
    *,
    runs: list[tuple[int, int, int]],
    src_base: int,
    dst_base: int,
) -> None:
    for out_channel, src_channel, count in runs:
        length = count * CHAN_BYTES
        data = reader.read(src_base + src_channel * CHAN_BYTES, length)
        pwrite_all(fd, data, dst_base + out_channel * CHAN_BYTES)

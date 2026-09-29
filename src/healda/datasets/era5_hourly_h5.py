# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read ERA5 HDF5 packs.

A pack is the 73varQ layout: one ``YYYY.h5`` per year holding a single ``fields``
dataset shaped ``(time, channel, lat, lon)``, alongside a ``metadata/data.json``
sidecar naming the grid, the ordered channels and the time step. Cadence comes
from that sidecar, so hourly and thinned (6-hourly) packs read the same way.

A channel list may span several packs, and a channel's years may be split across
them; each channel is read per year from the pack holding that year.
"""

import concurrent.futures
import json
import os
from collections.abc import Sequence

import h5py
import numpy as np
import pandas as pd


SIDECAR = "metadata/data.json"


class _Pack:
    """One pack directory: its grid metadata plus lazily opened year files."""

    def __init__(self, root: str):
        with open(os.path.join(root, SIDECAR)) as f:
            metadata = json.load(f)

        self.root = root
        self.h5_path = metadata["h5_path"]
        self.step = pd.Timedelta(hours=metadata["dhours"])
        self.lat = np.asarray(metadata["coords"]["lat"], dtype=np.float64)
        self.lon = np.asarray(metadata["coords"]["lon"], dtype=np.float64)
        self.channels = list(metadata["coords"]["channel"])
        self.years = {
            int(entry.name[:-3])
            for entry in os.scandir(root)
            if entry.name.endswith(".h5") and entry.name[:-3].isdigit()
        }

        # An h5py handle does not survive a fork, so years open on first read in
        # whichever dataloader worker process asks for them.
        self._files: dict[int, h5py.File] = {}
        self._nsteps: dict[int, int] = {}

    def fields(self, year: int):
        if year not in self._files:
            path = os.path.join(self.root, f"{year}.h5")
            self._files[year] = h5py.File(path, "r")
        return self._files[year][self.h5_path]

    def nsteps(self, year: int) -> int:
        if year not in self._nsteps:
            path = os.path.join(self.root, f"{year}.h5")
            with h5py.File(path, "r") as handle:
                self._nsteps[year] = int(handle[self.h5_path].shape[0])
        return self._nsteps[year]

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()


class Era5HourlyH5:
    """Random-access reader over one or more packs, in physical units.

    Packs are built one variable group at a time, so a channel list may span
    several of them, and a channel's years may be split across them (an archive
    root holding the older years beside a root holding newer ones). A channel is
    therefore resolved per year, to whichever pack holds that year. All packs must
    share a grid and time step.
    """

    def __init__(self, roots: str | Sequence[str], channels: list[str]):
        roots = [roots] if isinstance(roots, str) else list(roots)
        packs = [_Pack(root) for root in roots]
        grid = packs[0]
        for pack in packs[1:]:
            if (
                pack.step != grid.step
                or not np.array_equal(pack.lat, grid.lat)
                or not np.array_equal(pack.lon, grid.lon)
            ):
                raise ValueError(f"{pack.root} is not on the grid of {grid.root}")

        self.channels = list(channels)
        self.lat = grid.lat
        self.lon = grid.lon
        self.step = grid.step

        self._owners: dict[str, list[_Pack]] = {}
        for pack in packs:
            for name in pack.channels:
                self._owners.setdefault(name, []).append(pack)
        missing = [name for name in channels if name not in self._owners]
        if missing:
            raise ValueError(f"{roots} hold {sorted(self._owners)}, missing {missing}")

        # Only years every requested channel has a pack for: a year one channel is
        # missing is unreadable, and advertising it fails in _reads_for at load time.
        self.years = {year for pack in packs for year in pack.years}
        for name in self.channels:
            self.years &= {year for pack in self._owners[name] for year in pack.years}
        self._reads_cache: dict[int, list[tuple[_Pack, list[int], list[int]]]] = {}
        self._open: list[_Pack] = packs
        self.steps_by_year = {
            year: min(pack.nsteps(year) for pack, _, _ in self._reads_for(year))
            for year in self.years
        }

    def _reads_for(self, year: int) -> list[tuple[_Pack, list[int], list[int]]]:
        """Per pack, the store rows to read for ``year`` and the output rows they fill.

        h5py only accepts an increasing fancy index, hence reading in store order.
        """
        if year in self._reads_cache:
            return self._reads_cache[year]

        chosen: dict[str, _Pack] = {}
        for name in self.channels:
            for pack in self._owners[name]:
                if year in pack.years:
                    chosen[name] = pack
                    break
            else:
                raise ValueError(
                    f"no pack holds channel {name!r} for {year}; searched "
                    + ", ".join(pack.root for pack in self._owners[name])
                )

        reads = []
        for pack in self._open:
            pairs = sorted(
                (pack.channels.index(name), slot)
                for slot, name in enumerate(self.channels)
                if chosen[name] is pack
            )
            if pairs:
                reads.append(
                    (pack, [row for row, _ in pairs], [slot for _, slot in pairs])
                )
        self._reads_cache[year] = reads
        return reads

    def _time_index(self, time: pd.Timestamp) -> int:
        offset = time - pd.Timestamp(year=time.year, month=1, day=1)
        if offset % self.step != pd.Timedelta(0):
            raise ValueError(f"{time} is not aligned to step {self.step}")
        return offset // self.step

    def _read_one(self, pack, rows, slots, year: int, uniq, inv):
        fields = pack.fields(year)
        if uniq[0] < 0 or uniq[-1] >= fields.shape[0]:
            raise IndexError(
                f"year {year} hours {uniq[[0, -1]]} outside the "
                f"{fields.shape[0]} hours in {pack.root}"
            )
        # h5py releases the GIL during the actual HDF5 read, so threading across
        # packs overlaps their I/O wait instead of paying it out sequentially -
        # worthwhile whenever the channel list spans more than one pack.
        block = np.asarray(fields[uniq][:, rows])
        return slots, block[inv]

    def read(self, times) -> np.ndarray:
        """``(len(times), len(channels), nlat, nlon)`` float32."""
        times = pd.DatetimeIndex(times)
        nlat, nlon = int(self.lat.size), int(self.lon.size)
        out = np.empty((len(times), len(self.channels), nlat, nlon), dtype=np.float32)
        years = times.year.to_numpy()

        jobs: list[tuple] = []
        for year in np.unique(years):
            idxs = np.flatnonzero(years == year)
            hours = np.array(
                [self._time_index(times[int(i)]) for i in idxs], dtype=np.int64
            )
            uniq, inv = np.unique(hours, return_inverse=True)
            for pack, rows, slots in self._reads_for(int(year)):
                jobs.append((pack, rows, slots, int(year), idxs, uniq, inv))

        if len(jobs) <= 1:
            for pack, rows, slots, year, idxs, uniq, inv in jobs:
                slots, block = self._read_one(pack, rows, slots, year, uniq, inv)
                out[np.ix_(idxs, slots)] = block
            return out

        max_workers = min(8, len(jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                (idxs, pool.submit(self._read_one, pack, rows, slots, year, uniq, inv))
                for pack, rows, slots, year, idxs, uniq, inv in jobs
            ]
            for idxs, future in futures:
                slots, block = future.result()
                out[np.ix_(idxs, slots)] = block
        return out

    def close(self) -> None:
        for pack in self._open:
            pack.close()

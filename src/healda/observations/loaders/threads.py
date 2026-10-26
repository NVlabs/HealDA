# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow pool sizing and a persistent thread pool, shared by the observation loaders.

The observation path runs on these pools:

- The obs pool, ``thread_pool()``: ``WORKERS`` threads per process, 8 unless
  ``set_workers`` changes it (``DAModel.run_analysis`` sizes it from the CPU count). The loaders run one
  task per NNJA sensor, cycle file or UFS file on it from ``sel_time``, the conventional
  loader its conv-plevel expansion, and ``TransformV2`` one task per packed column.
- Arrow's CPU and I/O pools, sized once per process by ``configure_arrow_pools()``:
  Parquet reads and decodes issued from the obs pool's tasks run on them.
- ``NNJAWideLoader``'s read-ahead pool: one thread per sensor load, private to that
  call, that fetches the next row group while the calling obs-pool thread converts the
  current one. Submitting to the obs pool from a task already on it could deadlock.
- The pool of ``adapters.e2s_nnja.analysis_tables`` and ``analysis_tables_long``: by
  default the smaller of 32 and the CPUs, per call, adapting earth2studio frames into
  ``{cycle: table}``; the call returns before they finish.
  Loader tasks on the obs pool block on its futures when they look up a cycle, so it
  cannot be the obs pool: with every obs thread waiting, the adaptations queued behind
  them would never run.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

# Separate knobs despite sharing a value: Arrow's pools are per loading
# thread, so many dataloader workers in a rank multiply them, while the pool below is
# per process.
ARROW_CPU_THREADS = 4
ARROW_IO_THREADS = 4
# Concurrent tasks per process: a sensor for NNJA, a file for the replay, a tensor column
# for the transform's gather. 8 is better than 4 on GB300.
WORKERS = 8

_applied = False
_pools: dict[int, ThreadPoolExecutor] = {}


def configure_arrow_pools() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    pa.set_cpu_count(ARROW_CPU_THREADS)
    pa.set_io_thread_count(ARROW_IO_THREADS)


def set_workers(count: int) -> None:
    """Size this process's obs pool. A pool already built with another size is dropped,
    not shut down: tasks already on it finish there."""
    global WORKERS
    if count == WORKERS:
        return
    WORKERS = count
    _pools.pop(os.getpid(), None)


def thread_pool() -> ThreadPoolExecutor:
    # Persistent because asyncio.run tears down the default executor each sample, costing
    # ~0.5 ms to rebuild. Keyed by pid since threads do not survive fork into a worker.
    pid = os.getpid()
    if pid not in _pools:
        _pools[pid] = ThreadPoolExecutor(
            max_workers=WORKERS, thread_name_prefix="healda-obs"
        )
    return _pools[pid]

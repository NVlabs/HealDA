# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow pool sizing and a persistent thread pool, shared by the observation loaders."""

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


def thread_pool() -> ThreadPoolExecutor:
    # Persistent because asyncio.run tears down the default executor each sample, costing
    # ~0.5 ms to rebuild. Keyed by pid since threads do not survive fork into a worker.
    pid = os.getpid()
    if pid not in _pools:
        _pools[pid] = ThreadPoolExecutor(
            max_workers=WORKERS, thread_name_prefix="healda-obs"
        )
    return _pools[pid]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SIGTERM is recorded, not acted on: the training loop polls it at step boundaries
through ``distributed.should_stop``, so every rank stops at the same point."""

import signal

_received: int | None = None


def _record(signum, frame):
    global _received
    _received = signum


def install():
    signal.signal(signal.SIGTERM, _record)


def received() -> bool:
    return _received is not None


def clear():
    global _received
    _received = None


def ignore_in_worker(worker_id: int):
    """DataLoader ``worker_init_fn``: Slurm signals every process in the step, and a
    worker that dies on SIGTERM fails the main process's next fetch."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

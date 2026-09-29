# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch

NVTX_ENABLED = os.environ.get("HEALDA_NVTX", "0") == "1"
CUDA_TIMING_ENABLED = os.environ.get("HEALDA_CUDA_TIMING", "0") == "1"
CPU_TIMING_ENABLED = os.environ.get("HEALDA_CPU_TIMING", "0") == "1"


def nvtx(func=None, *, enabled: bool | None = None):
    def decorator(fn):
        use_nvtx = NVTX_ENABLED if enabled is None else enabled
        if not use_nvtx:
            return fn

        tag = fn.__module__ + ":" + fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _nvtx_range_impl(tag):
                return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@contextmanager
def _nvtx_range_impl(tag: str):
    torch.cuda.nvtx.range_push(tag)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_range(tag: str, enabled: bool | None = None):
    use_nvtx = NVTX_ENABLED if enabled is None else enabled
    if use_nvtx:
        return _nvtx_range_impl(tag)
    return nullcontext()


class _TimingAccumulator:
    """Shared per-tag time/call accumulation and reporting for region timers."""

    def __init__(self, label: str):
        self._label = label
        self._lock = threading.Lock()
        self.total_ms = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, tag: str, ms: float):
        with self._lock:
            self.total_ms[tag] += ms
            self.calls[tag] += 1

    def reset(self):
        with self._lock:
            self.total_ms.clear()
            self.calls.clear()

    def report(self) -> str:
        with self._lock:
            if not self.calls:
                return f"[{self._label}] no regions recorded"
            items = sorted(self.total_ms.items(), key=lambda kv: -kv[1])
            calls = dict(self.calls)
            grand_total = sum(self.total_ms.values())
        width = max(len(t) for t, _ in items)
        lines = [f"[{self._label}] mean time per region (since last reset):"]
        for tag, total in items:
            n = calls[tag]
            mean = total / n
            pct = 100.0 * total / grand_total if grand_total else 0.0
            lines.append(
                f"  {tag:<{width}}  {mean:8.3f} ms/call   total {total:9.1f} ms "
                f" {pct:5.1f}%  (n={n})"
            )
        lines.append(
            f"  {'TOTAL':<{width}}  {'':>8}            total {grand_total:9.1f} ms"
        )
        return "\n".join(lines)


class _CudaEventTimer:
    """Accumulating CUDA-event timer for measuring GPU time of named regions."""

    def __init__(self):
        self._pending = []  # (tag, start_event, end_event)
        self._acc = _TimingAccumulator("cuda-timing")

    @contextmanager
    def range(self, tag: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((tag, start, end))

    def flush(self):
        if not self._pending:
            return
        torch.cuda.synchronize()
        for tag, start, end in self._pending:
            self._acc.add(tag, start.elapsed_time(end))
        self._pending.clear()

    def reset(self):
        self._pending.clear()
        self._acc.reset()

    def report(self) -> str:
        self.flush()
        return self._acc.report()


class _CpuTimer:
    """Accumulating wall-clock timer for measuring CPU time of named regions.

    Records each region immediately (no flush needed). Instances are per-process,
    so to capture work done in DataLoader worker subprocesses, report from inside
    the worker (e.g. via ``worker_init_fn``/atexit) or run with ``num_workers=0``.
    """

    def __init__(self, label: str = "cpu-timing"):
        self._acc = _TimingAccumulator(label)

    @contextmanager
    def range(self, tag: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._acc.add(tag, (time.perf_counter() - start) * 1e3)

    def reset(self):
        self._acc.reset()

    def report(self) -> str:
        return self._acc.report()


_CUDA_TIMER = _CudaEventTimer()
_CPU_TIMERS: dict[str, _CpuTimer] = {}
_CPU_TIMERS_LOCK = threading.Lock()


def _cpu_timer(group: str) -> _CpuTimer:
    timer = _CPU_TIMERS.get(group)
    if timer is not None:
        return timer
    with _CPU_TIMERS_LOCK:
        timer = _CPU_TIMERS.get(group)
        if timer is None:
            timer = _CpuTimer(group)
            _CPU_TIMERS[group] = timer
        return timer


def cuda_timing_range(tag: str, enabled: bool | None = None):
    """GPU-time a code region without synchronizing until report/flush.

    No-op unless ``enabled`` is true or ``HEALDA_CUDA_TIMING=1`` is set.
    """
    use_timing = CUDA_TIMING_ENABLED if enabled is None else enabled
    if use_timing:
        return _CUDA_TIMER.range(tag)
    return nullcontext()


def cuda_timing_report() -> str:
    return _CUDA_TIMER.report()


def cuda_timing_reset():
    _CUDA_TIMER.reset()


def cpu_timing_range(
    tag: str, enabled: bool | None = None, *, group: str = "cpu-timing"
):
    """Wall-clock-time a CPU code region.

    No-op unless ``enabled`` is true or ``HEALDA_CPU_TIMING=1`` is set.
    ``group`` keeps concurrent work on separate reports so a TOTAL stays
    meaningful; a thread that times its own regions passes its own group.
    """
    use_timing = CPU_TIMING_ENABLED if enabled is None else enabled
    if use_timing:
        return _cpu_timer(group).range(tag)
    return nullcontext()


def _all_cpu_timers() -> list[_CpuTimer]:
    # Copy under the lock: a thread timing its own group can register mid-iteration.
    with _CPU_TIMERS_LOCK:
        return list(_CPU_TIMERS.values())


def cpu_timing_report(group: str | None = "cpu-timing") -> str:
    if group is None:
        timers = _all_cpu_timers()
        if not timers:
            return "[cpu-timing] no regions recorded"
        return "\n".join(timer.report() for timer in timers)
    return _cpu_timer(group).report()


def cpu_timing_reset(group: str | None = None):
    if group is None:
        for timer in _all_cpu_timers():
            timer.reset()
        return
    _cpu_timer(group).reset()

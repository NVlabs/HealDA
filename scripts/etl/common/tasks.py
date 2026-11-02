# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Array-job task helpers: time-block partitioning, per-task event logging, and
status aggregation. Used by the ETL build scripts via ``WorkManager``."""

import json
import os
import socket
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Time-block partitioning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeBlock:
    index: int
    start: int
    stop: int

    def slice(self) -> slice:
        return slice(self.start, self.stop)


def make_time_blocks(n_times: int, block_size: int) -> list[TimeBlock]:
    return [
        TimeBlock(index=i, start=start, stop=min(start + block_size, n_times))
        for i, start in enumerate(range(0, n_times, block_size))
    ]


def select_time_blocks(
    blocks: list[TimeBlock],
    block_start: int | None = None,
    block_stop: int | None = None,
) -> list[TimeBlock]:
    start = 0 if block_start is None else block_start
    stop = len(blocks) if block_stop is None else block_stop
    if start < 0 or stop < start or stop > len(blocks):
        raise ValueError(
            f"Invalid block range [{start}, {stop}) for {len(blocks)} available blocks"
        )
    return blocks[start:stop]


def make_times(start_date: str, end_date: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start_date, end_date, freq=freq)


def task_range(tasks: list[int], start: int | None, stop: int | None) -> list[int]:
    return tasks[start:stop]


# ---------------------------------------------------------------------------
# Per-task event logging
# ---------------------------------------------------------------------------
def append_task_event(events_dir: str | Path, event: dict) -> None:
    path = Path(events_dir)
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.time(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **event,
    }
    event_path = path / f"{os.getpid()}.jsonl"
    with open(event_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def task_event_context(
    events_dir: str | Path,
    *,
    task_id: int,
    stage: str,
    metadata: dict | None = None,
) -> Iterator[None]:
    base = {"task_id": task_id, "stage": stage, **(metadata or {})}
    start = time.time()
    append_task_event(events_dir, {"event": "started", **base})
    try:
        yield
    except Exception as exc:
        append_task_event(
            events_dir,
            {
                "event": "failed",
                "duration_seconds": time.time() - start,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                **base,
            },
        )
        raise
    append_task_event(
        events_dir,
        {"event": "succeeded", "duration_seconds": time.time() - start, **base},
    )


def read_task_events(events_dir: str | Path) -> list[dict]:
    events = []
    path = Path(events_dir)
    if not path.exists():
        return events
    for event_path in sorted(path.glob("*.jsonl")):
        with open(event_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def latest_task_events(events: list[dict]) -> dict[int, dict]:
    latest = {}
    for event in events:
        task_id = int(event["task_id"])
        if task_id not in latest or event["time"] >= latest[task_id]["time"]:
            latest[task_id] = event
    return latest


# ---------------------------------------------------------------------------
# Task status aggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskStatus:
    expected: set[int]
    completed: set[int]
    valid: set[int]
    attempted: set[int]
    failed: set[int]
    running: set[int]
    stale: set[int]
    repair: set[int]
    latest_events: dict[int, dict]

    def summary(self) -> dict[str, int | float]:
        summary = {
            "expected_tasks": len(self.expected),
            "attempted_tasks": len(self.attempted),
            "failed_tasks": len(self.failed),
            "running_tasks": len(self.running),
            "stale_running_tasks": len(self.stale),
            "completed_log_tasks": len(self.expected & self.completed),
            "valid_data_tasks": len(self.expected & self.valid),
            "log_without_valid_data": len(
                (self.expected & self.completed) - self.valid
            ),
            "valid_data_without_log": len(
                (self.expected & self.valid) - self.completed
            ),
            "repair_tasks": len(self.repair),
        }
        durations = sorted(
            float(event["duration_seconds"])
            for task, event in self.latest_events.items()
            if task in self.expected
            and event.get("event") in {"succeeded", "failed"}
            and "duration_seconds" in event
        )
        if durations:
            summary.update(
                {
                    "duration_count": len(durations),
                    "duration_mean_seconds": sum(durations) / len(durations),
                    "duration_median_seconds": _quantile(durations, 0.5),
                    "duration_p90_seconds": _quantile(durations, 0.9),
                    "duration_max_seconds": durations[-1],
                }
            )
        return summary

    def slowest_tasks(self, limit: int = 10) -> list[dict]:
        events = [
            event
            for task, event in self.latest_events.items()
            if task in self.expected and "duration_seconds" in event
        ]
        return sorted(
            events, key=lambda event: float(event["duration_seconds"]), reverse=True
        )[:limit]


def _quantile(values: list[float], q: float) -> float:
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def read_completed_tasks(jobs_dir: str | Path) -> set[int]:
    completed = set()
    path = Path(jobs_dir)
    if not path.exists():
        return completed
    for log_path in path.glob("*.log"):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(int(line))
    return completed


def compute_task_status(
    *,
    expected: set[int],
    jobs_dir: str | Path,
    events_dir: str | Path | None = None,
    valid: set[int] | None = None,
    stale_seconds: float = 3600,
) -> TaskStatus:
    completed = read_completed_tasks(jobs_dir)
    valid = valid or set()
    latest_events = (
        latest_task_events(read_task_events(events_dir)) if events_dir else {}
    )

    attempted = {task for task in expected if task in latest_events}
    failed = {
        task
        for task in expected
        if latest_events.get(task, {}).get("event") == "failed"
    }
    running = {
        task
        for task in expected
        if latest_events.get(task, {}).get("event") == "started"
    }
    now = time.time()
    stale = {
        task
        for task in running
        if now - float(latest_events[task]["time"]) > stale_seconds
    }
    repair = (expected - completed) | (expected - valid) | failed | stale
    return TaskStatus(
        expected=expected,
        completed=completed,
        valid=valid,
        attempted=attempted,
        failed=failed,
        running=running,
        stale=stale,
        repair=repair,
        latest_events=latest_events,
    )

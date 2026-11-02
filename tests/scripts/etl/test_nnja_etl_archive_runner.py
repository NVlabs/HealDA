# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A crashed worker must cost its own task, not the whole run.

When a worker dies abnormally every outstanding future raises
``BrokenProcessPool``. Workers are killed with ``os._exit``, since a raised exception does not break the pool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.etl.nnja.archive_common.archive_runner import run_tasks


@dataclass(frozen=True)
class _Task:
    index: int
    root: str
    mode: str = "ok"

    @property
    def output_path(self) -> Path:
        return Path(self.root) / f"{self.index:03d}.out"


def _process(task: _Task):
    if task.mode == "exit":
        os._exit(1)  # abnormal exit: no exception, worker just vanishes
    if task.mode == "publish_then_exit":
        if task.output_path.exists():
            raise FileExistsError(task.output_path)
        task.output_path.write_text(str(task.index))
        os._exit(1)
    if task.mode == "raise":
        raise ValueError(f"task {task.index} failed normally")
    task.output_path.write_text(str(task.index))
    return task.index


def _recover(task: _Task) -> int:
    return int(task.output_path.read_text())


def _record_failure(error_root: Path, task: _Task, exc: BaseException) -> Path:
    error_root = Path(error_root)
    error_root.mkdir(parents=True, exist_ok=True)
    marker = error_root / f"{task.index:03d}.error.json"
    marker.write_text(
        json.dumps({"index": task.index, "error": f"{type(exc).__name__}: {exc}"})
    )
    return marker


def test_worker_death_costs_only_its_own_task(tmp_path: Path):
    """One worker dying must not fail the other tasks."""
    root = tmp_path / "out"
    root.mkdir()
    errors = tmp_path / "errors"
    tasks = [_Task(i, str(root), "exit" if i == 5 else "ok") for i in range(20)]

    results = run_tasks(
        tasks,
        process=_process,
        max_workers=4,
        error_root=errors,
        continue_on_error=True,
        on_failure=_record_failure,
        describe=lambda t: f"task {t.index}",
    )

    assert sorted(results) == [i for i in range(20) if i != 5]
    written = sorted(int(p.stem) for p in root.glob("*.out"))
    assert written == [
        i for i in range(20) if i != 5
    ], "every healthy task must still run"

    markers = sorted(errors.glob("*.error.json"))
    assert (
        len(markers) == 1
    ), f"expected only the crashing task to fail, got {len(markers)}"
    assert json.loads(markers[0].read_text())["index"] == 5


def test_ordinary_exception_still_fails_only_that_task(tmp_path: Path):
    root = tmp_path / "out"
    root.mkdir()
    errors = tmp_path / "errors"
    tasks = [_Task(i, str(root), "raise" if i == 3 else "ok") for i in range(8)]

    results = run_tasks(
        tasks,
        process=_process,
        max_workers=3,
        error_root=errors,
        continue_on_error=True,
        on_failure=_record_failure,
        describe=lambda t: f"task {t.index}",
    )

    assert sorted(results) == [i for i in range(8) if i != 3]
    assert len(list(errors.glob("*.error.json"))) == 1


def test_two_separate_crashes_are_both_isolated(tmp_path: Path):
    """Recovery must survive more than one bad task in a run."""
    root = tmp_path / "out"
    root.mkdir()
    errors = tmp_path / "errors"
    bad = {4, 11}
    tasks = [_Task(i, str(root), "exit" if i in bad else "ok") for i in range(16)]

    results = run_tasks(
        tasks,
        process=_process,
        max_workers=4,
        error_root=errors,
        continue_on_error=True,
        on_failure=_record_failure,
        describe=lambda t: f"task {t.index}",
    )

    assert sorted(results) == [i for i in range(16) if i not in bad]
    assert sorted(
        int(p.stem.split(".")[0]) for p in errors.glob("*.error.json")
    ) == sorted(bad)


def test_crash_without_continue_on_error_still_raises(tmp_path: Path):
    """Without continue_on_error the run aborts."""
    root = tmp_path / "out"
    root.mkdir()
    tasks = [_Task(i, str(root), "exit" if i == 2 else "ok") for i in range(6)]

    with pytest.raises(RuntimeError):
        run_tasks(
            tasks, process=_process, max_workers=2, describe=lambda t: f"task {t.index}"
        )


def test_submit_time_pool_death_loses_no_task(tmp_path: Path, monkeypatch):
    """A pool that dies at submit() must not lose the task being submitted.

    The executor's sixth submit() raises BrokenProcessPool, standing in for a
    pool broken by a neighbouring worker between rounds.
    """
    from concurrent.futures.process import BrokenProcessPool

    from scripts.etl.nnja.archive_common import archive_runner as runner

    root = tmp_path / "out"
    root.mkdir()
    errors = tmp_path / "errors"
    tasks = [_Task(i, str(root)) for i in range(12)]

    real_executor = runner.ProcessPoolExecutor
    state = {"submits": 0, "died": False}

    class FlakyExecutor(real_executor):
        def submit(self, fn, *args, **kwargs):
            if not state["died"] and state["submits"] == 5:
                state["died"] = True
                raise BrokenProcessPool("synthetic submit-time pool death")
            state["submits"] += 1
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FlakyExecutor)

    results = run_tasks(
        tasks,
        process=_process,
        max_workers=3,
        error_root=errors,
        continue_on_error=True,
        on_failure=_record_failure,
        describe=lambda t: f"task {t.index}",
    )

    assert state["died"], "the synthetic pool death never fired; test proves nothing"
    assert sorted(results) == list(range(12))
    assert sorted(int(p.stem) for p in root.glob("*.out")) == list(range(12))
    assert (
        list(errors.glob("*.error.json")) == []
    ), "no task should be recorded as failed"


def test_output_published_before_worker_death_is_recovered(tmp_path: Path):
    tasks = [_Task(i, str(tmp_path), "publish_then_exit") for i in range(2)]

    results = run_tasks(
        tasks, process=_process, max_workers=2, recover_completed=_recover
    )

    assert results == (0, 1)


def test_duplicate_output_paths_are_rejected(tmp_path: Path):
    task = _Task(0, str(tmp_path))

    with pytest.raises(ValueError, match="duplicate output"):
        run_tasks([task, task], process=_process, max_workers=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor-agnostic task scheduling for NNJA archive ETL.

Tasks need distinct ``output_path`` values; the caller supplies how to process
one and how to initialize a worker, including its thread budgets.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import traceback
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = ["exception_traceback", "run_tasks", "write_json_atomic", "write_text_atomic"]

# Pool breaks a task may be caught in before it is re-run alone.
_RETRY_LIMIT = 2


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json_atomic(
    path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = True
) -> None:
    write_text_atomic(
        path, json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    )


def exception_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def run_tasks(
    tasks: Sequence[Any],
    *,
    process: Callable[[Any], Any],
    max_workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    progress: Callable[[Any, int, int], None] | None = None,
    describe: Callable[[Any], str] = repr,
    error_root: Path | None = None,
    continue_on_error: bool = False,
    on_failure: Callable[[Path, Any, BaseException], Path] | None = None,
    on_success: Callable[[Path, Any], None] | None = None,
    recover_completed: Callable[[Any], Any] | None = None,
) -> tuple[Any, ...]:
    """Run one task per unit of work on a persistent spawn pool.

    ``process`` must be a module-level callable so spawned workers can import
    it. Results are returned in submission order regardless of completion
    order.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if continue_on_error and error_root is None:
        raise ValueError("continue_on_error requires error_root")
    if continue_on_error and on_failure is None:
        raise ValueError("continue_on_error requires on_failure")

    submitted = tuple(tasks)
    if not submitted:
        return ()
    outputs = [Path(task.output_path).resolve() for task in submitted]
    if len(outputs) != len(set(outputs)):
        raise ValueError("tasks contain duplicate output paths")

    def _handle_failure(task: Any, exc: BaseException, completed: int) -> None:
        if not continue_on_error:
            raise RuntimeError(f"{describe(task)} ETL failed") from exc
        assert error_root is not None and on_failure is not None
        failure_path = on_failure(error_root, task, exc)
        print(
            f"[{completed}/{len(submitted)}] FAILED {describe(task)} -> {failure_path}",
            flush=True,
        )

    if max_workers == 1:
        if initializer is not None:
            initializer(*initargs)
        results: list[Any] = []
        for completed, task in enumerate(submitted, start=1):
            try:
                result = process(task)
            except Exception as exc:
                _handle_failure(task, exc, completed)
                continue
            results.append(result)
            if on_success is not None and error_root is not None:
                on_success(error_root, task)
            if progress is not None:
                progress(result, completed, len(submitted))
        return tuple(results)

    context = multiprocessing.get_context("spawn")
    ordered: list[Any | None] = [None] * len(submitted)
    pending: deque[tuple[int, Any]] = deque(enumerate(submitted))
    attempts: Counter[int] = Counter()
    completed = 0

    def _record(index: int, task: Any, result: Any) -> None:
        nonlocal completed
        ordered[index] = result
        completed += 1
        if on_success is not None and error_root is not None:
            on_success(error_root, task)
        if progress is not None:
            progress(result, completed, len(submitted))

    def _fail(task: Any, exc: BaseException) -> None:
        nonlocal completed
        completed += 1
        _handle_failure(task, exc, completed)

    def _new_pool(workers: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        )

    def _already_done(task: Any) -> Any | None:
        if recover_completed is None or not Path(task.output_path).exists():
            return None
        try:
            return recover_completed(task)
        except Exception:
            return None

    def _run_isolated(index: int, task: Any) -> None:
        """Run one task alone. If it still kills the pool, it is the culprit."""
        recovered = _already_done(task)
        if recovered is not None:
            _record(index, task, recovered)
            return
        pool = _new_pool(1)
        try:
            future = pool.submit(process, task)
            result = future.result()
        except Exception as exc:
            _fail(task, exc)
        else:
            _record(index, task, result)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # A worker that dies abnormally (segfault, OOM kill) breaks the executor:
    # outstanding futures and later submits raise BrokenProcessPool. A task
    # leaves `pending` only after submit succeeds, and a broken pool requeues
    # its in-flight tasks and is rebuilt. After _RETRY_LIMIT rounds a task runs
    # alone, since the executor cannot say which worker died.
    while pending:
        index, task = pending[0]
        if attempts[index] >= _RETRY_LIMIT:
            pending.popleft()
            _run_isolated(index, task)
            continue

        pool = _new_pool(max_workers)
        outstanding: dict[Any, tuple[int, Any]] = {}
        window = max_workers * 2
        broke = False
        try:
            while True:
                while pending and len(outstanding) < window:
                    next_index, next_task = pending[0]
                    if attempts[next_index] >= _RETRY_LIMIT:
                        break
                    if attempts[next_index]:
                        recovered = _already_done(next_task)
                        if recovered is not None:
                            pending.popleft()
                            _record(next_index, next_task, recovered)
                            continue
                    try:
                        future = pool.submit(process, next_task)
                    except (BrokenProcessPool, RuntimeError):
                        # The task is still at the head of `pending`.
                        broke = True
                        break
                    pending.popleft()
                    attempts[next_index] += 1
                    outstanding[future] = (next_index, next_task)
                if broke or not outstanding:
                    break

                done, _ = wait(list(outstanding), return_when=FIRST_COMPLETED)
                for future in done:
                    done_index, done_task = outstanding.pop(future)
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        pending.appendleft((done_index, done_task))
                        broke = True
                        break
                    except Exception as exc:
                        _fail(done_task, exc)
                    else:
                        _record(done_index, done_task, result)
                if broke:
                    break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if broke:
            # Requeue everything still in flight, in submission order.
            for still_index, still_task in reversed(sorted(outstanding.values())):
                pending.appendleft((still_index, still_task))

    return tuple(result for result in ordered if result is not None)

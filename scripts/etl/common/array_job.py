# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import os
from typing import Any, Callable
import random
import logging
from enum import Enum
from pathlib import Path

from scripts.etl.common.tasks import read_completed_tasks

logger = logging.getLogger(__name__)


def state_dir_for(root: str, output: str, kind: str) -> str:
    # Journals live under one root, keyed on the output's path below it: split
    # subdirectories share the root, and ARCHIVE/{train,val}/2018.h5 stay distinct.
    output_path = Path(output).resolve()
    try:
        key = output_path.relative_to(Path(root).resolve()).with_suffix("")
    except ValueError:
        key = Path(output_path.stem)
    return str(Path(root) / "state" / f"{'-'.join(key.parts)}.{kind}")


class LaunchMode(Enum):
    """Distributed launch backend detection mode."""

    SLURM = "slurm"  # SLURM job arrays
    TORCH = "torch"  # torch.distributed
    MANUAL = "manual"  # Explicit rank/world_size for single node, multi-process
    AUTO = "auto"  # Auto-detect, defaulting to SLURM when slurm and torch are present


class WorkManager:
    def __init__(
        self,
        path: str = "",
        *,
        launch_mode: LaunchMode = LaunchMode.AUTO,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        """
        Args:
            path: Directory for task completion logs
            launch_mode: Backend detection mode
            rank: Manual rank (required for MANUAL mode)
            world_size: Manual world size (required for MANUAL mode)
        """
        self.job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "0")

        if launch_mode == LaunchMode.MANUAL:
            if rank is None or world_size is None:
                raise ValueError("MANUAL mode requires rank and world_size")
            self.rank, self.world_size = rank, world_size

        elif launch_mode == LaunchMode.TORCH:
            self.rank, self.world_size = self._init_torch()

        elif launch_mode == LaunchMode.SLURM:
            self.rank, self.world_size = self._init_slurm()

        elif launch_mode == LaunchMode.AUTO:
            self.rank, self.world_size = self._auto_detect()

        else:
            raise ValueError(f"Unknown launch mode: {launch_mode}")

        logger.info(
            f"WorkManager(mode={launch_mode}): rank={self.rank}, world_size={self.world_size}"
        )

        if path:
            os.makedirs(path, exist_ok=True)
        self.path = path

    def _init_torch(self) -> tuple[int, int]:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )

    def _init_slurm(self) -> tuple[int, int]:
        slurm_rank = int(os.environ.get("SLURM_PROCID", 0))
        slurm_world = int(os.environ.get("SLURM_NPROCS", 1))

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", 1))

        array_rank = task_id - task_min
        array_world = task_max - task_min + 1

        rank = array_rank * slurm_world + slurm_rank
        world = array_world * slurm_world
        return rank, world

    def _auto_detect(self) -> tuple[int, int]:
        has_torch = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        has_slurm = "SLURM_PROCID" in os.environ
        has_slurm_array = "SLURM_ARRAY_TASK_ID" in os.environ

        # Priority order:
        # 1. SLURM array jobs: Must use SLURM logic to compute global rank across array
        #    (RANK/WORLD_SIZE from dist.init() are only per-task, not global)
        # 2. torch.distributed: Use when set by torchrun or explicit dist.init()
        #    (single-task srun + dist.init() sets RANK correctly for that task)
        # 3. SLURM non-array: Fallback for pure SLURM without torch

        if has_slurm_array and has_slurm:
            if has_torch:
                logger.info(
                    "SLURM array job detected. Using SLURM to compute global rank "
                    "across array tasks (RANK/WORLD_SIZE are per-task only)."
                )
            return self._init_slurm()

        if has_torch:
            return self._init_torch()

        if has_slurm:
            return self._init_slurm()

        # Single process fallback
        return 0, 1

    def split(self, tasks: list[int], seed: int = 0):
        # seed the random number generator with the job id
        # so that tasks are reshuffled across the array when a new job is
        # started
        rng = random.Random(self.job_id)
        tasks = tasks.copy()
        rng.shuffle(tasks)
        return tasks[self.rank :: self.world_size]

    def task_done(self, task: int):
        if not self.path:
            return

        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        filename = (
            f"{array_task_id}.{self.rank}.log"
            if array_task_id is not None
            else f"{self.rank}.log"
        )
        logger.info(f"marking {task} as done")
        with open(os.path.join(self.path, filename), "a") as f:
            print(task, file=f)

    def completed_tasks(self) -> set[int]:
        """Tasks journalled done by any rank."""
        return read_completed_tasks(self.path)

    def pending(
        self, tasks: list[int], *, extra_done: set[int] | None = None
    ) -> list[int]:
        """This rank's share of ``tasks``, minus anything already journalled done.

        ``extra_done`` is work known complete by inspection rather than the journal.
        """
        done = self.completed_tasks()
        if extra_done:
            done = done | extra_done
        return [task for task in self.split(tasks) if task not in done]

    def run_parallel(
        self,
        func: Callable[[int, Any], None],
        tasks: list[int],
        *,
        config: Any,
        workers: int,
        log_every: int = 25,
    ) -> list[int]:
        """Run ``func(task, config)`` over ``tasks`` in a process pool; return failures.

        Every task that succeeds is journalled even if a sibling fails: the pool
        drains on exit either way, so raising from the loop would do the remaining
        work and record none of it. Callers decide what a non-empty return means.
        """
        failed: list[int] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(func, task, config): task for task in tasks}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                task = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("task %s failed", task)
                    failed.append(task)
                else:
                    self.task_done(task)
                if i % log_every == 0 or i == len(tasks):
                    logger.info("rank=%s %s/%s tasks", self.rank, i, len(tasks))
        return failed

    def map(self, func: Callable[[int], None], tasks: list[int]):
        completed_tasks = self.completed_tasks()
        my_tasks = self.split(tasks)
        logger.info(
            f"Starting tasks: {len(completed_tasks)} total tasks already completed globally of {len(tasks)}. Have {len(my_tasks)} tasks to run on rank {self.rank} of {self.world_size}."
        )
        for task in my_tasks:
            if task in completed_tasks:
                logger.info(f"{task} already done. skipping.")
                continue

            func(task)
            self.task_done(task)

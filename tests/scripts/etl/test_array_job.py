# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import scripts.etl.common.array_job


def test_work_manager(tmp_path):
    m = scripts.etl.common.array_job.WorkManager(path=tmp_path)
    m.task_done(1)
    m.task_done(1)
    m.task_done(3)
    assert m.completed_tasks() == {1, 3}

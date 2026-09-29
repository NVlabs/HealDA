# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
import time

from healda.utils.profiling import _cpu_timer, cpu_timing_range, cpu_timing_reset


def test_cpu_timing_groups_do_not_mix():
    cpu_timing_reset()
    with cpu_timing_range("caller", enabled=True):
        time.sleep(0.01)
    with cpu_timing_range("io", enabled=True, group="write"):
        time.sleep(0.02)
    assert set(_cpu_timer("cpu-timing")._acc.total_ms) == {"caller"}
    assert set(_cpu_timer("write")._acc.total_ms) == {"io"}
    cpu_timing_reset()


def test_cpu_timing_add_is_thread_safe():
    cpu_timing_reset(group="write")

    def worker():
        for _ in range(50):
            with cpu_timing_range("t", enabled=True, group="write"):
                pass

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert _cpu_timer("write")._acc.calls["t"] == 200
    cpu_timing_reset()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import signal

from healda.utils import distributed, signals


def test_sigterm_is_recorded_not_raised():
    signals.install()
    try:
        assert not distributed.should_stop()
        os.kill(os.getpid(), signal.SIGTERM)
        assert signals.received()
        assert distributed.should_stop()
    finally:
        signals.clear()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

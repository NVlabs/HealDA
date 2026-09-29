# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for catching unix signals and gracefully exiting

Usage:
```
import healda.utils.signals
import signal

# now can catch signals with exceptions
signal.signal(healda.utils.signals.handler)


def do_stuff():
    do_optional_stuff()
    with healda.utils.signals.finish_before_quitting():
        do_stuff_i_need_to_finish()

try:
    do_stuff()
except healda.utils.signals.QuitEarly:
    cleanup()

```

"""


class QuitEarly(Exception):
    depth = 0
    quit_requested = False


def finish_before_quitting(func):
    """If signal caught defer quitting until the wrapped line of code completes

    Used to handle sensitive code blocks
    """

    def newfunc(*args, **kwargs):
        QuitEarly.depth += 1
        func(*args, **kwargs)
        QuitEarly.depth -= 1

        if QuitEarly.quit_requested and QuitEarly.depth == 0:
            QuitEarly.quit_requested = False
            raise QuitEarly()

    return newfunc


def handler(signum, frame):
    if QuitEarly.depth == 0:
        raise QuitEarly(signum, frame)
    else:
        QuitEarly.quit_requested = True

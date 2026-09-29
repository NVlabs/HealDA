# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named hooks the training loop resolves at run time.

Plain dicts, filled by assignment from whatever supplies the implementation. The loop
looks a provider up by name and fails with the list of registered names, so a missing
one is reported rather than silently substituted.
"""

from typing import Any

# name -> object exposing get_dataset / get_batch_info / ObsDataset
DATASET_PROVIDERS: dict[str, Any] = {}


def _get(registry: dict[str, Any], name: str, kind: str) -> Any:
    try:
        return registry[name]
    except KeyError:
        raise RuntimeError(
            f"no {kind} registered under {name!r}; registered: {sorted(registry)}"
        ) from None


def dataset_provider(name: str) -> Any:
    return _get(DATASET_PROVIDERS, name, "dataset provider")

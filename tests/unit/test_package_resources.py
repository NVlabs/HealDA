# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every data path the package computes at import time resolves, and ships.

A module-level ``Path(__file__).parent / ...`` does not move when its module moves,
and nothing routine catches that: the import succeeds, ``compileall`` is clean, ruff
is clean, and the failure only appears when something reads the file. The release
restructure broke four of these.

These tests **discover** the paths rather than listing them, by importing every
module under ``healda`` and inspecting its module-level Path attributes. A new
resource is covered the moment it is added; nobody has to remember to register it.
"""

import importlib
import pathlib
import pkgutil

import pytest

import healda

PACKAGE_ROOT = pathlib.Path(healda.__file__).parent


# Built inside a function or __post_init__, so walk_packages cannot see them.
FUNCTION_LOCAL = {
    "observations/normalizations": PACKAGE_ROOT / "observations" / "normalizations",
}


def _discover():
    """(module, attribute, path) for every module-level Path under healda."""
    found = []
    for info in pkgutil.walk_packages(healda.__path__, "healda."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue  # optional dependency; other suites skip these too
        for attr, value in vars(module).items():
            if attr.startswith("__") or not isinstance(value, pathlib.PurePath):
                continue
            # Only paths the package owns; anything else is caller-supplied.
            if PACKAGE_ROOT in pathlib.Path(value).parents:
                found.append((info.name, attr, pathlib.Path(value)))
    return sorted(set(found))


DISCOVERED = _discover() + [
    ("<function-local>", name, path) for name, path in FUNCTION_LOCAL.items()
]


def test_discovery_finds_the_known_resources():
    """Guard the guard: if this finds nothing, the other tests pass vacuously."""
    names = {attr for _, attr, _ in DISCOVERED}
    assert {"NORMALIZATION_DIR", "NORMALIZATIONS_DIR", "MASKS_DIR"} <= names
    assert len(DISCOVERED) >= 6


@pytest.mark.parametrize(
    "module,attr,path", DISCOVERED, ids=[f"{m}.{a}" for m, a, _ in DISCOVERED]
)
def test_package_path_exists(module, attr, path):
    assert path.exists(), f"{module}.{attr} -> {path}"

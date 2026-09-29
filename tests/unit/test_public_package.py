# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "healda",
        "healda.cli",
        "healda.config.models",
        "healda.observations",
        "healda.observations.protocols",
    ],
)
def test_public_modules_import(module_name):
    assert importlib.import_module(module_name).__name__ == module_name


def test_training_public_modules_import():
    pytest.importorskip("torch")
    module_names = [
        "healda.training.checkpoint",
        "healda.cli.train",
        "healda.datasets.da.packed_da_dataset",
        "healda.datasets.da.tasks",
        "healda.datasets.da.transform",
        "healda.models",
        "healda.observations.loaders",
        "healda.training.loop",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name


def test_packaged_modules_do_not_import_removed_namespaces():
    package_root = Path(__file__).parents[2] / "src" / "healda"
    forbidden_prefixes = (
        "catalog",
        "models",
        "private",
        "scripts",
        "train",
        "healda.datasets.legacy",
        "healda.etl",
    )
    violations = []

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        for module_name in imported_modules:
            if any(
                module_name == prefix or module_name.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            ):
                violations.append(
                    f"{path.relative_to(package_root)} imports {module_name}"
                )

    assert violations == []

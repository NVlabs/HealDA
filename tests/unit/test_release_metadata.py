# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tomllib
from pathlib import Path

import healda

REPO = Path(__file__).resolve().parents[2]


def _citation_field(name: str) -> str | None:
    match = re.search(rf"^{name}: (.+)$", (REPO / "CITATION.cff").read_text(), re.M)
    return match.group(1).strip() if match else None


def test_versions_agree():
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert _citation_field("version") == version
    assert healda.__version__ == version


def test_release_version_has_dated_changelog_section():
    version = healda.__version__
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return
    changelog = (REPO / "CHANGELOG.md").read_text()
    match = re.search(rf"^## \[{re.escape(version)}\] - (\S+)$", changelog, re.M)
    assert match, f"CHANGELOG.md has no ## [{version}] section"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", match.group(1)), match.group(1)
    assert _citation_field("date-released") == match.group(1)

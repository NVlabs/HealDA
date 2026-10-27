# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hooks: the home page is the repository README.

Links in the README are relative to the repository root. Targets under docs/ become
site pages; everything else points at the file on GitHub. Images under docs/ are
served from the site.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GITHUB_BLOB = "https://github.com/NVlabs/HealDA/blob/main/"

# Link text may itself be an image, as in a badge: [![alt](img)](target).
LINK_RE = re.compile(r"(?<!!)\[((?:[^\[\]]|!\[[^\]]*\]\([^)]*\))*)\]\(([^)\s]+)\)")
DOCS_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()docs/")


def _rewrite(match: re.Match) -> str:
    text, target = match.group(1), match.group(2)
    if re.match(r"^(?:[a-z]+:|#|/)", target):
        return match.group(0)
    if target.startswith("docs/"):
        return f"[{text}]({target[len('docs/'):]})"
    return f"[{text}]({GITHUB_BLOB}{target})"


def on_page_markdown(markdown: str, page, config, files) -> str:
    if page.file.src_uri != "index.md":
        return markdown
    page.edit_url = f"{config.repo_url}/edit/main/README.md"
    markdown = LINK_RE.sub(_rewrite, README.read_text(encoding="utf-8"))
    return DOCS_IMAGE_RE.sub(r"\1", markdown)

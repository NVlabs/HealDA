# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check (and optionally fix) the SPDX license header on every tracked source file.

Usage:
    python3 ci/check_licenses.py          # report files missing the header, exit 1 if any
    python3 ci/check_licenses.py --fix    # insert the header into files missing it
"""

import argparse
import datetime
import os
import subprocess
import sys

# Extension -> line-comment prefix for files that need an SPDX header.
COMMENT_BY_EXT = {
    ".py": "#",
    ".sh": "#",
    ".yaml": "#",
    ".yml": "#",
    ".f90": "!",
    ".F90": "!",
    ".env": "#",
    ".template": "#",
}

# Extensionless files identified by basename (or basename prefix) instead.
COMMENT_BY_BASENAME_PREFIX = {
    "Makefile": "#",
    "Dockerfile": "#",
}

# Files that are vendored/third-party or otherwise out of scope even though
# they match an extension/basename above.
EXCLUDE_PATHS = {
    "private/scripts/matplotlibrc",
}
EXCLUDE_DIR_PARTS = {"_regtest_outputs", ".git"}

SPDX_MARKER = "SPDX-License-Identifier: Apache-2.0"


def comment_prefix(path: str) -> str | None:
    if path in EXCLUDE_PATHS:
        return None
    if any(part in EXCLUDE_DIR_PARTS for part in path.split("/")):
        return None

    base = os.path.basename(path)
    _, ext = os.path.splitext(path)
    if ext in COMMENT_BY_EXT:
        return COMMENT_BY_EXT[ext]
    for prefix, comment in COMMENT_BY_BASENAME_PREFIX.items():
        if base == prefix or base.startswith(prefix + "."):
            return comment

    # Extensionless files: only in scope if they have a shebang.
    if not ext:
        try:
            with open(path, "rb") as f:
                first_two = f.read(2)
        except OSError:
            return None
        if first_two == b"#!":
            return "#"
    return None


def header_lines(comment: str, year: int) -> list[str]:
    return [
        f"{comment} SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
        f"{comment} SPDX-License-Identifier: Apache-2.0\n",
    ]


def has_header(text: str) -> bool:
    lines = text.splitlines()
    start = 1 if lines and lines[0].startswith("#!") else 0
    head = "\n".join(lines[start : start + 5])
    return "SPDX-FileCopyrightText:" in head and SPDX_MARKER in head


def insert_header(path: str, comment: str, year: int) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    new_lines = (
        lines[:insert_at] + header_lines(comment, year) + ["\n"] + lines[insert_at:]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"])
    return [line.decode() for line in out.split(b"\n") if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    year = datetime.date.today().year
    missing = []
    for path in tracked_files():
        comment = comment_prefix(path)
        if comment is None:
            continue
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not has_header(text):
            missing.append((path, comment))

    if not missing:
        print("All source files carry the SPDX license header.")
        return 0

    print("Missing SPDX license header:")
    print("-----------------------------")
    for path, comment in missing:
        if args.fix:
            insert_header(path, comment, year)
            print(f"fixed: {path}")
        else:
            print(path)

    if args.fix:
        return 0

    print()
    print(
        "Run `make fix-licenses` (or `python3 ci/check_licenses.py --fix`) to add them."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

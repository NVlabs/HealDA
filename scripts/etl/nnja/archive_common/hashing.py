# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content hashing for source provenance."""

from __future__ import annotations

from pathlib import Path

CHUNK_BYTES = 1 << 20


def sha256_file(path: str | Path) -> tuple[str, int]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size

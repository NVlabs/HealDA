# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import random
import shutil
import time
from pathlib import Path

import fsspec


class AtomicFileCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        wait_seconds: float = 600.0,
        download_attempts: int = 5,
        retry_base_seconds: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.wait_seconds = wait_seconds
        self.download_attempts = download_attempts
        self.retry_base_seconds = retry_base_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_path: str) -> Path:
        suffix = Path(source_path).suffix or ".dat"
        key = hashlib.sha256(source_path.encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}{suffix}"

    def discard(self, source_path: str) -> None:
        try:
            self.path_for(source_path).unlink()
        except FileNotFoundError:
            pass

    def ensure(self, source_path: str, storage_options: dict | None = None) -> Path:
        local_path = self.path_for(source_path)
        if local_path.exists():
            return local_path

        lock_path = local_path.with_suffix(local_path.suffix + ".lock")
        start = time.monotonic()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if local_path.exists():
                    return local_path
                if time.monotonic() - start > self.wait_seconds:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    start = time.monotonic()
                    continue
                time.sleep(2.0)

        tmp_path = local_path.with_suffix(local_path.suffix + f".tmp.{os.getpid()}")
        try:
            if local_path.exists():
                return local_path
            expected_size = _remote_size(source_path, storage_options or {})
            _download_with_retries(
                source_path,
                tmp_path,
                storage_options or {},
                expected_size=expected_size,
                attempts=self.download_attempts,
                retry_base_seconds=self.retry_base_seconds,
            )
            os.replace(tmp_path, local_path)
            return local_path
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            if lock_path.exists():
                lock_path.unlink()


def _remote_size(source_path: str, storage_options: dict) -> int | None:
    try:
        with fsspec.open(source_path, "rb", **storage_options) as src:
            return getattr(src, "size", None)
    except Exception:
        return None


def _download_once(source_path: str, tmp_path: Path, storage_options: dict) -> None:
    with fsspec.open(source_path, "rb", **storage_options) as src:
        with open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _download_with_retries(
    source_path: str,
    tmp_path: Path,
    storage_options: dict,
    *,
    expected_size: int | None,
    attempts: int,
    retry_base_seconds: float,
) -> None:
    last_error = None
    for attempt in range(attempts):
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            _download_once(source_path, tmp_path, storage_options)
            if expected_size is not None and tmp_path.stat().st_size != expected_size:
                raise IOError(
                    f"Incomplete download for {source_path}: got {tmp_path.stat().st_size} bytes, "
                    f"expected {expected_size}"
                )
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == attempts - 1:
                break
            sleep_seconds = retry_base_seconds * (2**attempt) + random.uniform(0.0, 1.0)
            time.sleep(sleep_seconds)
    raise last_error

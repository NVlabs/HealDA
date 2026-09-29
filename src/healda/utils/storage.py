# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import configparser
import os
import shutil
import tempfile
import sys

import fsspec

DEFAULT_PATH = os.path.expanduser("~/.config/rclone/rclone.conf")


class StorageConfigError(Exception):
    pass


def get_remote_config(remote_name, config_path=DEFAULT_PATH):
    if not remote_name:
        return None
    # Parse the rclone config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure the remote exists in the config
    if remote_name not in config:
        raise StorageConfigError(f"Remote '{remote_name}' not found in rclone config.")

    # Extract credentials from the config
    remote_config = config[remote_name]
    return remote_config


def get_storage_options(remote_name, config_path=DEFAULT_PATH):
    remote_config = get_remote_config(remote_name, config_path)

    if remote_config is None:
        return None

    if remote_config.get("type") != "s3":
        raise StorageConfigError(f"Remote '{remote_name}' is not an S3 remote.")

    access_key = remote_config.get("access_key_id")
    secret_key = remote_config.get("secret_access_key")
    endpoint_url = remote_config.get("endpoint", None)  # Optional endpoint

    if not access_key or not secret_key:
        raise StorageConfigError(
            f"Access key or secret key missing for remote '{remote_name}'."
        )

    # Instantiate and return the S3FileSystem object
    return dict(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else None,
    )


def get_duckdb_connection(profile):
    import duckdb

    opts = get_storage_options(profile)
    con = duckdb.connect()
    key = opts["key"]
    secret = opts["secret"]
    endpoint = opts["client_kwargs"]["endpoint_url"]
    if endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]
    con.execute(f"""
    CREATE SECRET (
        TYPE s3,
        PROVIDER config,
        ENDPOINT '{endpoint}',
        KEY_ID '{key}',
        SECRET '{secret}'
    );
    """)

    return con


def ensure_downloaded(url, local):
    if os.path.exists(local):
        return

    fs = fsspec.filesystem("http")
    print(f"Downloading from {url} to {local}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as d:
        tmpfile = os.path.join(d, "file")
        fs.get(url, tmpfile)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.move(tmpfile, local)


def _get_endpoint(opts):
    return opts["client_kwargs"]["endpoint_url"]


def get_filesystem(profile: str, **kwargs):
    """fsspec S3 filesystem for a remote, for listing/globbing.

    Use this over passing storage_options around when you need filesystem
    operations (ls, glob, exists) rather than a path to hand to a reader.
    """
    opts = get_storage_options(profile)
    if opts is None:
        return None

    return fsspec.filesystem("s3", **opts, **kwargs)


def get_pyarrow_filesystem(profile: str, **kwargs):
    import pyarrow.fs

    opts = get_storage_options(profile)
    if opts is None:
        return None

    return pyarrow.fs.S3FileSystem(
        access_key=opts.get("key"),
        secret_key=opts.get("secret"),
        region=opts.get("region", ""),
        endpoint_override=_get_endpoint(opts),
        **kwargs,
    )


def get_obstore(profile: str, bucket=None, **kwargs):
    from obstore.store import S3Store

    opts = get_storage_options(profile)
    if opts is None:
        return None

    return S3Store(
        bucket=bucket,
        access_key_id=opts.get("key"),
        secret_access_key=opts.get("secret"),
        endpoint=_get_endpoint(opts),
        **kwargs,
    )

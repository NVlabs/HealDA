# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import healda.utils.storage
import zarr
from dataclasses import dataclass
import xarray
from healda.config import environment
import zarr.storage
import urllib.parse


@dataclass
class _Zarr:
    path: str
    profile: str

    @property
    def storage_options(self):
        return healda.utils.storage.get_storage_options(self.profile)

    def to_store(self, obstore=True) -> zarr.storage.StoreLike:
        if self.profile == "":
            return self.path

        url = urllib.parse.urlparse(self.path)
        if obstore:
            bucket = url.netloc
            store = healda.utils.storage.get_obstore(
                self.profile, bucket=bucket, prefix=url.path
            )
            zarr_store = zarr.storage.ObjectStore(store)
        else:
            import fsspec

            fs = fsspec.filesystem(
                url.scheme, storage_options=self.storage_options, asyn=True
            )
            zarr_store = zarr.storage.FsspecStore(fs)

        return zarr_store

    def to_zarr(self, obstore=True, use_consolidated=True) -> zarr.Group:
        store = self.to_store(obstore=True)
        return zarr.open_group(store, use_consolidated=use_consolidated)

    def to_xarray(self, obstore: bool = True, **kwargs) -> xarray.Dataset:
        return xarray.open_zarr(
            self.to_store(obstore=obstore),
            **kwargs,
        )

    def consolidate_metadata(self):
        store = zarr.storage.FsspecStore.from_url(
            self.path,
            storage_options=healda.utils.storage.get_storage_options(self.profile),
        )
        zarr.consolidate_metadata(store)


def ufs():
    return _Zarr(path=environment.UFS_HPX6_ZARR, profile=environment.UFS_ZARR_PROFILE)


# Each store is configured by a paired `<NAME>` path and `<NAME>_PROFILE`
# constant on `environment`
def _get_zarr_from_env(name: str) -> _Zarr:
    return _Zarr(
        getattr(environment, name),
        getattr(environment, f"{name}_PROFILE"),
    )


# HPX64 104-channel packed era5 (2000-2026, 6-hourly). 1980-1999 pending. Adds
# (u10/v10/z10/t10) and tcw on top of the retired (deleted) 99ch store.
era5_104var = _get_zarr_from_env("ERA5_HPX64_104CH_ZARR")

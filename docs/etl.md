# Building the training data

HealDA trains on two datasets, each built by an ETL in `scripts/etl/`:

| dataset | HealDA reads it as | location setting | ETL | steps |
| --- | --- | --- | --- | --- |
| ERA5 0.25° hourly | targets (`Era5HourlyH5`) | `ERA5_HOURLY_73CH` (or `ERA5_HOURLY_ROOT`), `ERA5_HOURLY_EXT_ROOT` | `scripts/etl/era5/` | [Building the ERA5 target files](etl_era5.md) |
| NNJA observations | model inputs (the NNJA observation loaders) | `NNJA_ROOT` | `scripts/etl/nnja/` | [Building the NNJA archive](etl_nnja.md) |

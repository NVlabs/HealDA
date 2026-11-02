# Building the ERA5 target files

Builds the 0.25 degree ERA5 state that HealDA trains against and that
`healda.datasets.era5_hourly_h5.Era5HourlyH5` reads: one HDF5 file per year, with one
contiguous float32 dataset `fields` of shape `(hours, channels, 721, 1440)`, and a
`metadata/data.json` sidecar naming the channels. The values are a pure repack of the
NCAR ERA5 netCDFs (ds633.0).

The steps need the `etl` extra (`uv sync --extra etl`, or `uv pip install -e '.[etl]'`).
Run every step from the repository root with `python -m scripts.etl.era5.<step>`.

## What HealDA reads

The HealDA configuration reads two layouts, both on the same time axis:

- the 73-channel pack `c73` (8 surface fields, and u, v, z, t and q on 13 pressure
  levels), `ERA5_HOURLY_73CH/<year>.h5` (or `ERA5_HOURLY_ROOT`);
- one directory per variable, `ERA5_HOURLY_EXT_ROOT/<variable>/<year>.h5`: `w` with its
  13 levels, and the single-channel `sst`, `sic`, `d2m`, `skt`, `sp`, `tcc`, `lcc`,
  `mcc`, `hcc`, `stl1`, `stl2`, `swvl1`, `swvl2`, `sd`, `tcw`, `t10`, `u10`, `v10`, `z10`.

Both settings take a `:`-separated list of roots, so a layout's years can sit in more
than one directory. Each output directory also holds `metadata/data.json` and a `state/`
directory with the task journals.

Build `c73` and `w` directly into their directories (`--channels c73`, `--channels w`).
Build the single-channel variables as one pack and split it into one directory each
(see Thin and split):

```bash
--channels sst,sic,d2m,skt,sp,tcc,lcc,mcc,hcc,stl1,stl2,swvl1,swvl2,sd,tcw,t10,u10,v10,z10
```

`--channels` takes a preset, a comma separated list, or `@file` with one name per line.
The presets are in `channels.py`.

## Build a year

```bash
python -m scripts.etl.era5.build_year_h5 --init-only \
  --year 2020 --output "$ERA5_HOURLY_ROOT/2020.h5" --cache-dir "$CACHE" \
  --channels c73 --dhours 6
python -m scripts.etl.era5.build_year_h5 \
  --year 2020 --output "$ERA5_HOURLY_ROOT/2020.h5" --cache-dir "$CACHE" \
  --channels c73 --dhours 6 --workers 32 --rank 0 --world-size 1
```

`--init-only` allocates the file and writes the sidecar; run it once on one process. The
writers then fill it, one task per source variable and day, and can be relaunched: completed
tasks are recorded next to the output and skipped. Under Slurm the rank and world size come
from the job, so `--rank` and `--world-size` are only needed outside it. `--archive-end`
stops a partial final year.

The raw netCDFs are read from NCAR's public bucket `s3://nsf-ncar-era5` without
credentials. A mirror or a local copy is configured with `--sources`, a TOML file
described in `build_year_h5.py`; a source's `profile` names an S3 remote in
`~/.config/rclone/rclone.conf`, read for its credentials (the rclone program is not
needed). Remote files are cached in `--cache-dir`.

ERA5 keeps its provider's terms: CC BY 4.0, with attribution to ECMWF. The
[HealDA v2 model card](https://huggingface.co/nvidia/healda-v2#training-dataset) gives the
attribution text.

## Check a year

```bash
python -m scripts.etl.era5.verify_year_h5 "$ERA5_HOURLY_ROOT/2020.h5" --quick   # tasks done
python -m scripts.etl.era5.verify_year_h5 "$ERA5_HOURLY_ROOT/2020.h5"           # read every value
python -m scripts.etl.era5.verify_year_h5 NEW.h5 --reference OLD.h5             # bit-compare
```

`--quick` reads the task journals of `build_year_h5` and `write_arco_year_h5`; check a file
made by `thin_year_h5` without it.

## Thin and split

`thin_year_h5` copies every `--dhours`-th step out of an existing finer year file, which is
bit-identical to building at that cadence and reads no netCDF. `split_pack_to_ext` reslices a
multi-channel pack into one file per variable under `--ext-root`, the
`ERA5_HOURLY_EXT_ROOT/<variable>/<year>.h5` layout:

```bash
python -m scripts.etl.era5.split_pack_to_ext \
  --pack-dir "$PACK" --ext-root "$ERA5_HOURLY_EXT_ROOT" --year-start 2000 --year-stop 2024
```

## ERA5T from ARCO (scoring reference)

Google's ARCO ERA5 Zarr carries ERA5T, the preliminary release, after the final ERA5
ends; its root `.zattrs` `valid_time_stop` is the last final day. It is a scoring
reference, not training data.

`write_arco_year_h5` fills whole days of a year file that already exists and leaves every
other step as it was. Create that file first, with the same channels and cadence, either
empty (steps never written read as zeros):

```bash
python -m scripts.etl.era5.build_year_h5 --init-only \
  --year 2026 --output 2026.h5 --cache-dir "$CACHE" --channels c73 --dhours 6 \
  --archive-end 2026-05-31
```

or as a longer copy of an existing year file, whose steps are copied unchanged:

```bash
python -m scripts.etl.era5.thin_year_h5 --init-only --source OLD/2026.h5 \
  --source-root OLD --source-dhours 6 --dhours 6 --output 2026.h5 --output-steps 604
python -m scripts.etl.era5.thin_year_h5 --source OLD/2026.h5 \
  --source-root OLD --source-dhours 6 --dhours 6 --output 2026.h5 --output-steps 604 \
  --rank 0 --world-size 1
```

Then fetch the ARCO chunks for the range at the same cadence (`fetch_arco_curl.sh` needs
`curl`) and write them:

```bash
python -m scripts.etl.era5.make_arco_manifest --metadata arco/.zmetadata \
  --output arco.files --start 2026-05-01T00:00 --stop 2026-06-01T00:00 \
  --channels c73 --dhours 6
ARCO_LOCAL=arco bash scripts/etl/era5/fetch_arco_curl.sh arco.files 48
python -m scripts.etl.era5.write_arco_year_h5 --source arco --output 2026.h5 \
  --year 2026 --channels c73 --dhours 6 --day-start 120 --day-stop 151
```

The sidecar `sources` then names the range, split into ARCO ERA5 and ERA5T at
`valid_time_stop`. `make_arco_manifest --help` and `write_arco_year_h5 --help` list the
remaining flags.

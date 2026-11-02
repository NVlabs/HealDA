# Building the NNJA archive

The scripts in `scripts/etl/nnja/` decode NOAA's public NNJA BUFR into the Parquet
archive described in [NNJA archive](observations.md#nnja-archive). They are not part of
the installed package: run them from a checkout of the repository, from its root, as
`python -m scripts.etl.nnja.<...>`. Every step reads and writes under `NNJA_ROOT`.

## 1. Install

The decoder, earth2bufr, is the `etl` extra:

```bash
uv sync --extra etl        # or: uv pip install -e '.[etl]'
```

The commands below call `python`; run them in that environment (`source .venv/bin/activate`,
or prefix each with `uv run`).

Set `NNJA_ROOT` to the archive root, e.g. `export NNJA_ROOT=/path/to/nnja`. Every step
reads and writes under it unless given explicit paths, and HealDA's observation loaders
read the archive from it.

The pull scripts need [s5cmd](https://github.com/peak/s5cmd) installed.

## 2. Pull raw BUFR

The source is the public bucket `s3://noaa-reanalyses-pds/observations/reanalysis`, read
without credentials. Re-running a pull copies only objects whose size differs.

NNJA keeps its provider's terms: CC BY 4.0, with attribution to the NOAA–NASA Joint
Archive (NNJA), and the EUMETSAT `merged` winds carry their own acknowledgement. The
[HealDA v2 model card](https://huggingface.co/nvidia/healda-v2#training-dataset) gives the
attribution text.

Satellite sounders (`amsua amsub atms mhs iasi cris airs`), into
`$NNJA_ROOT/bufr/<sensor>/YYYY/MM/bufr/`. By default every sensor from 2000 through the
current year; `--sensors`, `--start-year`, `--end-year` and `--months` narrow it:

```bash
bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh
```

The bucket has no MHS files for 2005-01, 2005-03 to 2005-06, 2005-08, 2005-09 and
2022-05. s5cmd reports each of those months as `no object found`, and the script exits 1
after copying everything else.

Satellite winds, into `$NNJA_ROOT/bufr/<stream>/YYYY/MM/bufr/`. The decoder reads
`merged`, the EUMETSAT AMV Climate Data Record (CC BY 4.0, see the script header), through
2019 and NOAA's operational `satwnd` from 2020:

```bash
bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh --stream merged --start-year 2000 --end-year 2019
bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh --stream satwnd --start-year 2020
```

Both scripts take `--dry-run`, `--workers` and `--dst`. PrepBUFR and GPS-RO are copied
with s5cmd directly, into the layout their decoders read:

```bash
NOAA=s3://noaa-reanalyses-pds/observations/reanalysis
s5cmd --no-sign-request cp "$NOAA/conv/prepbufr/*" "$NNJA_ROOT/bufr/prepbufr/"
s5cmd --no-sign-request cp "$NOAA/gps/gpsro/*" "$NNJA_ROOT/bufr/gpsro/"
```

## 3. Decode

The commands below build 2000 through the current year:

```bash
START=2000 END=$(date -u +%Y)
```

PrepBUFR, one year per run, one file per cycle into `$NNJA_ROOT/parquet/cycles/YYYY/`.
Without `--resume` the decoder first deletes every `*.parquet` in the output directory:

```bash
for year in $(seq $START $END); do
  python -m scripts.etl.nnja.conv.nnja_prepbufr_to_parquet --year $year \
    --input-root "$NNJA_ROOT/bufr/prepbufr" --output-root "$NNJA_ROOT/parquet" \
    --workers 8 --resume
done
```

GPS-RO, one year per run, one file per cycle into `$NNJA_ROOT/parquet/gpsro_v3/YYYY/`,
same `--resume` rule:

```bash
for year in $(seq $START $END); do
  python -m scripts.etl.nnja.gpsro.nnja_gpsro_to_parquet --year $year \
    --input-root "$NNJA_ROOT/bufr/gpsro" --output-root "$NNJA_ROOT/parquet" \
    --workers 8 --resume
done
```

Satellite winds, one file per day into `$NNJA_ROOT/parquet/satwnd_v2/`, reading the
stream for each year from `$NNJA_ROOT/bufr`. Days already published with the same
revisions are skipped:

```bash
python -m scripts.etl.nnja.satwnd.run_production_etl --start-year $START --end-year $END \
  --output-root "$NNJA_ROOT/parquet" --sensor satwnd_v2 \
  --decoder-revision <earth2bufr version> --log-json satwnd.jsonl
```

Satellite sounders, one file per sensor-day into `$NNJA_ROOT/parquet/<sensor>/`, for
every day that has raw files under `$NNJA_ROOT/bufr/<sensor>/`. Published days are
skipped:

```bash
python -m scripts.etl.nnja.sat.archive_runner \
  --sensors amsua amsub atms mhs iasi cris airs \
  --workers 8 --decoder-revision <earth2bufr version>
```

`--start-day` and `--end-day` (`YYYY-MM-DD`) limit the run, and then every day in the
range is expected for every listed sensor: a day without raw files is recorded under
`$NNJA_ROOT/parquet/_errors/<sensor>/` and the run exits non-zero. Pass only sensors whose
record covers the range.

`--decoder-revision` is recorded in each file's footer. `scripts/etl/nnja/sat/README.md`
and `scripts/etl/nnja/satwnd/README.md` describe the two archive contracts.

## 4. Thin the IR sounders

The full-width IR files carry every published channel, 281 to 616 per sensor, one
Parquet column each, so a read fetches only the channels it asks for. HealDA trains on
32 per sensor (`ObsConfig.nnja_ir_channels = "ir32"`). To save storage, `thin_nnja_ir`
writes a copy with 48 selected channels per sensor (`ir48`, which contains `ir32`):
temperature, humidity and window channels whose weighting functions peak from about 30 to
1000 hPa. It goes to `$NNJA_ROOT/parquet/ir48/<sensor>/`: 4.6 TiB for the three sensors against
39.6 TiB at full width, with identical values on the kept channels. `--preset ir32` keeps
only the 32. The loader reads the narrowest such store that holds every requested
channel, and the full-width files otherwise. Days already present are skipped; by
default every day of the source is thinned:

```bash
python -m scripts.etl.nnja.sat.thin_nnja_ir --sensors cris iasi airs
```

`scripts/etl/nnja/sat/NNJA_IR_THINNING.md` lists what is kept.

## 5. Optional: radiosonde splice

NOAA posts each PrepBUFR cycle daily and has re-posted whole months later. In the 2026
months checked, the daily posting lacked about 45% of the radiosonde temperature and
humidity reports, which the re-posting added weeks later. For a month not yet re-posted,
the complete radiosondes are in NCAR RDA ds337.0 (PREPNR daily tars). `splice` writes
NOAA's file with its radiosonde block (data category 102) replaced by PREPNR's, into a new
tree with the NOAA layout; decode that tree with `--input-root` pointing at it:

```bash
python -m scripts.etl.nnja.conv.splice_radiosondes splice \
  --prepnr-root <prepnr dir> --out-root <spliced dir> $(seq -f 202608%02g 1 31)
```

`--prepnr-root` holds one extracted tar per day, `YYYYMMDD/prepbufr.gdas.YYYYMMDD.tHHz.nr`.
`census` prints messages, subsets and a digest per category for the sources. The module
docstring gives the download URLs.

## Output layout

```text
$NNJA_ROOT/parquet/
├── cycles/YYYY/gdas.YYYYMMDD.tHHz.prepbufr.nr.parquet
├── gpsro_v3/YYYY/gdas.YYYYMMDD.tHHz.gpsro.tm00.bufr_d.parquet
├── satwnd_v2/YYYYMMDD.parquet
├── <sensor>/YYYYMMDD.parquet       # amsua, amsub, atms, mhs, and full-width iasi, cris, airs
└── ir48/<sensor>/YYYYMMDD.parquet  # cris, iasi, airs
```

The loaders read this tree; see [NNJA archive](observations.md#nnja-archive) for the
schemas. The full archive, built through 2026-09:

| stream | span | files | size |
|---|---|---:|---:|
| `amsua/` | 1998-10-26 on | 10,109 | 601 GiB |
| `amsub/` | 1998-10-26 to 2014-06-06 | 5,703 | 789 GiB |
| `atms/` | 2012-02-15 on | 5,232 | 3.11 TiB |
| `mhs/` | 2005-02-01 on | 7,544 | 1.56 TiB |
| `iasi/` (616 channels) | 2008-01-01 on | 6,736 | 15.74 TiB |
| `cris/` (431 channels) | 2018-01-16 on | 3,075 | 16.25 TiB |
| `airs/` (281 channels) | 2002-08-31 to 2023-01-18 | 7,377 | 7.63 TiB |
| `ir48/` (48 channels each) | as above | 17,188 | 4.55 TiB |
| `cycles/` (PrepBUFR) | | 38,624 | 633 GiB |
| `gpsro_v3/` | | 35,340 | 575 GiB |
| `satwnd_v2/` | | 9,485 | 2.40 TiB |

`ir48/` is a thinned copy of the full-width IR files with the same values on its 48
channels; the loader reads either. With `ir48/` in their place the archive is about 14 TiB.

## Runtime

Decode only, with the raw BUFR already local, on one CPU node. A worker is one process:
a sounder worker runs 4 earth2bufr decode threads (plus 2 Arrow and 2 torch threads), a
satellite-wind worker 4 decode threads, and PrepBUFR and GPS-RO workers earth2bufr's default.

| stream | measured |
|---|---|
| CrIS | 112 s per day per worker |
| IASI | 62 s per day per worker |
| AIRS | 55 s per day per worker |
| ATMS | 26 s per day per worker |
| MHS / AMSU-B / AMSU-A | 9 / 7 / 6 s per day per worker |
| PrepBUFR | 600 cycles in about 100 s with 48 workers |
| GPS-RO | 120 cycles in about 5 min with 32 workers |
| Satellite winds | 30 days in about 6.5 min with 24 workers |

The worker counts are the ones these runs used, on one node; set `--workers` from the
node's cores (times the threads per worker) and memory. Workers split the work by day or
cycle and share nothing, and each file is written to a temporary name and renamed into
place, so runs over disjoint ranges (years for PrepBUFR, GPS-RO and satellite winds, day
ranges for the sounders) can also go on separate nodes. A year of CrIS is about 11
worker-hours.

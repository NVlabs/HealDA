# NNJA IR Channel Thinning

Derives a channel-thinned store from the source-faithful wide NNJA Parquet archive: 48 selected
channels per IR sounder instead of every published one, at a tenth of the bytes and identical
values. Written by `thin_nnja_ir.py`, whose module docstring covers what is kept and why. Its
input is the archive `archive_runner.py` writes (README.md).

Each channel is its own Parquet column, so the store saves storage, not read work. HealDA trains
on `ir32`; `ir48` adds 16 more channels to it. The channels are temperature,
humidity and window channels with weighting-function peaks spread from about 30 to 1000 hPa.

## Run it

```bash
python3 -m scripts.etl.nnja.sat.thin_nnja_ir --sensors cris --workers 64
python3 -m scripts.etl.nnja.sat.thin_nnja_ir --sensors iasi airs --workers 64
```

Defaults cover the rest: `--preset ir48`, `--source-root` the published archive, `--output-root`
`<source-root>/<preset>`, and every published day. Each worker writes one day at a time, so
throughput scales with `--workers` until the file system's write bandwidth is the limit.

Resumable: a day already present in the output is skipped, so an interrupted or killed run
resumes by re-running the same command. Days are written to `YYYYMMDD.parquet.tmp` and renamed
only after `fsync`, so a file at the final path is a complete one and a reader never sees a
partial day. Pass `--overwrite` to redo days that are already there.

## What it produced

At `ir48`, 17,188 sensor-days:

| sensor | channels kept | days | published | thinned |
| --- | --- | --- | --- | --- |
| CrIS | 48 of 431 | 3075 | 17.7 TB | 1.8 TB |
| IASI | 48 of 616 | 6736 | 15.0 TB | 1.4 TB |
| AIRS | 48 of 281 | 7377 | 7.7 TB | 1.5 TB |

Channel numbers per sensor are ranked in `obs/ir_channel_ranking.csv` and resolved by
`ir_spectral.ir_channel_preset`; each file also records its preset and channel numbers in schema
metadata.

## Reading it

Nothing to configure. `NNJAWideLoader` points at the archive root as always, and `_sensor_root`
prefers the narrowest thinned store under it that holds every channel `ir_channels` asked for:

```python
NNJAWideLoader(sensors=["cris", "iasi", "airs", "atms"], ir_channels="ir32")
```

reads `ir48/{cris,iasi,airs}` for the sounders, ir32 being a subset of what that store holds,
and the published `atms` full width. A full-axis read (`ir_channels=None`) and an explicit
channel list reaching outside the presets both stay on the published archive, as does any day
the thinned store lacks.

Verified bit-identical against a full-width read on 2018-02-01 across all three sounders, 3.37M
rows and every emitted column, at both `ir32` and `ir48`. That includes `Observation`, which for
CrIS runs from the integer BUFR code through the `value_encoding` metadata to Planck-inverted
kelvin.

## Output structure

```text
nnja/parquet/
├── cris/YYYYMMDD.parquet        # source-faithful, every published channel
├── iasi/YYYYMMDD.parquet
├── airs/YYYYMMDD.parquet
├── atms/YYYYMMDD.parquet        # microwave, never thinned
└── ir48/                        # this store, a sibling like cycles and gpsro
    ├── cris/YYYYMMDD.parquet
    ├── iasi/YYYYMMDD.parquet
    └── airs/YYYYMMDD.parquet
```

Sensor directories keep the archive's own names and column names, which is what lets the loader
read a thinned day with no change.

# NNJA SATWND (AMV) Archive

Raw satellite-wind (AMV) BUFR to daily Parquet. The public loader
`healda.observations.loaders.nnja_satwnd.NNJASatwndLoader` reads the result from
`$NNJA_ROOT/parquet/satwnd_v2`.

## Why this exists

The merged-PrepBUFR route that the NNJA conventional loader reads keeps only ~5% of
raw SATWND for a representative cycle. Most of that loss is decode routing rather
than deliberate thinning: modern GOES (74% of raw volume), VIIRS and AVHRR use NCEP
subtype numbers that PREPDATA's decoder does not recognize, and LEO-GEO has no
PREPDATA branch at all, so those products are dropped before GSI runs. Operational
GFS v16 reads the raw dump directly (`use_prepb_satwnd=.false.`). Decoding the raw
dump matches that contract and recovers data the replay-based archive cannot contain.

## Source streams

One stream per era, chosen by `archive_etl.streams_for_year`:

- **1979-2019: `merged`**, the EUMETSAT AMV Climate Data Record, Release 2. Meteosat
  rows in it are EUMETSAT-sourced (`OGCE`=254); GOES, GOES-R and MODIS rows are
  NESDIS-sourced (`OGCE` 176/160/34), the same producer as `satwnd`.
- **2020 onward: NCEP `satwnd`**, from when the CDR ends.

The one-stream rule follows GSI's own ncdiag output, which reads `merged` at every
era tested.

## Data model

One row per atmospheric motion vector. Unlike conventional PrepBUFR, which splits
reports into MASS (`POB`/`TOB`/`QOB`) and WIND (`POB`/`UOB`/`VOB`) rows, an AMV is
wind-only at the source, so each row carries a complete wind vector
(`wind_direction`/`wind_speed` and derived `wind_u_derived`/`wind_v_derived`) with
its own assigned pressure and quality metadata.

- Column names come from `descriptor_catalog.csv`, derived from each descriptor's
  WMO Table B element name. Every column carries its mnemonic, F-X-Y, units, base
  packing and code-table key as Arrow field metadata. There is no raw-mnemonic
  catch-all.
- Repeated occurrences are grouped into `list<struct>` columns (`wind_candidates`,
  `height_assignments`, `quality_diagnostics`, `model_winds`,
  `measurement_uncertainty`) by descriptor path, each entry keeping its raw `TSIG`
  (`tsig_raw`), never by occurrence index: the same mnemonic means different things at
  different paths. The wind blocks' own `HOUR`/`MINU`/`SECO` occurrences are not
  kept; every other decoded occurrence is.
- Fields the source can omit are Arrow nulls, except inside the numeric
  `*_occurrences` lists, where a missing occurrence is NaN.
- Every file has the columns of `declared_schema.arrow` whatever templates that day
  contained.
- `gsi_observation_type` is resolved at write time from (subset, `SAID`, `SWCM`) with
  `gsi_satwnd_type_table.csv`. Its mappings are derived from the `sattab` array and provider
  cases in NOAA-EMC GSI `src/gsi/read_satwnd.f90` at commit `860d1374` (GSI is LGPL-3.0;
  the table holds the resulting codes only). Rows with `mapping_tier` `2-pre-refactor-gsi`
  or `3-prepdata` add GSI's earlier SWCM rule and the NCEP PREPDATA route.

## File layout

- One Parquet file per UTC day at `<output-root>/<sensor>/<YYYYMMDD>.parquet`.
- One row group per 3-hour `da_window` (`time_utc <= cycle` -> `cycle`, otherwise
  `cycle + 3h`), up to 8 per day. The loader reads one row group per window.
- Atomic publish: write `.tmp`, reopen and validate the footer (required keys,
  one `da_window` per row group in increasing order), then `replace()` onto the
  final path.
- Footer metadata: schema, decoder and ETL revisions, declared-schema hash, build
  time, UTC day, per-source-file path, SHA-256 and size, per-template source
  structures, and the SHA-256 of every lookup table used.
- Resume: the published file is the completion record. A day is skipped only if its
  footer validates, it carries the same decoder and ETL revisions, has exactly the
  columns of `declared_schema.arrow`, and was built from the same source bytes
  (matched by cycle, SHA-256 and size, not path). Any other day is rebuilt, and the
  reason is printed. There is no side-car status index.

Physical writing is the sensor-agnostic `../archive_common` day writer; this package
overrides its `bind_metadata` and `validate_contract` hooks.

## Running it

### 1. Install the decoder

The ETL decodes with earth2bufr, installed by the `etl` extra:

```bash
pip install -e ".[etl]"
```

### 2. Pull the BUFR

```bash
bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh --stream merged --start-year 2000 --end-year 2019
bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh --stream satwnd --start-year 2020 --end-year 2025
```

Source is the public `s3://noaa-reanalyses-pds`, mirrored to
`$NNJA_ROOT/bufr/<stream>/<YYYY>/<MM>/bufr/`, where the ETL reads it. The `merged` CDR
is CC BY 4.0; see the attribution note at the top of the script.

### 3. Build the archive

From the repository root, one era per node:

```bash
python -m scripts.etl.nnja.satwnd.run_production_etl \
    --start-year <Y0> --end-year <Y1> \
    --output-root "$NNJA_ROOT/parquet" --sensor satwnd_v2 \
    --decoder-revision <earth2bufr version> \
    --log-json <log.jsonl>
```

`--sensor` is appended to `--output-root`; do not repeat it there. `--etl-revision`
defaults to `ETL_CONTENT_REVISION`, the resume key, and should not normally be
passed: a different value rebuilds every day stamped otherwise. On a resumed run,
check that the first days report `skipped_validated`. The driver exits with an error
naming any requested year that has no source cycles on disk. Each worker decodes with
4 earth2bufr threads.

## Tests

`tests/scripts/etl/test_nnja_etl_*.py`. The projection, streams and determinism tests read real
cycles from `$NNJA_ROOT/bufr` and skip when those are absent.

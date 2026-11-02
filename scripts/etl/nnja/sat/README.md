# NNJA Satellite Radiance Archive

Raw NCEP NNJA sounder BUFR (AMSU-A, AMSU-B, MHS, ATMS, CrIS FSR, IASI, AIRS) to one Parquet
file per sensor per UTC day, `<output root>/<sensor>/<YYYYMMDD>.parquet`, one row per source
subset. Fields, types, source bindings and footer provenance are defined in
`archive_schema.py`. `thin_nnja_ir.py` builds the `ir48` stores from these files
(`NNJA_IR_THINNING.md`).

Sources are found at `<raw root>/<sensor>/<YYYY>/<MM>/bufr/`, one file per cycle with
`YYYYMMDD` and `tHHz` in its name. AIRS is NOAA's `airs/nasa/aqua` route.

## Incremental refresh

New months are pulled and decoded with the full-build commands below, restricted to the new
range; published days are skipped:

```bash
bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh --sensors "atms cris" --start-year 2025 --end-year 2025 --months 01
python3 -m scripts.etl.nnja.sat.archive_runner --sensors atms cris --workers 8 \
  --decoder-revision <earth2bufr version> --start-day 2025-01-01 --end-day 2025-01-31
python3 -m scripts.etl.nnja.sat.thin_nnja_ir --sensors cris --start 20250101 --end 20250131
```

## Full-archive builds

All steps run from the repository root. Paths default to `$NNJA_ROOT/bufr` (source) and
`$NNJA_ROOT/parquet` (output).

1. Pull the BUFR (`--dry-run` prints without downloading):

   ```bash
   bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh --sensors "atms iasi" --start-year 2024 --end-year 2025
   ```

2. Install the decoder: `pip install -e ".[etl]"` (earth2bufr from PyPI).

3. Build the archive:

   ```bash
   python3 -m scripts.etl.nnja.sat.archive_runner --sensors atms iasi \
     --raw-root $NNJA_ROOT/bufr --output-root <dir> --workers 8 \
     --decoder-revision <earth2bufr version> --start-day 2025-01-01 --end-day 2025-01-31
   ```

   `--task-index <file>` keeps the scanned sensor-day sources so later runs do not rescan
   the raw tree; `--reindex --index-only` rebuilds it without decoding.

A day is written to a temporary file, validated and renamed into place; reruns skip
published days. `--audit-existing` revalidates published days and rebuilds those whose
sources (cycle, object key, size) or contract changed; `--overwrite` rebuilds every day in
the range. The task index records the source root, so moving the BUFR needs one
`--reindex` per sensor.

## Ledgers

Warnings, one JSON per day, under `<output root>/_warnings/<sensor>/`:
`missing_source_cycles` (a partial day is still published), `empty_da_windows`,
`zero_byte_source_objects`, `no_observation_subsets`, `missing_template_fields`,
`truncated_source_messages`.

Failures, one JSON per day, under `<output root>/_errors/<sensor>/` (duplicate source
cycles, malformed BUFR, unsupported structure, conversion or validation failures); a
successful retry removes the record. The runner exits non-zero when any day fails.

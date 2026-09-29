# Prepared Parquet observation input

HealDA reads prepared observation archives. Raw observation decoding,
calibration, quality-control production, and archive construction are outside
the package boundary. A directory is compatible only when its layout, Arrow
schema, channel identities, and normalization metadata match the selected
loader and checkpoint.

There are two sources, and a recipe uses one or the other:

| source | root | loader | sensor vocabulary |
| --- | --- | --- | --- |
| UFS | `UFS_OBS_PATH` | `observations/loaders/ufs.py` | `observations/sensors.py` |
| NNJA | `NNJA_ROOT` | `observations/loaders/nnja_*.py` | `observations/sensors_nnja.py` |

Each source is self-contained: an NNJA run does not need the UFS archive.

The two vocabularies assign different `Global_Channel_ID` values, so a
checkpoint trained against one archive cannot read the other. Pick the source
the checkpoint was trained on.

## UFS archive

Set `UFS_OBS_PATH` to a local directory or object-store URL. The root contains:

```text
channel_table.parquet
<sensor>/YYYYMMDD/0.parquet
```

Each observation file uses the long schema: one row per observation value.
Required model-facing columns are `Latitude`, `Longitude`,
`Absolute_Obs_Time`, `DA_window`, `Platform_ID`, `Observation`, and
`Global_Channel_ID`. Sensor-specific nullable fields are
`Sat_Zenith_Angle`, `Sol_Zenith_Angle`, `Scan_Angle`, `Pressure`, `Height`,
`Observation_Type`, `QC_Flag`, and `Analysis_Use_Flag`.

`channel_table.parquet` maps every `Global_Channel_ID` to normalization and
identity metadata. It must contain `Global_Channel_ID`, `min_valid`,
`max_valid`, `sensor_id`, `is_conv`, `name`, `mean`, and `stddev`. IDs must be
unique and consistent with the package vocabulary.

## NNJA archive

Set `NNJA_ROOT`. Files live under `NNJA_ROOT/parquet` in the following
layouts, one per observation family:

```text
parquet/<sensor>/YYYYMMDD.parquet
parquet/cycles/YYYY/gdas.YYYYMMDD.tHHz.prepbufr.nr.parquet
parquet/gpsro_v3/YYYY/gdas.YYYYMMDD.tHHz.gpsro.tm00.bufr_d.parquet
parquet/satwnd_v2/YYYYMMDD.parquet
```

Wide satellite files require `latitude`, `longitude`, `time_utc`, `da_window`,
`platform_id`, `satellite_zenith_angle`, `solar_zenith_angle`,
`field_of_view`, and `hpx2048_nest`, plus sensor-specific observation columns
whose names and metadata match the packaged sensor vocabulary.

Conventional cycle files require `XOB`, `YOB`, `DHR`, `POB`, and `TYP`, plus
the value and quality-mark columns for each enabled variable. GPS radio
occultation files require `bending_angle`, `qfro`, `obs_time`,
`impact_parameter`, `earth_radius_curvature`, and `satellite_id`. Optional
height and pressure columns improve GPS metadata; the loader provides defined
fallbacks when they are absent.

Optional atmospheric-motion-vector files require `time_utc`, `da_window`,
`latitude`, `longitude`, `hpx4096_nest`, `wind_u_derived`,
`wind_v_derived`, `assigned_pressure`, `gsi_observation_type`,
`gsi_type_mapping_tier`, `wind_computation_method`, `ncep_dump_subtype`, and
`cycle`.

An NNJA run needs only `NNJA_ROOT`. The conventional channel vocabulary, its
validity bounds and its normalization statistics all ship with the package.

## Invariants

- Timestamps are timezone-naive UTC values and archive partitions cover the
  requested context window.
- Coordinates and physical values use the units expected by the loader.
- Arrow types are safely castable to the package schema.
- Global channel, local channel, sensor, and platform identities remain stable.
- Every channel has finite normalization and validity metadata.
- Row-group statistics for time-selection columns are present when required.

These contracts describe loader compatibility, not scientific suitability.
Prepared data should be independently reviewed before it is used in a training
claim.

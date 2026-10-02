# Prepared Parquet observation input

HealDA reads prepared observation archives. Raw observation decoding,
calibration, quality-control production, and archive construction are outside
the package boundary. A directory is compatible only when its layout, Arrow
schema, channel identities, and normalization metadata match the selected
loader and checkpoint.

There are two sources, and a recipe uses one or the other:

| source | root | loader | sensor vocabulary |
| --- | --- | --- | --- |
| NNJA | `NNJA_ROOT` | `observations/loaders/nnja_*.py` | `observations/sensors_nnja.py` |
| UFS replay (legacy) | `UFS_OBS_PATH` | `observations/loaders/ufs.py` | `observations/sensors.py` |

NNJA is the source the checkpoints are trained on. The UFS replay archive is an
earlier source, kept for older runs and described at the end. Each source is
self-contained: an NNJA run does not need the UFS archive.

The two vocabularies assign different `Global_Channel_ID` values, so a
checkpoint trained against one archive cannot read the other. Pick the source
the checkpoint was trained on.

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
`impact_parameter`, `earth_radius_curvature`, and `satellite_id`.

Satellite wind (atmospheric motion vector) files require `time_utc`, `da_window`,
`latitude`, `longitude`, `hpx4096_nest`, `wind_u_derived`,
`wind_v_derived`, `assigned_pressure`, `gsi_observation_type`,
`gsi_type_mapping_tier`, `wind_computation_method`, `ncep_dump_subtype`, and
`cycle`.

An NNJA run needs only `NNJA_ROOT`. The conventional channel vocabulary, its
validity bounds and its normalization statistics all ship with the package.

## Matching a trained checkpoint

The checkpoints were trained on the NNJA archive, built from the NNJA BUFR feeds,
with the `v2-nnja-latlon-final` configuration in `healda/cli/train.py`. Settings
named below take that configuration's values.

A checkpoint gives its intended results only on data prepared the same way. Data
from another source, such as a near-real-time feed, has to reproduce the
processing below. Columns can be renamed to the archive's, but the values have to
mean the same thing. Fine-tuning or training a new model on differently prepared
data is outside this contract.

Two places in the code show the format in full:

- The loaders, `observations/loaders/nnja_*.py`, define each family's archive rows
  and what is done to them on load.
- `observations/adapters/e2s_nnja.py` converts Earth2Studio's NNJA data into those
  archive rows. It renames columns, derives what the archive derives with the same
  code (for example the GPS-RO pressure below), and reads ATMS antenna temperature
  rather than brightness temperature. Its functions return per-cycle tables that
  the loaders take through their `table_source` arguments, so nothing has to be
  written to disk. It is the model to follow for a new source.

### Inputs the model was trained with

- **Satellite sounders**, from every platform below that was flying. Each sensor's
  platforms are those in its normalization file under
  `observations/normalizations/nnja/`.
  - ATMS: Suomi NPP, NOAA-20, NOAA-21
  - AMSU-A: NOAA-15 to NOAA-19, MetOp-A, MetOp-B, MetOp-C
  - AMSU-B: NOAA-15 to NOAA-17 (training only, no longer flying)
  - MHS: NOAA-18, NOAA-19, MetOp-A, MetOp-B, MetOp-C
  - AIRS: Aqua
  - IASI: MetOp-A, MetOp-B, MetOp-C
  - CrIS: Suomi NPP, NOAA-20, NOAA-21
- **Conventional PrepBUFR reports** of temperature, humidity, wind and surface
  pressure from:
  - radiosondes, dropsondes and pibals
  - aircraft, including the restricted AMDAR, ACARS and TAMDAR reports
  - surface land stations (synoptic and METAR)
  - surface marine stations (ships and buoys)
  - wind profilers, NEXRAD VAD winds and RASS
  - scatterometers (ASCAT, QuikSCAT, ERS, WindSat)

  The report types each variable accepts are `T_REPORT_TYPES`, `Q_REPORT_TYPES`,
  `UV_REPORT_TYPES` and `PRESSURE_REPORT_TYPES` in `loaders/nnja_conventional.py`.
- **GPS radio occultation** bending angle, from every satellite in the archive.
- **Satellite winds** (atmospheric motion vectors) from the SATWND archive. Their
  PrepBUFR copies are dropped.

Each of these sources contributes positively to analysis skill. For the best
analyses, provide all of them at inference.

### Conventional observations

This covers PrepBUFR reports, GPS radio occultation and satellite winds, and
concerns their vertical coordinates. The observation featurization requires a
valid height and pressure on every conventional observation and drops rows that
lack either (`preprocessing/filtering.py`, `QCLimits` in `observations/sensors.py`):

- height: 0 to 60,000 m
- pressure: 0.5 to 1,100 hPa for GPS radio occultation, and
  `conv_min_pressure_hpa` (1 hPa) to 1,100 hPa for everything else

Some families report both coordinates. For the others the loader fills one in.

- **PrepBUFR conventional** (`loaders/nnja_conventional.py`): pressure is `POB` and
  height is `ZOB`. Rows without `ZOB` are dropped, with two exceptions.
  - Scatterometer winds report a 10 m wind and no `ZOB`. Their height is set to
    10 m.
  - Surface winds of report types 280, 281 and 287 (`nnja_surface_winds`) get
    height `ELV` + 10 m, as in GSI. Types 282 (ATLAS buoys) and 284 are excluded
    and dropped, because their `POB` is a nominal 1013 hPa or reduced from
    sea-level pressure rather than measured.
- **GPS radio occultation** (`loaders/nnja_gpsro.py`) reports height only. Height
  is `heit_at_impact_m`. Pressure is `blended_pressure_5km_hpa`, which
  `derive_level_coordinates` in `preprocessing/gpsro_pressure.py` derives from the
  occultation's own refractivity, with no model background. Below 5 km it is 0.8
  standard atmosphere plus 0.2 dry hydrostatic, and above 5 km dry hydrostatic.
- **Satellite winds** (`loaders/nnja_satwnd.py`) report pressure only. Pressure is
  `assigned_pressure`, the producing centre's final height assignment. Height is
  that pressure's US Standard Atmosphere 1976 altitude, floored at 0 m.

The model is sensitive to the coordinates it was trained on. A coordinate from a
substantially different algorithm, even a more accurate one, changes what the
checkpoint sees and is likely to degrade its analyses. A GPS pressure taken from a
model background is one example. Compare the two before relying on such a
coordinate.

### Satellite sounders

What the wide satellite files (`loaders/nnja_wide.py`) must hold, and what the
loader changes on load:

- **Channels** are identified by the instrument's own channel number. A loaded
  model's `DAModel.ir_channels` lists the infrared channels its recipe reads, and
  it reads no others. For the trained configuration that is the `ir32` preset, 32
  channels each for AIRS, IASI and CrIS (`ir_channel_preset` in
  `preprocessing/ir_spectral.py`). Microwave sounders are read on all their
  channels. `healda.inference.DENIALS` drops channels from a given date: MetOp-B
  AMSU-A 3 and 6 from 2025-01-01, and MetOp-C AMSU-A 4 from 2026-03-17.
- **Microwave sounders** carry no antenna-to-brightness-temperature correction.
  ATMS is read from `antenna_temperature`. The AMSU-A, AMSU-B and MHS archives
  name their column `brightness_temperature`, but apart from NOAA-15 and NOAA-16
  AMSU-A it holds antenna temperature. Supply these values uncorrected.
- **Hyperspectral infrared**: IASI is archived as spectral radiance and CrIS as
  integer-coded radiance. The loader converts both to brightness temperature by
  monochromatic Planck inversion (`brightness_temperature` in
  `preprocessing/ir_spectral.py`), with CODATA 2018 constants and the channel
  wavenumbers defined in that module. AIRS is archived as brightness temperature.
  To supply IASI or CrIS brightness temperature, convert it to radiance with
  `spectral_radiance_mw` from the same module, as `observations/adapters/e2s_nnja.py`
  does, so the loader recovers the same values. Brightness temperatures computed
  with other constants, channel wavenumbers or a band correction may carry small
  biases relative to the training data.
- **Geometry**: longitude is -180 to 180 in the archive and moved to 0 to 360 on
  load. Satellite zenith angle is an unsigned magnitude in the archive, and the
  loader gives it GSI's sign, negative on the first half of the scan.

## Invariants

- Timestamps are timezone-naive UTC values and archive partitions cover the
  requested context window.
- Coordinates and physical values use the units expected by the loader.
- Arrow types are safely castable to the package schema.
- Global channel, local channel, sensor, and platform identities remain stable.
- Every channel has finite normalization and validity metadata.
- Row-group statistics for time-selection columns are present when required.

## UFS replay archive (legacy)

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

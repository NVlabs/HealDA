# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wide-to-long conversion, thinning, and winner policies, on a synthetic archive.

build_archive writes a Parquet day in the archive's layout, one row group per DA
window, so the loader is exercised end to end without Lustre.
"""

import asyncio
import atexit
import functools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from healda.config.models import ModelConfigV1, ObsConfig
from healda.observations.loaders import nnja_wide as wide
from healda.observations.loaders.nnja_wide import NNJAWideLoader, winner_hash
from healda.observations.sensors import PLATFORM_NAME_TO_ID
from healda.observations.sensors_nnja import (
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
    get_global_channel_id,
)
from healda.observations.schema import GLOBAL_CHANNEL_ID
from healda.observations.preprocessing import ir_spectral, scan_geometry

# Publish the curated set plus a few channels outside it, so subsetting has to do work. The
# extras come from the archive's own vocabulary, which the registry validates against.
IASI_CURATED = ir_spectral.ir_channel_sets()["iasi"]
IASI_EXTRAS = [
    channel
    for channel in SENSOR_CONFIGS["iasi"].channels.tolist()
    if channel not in set(IASI_CURATED)
][:3]
IASI_CHANNELS = np.array(sorted(set(IASI_CURATED) | set(IASI_EXTRAS)))
CRIS_CHANNELS = np.array(sorted(ir_spectral.ir_channel_sets()["cris"]))

# What every CrIS column of the archive states, and without which the code cannot be
# reconstructed.
CRIS_ENCODING = {"reference": -100000, "scale": 7}

# Source channels, the SAIDs the sensor flies on, and its scan width. IASI's width is in
# source field-of-view units, two detectors to a model scan position; CrIS's is in looks,
# which it publishes apart from the detector.
SENSOR_SHAPE = {
    "atms": (np.arange(1, 23), (224, 225), 96),  # npp, n20
    "amsua": (np.arange(1, 16), (209, 223), 30),  # n18, n19
    "iasi": (IASI_CHANNELS, (3, 4), 120),  # metop-b, metop-a
    "cris": (CRIS_CHANNELS, (25, 26), 30),  # npp, n20
}
# Unless a test names another sensor it runs on ATMS, a microwave sounder the loader reads
# straight through in kelvin. MICROWAVE is the pair the fan-out test needs; IASI carries the
# infrared path, on its own archive.
MICROWAVE = ("atms", "amsua")
ATMS_CHANNELS = SENSOR_SHAPE["atms"][0]
ATMS_KEEP = scan_geometry.SCAN_GEOMETRY["atms"].keep

TARGET = pd.Timestamp("2022-09-05T00:00:00")
DAY = TARGET.strftime("%Y%m%d")
WINDOWS = (TARGET, TARGET + pd.Timedelta(hours=3))
ROWS = 512

# Coarse enough that the 512 footprints of a window collide in every cell.
NSIDE = 16
PARENTS = 64
CHILD_BITS = 2 * (wide.ARCHIVE_HPX_ORDER - (NSIDE.bit_length() - 1))

MEAN = 250.0
STDDEV = 5.0
MIN_VALID = 100.0
MAX_VALID = 400.0


def _wide_schema(sensor: str, nullable_pixel: bool) -> pa.Schema:
    channels, _platforms, _scan = SENSOR_SHAPE[sensor]
    prefix = wide.SENSOR_VALUE_PREFIX[sensor]
    fields = [
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("time_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("da_window", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("platform_id", pa.uint16()),
        pa.field("satellite_zenith_angle", pa.float32()),
        pa.field("solar_zenith_angle", pa.float32()),
        pa.field("field_of_view", pa.uint16()),
        pa.field("hpx2048_nest", pa.uint32(), nullable=nullable_pixel),
    ]
    if sensor in wide.SCAN_POSITION_COLUMN:
        fields.append(pa.field(wide.SCAN_POSITION_COLUMN[sensor], pa.uint16()))
    metadata = (
        {b"nnja.archive.value_encoding": json.dumps(CRIS_ENCODING).encode()}
        if sensor == "cris"
        else None
    )
    fields += [
        pa.field(f"{prefix}__ch_{channel:05d}", pa.float32(), metadata=metadata)
        for channel in channels
    ]
    return pa.schema(fields)


def _window_table(
    sensor: str, window: pd.Timestamp, seed: int, *, nulls: bool, invalid: bool
) -> pa.Table:
    channels, platforms, scan_positions = SENSOR_SHAPE[sensor]
    prefix = wide.SENSOR_VALUE_PREFIX[sensor]
    rng = np.random.default_rng(seed)
    rows = np.arange(ROWS)

    # Parent cells are drawn from a small set so most cells hold several footprints,
    # and the child bits below them vary so the coarsening is doing real work.
    parent = rng.integers(0, PARENTS, ROWS)
    pixel = ((parent << CHILD_BITS) | rng.integers(0, 1 << CHILD_BITS, ROWS)).astype(
        np.uint32
    )
    pixel_column = pa.array(pixel, type=pa.uint32())
    if nulls:
        mask = np.zeros(ROWS, dtype=bool)
        mask[[3, 17, 400]] = True
        pixel_column = pa.array(pixel, type=pa.uint32(), mask=mask)

    values = rng.normal(MEAN, STDDEV, (len(channels), ROWS)).astype(np.float32)
    if invalid:
        values[0, :7] = np.nan
        values[3, 11] = 1e4
        values[5, 12] = -1e4

    # A radiance sensor publishes the Planck radiance of that scene, so the loader has to
    # invert it to get back to kelvin.
    if wide.SENSOR_VALUE_PREFIX[sensor].startswith(wide.RADIANCE_PREFIXES):
        wavenumber = ir_spectral.wavenumber_cm_inverse(sensor, channels)[:, None]
        values = ir_spectral.spectral_radiance_mw(values, wavenumber)
        if sensor == "cris":
            # The code its column decodes from, the inverse of ir_spectral.radiance_mw.
            values = (
                values * 10.0 ** (CRIS_ENCODING["scale"] - 3)
                - CRIS_ENCODING["reference"]
            )
        values = values.astype(np.float32)

    columns = {
        "latitude": rng.uniform(-90, 90, ROWS),
        "longitude": rng.uniform(-180, 180, ROWS),
        # Spread across the window so the synoptic policy has something to rank.
        "time_utc": window + pd.to_timedelta(rng.integers(0, 10800, ROWS), unit="s"),
        "da_window": np.full(ROWS, window),
        "platform_id": np.array(platforms, dtype=np.uint16)[rows % len(platforms)],
        "satellite_zenith_angle": rng.uniform(0, 60, ROWS).astype(np.float32),
        "solar_zenith_angle": rng.uniform(0, 180, ROWS).astype(np.float32),
        "field_of_view": (1 + rows % scan_positions).astype(np.uint16),
        "hpx2048_nest": pixel_column,
    }
    if sensor in wide.SCAN_POSITION_COLUMN:
        # CrIS numbers the detector of its 3x3 array in field_of_view and the cross-track
        # look, which is the scan position, in field_of_regard.
        detectors = scan_geometry.SCAN_GEOMETRY[sensor].detectors
        columns["field_of_view"] = (1 + rows % detectors).astype(np.uint16)
        columns[wide.SCAN_POSITION_COLUMN[sensor]] = (1 + rows % scan_positions).astype(
            np.uint16
        )
    for index, channel in enumerate(channels):
        columns[f"{prefix}__ch_{channel:05d}"] = values[index]
    schema = _wide_schema(sensor, nulls)
    return pa.table([columns[field.name] for field in schema], schema=schema)


@functools.cache
def _scratch() -> Path:
    root = Path(tempfile.mkdtemp(prefix="nnja-loader-test-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


@functools.cache
def build_archive(
    *sensors: str,
    nulls: bool = False,
    invalid: bool = False,
    groups_per_window: int = 1,
) -> str:
    """Write one file per sensor-day, one row group per DA window, and return its root.

    The cache is on the write, not the path: each shape the tests ask for is built once,
    however many tests ask for it.
    """
    sensors = sensors or ("atms",)
    root = _scratch() / "-".join(
        (
            *sensors,
            f"g{groups_per_window}",
            *(["nulls"] * nulls),
            *(["invalid"] * invalid),
        )
    )
    for sensor in sensors:
        directory = root / sensor
        directory.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(
            directory / f"{DAY}.parquet", _wide_schema(sensor, nulls)
        ) as writer:
            for seed, window in enumerate(WINDOWS):
                table = _window_table(
                    sensor, window, seed, nulls=nulls, invalid=invalid
                )
                step = -(-table.num_rows // groups_per_window)
                for start in range(0, table.num_rows, step):
                    writer.write_table(table.slice(start, step))
    return str(root)


@functools.cache
def build_channel_table() -> str:
    # Every sensor always, so a loader can be pointed at any archive without rewriting it.
    global_ids = np.concatenate(
        [get_global_channel_id(name, SENSOR_SHAPE[name][0]) for name in SENSOR_SHAPE]
    )
    channels = len(global_ids)
    path = _scratch() / "channel_table.parquet"
    pq.write_table(
        pa.table(
            {
                GLOBAL_CHANNEL_ID.name: pa.array(
                    global_ids, type=GLOBAL_CHANNEL_ID.type
                ),
                "mean": np.full(channels, MEAN, dtype=np.float32),
                "stddev": np.full(channels, STDDEV, dtype=np.float32),
                "min_valid": np.full(channels, MIN_VALID, dtype=np.float32),
                "max_valid": np.full(channels, MAX_VALID, dtype=np.float32),
            }
        ),
        path,
    )
    return str(path)


def loader_for(archive_root: str | None = None, **overrides) -> NNJAWideLoader:
    """An ATMS loader over the plain archive unless the caller says otherwise."""
    options = {
        "channel_table_path": build_channel_table(),
        "sensors": ("atms",),
        "archive_root": archive_root or build_archive(),
        "thin_nside": NSIDE,
        "fov_keep_range": {s: scan_geometry.SCAN_GEOMETRY[s].keep for s in MICROWAVE},
    }
    options.update(overrides)
    return NNJAWideLoader(**options)


def plan_for(
    loader: NNJAWideLoader, sensor: str = "atms", day: str = DAY
) -> wide.ReadPlan:
    return loader.read_plan(
        sensor, day, pq.ParquetFile(loader._path(sensor, day)).schema_arrow
    )


def read_window(
    loader: NNJAWideLoader, group: int = 0, sensor: str = "atms"
) -> pa.Table:
    path = loader._path(sensor, DAY)
    return pq.ParquetFile(path).read_row_group(
        group, columns=plan_for(loader, sensor=sensor).columns
    )


def cell_keys(table: pa.Table, index: np.ndarray) -> np.ndarray:
    """The (platform, coarse pixel) cell each selected footprint represents."""
    pixel = wide._numpy(table["hpx2048_nest"]).astype(np.int64)[index] >> CHILD_BITS
    platform = wide._numpy(table["platform_id"])[index].astype(np.int64)
    return platform * (12 * NSIDE * NSIDE) + pixel


def test_winner_hash_is_deterministic_and_positionally_unbiased():
    rows = np.arange(1 << 16)
    assert np.array_equal(winner_hash(rows, 0), winner_hash(rows, 0))
    assert winner_hash(rows, 0).max() < 1 << 32
    assert (winner_hash(rows, 1) != winner_hash(rows, 0)).mean() > 0.99

    # Footprints sharing a cell are near-consecutive in the source, so a hash left
    # monotone in the row index would hand every cell to the same position.
    ranks = winner_hash(rows, 0).reshape(-1, 16).argmin(axis=1).mean() / 15
    assert 0.45 < ranks < 0.55


@pytest.mark.parametrize("winner", wide.WINNER_POLICIES)
def test_compiled_and_staged_selection_agree(winner):
    if wide._FUSED_SELECT is None:
        pytest.skip("numba is unavailable, so only the staged path exists")
    loader = loader_for(winner=winner)
    table = read_window(loader)

    fused = loader.select(table, "atms")
    saved, wide._FUSED_SELECT = wide._FUSED_SELECT, None
    try:
        staged = loader.select(table, "atms")
    finally:
        wide._FUSED_SELECT = saved
    np.testing.assert_array_equal(fused, staged)


def test_thinning_keeps_one_screened_footprint_per_cell():
    loader = loader_for()
    table = read_window(loader)
    index = loader.select(table, "atms")

    scan = wide._numpy(table["field_of_view"])
    eligible = np.flatnonzero((scan >= ATMS_KEEP[0]) & (scan <= ATMS_KEEP[1]))
    assert eligible.size < table.num_rows, "the FOV screen should drop scan edges"
    # Screening runs first, so the winners come from the survivors and cover every
    # cell the survivors occupy, exactly once.
    assert np.isin(index, eligible).all()
    np.testing.assert_array_equal(
        np.unique(cell_keys(table, index)),
        np.unique(cell_keys(table, eligible)),
    )


def test_random_repeats_and_resample_redraws():
    fixed = loader_for(winner="random")
    table = read_window(fixed)
    np.testing.assert_array_equal(
        fixed.select(table, "atms"), fixed.select(table, "atms")
    )

    redrawn = loader_for(winner="resample")
    before = redrawn.select(table, "atms")
    asyncio.run(redrawn.sel_time(pd.DatetimeIndex([TARGET])))
    after = redrawn.select(table, "atms")
    assert (before != after).mean() > 0.5
    # A different footprint per cell, never a different set of cells. The index is in
    # row order, and a redraw reorders the cells within it, so compare them sorted.
    np.testing.assert_array_equal(
        np.sort(cell_keys(table, before)), np.sort(cell_keys(table, after))
    )


def test_synoptic_keeps_the_footprint_nearest_an_analysis_hour():
    loader = loader_for(winner="synoptic")
    table = read_window(loader)
    index = loader.select(table, "atms")

    distance = wide.synoptic_distance(wide._numpy(table["time_utc"]))
    scan = wide._numpy(table["field_of_view"])
    eligible = np.flatnonzero((scan >= ATMS_KEEP[0]) & (scan <= ATMS_KEEP[1]))
    nearest = {}
    for row, key in zip(eligible, cell_keys(table, eligible)):
        if key not in nearest or distance[row] < distance[nearest[key]]:
            nearest[key] = row
    np.testing.assert_array_equal(
        index, np.sort(np.fromiter(nearest.values(), dtype=np.int64))
    )


@pytest.mark.parametrize("thin_nside", [None, NSIDE])
def test_footprints_without_a_pixel_are_dropped(thin_nside):
    loader = loader_for(build_archive(nulls=True), thin_nside=thin_nside)
    table = read_window(loader)
    unlocated = np.flatnonzero(
        ~table["hpx2048_nest"].is_valid().to_numpy(zero_copy_only=False)
    )
    assert unlocated.size, "the archive should carry null pixels"

    index = loader.select(table, "atms")
    assert index is not None
    assert not np.isin(index, unlocated).any()


@pytest.mark.parametrize("nulls", [False, True])
def test_screening_every_footprint_out_selects_nothing(nulls):
    # Null pixels force the staged path, otherwise the fused kernel runs. Both size a
    # reduction from the survivors, so both have to hold up when there are none.
    loader = loader_for(
        build_archive(nulls=nulls), fov_keep_range={"atms": (10_000, 10_001)}
    )
    assert loader.select(read_window(loader), "atms").size == 0

    combined = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"][0]
    assert combined.num_rows == 0
    assert combined.schema == loader.output_schema


@pytest.mark.parametrize("fused", [True, False])
def test_a_row_group_with_no_rows_selects_nothing(fused, monkeypatch):
    # The fused path bounds its key space before the kernel runs, so an empty table
    # reaches that bound without the kernel's own screening ever getting a say.
    if not fused:
        monkeypatch.setattr(wide, "_FUSED_SELECT", None)
    loader = loader_for()
    assert loader.select(read_window(loader).slice(0, 0), "atms").size == 0


def test_read_plan_is_cached():
    # Uncached, every row group would resolve the channel axis again.
    loader = loader_for()
    assert plan_for(loader) is plan_for(loader)


def test_expansion_reproduces_the_wide_values():
    loader = loader_for(thin_nside=None, normalize=False)
    plan = plan_for(loader)
    table = read_window(loader)
    kept = loader.select(table, "atms")
    long = loader.expand(loader.transform(table, plan, "atms", kept), plan, "atms")
    footprint_count = kept.size
    assert long.schema == loader.output_schema
    assert long.num_rows == footprint_count * plan.channel_count

    # Channel-major: long row c * footprints + f is footprint f of channel c.
    source = np.stack([wide._numpy(table[name])[kept] for name in plan.value_columns])
    np.testing.assert_array_equal(
        np.asarray(long["Observation"]).reshape(plan.channel_count, footprint_count),
        source,
    )
    np.testing.assert_array_equal(
        np.asarray(long["local_channel_id"]).reshape(
            plan.channel_count, footprint_count
        ),
        np.broadcast_to(plan.local_channel_id[:, None], source.shape),
    )
    np.testing.assert_array_equal(
        np.asarray(long["Latitude"]),
        np.tile(
            wide._numpy(table["latitude"])[kept].astype(np.float32), plan.channel_count
        ),
    )
    assert set(np.asarray(long["sensor_id"])) == {SENSOR_NAME_TO_ID["atms"]}
    for name in wide.NULL_COLUMNS:
        assert long[name].null_count == long.num_rows


def test_platform_channel_dropout_targets_metop_b_amsua_only():
    loader = loader_for(
        archive_root=build_archive("amsua"),
        sensors=("amsua",),
        thin_nside=None,
        fov_keep_range={},
        normalize=False,
        platform_channel_dropout=(
            ("amsua", "metop-b", 3, 1.0),
            ("amsua", "metop-b", 6, 1.0),
        ),
    )
    plan = plan_for(loader, sensor="amsua")
    table = read_window(loader, sensor="amsua")
    source_platform = np.full(table.num_rows, 3, dtype=np.uint16)
    source_platform[table.num_rows // 2 :] = 4
    table = table.set_column(
        table.schema.get_field_index("platform_id"),
        "platform_id",
        pa.array(source_platform, type=pa.uint16()),
    )

    # Rules are resolved per sample in sel_time and passed in, so state them here too.
    rules = loader._platform_channel_dropout["amsua"]
    long = loader.to_long(table, plan, "amsua", np.random.default_rng(0), rules)

    channel = np.asarray(long[GLOBAL_CHANNEL_ID.name])
    platform = np.asarray(long["Platform_ID"])
    target_channels = get_global_channel_id("amsua", [3, 6])
    metop_b = platform == PLATFORM_NAME_TO_ID["metop-b"]
    metop_a = platform == PLATFORM_NAME_TO_ID["metop-a"]
    assert not np.any(metop_b & np.isin(channel, target_channels))
    for global_channel_id in target_channels:
        assert np.count_nonzero(metop_a & (channel == global_channel_id)) == ROWS // 2
    assert long.num_rows == ROWS * len(SENSOR_SHAPE["amsua"][0]) - ROWS


def iasi_window(**overrides):
    """An IASI loader, its read plan, and the first row group of its day."""
    loader = loader_for(
        build_archive("iasi"),
        sensors=("iasi",),
        normalize=False,
        thin_nside=None,
        **overrides,
    )
    plan = plan_for(loader, sensor="iasi")
    path = loader._path("iasi", DAY)
    return loader, plan, pq.ParquetFile(path).read_row_group(0, columns=plan.columns)


@pytest.mark.parametrize("sensor", ["airs", "iasi", "cris"])
def test_every_curated_channel_has_a_wavenumber(sensor):
    # A typo in the CSV would otherwise surface as a missing archive column much later.
    channels = ir_spectral.ir_channel_sets()[sensor]
    assert len(set(channels)) == ir_spectral.IR_CHANNEL_SET_LENGTH
    assert np.all(ir_spectral.wavenumber_cm_inverse(sensor, channels) > 0)


def test_an_unknown_channel_preset_is_rejected():
    with pytest.raises(ValueError, match="preset"):
        loader_for(ir_channels="ir17")


@pytest.mark.parametrize("ir_channels", ["ir32", "ir48", None])
def test_iasi_reads_the_channels_a_preset_names(ir_channels):
    # The archive publishes the 48 curated channels plus three the set does not name, so
    # None is visibly wider than the whole curated set rather than equal to it.
    _loader, plan, _table = iasi_window(ir_channels=ir_channels)
    expected = (
        IASI_CHANNELS
        if ir_channels is None
        else IASI_CURATED[: ir_spectral.IR_CHANNEL_PRESETS[ir_channels]]
    )
    assert plan.channel_count == len(expected)
    assert [int(name[-5:]) for name in plan.value_columns] == sorted(expected)


def test_ir32_is_a_prefix_of_ir48():
    # Presets are prefix lengths of one ranking, so the smaller cannot name a channel the
    # larger leaves out.
    small, large = (ir_spectral.ir_channel_preset(name) for name in ("ir32", "ir48"))
    for sensor, channels in small.items():
        assert channels == large[sensor][: len(channels)]


def test_explicit_channel_numbers_are_read_in_archive_order():
    # The analysis path: name channels outright rather than take a prefix of the ranking.
    wanted = [IASI_CURATED[40], IASI_CURATED[0], IASI_CHANNELS[-1]]
    _loader, plan, _table = iasi_window(ir_channels={"iasi": wanted})
    assert [int(name[-5:]) for name in plan.value_columns] == sorted(wanted)


def test_a_channel_the_archive_lacks_is_rejected():
    with pytest.raises(ValueError, match="absent"):
        iasi_window(ir_channels={"iasi": [1, 999_99]})


def test_naming_no_channel_for_a_sensor_is_rejected():
    with pytest.raises(ValueError, match="no channel"):
        loader_for(ir_channels={"iasi": []})


def test_a_config_written_before_the_presets_still_loads():
    # Checkpoints hold a count of the ranked set where the loader now wants a preset name, and
    # come back through json, which has no tuple.
    stored = {
        "obs_config": {
            "use_nnja_sat": True,
            "conv_level_channels": True,
            "nnja_ir_channels": 32,
            "nnja_sensors": ["atms", "iasi"],
        }
    }
    config = ModelConfigV1.loads(json.dumps(stored)).obs_config
    assert config.nnja_ir_channels == "ir32"
    assert len(ir_spectral.ir_channel_preset(config.nnja_ir_channels)["iasi"]) == 32
    assert config.nnja_sensors == ("atms", "iasi")
    # Checkpoints older still have no sensor field, which means every archive sensor.
    assert ObsConfig(use_nnja_sat=True, conv_level_channels=True).nnja_sensors is None


def test_iasi_radiance_reads_back_as_kelvin():
    # The fixture published the Planck radiance of a scene at MEAN kelvin, so inverting it
    # has to land back on that scene and not merely on something finite.
    loader, plan, table = iasi_window()
    assert plan.wavenumber is not None

    kelvin = loader.transform(table, plan, "iasi", None).values
    assert np.abs(kelvin.mean() - MEAN) < 0.5
    assert np.abs(kelvin.std() - STDDEV) < 0.5


def test_iasi_can_come_back_as_radiance_screened_in_kelvin():
    # The bounds in the channel table are kelvin, so a radiance read still has to screen on
    # the inverted value: here one radiance is positive, finite, and far too cold.
    loader, plan, table = iasi_window(ir_units="radiance")
    name = plan.value_columns[0]
    spoiled = table.column(name).to_numpy(zero_copy_only=False).copy()
    spoiled[0] = 1e-8
    table = table.set_column(
        table.schema.get_field_index(name), name, pa.array(spoiled, pa.float32())
    )

    result = loader.transform(table, plan, "iasi", None)
    assert result.values[0, 0] == pytest.approx(1e-8)
    assert result.valid is not None and not result.valid[0, 0]
    # The rest are radiances, orders of magnitude away from the kelvin they invert to.
    assert result.values[result.valid].max() < MEAN


def test_radiance_with_normalization_is_rejected():
    with pytest.raises(ValueError, match="normalize=False"):
        loader_for(ir_units="radiance")


def test_iasi_radiance_at_or_below_zero_is_invalid():
    # Planck inversion is undefined there and returns NaN, which has to be caught by the
    # range check rather than reaching the model as a kelvin value.
    loader, plan, table = iasi_window()
    name = plan.value_columns[0]
    spoiled = table.column(name).to_numpy(zero_copy_only=False).copy()
    spoiled[:3] = (0.0, -1.0, np.nan)
    table = table.set_column(
        table.schema.get_field_index(name), name, pa.array(spoiled, pa.float32())
    )

    result = loader.transform(table, plan, "iasi", None)
    assert result.valid is not None
    assert not result.valid[0, :3].any()
    assert result.valid[0, 3:].all()
    assert np.isfinite(result.values[result.valid]).all()


@pytest.mark.parametrize(
    "column", ["satellite_zenith_angle", "solar_zenith_angle", "latitude"]
)
def test_a_footprint_without_geometry_is_dropped_in_every_channel(column):
    # The metadata featurizer fouriers these unguarded, so a null here reaches the model as
    # NaN and poisons the step's whole gradient. Real archive days carry a few.
    loader = loader_for(thin_nside=None, normalize=False)
    plan = plan_for(loader)
    table = read_window(loader)
    spoiled = wide._numpy(table[column]).copy()
    spoiled[:2] = np.nan
    index = table.schema.get_field_index(column)
    table = table.set_column(index, column, pa.array(spoiled, table.field(index).type))

    result = loader.transform(table, plan, "atms", None)
    assert result.valid is not None
    assert not result.valid[:, :2].any()
    assert result.valid[:, 2:].all()

    long = loader.expand(result, plan, "atms")
    assert long.num_rows == (table.num_rows - 2) * plan.channel_count
    for name in ("Latitude", "Sat_Zenith_Angle", "Sol_Zenith_Angle", "Scan_Angle"):
        assert np.isfinite(np.asarray(long[name])).all()


@pytest.mark.parametrize("ir_channels", ["ir32", None, {"iasi": [1]}])
def test_an_ir_selection_leaves_a_microwave_sensor_alone(ir_channels):
    assert "atms" not in ir_spectral.ir_channel_sets()
    plan = plan_for(loader_for(ir_channels=ir_channels))
    assert plan.channel_count == len(ATMS_CHANNELS) == 22


def touch_days(root: Path, sensors: list[str], days: list[str]) -> Path:
    # _path decides on names alone, so empty files are enough.
    for sensor in sensors:
        (root / sensor).mkdir(parents=True)
        for day in days:
            (root / sensor / f"{day}.parquet").touch()
    return root


def fresh_archive(sensors: list[str], days: list[str]) -> Path:
    # A root per call: _scratch is shared for the session.
    return touch_days(Path(tempfile.mkdtemp(dir=_scratch())), sensors, days)


def test_a_thinned_store_serves_the_sounders_it_covers():
    root = fresh_archive(["atms", "iasi"], [DAY])
    touch_days(root / "ir48", ["iasi"], [DAY])
    loader = loader_for(str(root), sensors=("atms", "iasi"), ir_channels="ir32")
    # ir32 is a subset of ir48, so that store serves it; microwave has no store at all.
    assert loader._path("iasi", DAY) == str(root / "ir48" / "iasi" / f"{DAY}.parquet")
    assert loader._path("atms", DAY) == str(root / "atms" / f"{DAY}.parquet")
    full = loader_for(str(root), sensors=("iasi",), ir_channels=None)
    assert full._path("iasi", DAY) == str(root / "iasi" / f"{DAY}.parquet")


def test_a_day_the_thinned_store_lacks_reads_full_width():
    # The caller skips a day it finds no file for, so a partial store would drop those days.
    root = fresh_archive(["cris"], [DAY, "20240102"])
    touch_days(root / "ir48", ["cris"], [DAY])
    loader = loader_for(str(root), sensors=("cris",), ir_channels="ir48")
    assert loader._path("cris", DAY) == str(root / "ir48" / "cris" / f"{DAY}.parquet")
    assert loader._path("cris", "20240102") == str(root / "cris" / "20240102.parquet")


def test_cris_screens_and_angles_on_field_of_regard():
    # CrIS is the one sensor whose scan position is not field_of_view: that column holds the
    # detector, 1..9. A loader reading it as the position would screen every row out here and
    # emit an angle nine positions wide instead of thirty.
    keep = (10, 30)
    loader = loader_for(
        build_archive("cris"),
        sensors=("cris",),
        thin_nside=None,
        normalize=False,
        fov_keep_range={"cris": keep},
    )
    plan = plan_for(loader, sensor="cris")
    assert plan.scan_columns == ("field_of_regard",)

    table = read_window(loader, sensor="cris")
    kept = loader.select(table, "cris")
    regard = wide._numpy(table["field_of_regard"])[kept]
    assert kept.size
    assert ((regard >= keep[0]) & (regard <= keep[1])).all()

    long = loader.expand(loader.transform(table, plan, "cris", kept), plan, "cris")
    expected = scan_geometry.scan_angle(
        "cris", wide._numpy(table["field_of_view"])[kept], regard
    )
    np.testing.assert_allclose(
        np.asarray(long["Scan_Angle"]),
        np.tile(expected, plan.channel_count),
        rtol=0,
        atol=1e-4,
    )
    # The Planck inversion of the decoded code, so the whole radiance path is covered too.
    assert np.isfinite(np.asarray(long["Observation"])).all()
    np.testing.assert_allclose(
        np.asarray(long["Observation"]).mean(), MEAN, rtol=0, atol=1.0
    )


def test_zenith_is_emitted_signed_by_scan_side():
    loader = loader_for(thin_nside=None, normalize=False)
    plan = plan_for(loader)
    table = read_window(loader)
    kept = loader.select(table, "atms")
    long = loader.expand(loader.transform(table, plan, "atms", kept), plan, "atms")

    source = wide._numpy(table["satellite_zenith_angle"])[kept]
    assert (source > 0).all(), "the archive stores an unsigned magnitude"
    zenith = np.asarray(long["Sat_Zenith_Angle"])
    scan = np.asarray(long["Scan_Angle"])
    # Both halves have to be present, or comparing two all-positive sides proves nothing.
    assert (scan < 0).any() and (scan > 0).any()
    np.testing.assert_array_equal(zenith < 0, scan < 0)
    np.testing.assert_allclose(
        np.abs(zenith), np.tile(source, plan.channel_count), rtol=0, atol=1e-5
    )


def test_out_of_range_cells_are_dropped_and_the_rest_normalized():
    loader = loader_for(build_archive(invalid=True), thin_nside=None)
    plan = plan_for(loader)
    table = read_window(loader)
    kept = loader.select(table, "atms")
    long = loader.expand(loader.transform(table, plan, "atms", kept), plan, "atms")
    source = np.stack([wide._numpy(table[name])[kept] for name in plan.value_columns])
    valid = (source >= MIN_VALID) & (source <= MAX_VALID)
    assert not valid.all(), "the archive should exercise the masked path"
    assert long.num_rows == int(valid.sum())
    np.testing.assert_allclose(
        np.asarray(long["Observation"]),
        (source[valid] - MEAN) / STDDEV,
        rtol=0,
        atol=1e-3,
    )


def test_sel_time_returns_the_context_windows():
    loader = loader_for()
    tables = asyncio.run(loader.sel_time(pd.DatetimeIndex([TARGET])))["obs_v2"]
    assert len(tables) == 1

    combined = tables[0]
    assert combined.schema == loader.output_schema
    assert set(np.asarray(combined["DA_window"])) == {
        window.to_datetime64() for window in WINDOWS
    }
    per_window = [
        loader.select(read_window(loader, group), "atms").size
        for group in range(len(WINDOWS))
    ]
    channels = plan_for(loader).channel_count
    assert combined.num_rows == sum(per_window) * channels


def test_read_ahead_agrees_with_the_serial_path():
    times = pd.DatetimeIndex([TARGET])
    serial = loader_for(build_archive(groups_per_window=2), read_ahead=False)
    pipelined = loader_for(build_archive(groups_per_window=2), read_ahead=True)

    jobs = len(pipelined._row_group_jobs("atms", pipelined._interval_times(TARGET)))
    assert jobs > 2, "the queue should have to refill past what priming covers"

    expected = asyncio.run(serial.sel_time(times))["obs_v2"][0]
    assert expected.num_rows, "the archive should produce rows on both paths"
    assert expected.equals(asyncio.run(pipelined.sel_time(times))["obs_v2"][0])


def test_sensors_fan_out_into_one_table():
    times = pd.DatetimeIndex([TARGET])
    loader = loader_for(build_archive("atms", "amsua"), sensors=MICROWAVE)
    combined = asyncio.run(loader.sel_time(times))["obs_v2"][0]

    assert combined.schema == loader.output_schema
    assert set(np.asarray(combined["sensor_id"])) == {
        SENSOR_NAME_TO_ID[sensor] for sensor in MICROWAVE
    }
    # Sensors are read on separate threads and staged by window, so each must
    # contribute exactly what it does alone.
    alone = sum(
        asyncio.run(
            loader_for(build_archive("atms", "amsua"), sensors=(sensor,)).sel_time(
                times
            )
        )["obs_v2"][0].num_rows
        for sensor in MICROWAVE
    )
    assert combined.num_rows == alone

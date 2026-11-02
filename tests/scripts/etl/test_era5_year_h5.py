# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import json
import subprocess
import sys
import types
from pathlib import Path

import h5py
import pytest

from scripts.etl.era5.channels import ARCO_VARIABLE, hour_index
from scripts.etl.era5.channels import (
    C73_LEVELS,
    CHAN_BYTES,
    CHANNELS_99,
    HOURS_PER_BLOCK,
    channel_runs,
    resolve_channels,
    source_groups,
    write_plan,
)
from scripts.etl.era5.year_h5 import RangeReader


def test_c73_layout_matches_the_archive():
    names = resolve_channels("c73")
    assert len(names) == 73
    assert names[:8] == ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv"]
    assert (names[8], names[72]) == ("u50", "q1000")
    assert resolve_channels("c75")[73:] == ["sst", "sic"]
    assert resolve_channels("w") == [f"w{level}" for level in C73_LEVELS]
    assert resolve_channels("d2m_skt_sp") == ["d2m", "skt", "sp"]


def test_channels_99_order_is_frozen():
    # The store is written once against this exact order, so pin the seams and
    # uniqueness, not just the count.
    assert len(CHANNELS_99) == 99
    assert len(set(CHANNELS_99)) == 99
    assert CHANNELS_99[74:76] == ("sp", "W1000")
    assert CHANNELS_99[-5:] == ("stl1", "stl2", "swvl1", "swvl2", "sd")


def test_resolve_channels_rejects_a_repeated_name():
    # A repeat would put one output channel in a group twice, so two writes would
    # overlap. resolve_channels is the only thing that stops it.
    assert resolve_channels("sst,sic,z500") == ["sst", "sic", "z500"]
    with pytest.raises(ValueError):
        resolve_channels("sst,sst")


def test_uppercase_pressure_channels_use_lowercase_source_names():
    groups = source_groups(["T10", "U10", "V10", "Z10"])
    assert [group.raw_name for group in groups] == ["t", "u", "v", "z"]


def test_channel_runs_split_on_a_gap():
    # Writes are one pwrite per run, so a group's scattered outputs must not merge.
    groups = {g.raw_name: g for g in source_groups(["z500", "t2m", "z850", "z1000"])}
    assert channel_runs(groups["z"]) == [(0, [500]), (2, [850, 1000])]
    assert channel_runs(groups["2t"]) == [(1, [None])]
    # One group per raw variable, holding every level the set wants, so a pressure
    # chunk is inflated once.
    c75 = source_groups(resolve_channels("c75"))
    assert len(c75) == 15
    assert all(len(g.levels) == 13 for g in c75 if g.product == "pl")


@pytest.mark.parametrize("spec", ["c73", "c75", "sst_sic", "q1000,sic,u50,sst"])
def test_write_plan_tiles_the_dataset_exactly(spec):
    """Workers pwrite by offset with nothing checking them, so the ranges every task of
    a year writes must cover the dataset once: a gap reads back as zeros and an overlap
    silently destroys another variable."""
    names = resolve_channels(spec)
    ndays, data_offset = 3, 2048
    stride = len(names) * CHAN_BYTES
    nslabs = ndays * HOURS_PER_BLOCK * len(names)

    # Every write is a whole number of channel slabs, so count in slabs, not bytes.
    written = bytearray(nslabs)
    for group in source_groups(names):
        for day in range(ndays):
            for _, offset, keys in write_plan(group, day, data_offset, stride):
                start, remainder = divmod(offset - data_offset, CHAN_BYTES)
                assert remainder == 0, "write is not channel aligned"
                assert 0 <= start and start + len(keys) <= nslabs, "write runs past end"
                region = written[start : start + len(keys)]
                assert not any(region), "two tasks write the same bytes"
                written[start : start + len(keys)] = b"\x01" * len(keys)

    assert all(written), "some bytes are never written"


def test_write_plan_supports_six_hourly_days():
    names = resolve_channels("sst_sic")
    stride = len(names) * CHAN_BYTES
    group = source_groups(names)[0]
    plan = list(
        write_plan(
            group,
            day=2,
            data_offset=2048,
            stride=stride,
            steps_per_day=4,
        )
    )
    assert len(plan) == 4
    assert plan[-1][1] == 2048 + (11 * len(names)) * CHAN_BYTES


def test_write_plan_addresses_match_c_order():
    names = resolve_channels("c75")
    stride = len(names) * CHAN_BYTES
    group = next(g for g in source_groups(names) if g.raw_name == "z")
    channel = group.channel_indices[0]

    plan = list(write_plan(group, day=2, data_offset=2048, stride=stride))
    assert len(plan) == HOURS_PER_BLOCK
    hour, offset, keys = plan[5]
    time_index = 2 * HOURS_PER_BLOCK + 5
    assert hour == 5
    assert len(keys) == 13
    assert offset == 2048 + (time_index * len(names) + channel) * CHAN_BYTES


def test_arco_channel_mapping_and_hour_index():
    assert ARCO_VARIABLE["z"] == "geopotential"
    assert ARCO_VARIABLE["tcw"] == "total_column_water"
    assert hour_index(datetime.datetime(1900, 1, 2, 6)) == 30


def test_range_reader_retries_s3_reads(monkeypatch):
    calls = 0

    def get_range(handle, key, *, start, length):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("transient")
        return b"data"

    monkeypatch.setitem(
        sys.modules, "obstore", types.SimpleNamespace(get_range=get_range)
    )
    monkeypatch.setattr("scripts.etl.era5.year_h5.time.sleep", lambda _: None)
    reader = RangeReader.__new__(RangeReader)
    reader.source = "s3://bucket/key"
    reader.is_s3 = True
    reader.key = "key"
    reader._handle = lambda: object()

    assert reader.read(0, 4) == b"data"
    assert calls == 3


def test_verify_rejects_sidecar_and_h5_channel_disagreement(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "data.json").write_text(
        json.dumps({"dhours": 6, "coords": {"channel": ["t10"]}})
    )
    year = tmp_path / "2026.h5"
    with h5py.File(year, "w") as handle:
        handle.create_dataset("fields", shape=(4, 1, 1, 1), dtype="f4")
        handle.attrs["channel"] = ["T10"]

    repo = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.etl.era5.verify_year_h5", year, "--quick"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "embedded channels differ from its sidecar" in result.stderr


def test_each_day_reads_the_last_source_started_by_then(tmp_path):
    from scripts.etl.era5.build_year_h5 import read_sources, roots_for, source_for

    path = tmp_path / "sources.toml"
    path.write_text(
        """
[[source]]
start = 2023-11-01
pl = "s3://mirror/e5.oper.an.pl"
sfc = "s3://mirror/e5.oper.an.sfc"
profile = "mine"

[[source]]
start = 1940-01-01
pl = "s3://flat_pl"
sfc = "s3://flat_sfc"
monthly = false
"""
    )
    sources = read_sources(str(path))
    assert source_for(datetime.date(2023, 10, 31), sources) == 0
    assert source_for(datetime.date(2023, 11, 1), sources) == 1
    assert sources[1].profile == "mine" and sources[0].profile is None
    assert roots_for(sources, 0, 2023, 365) == ["s3://flat_pl", "s3://flat_sfc"]
    assert roots_for(sources, 1, 2023, 365) == [
        "s3://mirror/e5.oper.an.pl/202311",
        "s3://mirror/e5.oper.an.sfc/202311",
        "s3://mirror/e5.oper.an.pl/202312",
        "s3://mirror/e5.oper.an.sfc/202312",
    ]


def run_module(module: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=Path(__file__).parents[3],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_init_writes_the_sidecar_onto_the_file(tmp_path):
    sources = tmp_path / "sources.toml"
    sources.write_text(
        f"""
[[source]]
start = 1940-01-01
pl = "{tmp_path / "pl"}"
sfc = "{tmp_path / "sfc"}"
monthly = false
"""
    )
    year = tmp_path / "2025.h5"
    result = run_module(
        "scripts.etl.era5.build_year_h5",
        "--init-only",
        "--year", 2025,
        "--archive-end", "2025-01-01",
        "--output", year,
        "--cache-dir", tmp_path / "cache",
        "--sources", sources,
        "--channels", "t2m,z500",
        "--dhours", 6,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr

    sidecar = json.loads((tmp_path / "metadata" / "data.json").read_text())
    assert sidecar["dataset_name"] == "era5-6hourly-t2m,z500"
    assert sidecar["attrs"]["sources"] == f"{tmp_path / 'pl'}, {tmp_path / 'sfc'}"
    with h5py.File(year, "r") as f:
        assert f["fields"].shape == (4, 2, 721, 1440)
        assert list(f.attrs["channel"]) == sidecar["coords"]["channel"]
        assert f.attrs["dhours"] == sidecar["dhours"]
        assert f.attrs["dataset_name"] == sidecar["dataset_name"]
        assert f.attrs["sources"] == sidecar["attrs"]["sources"]


def test_arco_manifest_lists_only_the_hours_at_the_cadence(tmp_path):
    def array(dims: list[str], shape: list[int], chunks: list[int]) -> dict:
        return {
            "zarray": {"shape": shape, "chunks": chunks},
            "zattrs": {"_ARRAY_DIMENSIONS": dims},
        }

    nhours = 2_000_000
    arrays = {
        "latitude": array(["latitude"], [721], [721]),
        "longitude": array(["longitude"], [1440], [1440]),
        "level": array(["level"], [37], [37]),
        "time": array(["time"], [nhours], [nhours]),
        "2m_temperature": array(
            ["time", "latitude", "longitude"], [nhours, 721, 1440], [1, 721, 1440]
        ),
    }
    metadata = {}
    for name, parts in arrays.items():
        metadata[f"{name}/.zarray"] = parts["zarray"]
        metadata[f"{name}/.zattrs"] = parts["zattrs"]
    zmetadata = tmp_path / ".zmetadata"
    zmetadata.write_text(json.dumps({"metadata": metadata}))

    output = tmp_path / "arco.files"
    result = run_module(
        "scripts.etl.era5.make_arco_manifest",
        "--metadata", zmetadata,
        "--output", output,
        "--start", "2025-01-03T00:00",
        "--stop", "2025-01-04T00:00",
        "--channels", "t2m",
        "--dhours", 6,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr

    first = hour_index(datetime.datetime(2025, 1, 3))
    keys = output.read_text().split()
    assert [key for key in keys if key.startswith("2m_temperature/")] == [
        "2m_temperature/.zarray",
        "2m_temperature/.zattrs",
        *(f"2m_temperature/{first + hour}.0.0" for hour in (0, 6, 12, 18)),
    ]
    assert "time/0" in keys


def test_arco_label_splits_final_era5_from_era5t():
    from scripts.etl.era5.write_arco_year_h5 import arco_label

    day = datetime.date
    assert arco_label(day(2026, 6, 29), day(2026, 7, 2), day(2026, 6, 30)) == (
        "ARCO ERA5 2026-06-29 through 2026-06-30;"
        " ARCO ERA5T 2026-07-01 through 2026-07-02"
    )
    assert arco_label(day(2025, 1, 3), day(2025, 1, 3), day(2026, 6, 30)) == (
        "ARCO ERA5 2025-01-03 through 2025-01-03"
    )


def test_verify_reads_build_and_arco_journals_over_the_checked_days(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "data.json").write_text(
        json.dumps({"dhours": 6, "coords": {"channel": ["t2m", "msl"]}})
    )
    year = tmp_path / "2025.h5"
    with h5py.File(year, "w") as handle:
        handle.create_dataset("fields", shape=(12, 2, 1, 1), dtype="f4")

    # Two variables, three days. build_year_h5 ids are group * 3 + day (the file's
    # days), write_arco_year_h5 ids group * 365 + day (the calendar year's).
    state = tmp_path / "state"
    (state / "2025.jobs").mkdir(parents=True)
    (state / "2025.jobs" / "0.log").write_text("0\n3\n")
    (state / "2025.arco-jobs").mkdir()
    (state / "2025.arco-jobs" / "0.log").write_text("1\n366\n")

    result = run_module("scripts.etl.era5.verify_year_h5", year, "--quick")
    assert result.returncode == 1
    assert "rebuild days: 2\n" in result.stdout

    result = run_module(
        "scripts.etl.era5.verify_year_h5", year, "--quick", "--day-stop", 2
    )
    assert result.returncode == 0, result.stdout
    assert "all 4 tasks recorded complete" in result.stdout

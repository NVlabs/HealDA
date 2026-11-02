# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Splice NCAR radiosonde messages into NOAA GDAS PrepBUFR cycles.

NOAA's daily first posting of a PrepBUFR cycle lacks the high-resolution radiosonde reports
that its later re-posting adds. NCAR RDA ds337.0 publishes two copies of the same cycles:

- PREPNR daily tars, ``https://data.gdex.ucar.edu/d337000/tarfiles/YYYY/prepbufr.YYYYMMDD.nr.tar.gz``:
  complete radiosondes (BUFR data category 102), restricted aircraft removed;
- PREP48H, ``https://data.gdex.ucar.edu/d337000/prep48h/YYYY/prepbufr.gdas.YYYYMMDD.tHHz.nr.48h``:
  identical to NOAA's first posting.

``splice`` writes NOAA's file with its category-102 block replaced by PREPNR's category-102
messages; ``census`` prints messages, subsets and a digest per category for the three sources,
which is what the splice relies on (table messages equal, PREPNR radiosondes equal to a NOAA
re-post where one exists).

    python -m scripts.etl.nnja.conv.splice_radiosondes census --prepnr-root DIR --prep48h-root DIR 20260801 ... > census.tsv
    python -m scripts.etl.nnja.conv.splice_radiosondes splice --prepnr-root DIR --out-root DIR 20260801 ...

Layouts: NOAA ``<noaa-root>/YYYY/MM/prepbufr/gdas.YYYYMMDD.tHHz.prepbufr.nr`` (``--noaa-root``
defaults to ``$NNJA_ROOT/bufr/prepbufr``), PREPNR ``<prepnr-root>/YYYYMMDD/prepbufr.gdas.YYYYMMDD.tHHz.nr``
(one extracted tar per day), PREP48H ``<prep48h-root>/prepbufr.gdas.YYYYMMDD.tHHz.nr.48h``.
The output tree has the NOAA layout.
"""

import argparse
import hashlib
import os
from dataclasses import dataclass

RADIOSONDES = 102
TABLES = 11
CENSUS_CATEGORIES = (TABLES, RADIOSONDES, 103, 104, 105)
CYCLES = ("00", "06", "12", "18")


@dataclass(frozen=True)
class Message:
    category: int
    subsets: int
    body: bytes
    # NCEP writes each message zero-padded to the next 8-byte boundary.
    padded: bytes


def read_messages(data: bytes) -> list[Message]:
    messages, pos = [], 0
    while pos < len(data):
        if data[pos : pos + 4] != b"BUFR":
            raise ValueError(f"no BUFR message at byte {pos}")
        length = int.from_bytes(data[pos + 4 : pos + 7], "big")
        if data[pos + length - 4 : pos + length] != b"7777":
            raise ValueError(f"message at byte {pos} does not end in 7777")
        end = data.find(b"BUFR", pos + length)
        end = len(data) if end < 0 else end
        if data[pos + length : end].strip(b"\x00") or (end - pos) % 8:
            raise ValueError(
                f"message at byte {pos}: padding is not zeros to an 8-byte boundary"
            )
        edition = data[pos + 7]
        s1 = pos + 8
        s1_length = int.from_bytes(data[s1 : s1 + 3], "big")
        category, flags = (
            (data[s1 + 10], data[s1 + 9])
            if edition >= 4
            else (data[s1 + 8], data[s1 + 7])
        )
        s3 = s1 + s1_length
        if flags & 0x80:
            s3 += int.from_bytes(data[s3 : s3 + 3], "big")
        subsets = int.from_bytes(data[s3 + 4 : s3 + 6], "big")
        messages.append(
            Message(category, subsets, data[pos : pos + length], data[pos:end])
        )
        pos = end
    return messages


def digest(messages) -> str:
    sha = hashlib.sha256()
    for message in messages:
        sha.update(message.body)
    return sha.hexdigest()


def splice(noaa: bytes, prepnr: bytes) -> tuple[bytes, int, int]:
    """NOAA's file with its radiosonde block replaced by PREPNR's, plus both block sizes."""
    old, new = read_messages(noaa), read_messages(prepnr)
    if digest(m for m in old if m.category == TABLES) != digest(
        m for m in new if m.category == TABLES
    ):
        raise ValueError("table messages differ")
    at = [i for i, m in enumerate(old) if m.category == RADIOSONDES]
    if not at or at != list(range(at[0], at[-1] + 1)):
        raise ValueError("NOAA radiosonde messages are missing or not contiguous")
    sondes = [m for m in new if m.category == RADIOSONDES]
    if sum(m.subsets for m in sondes) < sum(old[i].subsets for i in at):
        raise ValueError("PREPNR has fewer radiosonde subsets than NOAA")
    out = old[: at[0]] + sondes + old[at[-1] + 1 :]
    data = b"".join(m.padded for m in out)
    check = read_messages(data)
    if digest(m for m in check if m.category != RADIOSONDES) != digest(
        m for m in old if m.category != RADIOSONDES
    ):
        raise ValueError("non-radiosonde messages changed")
    if digest(m for m in check if m.category == RADIOSONDES) != digest(sondes):
        raise ValueError("radiosonde messages do not match PREPNR")
    return data, len(at), len(sondes)


def noaa_path(root, day, hour):
    return f"{root}/{day[:4]}/{day[4:6]}/prepbufr/gdas.{day}.t{hour}z.prepbufr.nr"


def prepnr_path(root, day, hour):
    return f"{root}/{day}/prepbufr.gdas.{day}.t{hour}z.nr"


def prep48h_path(root, day, hour):
    return f"{root}/prepbufr.gdas.{day}.t{hour}z.nr.48h"


def read(path):
    with open(path, "rb") as handle:
        return handle.read()


def run_splice(args):
    print("cycle\tnoaa_sonde_msgs\tprepnr_sonde_msgs\tbytes_added\tstatus")
    for day in args.days:
        for hour in CYCLES:
            source = noaa_path(args.noaa_root, day, hour)
            target = noaa_path(args.out_root, day, hour)
            try:
                noaa = read(source)
                data, old, new = splice(
                    noaa, read(prepnr_path(args.prepnr_root, day, hour))
                )
            except (OSError, ValueError) as error:
                print(f"{day}{hour}\t\t\t\tFAIL {error}")
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target + ".tmp", "wb") as handle:
                handle.write(data)
            os.replace(target + ".tmp", target)
            print(f"{day}{hour}\t{old}\t{new}\t{len(data) - len(noaa)}\tok")


def run_census(args):
    sources = {
        "noaa": (noaa_path, args.noaa_root),
        "prepnr": (prepnr_path, args.prepnr_root),
    }
    if args.prep48h_root:
        sources["prep48h"] = (prep48h_path, args.prep48h_root)
    print("day\tcycle\tsource\tcategory\tmessages\tsubsets\tsha")
    for day in args.days:
        for hour in CYCLES:
            for name, (path_of, root) in sources.items():
                path = path_of(root, day, hour)
                if not os.path.exists(path):
                    print(day, hour, name, "missing", "", "", "", sep="\t")
                    continue
                messages = read_messages(read(path))
                for category in CENSUS_CATEGORIES:
                    part = [m for m in messages if m.category == category]
                    subsets = sum(m.subsets for m in part)
                    print(
                        day,
                        hour,
                        name,
                        category,
                        len(part),
                        subsets,
                        digest(part)[:12],
                        sep="\t",
                    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("splice", "census"):
        command = commands.add_parser(name)
        command.add_argument(
            "days", nargs="+", help="YYYYMMDD; all four cycles of each day"
        )
        command.add_argument(
            "--noaa-root",
            default=os.path.join(os.environ.get("NNJA_ROOT", ""), "bufr", "prepbufr"),
        )
        command.add_argument("--prepnr-root", required=True)
    commands.choices["splice"].add_argument("--out-root", required=True)
    commands.choices["census"].add_argument("--prep48h-root")
    args = parser.parse_args()
    if args.command == "splice" and os.path.abspath(args.out_root) == os.path.abspath(
        args.noaa_root
    ):
        parser.error("--out-root must differ from --noaa-root")
    if args.command == "splice":
        run_splice(args)
    else:
        run_census(args)


if __name__ == "__main__":
    main()

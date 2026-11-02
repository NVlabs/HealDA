# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader for a BUFR Table-B descriptor catalogue CSV.

A catalogue is a curated subset: a missing row does not mean a template lacks
the field (the footer's source-structure record answers that). The recorded
packing is base Table B only; template operators can change the effective
packing, so observed values are not validated against it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Descriptor:
    """One Table-B descriptor and its base packing."""

    descriptor: str
    mnemonic: str
    meaning: str
    units: str
    base_scale: int
    base_reference: int
    base_bits: int
    value_kind: str
    interpretation_table_key: str | None
    # Optional columns. ``column_name`` is the logical Parquet column;
    # ``slot1_promoted`` promotes the first occurrence to a flat scalar.
    column_name: str = ""
    slot1_promoted: bool = False


def descriptor_value_kind(units: str) -> str:
    if units == "CODE TABLE":
        return "code"
    if units == "FLAG TABLE":
        return "flag"
    if units == "CCITT IA5":
        return "text"
    return "numeric"


def _descriptor_from_row(row: dict[str, str]) -> Descriptor:
    value_kind = descriptor_value_kind(row["units"])
    return Descriptor(
        descriptor=row["descriptor"],
        mnemonic=row["mnemonic"],
        meaning=row["meaning"],
        units=row["units"],
        base_scale=int(row["base_scale"]),
        base_reference=int(row["base_reference"]),
        base_bits=int(row["base_bits"]),
        value_kind=value_kind,
        interpretation_table_key=(
            row["descriptor"] if value_kind in {"code", "flag"} else None
        ),
        column_name=row.get("column_name", ""),
        slot1_promoted=row.get("slot1_promoted", "") == "yes",
    )


def _load_indexed(path: str | Path, key: str) -> dict[str, dict[str, Descriptor]]:
    result: dict[str, dict[str, Descriptor]] = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["sensor"], {})[row[key]] = _descriptor_from_row(row)
    return result


def load_descriptor_catalog(path: str | Path) -> dict[str, dict[str, Descriptor]]:
    return _load_indexed(path, "mnemonic")


def load_descriptor_catalog_by_descriptor(
    path: str | Path,
) -> dict[str, dict[str, Descriptor]]:
    """Read a catalogue CSV into ``{sensor: {descriptor: Descriptor}}``.

    Mnemonics are not unique in satwnd (``TMDB`` is 0-12-101 and 0-12-225,
    ``RFFL`` is 0-25-202 and 0-33-196), so keying by mnemonic drops one of each.
    """

    return _load_indexed(path, "descriptor")

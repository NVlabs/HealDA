# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group repeated BUFR occurrences into semantic ``list<struct>`` columns.

Occurrence counts differ per mnemonic within one template (e.g. 5 winds, 11
pressure candidates, 2-12 quality blocks), so parallel per-mnemonic lists cannot
be zipped. Grouping is decided at write time from the template's descriptor
expansion.

``earth2bufr``'s ``source_structure()`` returns ``effective_occurrences`` in
expansion order; ``archive_etl`` stores each one's ``decoder_path`` (its
replication nesting) as ``source_path``. A
:class:`GroupSpec` names an anchor mnemonic; each anchor occurrence opens one
entry, whose members are the listed occurrences just before it (qualifiers such
as ``TSIG``) or just after it (``WSPD`` after ``WDIR``). Members must share the
anchor's container path, so fields repeated inside provider diagnostic
sequences (GOES-R ``DRF8BIT_3.UWND``, ...) never join a main-body group.

Per-field metadata for the group columns comes from ``declared_schema.arrow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class GroupSpec:
    """One ``list<struct>`` output column."""

    name: str
    anchor: str
    before: Mapping[str, str] = field(default_factory=dict)
    after: Mapping[str, str] = field(default_factory=dict)


WIND_CANDIDATES = GroupSpec(
    name="wind_candidates",
    anchor="WDIR",
    before={"TSIG": "tsig_raw", "HOUR": "hour", "MINU": "minute", "SECO": "second"},
    after={"WSPD": "speed_m_s"},
)

HEIGHT_ASSIGNMENTS = GroupSpec(
    name="height_assignments",
    anchor="PRLC",
    before={"HAMD": "method_code", "TSIG": "tsig_raw", "EHAM": "estimated_error"},
    after={"TMDBST": "temperature_k"},
)

MODEL_WINDS = GroupSpec(
    name="model_winds",
    anchor="UMWV",
    before={"TSIG": "tsig_raw", "PRLC": "pressure_pa"},
    after={"VWMV": "v_model"},
)

MEASUREMENT_UNCERTAINTY = GroupSpec(
    name="measurement_uncertainty",
    anchor="MUNCEX",
    before={},
    after={
        "PRLC": "pressure_uncertainty",
        "UWND": "u_standard_uncertainty",
        "VWND": "v_standard_uncertainty",
    },
)

QUALITY_DIAGNOSTICS = GroupSpec(
    name="quality_diagnostics",
    anchor="PCCF",
    before={
        "GNAPS": "standard_generating_application",
        "GNAP": "generating_application",
        "OGCE": "originating_centre_id",
        "GCLONG": "originating_centre_id_long",
        "QFGU": "first_guess_use_qualifier",
    },
    after={"MAQC": "manual_automatic_qc", "NCTH": "nominal_confidence_threshold"},
)

GROUP_SPECS: tuple[GroupSpec, ...] = (
    WIND_CANDIDATES,
    HEIGHT_ASSIGNMENTS,
    MODEL_WINDS,
    MEASUREMENT_UNCERTAINTY,
    QUALITY_DIAGNOSTICS,
)


def plan_groups(
    effective_occurrences: Sequence[Mapping[str, object]],
    spec: GroupSpec,
) -> list[dict[str, int]]:
    """Occurrence indices per struct entry, from the expansion order.

    Returns one dict per entry mapping mnemonic -> zero-based occurrence index
    of that mnemonic (the ``_col_positions`` order). ``before`` members also get
    ``<mnemonic>@first`` for the first in the window.
    """
    counters: dict[str, int] = {}
    ordered: list[tuple[str, int, str]] = []
    for occurrence in effective_occurrences:
        mnemonic = str(occurrence["mnemonic"])
        index = counters.get(mnemonic, 0)
        counters[mnemonic] = index + 1
        path = str(occurrence["source_path"])
        ordered.append((mnemonic, index, path))

    def container(path: str, mnemonic: str) -> str:
        # "DRF8BIT_3.DRF8BIT.UWND" -> "DRF8BIT_3.DRF8BIT"; "UWND" -> "".
        return path[: -len(mnemonic)].rstrip(".") if path.endswith(mnemonic) else path

    anchors = [i for i, (m, _, _) in enumerate(ordered) if m == spec.anchor]
    entries: list[dict[str, int]] = []
    for position, anchor_index in enumerate(anchors):
        mnemonic, occurrence_index, path = ordered[anchor_index]
        scope = container(path, mnemonic)
        entry = {spec.anchor: occurrence_index}
        previous_anchor = anchors[position - 1] if position else -1

        # Qualifiers precede their value, in the same container and after the
        # previous anchor. A wind block is `TSIG,HH,MM,SS, TSIG,HH,MM,SS,
        # WDIR,WSPD`: the first TSIG carries the role, the second (often
        # unpopulated) bounds the image interval.
        window: dict[str, list[int]] = {}
        for step in range(previous_anchor + 1, anchor_index):
            other, other_index, other_path = ordered[step]
            if other not in spec.before:
                continue
            if container(other_path, other) != scope:
                continue
            window.setdefault(other, []).append(other_index)
        for other, found in window.items():
            entry[other] = found[-1]
            entry[f"{other}@first"] = found[0]

        next_anchor = (
            anchors[position + 1] if position + 1 < len(anchors) else len(ordered)
        )
        for step in range(anchor_index + 1, next_anchor):
            other, other_index, other_path = ordered[step]
            if other not in spec.after or other in entry:
                continue
            if container(other_path, other) != scope:
                continue
            entry[other] = other_index
        entries.append(entry)
    return entries


def build_group_array(
    spec: GroupSpec,
    entries: Sequence[Mapping[str, int]],
    values_for: Callable[[str, int], np.ndarray | None],
    n_rows: int,
    member_arrow_type: Callable[[str], pa.DataType],
) -> pa.Array | None:
    """Build the ``list<struct>`` array for one group spec.

    ``values_for(mnemonic, occurrence_index)`` returns that occurrence's column
    as a float64 NumPy array with NaN for missing, or None if it was not decoded.
    """
    if not entries:
        return None

    fields: list[pa.Field] = [
        pa.field("source_occurrence", pa.uint8()),
    ]
    if spec is WIND_CANDIDATES:
        fields += [
            pa.field("direction_deg", pa.float32()),
            pa.field("speed_m_s", pa.float32()),
            pa.field("u_m_s", pa.float32()),
            pa.field("v_m_s", pa.float32()),
        ]
    else:
        fields.append(pa.field(_anchor_field(spec), pa.float64()))

    # Block time components are not struct members; speed_m_s is added above.
    absorbed = {"speed_m_s", "hour", "minute", "second"}
    member_names = sorted(
        (set(spec.before.values()) | set(spec.after.values())) - absorbed
    )
    # Members follow the flat-column typing rule (code/flag tables unsigned).
    reverse_members = {
        **{v: k for k, v in spec.before.items()},
        **{v: k for k, v in spec.after.items()},
    }
    for name in member_names:
        fields.append(pa.field(name, member_arrow_type(reverse_members[name])))

    # Members no entry decodes stay None and become an all-null child.
    def values_or_none(mnemonic: str, index: int | None) -> np.ndarray | None:
        return values_for(mnemonic, index) if index is not None else None

    columns: dict[str, list[np.ndarray | None]] = {f.name: [] for f in fields}
    for entry in entries:
        anchor_values = values_for(spec.anchor, entry[spec.anchor])
        if anchor_values is None:
            continue
        columns["source_occurrence"].append(
            np.full(n_rows, entry[spec.anchor], dtype="uint8")
        )

        if spec is WIND_CANDIDATES:
            speed = values_or_none("WSPD", entry.get("WSPD"))
            if speed is None:
                speed = np.full(n_rows, np.nan)
            radians = np.deg2rad(anchor_values)
            columns["direction_deg"].append(anchor_values)
            columns["speed_m_s"].append(speed)
            columns["u_m_s"].append(-speed * np.sin(radians))
            columns["v_m_s"].append(-speed * np.cos(radians))
        else:
            columns[_anchor_field(spec)].append(anchor_values)

        for name in member_names:
            mnemonic = reverse_members[name]
            # tsig_raw is the first TSIG in the window, the one carrying the role.
            index = (
                entry.get(f"{mnemonic}@first")
                if name == "tsig_raw"
                else entry.get(mnemonic)
            )
            columns[name].append(values_or_none(mnemonic, index))

    if not columns["source_occurrence"]:
        return None

    # Variable-length lists: a row gets an entry only where the anchor value is
    # present, so list length is the number of real candidates.
    anchor_name = "direction_deg" if spec is WIND_CANDIDATES else _anchor_field(spec)
    present = np.stack(
        [np.isfinite(values) for values in columns[anchor_name]], axis=1
    )  # (n_rows, entries)
    counts = present.sum(axis=1).astype("int64")
    offsets = pa.array(np.concatenate([[0], np.cumsum(counts)]).astype("int32"))

    child_arrays: list[pa.Array] = []
    total = int(offsets[-1].as_py())
    for f in fields:
        if all(values is None for values in columns[f.name]):
            child_arrays.append(pa.nulls(total, type=f.type))
            continue
        stacked = np.stack(
            [
                np.full(n_rows, np.nan) if values is None else values
                for values in columns[f.name]
            ],
            axis=1,
        )
        values = stacked[present].astype("float64")  # row-major
        finite = np.isfinite(values)
        if pa.types.is_unsigned_integer(f.type):
            child = pa.array(
                np.where(finite, values, 0).astype("int64"), type=f.type, mask=~finite
            )
        else:
            # All-null members are kept so the struct type depends only on the
            # GroupSpec.
            child = pa.array(values, type=f.type, mask=~finite)
        child_arrays.append(child)

    struct_array = pa.StructArray.from_arrays(child_arrays, fields=fields)
    return pa.ListArray.from_arrays(offsets, struct_array)


def _anchor_field(spec: GroupSpec) -> str:
    return {
        "PRLC": "pressure_pa",
        "PCCF": "per_cent_confidence",
        "UMWV": "u_model",
        "MUNCEX": "expression_code",
    }.get(spec.anchor, spec.anchor.lower())

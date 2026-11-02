# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode and projection for the raw NCEP SATWND (AMV) archive.

Raw satwnd BUFR -> Arrow tables in the AMV archive's logical shape: one row
per atmospheric motion vector, flat columns for single and promoted values, and
repeated source fields kept as grouped ``list<struct>`` columns or
``<column_name>_occurrences`` lists.

Logical column names come from ``descriptor_catalog.csv``, which is built
from the WMO Table-B element names. Per-field metadata comes from
``declared_schema.arrow``, applied by ``conform_to_declared``.

Physical Parquet writing lives in ``amv_writer``, which composes the
shared ``archive_common`` day writer.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc
import earth2bufr

from healda.observations.loaders.nnja_base import archive_cycle_and_window
from healda.observations.loaders.nnja_satwnd import ARCHIVE_HPX_LEVEL
from healda.observations.preprocessing.sat_helpers import hpx_nest_pixels
from healda.observations.preprocessing.satwnd_qc import (
    AMVQIC_CASES,
    AMVQIC_EE_GNAPS,
    AMVQIC_QIFN_GNAPS,
    QIFN_GNAP_BY_CASE,
    QIFY_GNAP_BY_CASE,
)

from ..archive_common.descriptor_catalog import (
    Descriptor,
    load_descriptor_catalog,
    load_descriptor_catalog_by_descriptor,
)
from .candidate_groups import (
    GROUP_SPECS,
    build_group_array,
    plan_groups,
)

SCHEMA_VERSION = "amv-v0.3-descriptor-grounded"

# Cache key for resume: a published day is reused only if its footer carries
# this revision. Bump for a schema change, a change to any decoded or derived
# value, or the stream-selection rule. Do not bump for
# fixes that only turn failures into successes, docs, tests, logging or
# performance work. The exact commit is recorded separately as amv.etl_git_sha.
ETL_CONTENT_REVISION = "amv-etl-r4"


def etl_git_sha() -> str:
    """The exact commit, for provenance only; 'unknown' if git is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = result.stdout.strip()
        if sha and result.returncode == 0:
            dirty = subprocess.run(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parent),
                    "status",
                    "--porcelain",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except Exception:
        pass
    return "unknown"


CATALOG_PATH = Path(__file__).resolve().parent / "descriptor_catalog.csv"
GSI_TYPE_TABLE_PATH = Path(__file__).resolve().parent / "gsi_satwnd_type_table.csv"
WIND_SLOT_GATE_PATH = Path(__file__).resolve().parent / "wind_slot_distinctness.csv"
DECLARED_SCHEMA_PATH = Path(__file__).resolve().parent / "declared_schema.arrow"
DESCRIPTORS = load_descriptor_catalog(CATALOG_PATH)["satwnd"]
DESCRIPTORS_BY_FXY = load_descriptor_catalog_by_descriptor(CATALOG_PATH)["satwnd"]


# (bufr_subset, SAID, SWCM) -> (type, subtype, provider, subprovider, tier), derived
# from the sattab array in NOAA-EMC/GSI src/gsi/read_satwnd.f90 @860d1374.
def _load_gsi_type_table() -> (
    dict[tuple[str, int, int], tuple[int, int, str, str, str]]
):
    table: dict[tuple[str, int, int], tuple[int, int, str, str, str]] = {}
    with GSI_TYPE_TABLE_PATH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            table[(row["bufr_subset"], int(row["said"]), int(row["swcm"]))] = (
                int(row["gsi_observation_type"]),
                int(row["gsi_internal_subtype"]),
                row["gsi_provider"],
                row["gsi_subprovider"],
                row["mapping_tier"],
            )
    return table


GSI_TYPE_TABLE = _load_gsi_type_table()


# (bufr_subset, slot) -> whether the slot differs from the final wind; some subsets
# (all JMA) fill the first-guess slot with a copy of the final wind.
def _load_wind_slot_gate() -> dict[tuple[str, int], bool]:
    gate: dict[tuple[str, int], bool] = {}
    with WIND_SLOT_GATE_PATH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            gate[(row["bufr_subset"], int(row["slot"]))] = row["informative"] == "yes"
    return gate


WIND_SLOT_GATE = _load_wind_slot_gate()

# Column naming is owned by descriptor_catalog.csv: `column_name` names each
# descriptor and `slot1_promoted` says whether its first occurrence is also a
# flat scalar column. A mnemonic without a catalogue entry is emitted as
# `unclassified__<MNEMONIC>`. Repeated fields are list columns named
# `<column_name>_occurrences`; slots are keyed by time significance (0-08-021),
# and in five-slot templates slot 2 is TSIG=27 "first guess", a model background.
CODE_KINDS = frozenset({"code", "flag"})

# ---------------------------------------------------------------------------
# Stream selection
# ---------------------------------------------------------------------------

# (first_year, last_year, streams), closed on both ends. One stream per era: the
# `merged` stream through 2019, NCEP `satwnd` from 2020 when `merged` stops.
# The positions GSI assimilated are contained in `merged` wherever it exists.
STREAM_ERAS: tuple[tuple[int, int, tuple[str, ...]], ...] = (
    (1979, 2019, ("merged",)),
    (2020, 9999, ("satwnd",)),
)


def streams_for_year(year: int) -> tuple[str, ...]:
    """Source streams to read for a UTC year; raises for a year not in STREAM_ERAS."""
    for first, last, streams in STREAM_ERAS:
        if first <= year <= last:
            return streams
    raise ValueError(
        f"year {year} matches no row in STREAM_ERAS; extend the table rather "
        "than letting the year fall through"
    )


def primary_stream_for_year(year: int) -> str:
    return streams_for_year(year)[0]


# A closed two-value vocabulary, so a dictionary type is safe. The values array
# is built once and shared by every batch: the writer spools through Arrow IPC
# files, which allow one non-delta dictionary per field, so a per-batch
# dictionary raises "Dictionary replacement detected" on the second batch.
SOURCE_STREAM_VALUES = pa.array(["merged", "satwnd"], type=pa.string())
SOURCE_STREAM_TYPE = pa.dictionary(pa.int8(), pa.string())
_SOURCE_STREAM_INDEX = {"merged": 0, "satwnd": 1}


def source_stream_array(stream: str, length: int) -> pa.DictionaryArray:
    if stream not in _SOURCE_STREAM_INDEX:
        raise ValueError(
            f"unknown source stream {stream!r}; "
            f"expected one of {sorted(_SOURCE_STREAM_INDEX)}"
        )
    indices = pa.array(
        np.full(length, _SOURCE_STREAM_INDEX[stream], dtype="int8"), type=pa.int8()
    )
    return pa.DictionaryArray.from_arrays(indices, SOURCE_STREAM_VALUES)


UNCLASSIFIED_PREFIX = "unclassified__"


def _arrow_type_for(catalog: Descriptor) -> pa.DataType:
    """Code and flag tables are uint32, everything else float64.

    The width is fixed rather than taken from base Table-B packing: a template's
    width operator can widen a code field beyond its base bits, and a per-file
    width would make the schema differ between day files.
    """
    if catalog.value_kind in CODE_KINDS:
        return pa.uint32()
    return pa.float64()


def _resolve(mnemonic: str, fxy_by_mnemonic: Mapping[str, str]) -> Descriptor | None:
    """Catalogue entry for one mnemonic within one template.

    Mnemonics are not unique in satwnd (TMDB is 0-12-101 and 0-12-225; RFFL is
    0-25-202 and 0-33-196), so the template's own mnemonic -> FXY map is tried
    first and the mnemonic index is the fallback.
    """
    fxy = fxy_by_mnemonic.get(mnemonic)
    if fxy is not None:
        resolved = DESCRIPTORS_BY_FXY.get(fxy)
        if resolved is not None:
            return resolved
    return DESCRIPTORS.get(mnemonic)


# The promoted PCCF scalar is the first populated occurrence, whose generating
# application varies by provider (qifn for NESDIS, qify -- with forecast -- for
# EUMETSAT and JMA). The name marks it ambiguous; `amv_qi_without_forecast` is
# the forecast-independent QI.
RENAMED_COLUMNS: Mapping[str, str] = {
    "per_cent_confidence": "per_cent_confidence_first_occurrence_ambiguous",
}


def _column_name_for(catalog: Descriptor | None, mnemonic: str) -> str:
    if catalog is not None and catalog.column_name:
        return RENAMED_COLUMNS.get(catalog.column_name, catalog.column_name)
    # Decoder-generated names absent from Table B: earth2bufr suffixes repeated
    # delayed-replication counters (DRF8BIT_2, ...) and adds `_data_category`.
    if mnemonic.startswith("DRF8BIT_"):
        return f"delayed_replication_count_{mnemonic.rsplit('_', 1)[-1]}"
    if mnemonic == "_data_category":
        return "bufr_data_category"
    return f"{UNCLASSIFIED_PREFIX}{mnemonic}"


# Consumed into `time_utc` and not emitted as scalar columns. HOUR/MINU/SECO
# are also members of the wind_candidates group, which does not store them, so
# their repeated wind-block occurrences are not emitted in any column.
TIME_COMPONENTS = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")

# Accuracy variants of one quantity. The catalogue already maps both spellings
# to the same `column_name`, so this only records the preference order.
COALESCED_COLUMNS: dict[str, tuple[str, ...]] = {
    "latitude": ("CLATH", "CLAT"),
    "longitude": ("CLONH", "CLON"),
}

# Columns computed rather than decoded, with their per-field derivation text.
# Also written to the footer as `amv.derived_columns`.
_SLOT_DOC = (
    "PREPDATA wind slot {slot} ({stage}), converted to a Cartesian component "
    "from that slot's own WDIR/WSPD. Emitted only for the legacy five-slot "
    "templates (exactly five WDIR occurrences); null everywhere else and never "
    "inferred for modern templates, whose repeated fields belong to unrelated "
    "provider diagnostic sequences. Slot 1 (final observed) is NOT duplicated "
    "here -- it is wind_u_derived / wind_v_derived. DERIVED.{extra}"
)

DERIVED_COLUMNS: dict[str, str] = {
    "source_stream": (
        "which source stream supplied this row: 'satwnd' (the NCEP operational "
        "dump) or 'merged' (the EUMETSAT AMV Climate Data Record Release 2). "
        "Both are read for 2000-2019; only satwnd exists from 2020"
    ),
    "also_in_other_stream": (
        "the other source stream supplied an observation at the same "
        "(latitude, longitude, satellite, time) key in the same cycle. FALSE "
        "means NOT CORROBORATED, not 'disagreed' -- in particular it is false "
        "for every row from 2020 onward, where only one stream is read"
    ),
    "source_template": (
        "root Table-D descriptor (F-X-Y) of the BUFR message this row was decoded "
        "from, read from the message's own descriptor list"
    ),
    "ncep_dump_subtype": (
        "Table-D sequence mnemonic for source_template, read from the BUFR file's "
        "embedded DX table (NOT a hard-coded lookup); this is NCEP's own name for "
        "the dump subtype, e.g. NC005032"
    ),
    "time_utc": (
        "UTC observation time composed from the first occurrence of "
        "YEAR/MNTH/DAYS/HOUR/MINU/SECO; null when those cannot form a valid date"
    ),
    "da_window": (
        "3h routing label: the cycle timestamp when time_utc <= cycle, else "
        "cycle + 3h; identical rule to nnja_sat_archive.assign_da_windows"
    ),
    "cycle": "synoptic cycle (00/06/12/18Z) of the source file this row came from",
    "amv_qi_without_forecast": (
        "GSI's `qifn`: per_cent_confidence selected by GENERATING APPLICATION, "
        "dispatched on gsi_internal_subtype -- GNAP 2 for EUMETSAT, 102 for "
        "JMA, 1 for NESDIS/MODIS/AVHRR/VIIRS, and GNAPS 5 for the AMVQIC "
        "families. NULL means NOT RESOLVABLE for this row (no such generating "
        "application present), which is NOT the same as a low QI and must not "
        "be read as zero: 2019 GOES-R carries AMVQIC with GNAPS null on ~45% "
        "of rows. Use this, never per_cent_confidence_first_occurrence_ambiguous"
    ),
    "amv_qi_with_forecast": (
        "GSI's `qify`, resolved the same way (GNAP 1 EUMETSAT, 101 JMA, 3 "
        "NESDIS). Carried so the forecast dependence is VISIBLE rather than "
        "implicit -- it is the value the old promoted scalar was silently "
        "returning for EUMETSAT and JMA. Always null for the AMVQIC families, "
        "which carry no with-forecast variant. Do not screen on it: it depends "
        "on a first guess, which is exactly what a forecast-independent gate "
        "must not use"
    ),
    "amv_expected_error": (
        "GSI's `ee`: per_cent_confidence at GNAP 4 (104 for JMA) or GNAPS 2 for "
        "the AMVQIC families. For GOES-R this is in m/s, not per cent -- GSI's "
        "own comment says so -- so it is stored raw and units are per family. "
        "Feeds the normalised expected-error screen (10 - 0.1*ee)/speed > 0.9"
    ),
    "amv_coefficient_of_variation": (
        "CVWD (0-25-148) from intermediate vector 0 of the AMVIVR (3-62-197) "
        "sequence -- GSI's `pct1`, read_satwnd.f90:620-621. Null for every "
        "template that does not carry AMVIVR, which is everything outside the "
        "GOES-R families. GSI screens 0.04 < pct1 < 0.50 for types 240/245/246/"
        "251 (read_satwnd.f90:948-951); the value is stored unscreened"
    ),
    "amv_tracking_correlation": (
        "TCOV (0-25-149) from intermediate vector 0 of the same AMVIVR "
        "sequence. GSI reads it alongside CVWD and then uses only CVWD; it is "
        "carried because it is free once the sequence is opened and it is the "
        "only tracking-quality measure in the GOES-R feed that is not a QI"
    ),
    # Satwnd winds are polar (WDIR/WSPD); NCEP converts to U/V downstream in
    # PREPDATA GETC06. Templates that natively carry 0-11-003/4 keep those in
    # separate `_source` columns.
    "wind_u_derived": (
        "-wind_speed * sin(wind_direction), computed at write time from the "
        "POLAR source fields only -- the same choice both NCEP routes make "
        "(PREPDATA GETC06 calls UV(WDIR,WSPD); read_satwnd derives it too). "
        "Never falls back to, and is never coalesced with, the native "
        "0-11-003 UWND column wind_u_source, which is a numerically different "
        "quantity (0.068 m/s RMS from PREPRO vs 0.029 m/s for this one)"
    ),
    "wind_v_derived": (
        "-wind_speed * cos(wind_direction), computed at write time from the "
        "POLAR source fields only; see wind_u_derived. Never coalesced with "
        "the native 0-11-004 VWND column wind_v_source"
    ),
    "gsi_observation_type": (
        "GSI conventional observation type (240-260) that read_satwnd.f90 would "
        "assign this report, from sattab(subset_number, SAID, SWCM). DERIVED: raw "
        "SATWND BUFR does not contain it. Null means GSI's own reader would have "
        "rejected the report -- SWCM outside 1-7, or an unmapped (subset, SAID, "
        "SWCM) cell, both of which are `cycle loop_report` in GSI. A null is "
        "therefore evidence about GSI, not a decode failure. It does not replace "
        "the raw identity: ncep_dump_subtype, satellite_id and "
        "wind_computation_method are all retained"
    ),
    "gsi_internal_subtype": (
        "GSI's internal processing-case index, sattab(...,2); selects the "
        "select-case branch in read_satwnd.f90. DERIVED, same source and same "
        "null semantics as gsi_observation_type"
    ),
    "gsi_provider": (
        "GSI c_prvstg provider string for gsi_internal_subtype (e.g. GOESR, "
        "EUMETSAT, JMA, NESDIS, MODIS, AVHRR, METOP, LEOGEO, VIIRS). DERIVED"
    ),
    "gsi_subprovider": (
        "GSI c_sprvstg sub-provider string (e.g. IRSW for GOES short-wave IR, "
        "which is what distinguishes type 240 from 245 -- both are SWCM=1). "
        "DERIVED"
    ),
    "gsi_type_mapping_tier": (
        "which sourced route produced gsi_observation_type: "
        "`1-direct-gsi-sattab` (today's read_satwnd.f90 sattab), "
        "`1-direct-gsi-sattab-unusable-istype-1`, "
        "`2-pre-refactor-gsi` (a rule present before the 8e5464277 refactor "
        "regression and restored here), or `3-prepdata` (NCEP PREPDATA "
        "JNSTR, the route the direct reader cannot ingest -- INSAT). "
        "NOT decoration: type 241 is ambiguous, used both for INSAT here and "
        "for CIMSS tropical-cyclone AMVs whose QC setupw.f90 keys off 241, so "
        "a consumer reproducing GSI QC must filter on this. DERIVED"
    ),
    "wind_u_first_guess": _SLOT_DOC.format(
        slot=2,
        stage="first guess",
        extra=(
            " LEAKAGE: this IS a provider NWP forecast wind, per provider "
            "documentation -- GOES-R's ATBD uses a short-term GFS forecast to "
            "centre the tracking search box and for forecast-consistency QC; JMA "
            "uses GSM first-guess fields; VIIRS documents forecast-consistency "
            "rejection; LEO-GEO (Lazzara et al. 2014) compares pairwise vectors "
            "against a background before averaging. It is NOT the GSI/NCEP "
            "background: both are model-derived but they are different pipeline "
            "stages from potentially different forecasts, so they are not "
            "interchangeable and not the same leakage. Treat this column as a "
            "forecast feature, never as an independent observation. By the same "
            "chain, even the FINAL wind is not strictly background-free, because "
            "provider forecast guidance influenced tracking, height assignment "
            "and QC upstream of it."
        ),
    ),
    "wind_v_first_guess": _SLOT_DOC.format(
        slot=2,
        stage="first guess",
        extra=(" See wind_u_first_guess for the provider-forecast leakage note."),
    ),
    "wind_u_original": _SLOT_DOC.format(
        slot=3,
        stage="original",
        extra=(
            " R06UBF calls slot 3 'Original'; the file's own TSIG says 'Time "
            "averaged'. The disagreement is recorded, not resolved."
        ),
    ),
    "wind_v_original": _SLOT_DOC.format(slot=3, stage="original", extra=""),
    "wind_u_pair12": _SLOT_DOC.format(slot=4, stage="image 1->2 track", extra=""),
    "wind_v_pair12": _SLOT_DOC.format(slot=4, stage="image 1->2 track", extra=""),
    "wind_u_pair23": _SLOT_DOC.format(slot=5, stage="image 2->3 track", extra=""),
    "wind_v_pair23": _SLOT_DOC.format(slot=5, stage="image 2->3 track", extra=""),
    "wind_slot_availability": (
        "bitmask over the five PREPDATA wind slots; bit k-1 set when slot k has "
        "both a finite WDIR and WSPD. 0 for every template that is not "
        "five-slot-capable. DERIVED"
    ),
    "wind_slot_spread": (
        "largest vector separation sqrt((u_k-u_1)^2+(v_k-v_1)^2) between any "
        "populated non-final slot and the final wind, m/s. A provenance/QC "
        "feature ONLY: the slots are correlated processing stages, so this is "
        "NOT a calibrated observation-error estimate. DERIVED"
    ),
    "hpx4096_nest": (
        "earth2grid.healpix.Grid(level=12, NEST).ang2pix(longitude, latitude); "
        "null where geolocation is missing or out of range"
    ),
}


def _root_of(message: earth2bufr.MessageInfo) -> str:
    descriptors = message.descriptors
    return descriptors[1] if len(descriptors) > 1 else descriptors[0]


def read_source_structures(path: str) -> dict[str, dict[str, Any]]:
    """Per-root-template structure resolved from the file's own DX tables.

    This, not the descriptor catalogue, answers whether a template carries a
    descriptor. Keys are the root sequence, suffixed with table and template
    hashes when a root has more than one structure in the file.
    """
    with earth2bufr.open_bufr(path) as handle:
        return _source_structures(path, handle)


def _source_structures(
    path: str, handle: earth2bufr.BufrFile
) -> dict[str, dict[str, Any]]:
    by_identity: dict[tuple[str, int, tuple[str, ...]], dict[str, Any]] = {}
    for message in handle.info.data_messages:
        index = message.source_message_index
        root = _root_of(message)
        epoch_index = message.table_epoch
        observation_descriptors = message.descriptors
        identity = (root, epoch_index, observation_descriptors)
        if identity in by_identity:
            continue
        try:
            resolved = handle.source_structure(index)
        except Exception as exc:
            raise RuntimeError(
                f"{path}: cannot resolve {root} at source message {index}"
            ) from exc

        descriptors_by_mnemonic: dict[str, set[str]] = {}
        for element in resolved["active_elements"]:
            descriptors_by_mnemonic.setdefault(element["mnemonic"], set()).add(
                element["descriptor"]
            )
        ambiguous = {
            mnemonic: sorted(descriptors)
            for mnemonic, descriptors in descriptors_by_mnemonic.items()
            if len(descriptors) > 1
        }
        if ambiguous:
            raise RuntimeError(
                f"{path}: {root} reuses mnemonics for different descriptors: {ambiguous}"
            )

        sequences = {
            s["descriptor"]: s["mnemonic"] for s in resolved["active_sequences"]
        }
        by_identity[identity] = {
            "root_sequence": root,
            "source_epoch": epoch_index,
            "observation_descriptors": list(observation_descriptors),
            "ncep_dump_subtype": sequences[root],
            "embedded_table_sha256": resolved["embedded_table_sha256"],
            "decoded_column_count": len(resolved["decoded_physical_schema"]),
            "active_elements": [
                [element["descriptor"], element["mnemonic"]]
                for element in resolved["active_elements"]
            ],
            "active_sequences": [
                [sequence["descriptor"], sequence["mnemonic"]]
                for sequence in resolved["active_sequences"]
            ],
            # Used by candidate_groups (expansion order). Too large for the
            # footer; see FOOTER_EXCLUDED_STRUCTURE_KEYS.
            "decoded_physical_schema": resolved["decoded_physical_schema"],
            "effective_occurrences": [
                {
                    "descriptor": occurrence["descriptor"],
                    "mnemonic": occurrence["mnemonic"],
                    "source_path": occurrence["decoder_path"],
                }
                for occurrence in resolved["effective_occurrences"]
            ],
        }

    counts: dict[str, int] = {}
    for root, _epoch, _descriptors in by_identity:
        counts[root] = counts.get(root, 0) + 1
    structures: dict[str, dict[str, Any]] = {}
    for (root, _epoch, descriptors), structure in by_identity.items():
        template_sha256 = hashlib.sha256("|".join(descriptors).encode()).hexdigest()
        key = (
            root
            if counts[root] == 1
            else (
                f"{root}@{structure['embedded_table_sha256'][:12]}"
                f"@{template_sha256[:12]}"
            )
        )
        structures[key] = structure
    return structures


def _structure_for_message(
    structures: Mapping[str, Mapping[str, Any]],
    root: str,
    embedded_table_sha256: str | None,
    observation_descriptors: Sequence[str],
) -> Mapping[str, Any]:
    expected_descriptors = tuple(observation_descriptors)
    for structure in structures.values():
        if (
            structure["root_sequence"] == root
            and structure["embedded_table_sha256"] == embedded_table_sha256
            and tuple(structure["observation_descriptors"]) == expected_descriptors
        ):
            return structure
    raise RuntimeError(
        f"no source structure matches {root} with table {embedded_table_sha256} "
        f"and descriptors {list(expected_descriptors)}"
    )


FOOTER_EXCLUDED_STRUCTURE_KEYS = ("effective_occurrences", "decoded_physical_schema")


def footer_source_structures(structures: Mapping[str, Any]) -> dict[str, Any]:
    return {
        root: {
            k: v
            for k, v in structure.items()
            if k not in FOOTER_EXCLUDED_STRUCTURE_KEYS
        }
        for root, structure in structures.items()
    }


def _col_positions(tbl: pa.Table, name: str) -> list[int]:
    return tbl.schema.get_all_field_indices(name)


# combine_chunks copies even a single chunk.
def _one_array(column: pa.ChunkedArray) -> pa.Array:
    return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()


def _to_np(tbl: pa.Table, i: int) -> np.ndarray:
    # earth2bufr returns delayed-replication sequences (AMVIVR etc.) as nested
    # list<struct> columns; to_numpy on those builds Python objects for the whole
    # column before failing, so non-numeric types are rejected by type first.
    field_type = tbl.schema.field(i).type
    if not (
        pa.types.is_floating(field_type)
        or pa.types.is_integer(field_type)
        or pa.types.is_boolean(field_type)
    ):
        return np.full(tbl.num_rows, np.nan)
    column = _one_array(tbl.column(i))
    return column.to_numpy(zero_copy_only=False).astype("float64", copy=False)


def _is_text_column(tbl: pa.Table, i: int) -> bool:
    field_type = tbl.schema.field(i).type
    return pa.types.is_string(field_type) or pa.types.is_large_string(field_type)


# The decoder returns '' for an absent CCITT IA5 value; that becomes null.
def _text_values(tbl: pa.Table, i: int) -> pa.Array:
    column = _one_array(tbl.column(i))
    stripped = pc.utf8_trim_whitespace(column.cast(pa.string()))
    return pc.if_else(pc.equal(stripped, ""), pa.scalar(None, pa.string()), stripped)


# The scalar, group and _occurrences builders read the same columns of one table.
def _column_cache(tbl: pa.Table):
    cache: dict[int, np.ndarray] = {}

    def values(i: int) -> np.ndarray:
        cached = cache.get(i)
        if cached is None:
            cached = _to_np(tbl, i)
            cache[i] = cached
        return cached

    return values


def resolve_promoted_occurrence(
    tbl: pa.Table,
    mnemonic: str,
    column_values,
) -> np.ndarray:
    """Which occurrence of a repeated field supplies the promoted scalar.

    Occurrence 0 is not always the populated one (GOES-R's PCCF starts at 1).
    Resolution is per batch, and a batch is one (template, epoch) run: the first
    position with any finite value is returned whole, so per-row nulls in it
    stay null rather than being filled from a later occurrence.

    ``column_values`` must return float64 with NaN for missing (as ``_to_np``
    does); ``np.isfinite`` is always true for integer arrays.
    """
    positions = _col_positions(tbl, mnemonic)
    if not positions:
        return np.full(tbl.num_rows, np.nan)
    for i in positions:
        values = column_values(i)
        if np.isfinite(values).any():
            return values
    return column_values(positions[0])  # every occurrence empty: all null


# datetime64[ns] spans 1677-09-21T00:12:43 .. 2262-04-11T23:47:16. Held in
# seconds so out-of-range timestamps can be rejected before any ns cast.
_NS_MIN_SECONDS = np.datetime64("1677-09-22T00:00:00", "s")
_NS_MAX_SECONDS = np.datetime64("2262-04-10T23:59:59", "s")


def _compose_utc(year, month, day, hour, minute, second) -> np.ndarray:
    """Compose ``time_utc`` from the six source components, vectorised.

    Missing MINU/SECO count as zero (pre-2010 templates carry no SECO). Missing
    or out-of-range YEAR/MNTH/DAYS/HOUR, or a day past the month's length, give
    NaT, which becomes an Arrow null.
    """
    minute = np.where(np.isfinite(minute), minute, 0.0)
    second = np.where(np.isfinite(second), second, 0.0)
    valid = (
        np.isfinite(year) & np.isfinite(month) & np.isfinite(day) & np.isfinite(hour)
    )
    valid &= (month >= 1) & (month <= 12)
    valid &= (hour >= 0) & (hour <= 23)
    valid &= (minute >= 0) & (minute <= 59)
    valid &= (second >= 0) & (second <= 60)
    valid &= (year >= 1600) & (year <= 2500)
    # Corrupt historical files carry out-of-range days.
    valid &= (day >= 1) & (day <= 31)

    out = np.full(len(year), np.datetime64("NaT", "ns"))
    if not valid.any():
        return out

    months = ((year[valid] - 1970) * 12 + (month[valid] - 1)).astype("int64")
    month_start = months.astype("datetime64[M]")
    month_length = (
        (month_start + 1).astype("datetime64[D]") - month_start.astype("datetime64[D]")
    ).astype("int64")
    days = day[valid].astype("int64")
    in_month = (days >= 1) & (days <= month_length)

    seconds = (hour[valid] * 3600 + minute[valid] * 60 + second[valid]).astype("int64")
    # Compose in seconds and drop values outside the datetime64[ns] range before
    # casting: corrupt pre-2010 files decode years such as 1600 or 2497, and
    # numpy's out-of-range datetime cast segfaults rather than raising.
    stamps = (
        month_start.astype("datetime64[s]")
        + (days - 1).astype("timedelta64[D]").astype("timedelta64[s]")
        + seconds.astype("timedelta64[s]")
    )
    representable = in_month & (stamps >= _NS_MIN_SECONDS) & (stamps <= _NS_MAX_SECONDS)

    resolved = np.full(stamps.shape, np.datetime64("NaT", "ns"))
    resolved[representable] = stamps[representable].astype("datetime64[ns]")

    out[valid] = resolved
    return out


UNSIGNED_WIDTHS = (
    (pa.uint8(), 0xFF),
    (pa.uint16(), 0xFFFF),
    (pa.uint32(), 0xFFFFFFFF),
    (pa.uint64(), 2**64 - 1),
)


def fit_integer_type(
    values: np.ndarray, valid: np.ndarray, preferred: pa.DataType
) -> pa.DataType:
    """Widen ``preferred`` if the observed values do not fit it; never narrows.

    Values that fit no unsigned width (negative, fractional, too large) give
    float64.
    """
    if not valid.any():
        return preferred
    finite = values[valid]
    if finite.size == 0:
        return preferred
    low, high = float(np.min(finite)), float(np.max(finite))
    if low < 0 or high != np.floor(high):
        return pa.float64()
    for arrow_type, limit in UNSIGNED_WIDTHS:
        if high <= limit and _width_of(arrow_type) >= _width_of(preferred):
            return arrow_type
    return pa.float64()


def _width_of(arrow_type: pa.DataType) -> int:
    return arrow_type.bit_width if pa.types.is_integer(arrow_type) else 64


# NaN is not an Arrow null, so missing values need an explicit mask.
def _masked(values: np.ndarray, arrow_type: pa.DataType, valid: np.ndarray) -> pa.Array:
    if pa.types.is_integer(arrow_type):
        arrow_type = fit_integer_type(values, valid, arrow_type)
    if pa.types.is_integer(arrow_type):
        safe = np.where(valid, values, 0).astype("int64")
        return pa.array(safe, type=arrow_type, mask=~valid)
    return pa.array(values, type=arrow_type, mask=~valid)


# The five positional WDIR/WSPD occurrences as interpreted by PREPDATA R06UBF
# (NOAA-EMC/prepobs iw3unpbf.f), which keeps only slot 1: 1 final, 2 first guess,
# 3 original, 4 image 1->2, 5 image 2->3. They are correlated processing stages,
# not independent observations. The files' TSIG values agree except for slot 3,
# which TSIG labels "time averaged".
WIND_SLOT_STAGES: tuple[tuple[int, str], ...] = (
    (1, "final"),
    (2, "first_guess"),
    (3, "original"),
    (4, "pair12"),
    (5, "pair23"),
)
FIVE_SLOT_WIDTH = 5

# ---------------------------------------------------------------------------
# QI resolution
# ---------------------------------------------------------------------------

# QIFN/QIFY codes and the AMVQIC families are the loader's (satwnd_qc). GSI's
# expected-error codes for the GNAP-triplet families, which the loader does not use:
EE_GNAP_BY_CASE: Mapping[int, int] = {
    1: 4,
    4: 104,
    7: 4,
    8: 4,
    9: 4,
    11: 4,
    13: 4,
}

# AMVIVR (3-62-197), the intermediate-vector sequence, decoded as
# list<struct>. Its decoder name is positional (e.g. DRF8BIT_3) and shifts with
# replication order, so it is found by the fields it carries.
AMVIVR_MARKERS = frozenset({"TCOV", "CVWD"})


def _amvivr_column_index(schema: pa.Schema) -> int | None:
    """Index of the one list<struct<...>> column carrying the AMVIVR markers."""
    found = []
    for i in range(len(schema)):
        field_type = schema.field(i).type
        if not pa.types.is_list(field_type):
            continue
        value_type = field_type.value_type
        if not pa.types.is_struct(value_type):
            continue
        names = {value_type.field(j).name for j in range(value_type.num_fields)}
        if AMVIVR_MARKERS <= names:
            found.append(i)
    if len(found) > 1:
        raise RuntimeError(
            "AMVIVR marker fields {TCOV, CVWD} appear in "
            f"{[schema.field(i).name for i in found]}; the sequence is supposed "
            "to be unique per subset and the selection would be arbitrary"
        )
    return found[0] if found else None


def _amvivr_first(column: pa.ChunkedArray | pa.Array, field_name: str) -> np.ndarray:
    """`field_name` from intermediate vector 0 of each row, NaN where absent.

    Matches GSI's ``amvivr(2,1)`` (read_satwnd.f90). Struct fields are
    addressed by index because the structs carry duplicate names (LTDS twice).
    """
    if isinstance(column, pa.ChunkedArray):
        column = _one_array(column)
    struct_type = column.type.value_type
    matches = [
        j
        for j in range(struct_type.num_fields)
        if struct_type.field(j).name == field_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{field_name} appears {len(matches)}x in the AMVIVR struct; "
            "cannot resolve which occurrence GSI's amvivr(2,1) refers to"
        )
    inner = np.asarray(
        column.values.field(matches[0]).to_numpy(zero_copy_only=False),
        dtype="float64",
    )
    offsets = np.asarray(column.offsets)
    starts, ends = offsets[:-1], offsets[1:]
    out = np.full(len(column), np.nan)
    non_empty = ends > starts
    out[non_empty] = inner[starts[non_empty]]
    if column.null_count:
        out[np.asarray(column.is_null())] = np.nan
    return out


def _add_amvivr_columns(add, tbl: pa.Table) -> None:
    index = _amvivr_column_index(tbl.schema)
    if index is None:
        return
    column = tbl.column(index)
    for mnemonic, name in (
        ("CVWD", "amv_coefficient_of_variation"),
        ("TCOV", "amv_tracking_correlation"),
    ):
        values = _amvivr_first(column, mnemonic)
        add(name, _masked(values, pa.float64(), np.isfinite(values)))


def _add_qi_columns(
    add,
    n: int,
    diagnostics: pa.Array | None,
    istype: np.ndarray,
    istype_valid: np.ndarray,
) -> None:
    """Materialise qifn / qify / ee, selected by generating application.

    Null means not resolvable for the row, not a low QI (GOES-R often carries
    AMVQIC with GNAPS null).
    """
    qifn = np.full(n, np.nan)
    qify = np.full(n, np.nan)
    expected_error = np.full(n, np.nan)

    if diagnostics is not None and len(diagnostics):
        entries = diagnostics.values
        offsets = np.asarray(diagnostics.offsets)
        parent = np.repeat(np.arange(n), np.diff(offsets))

        pccf = np.asarray(
            entries.field("per_cent_confidence").to_numpy(zero_copy_only=False),
            dtype="float64",
        )
        gnap = _nullable_int(entries, "generating_application")
        gnaps = _nullable_int(entries, "standard_generating_application")

        row_istype = np.where(istype_valid, istype, -1)[parent]
        amvqic = np.isin(row_istype, sorted(AMVQIC_CASES))
        finite = np.isfinite(pccf)

        def select(
            target: np.ndarray,
            table: Mapping[int, int],
            codes: np.ndarray,
            applies: np.ndarray,
        ) -> None:
            want = np.full(len(codes), -1, dtype="int64")
            for case, code in table.items():
                want[row_istype == case] = code
            hit = applies & finite & (codes == want) & (want >= 0)
            target[parent[hit]] = pccf[hit]

        select(qifn, QIFN_GNAP_BY_CASE, gnap, ~amvqic)
        select(qify, QIFY_GNAP_BY_CASE, gnap, ~amvqic)
        select(expected_error, EE_GNAP_BY_CASE, gnap, ~amvqic)
        # AMVQIC families have no with-forecast variant, so qify stays null.
        hit = amvqic & finite & (gnaps == AMVQIC_QIFN_GNAPS)
        qifn[parent[hit]] = pccf[hit]
        hit = amvqic & finite & (gnaps == AMVQIC_EE_GNAPS)
        expected_error[parent[hit]] = pccf[hit]

    for values, name in (
        (qifn, "amv_qi_without_forecast"),
        (qify, "amv_qi_with_forecast"),
        (expected_error, "amv_expected_error"),
    ):
        add(name, _masked(values, pa.float64(), np.isfinite(values)))


def _nullable_int(entries: pa.StructArray, field: str) -> np.ndarray:
    column = entries.field(field)
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype="float64")
    return np.where(np.isfinite(values), values, -1).astype("int64")


def _add_wind_stage_columns(
    add,
    n: int,
    subset: str,
    wdir_positions: list[int],
    wspd_positions: list[int],
    final_u: np.ndarray,
    final_v: np.ndarray,
    column_values,
) -> None:
    """Named per-stage U/V for the legacy five-slot templates only.

    Templates carry either one or five WDIR occurrences. Modern templates repeat
    other fields inside unrelated provider diagnostic sequences, which are not
    wind slots.
    """
    directions = [column_values(p) for p in wdir_positions]
    speeds = [column_values(p) for p in wspd_positions]

    availability = np.zeros(n, dtype="uint8")
    spread = np.full(n, np.nan)
    for index, (slot, stage) in enumerate(WIND_SLOT_STAGES):
        direction, speed = directions[index], speeds[index]
        valid = np.isfinite(direction) & np.isfinite(speed)
        availability |= valid.astype("uint8") << (slot - 1)
        radians = np.deg2rad(direction)
        u, v = -speed * np.sin(radians), -speed * np.cos(radians)
        if slot == 1:
            # Slot 1 is wind_direction/wind_speed/wind_u_derived/wind_v_derived.
            continue
        # Skip slots that copy the final wind for this subset (see
        # WIND_SLOT_GATE); the availability mask still records them.
        if not WIND_SLOT_GATE.get((subset, slot), False):
            continue
        for component, values in (("u", u), ("v", v)):
            add(f"wind_{component}_{stage}", _masked(values, pa.float64(), valid))
        separation = np.sqrt((u - final_u) ** 2 + (v - final_v) ** 2)
        usable = valid & np.isfinite(separation)
        spread = np.where(
            usable & (~np.isfinite(spread) | (separation > spread)), separation, spread
        )

    add("wind_slot_availability", pa.array(availability, type=pa.uint8()))
    add("wind_slot_spread", _masked(spread, pa.float64(), np.isfinite(spread)))


def _add_gsi_columns(
    add, n: int, subset: str, said: np.ndarray, swcm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Emit the derived GSI routing columns; return (istype, istype_valid).

    Looks up each distinct (SAID, SWCM) pair in the batch once.
    """
    usable = np.isfinite(said) & np.isfinite(swcm)
    # GSI: `if(ihdr9<1 .or. ihdr9>7) cycle loop_report`.
    usable &= (swcm >= 1) & (swcm <= 7)
    # SWCM is 1..7 here, so SAID * 8 + SWCM identifies the pair.
    keys = said[usable].astype("int64") * 8 + swcm[usable].astype("int64")
    unique, inverse = np.unique(keys, return_inverse=True)
    # One entry per distinct pair, then a null entry for unusable rows. An
    # unmapped pair yields no type at all.
    entries = [GSI_TYPE_TABLE.get((subset, int(k) // 8, int(k) % 8)) for k in unique]
    entries.append(None)
    slot = np.full(n, len(unique), dtype="int64")
    slot[usable] = inverse

    def per_row(values: list, arrow_type: pa.DataType) -> pa.Array:
        return pa.array(values, type=arrow_type).take(slot)

    # istype -1: the direct GSI reader has no processing case, but PREPDATA
    # types the report (INSAT, tier `3-prepdata`). Keep the type and leave the
    # subtype null.
    has_subtype = [e is not None and e[1] != -1 for e in entries]
    add(
        "gsi_observation_type",
        per_row([e[0] if e is not None else None for e in entries], pa.uint16()),
    )
    add(
        "gsi_internal_subtype",
        per_row(
            [e[1] if h else None for e, h in zip(entries, has_subtype)], pa.uint8()
        ),
    )
    for name, field in (
        ("gsi_provider", 2),
        ("gsi_subprovider", 3),
        ("gsi_type_mapping_tier", 4),
    ):
        add(
            name,
            per_row(
                [e[field] if e is not None else None for e in entries], pa.string()
            ),
        )
    subtypes = np.array([e[1] if h else 0 for e, h in zip(entries, has_subtype)])[slot]
    subtype_valid = np.array(has_subtype)[slot]
    return subtypes, subtype_valid


def _add_candidate_groups(
    add,
    tbl: pa.Table,
    n: int,
    effective_occurrences: Sequence[Mapping[str, Any]],
    fxy_by_mnemonic: Mapping[str, str],
    column_values,
) -> set[str]:
    """Emit the semantic ``list<struct>`` columns; return the mnemonics consumed.

    Grouping is decided here from the template's own expansion order (see
    ``candidate_groups``), because occurrence counts differ per mnemonic and
    parallel per-mnemonic lists therefore have no well-defined zip.
    """
    if not effective_occurrences:
        return set()

    def values_for(mnemonic: str, occurrence: int) -> np.ndarray | None:
        positions = _col_positions(tbl, mnemonic)
        if occurrence >= len(positions):
            return None
        return column_values(positions[occurrence])

    def member_arrow_type(mnemonic: str) -> pa.DataType:
        resolved = _resolve(mnemonic, fxy_by_mnemonic)
        return _arrow_type_for(resolved) if resolved else pa.float64()

    consumed: set[str] = set()
    for spec in GROUP_SPECS:
        entries = plan_groups(effective_occurrences, spec)
        array = build_group_array(spec, entries, values_for, n, member_arrow_type)
        if array is None:
            continue
        add(spec.name, array)
        consumed |= {spec.anchor, *spec.before, *spec.after}
    return consumed


def project_table(
    tbl: pa.Table,
    root_sequence: str,
    cycle: datetime,
    ncep_dump_subtype: str = "",
    active_elements: Sequence[Sequence[str]] = (),
    effective_occurrences: Sequence[Mapping[str, Any]] = (),
    source_stream: str = "satwnd",
) -> pa.Table:
    """Project one decoded batch; field metadata is applied by ``conform_to_declared``."""
    n = tbl.num_rows
    fxy_by_mnemonic = {mnemonic: fxy for fxy, mnemonic in active_elements}
    column_values = _column_cache(tbl)
    arrays: list[pa.Array] = []
    names: list[str] = []

    def add(name: str, array: pa.Array) -> None:
        arrays.append(array)
        names.append(name)

    def first(mnemonic: str) -> np.ndarray:
        return resolve_promoted_occurrence(tbl, mnemonic, column_values)

    # --- template provenance ---
    add("source_template", pa.array([root_sequence] * n, type=pa.string()))
    add("ncep_dump_subtype", pa.array([ncep_dump_subtype] * n, type=pa.string()))

    # --- GSI routing identity ---
    # Keyed on (subset, SAID, SWCM): SWCM=1 covers both GOES long-wave (245)
    # and short-wave (240) IR, which only the subset separates.
    said = first("SAID")
    # Some templates carry only CMCM, the older NCEP-local spelling of SWCM.
    swcm = first("SWCM")
    if not _col_positions(tbl, "SWCM"):
        swcm = first("CMCM")
    istype, istype_valid = _add_gsi_columns(add, n, ncep_dump_subtype, said, swcm)

    # --- time ---
    obs_time_ns = _compose_utc(
        first("YEAR"),
        first("MNTH"),
        first("DAYS"),
        first("HOUR"),
        first("MINU"),
        first("SECO"),
    )
    add("time_utc", pa.array(obs_time_ns, type=pa.timestamp("ns")))
    add(
        "da_window",
        pa.array(
            archive_cycle_and_window(obs_time_ns)[1],
            type=pa.timestamp("ns"),
        ),
    )
    add(
        "cycle",
        pa.array(
            [np.datetime64(cycle.replace(tzinfo=None), "ns")] * n,
            type=pa.timestamp("ns"),
        ),
    )

    # --- geolocation ---
    coordinates: dict[str, np.ndarray] = {}
    for name, candidates in COALESCED_COLUMNS.items():
        values = np.full(n, np.nan)
        for mnemonic in candidates:
            if _col_positions(tbl, mnemonic):
                values = first(mnemonic)
                break
        coordinates[name] = values
        add(name, _masked(values, pa.float64(), np.isfinite(values)))

    pixels, pixels_valid = hpx_nest_pixels(
        coordinates["latitude"], coordinates["longitude"], ARCHIVE_HPX_LEVEL
    )
    add("hpx4096_nest", pa.array(pixels, type=pa.uint32(), mask=~pixels_valid))

    # --- stream provenance ---
    # One stream is read per era, so no row is corroborated by another stream.
    add("source_stream", source_stream_array(source_stream, n))
    add("also_in_other_stream", pa.array(np.zeros(n, dtype=bool), type=pa.bool_()))

    # --- wind ---
    # U/V is derived from WDIR/WSPD only; native 0-11-003/4 are emitted below
    # as wind_u_source / wind_v_source.
    direction, speed = first("WDIR"), first("WSPD")
    wind_valid = np.isfinite(direction) & np.isfinite(speed)
    radians = np.deg2rad(direction)
    final_u = -speed * np.sin(radians)
    final_v = -speed * np.cos(radians)
    add("wind_u_derived", _masked(final_u, pa.float64(), wind_valid))
    add("wind_v_derived", _masked(final_v, pa.float64(), wind_valid))

    wdir_positions = _col_positions(tbl, "WDIR")
    wspd_positions = _col_positions(tbl, "WSPD")
    if (
        len(wdir_positions) == FIVE_SLOT_WIDTH
        and len(wspd_positions) == FIVE_SLOT_WIDTH
    ):
        _add_wind_stage_columns(
            add,
            n,
            ncep_dump_subtype,
            wdir_positions,
            wspd_positions,
            final_u,
            final_v,
            column_values,
        )

    # --- GSI's pct1 from the AMVIVR sequence ---
    _add_amvivr_columns(add, tbl)

    # --- remaining source fields, named from the catalogue ---
    # Each mnemonic yields a flat scalar if it occurs once or is slot1_promoted,
    # and an `<name>_occurrences` list if repeated and not consumed by a group.
    grouped = _add_candidate_groups(
        add, tbl, n, effective_occurrences, fxy_by_mnemonic, column_values
    )

    # --- qifn / qify / ee ---
    # Resolved from the emitted quality_diagnostics group so the flat columns
    # and the nested list agree.
    diagnostics = next(
        (array for name, array in zip(names, arrays) if name == "quality_diagnostics"),
        None,
    )
    _add_qi_columns(add, n, diagnostics, istype, istype_valid)
    consumed = set(TIME_COMPONENTS) | {
        c for cs in COALESCED_COLUMNS.values() for c in cs
    }

    positions_by_mnemonic: dict[str, list[int]] = {}
    for i in range(tbl.num_columns):
        positions_by_mnemonic.setdefault(tbl.schema.field(i).name, []).append(i)

    for mnemonic, positions in positions_by_mnemonic.items():
        catalog = _resolve(mnemonic, fxy_by_mnemonic)
        name = _column_name_for(catalog, mnemonic)
        repeated = len(positions) > 1
        promoted = catalog is not None and catalog.slot1_promoted

        # CCITT IA5 text is carried as strings; numeric-looking text such as
        # SEQNUM and BULTIM has leading zeros.
        if _is_text_column(tbl, positions[0]):
            if mnemonic in consumed:
                continue
            add(name, _text_values(tbl, positions[0]))
            continue

        if mnemonic not in consumed and (not repeated or promoted):
            values = first(mnemonic)
            valid = np.isfinite(values)
            if mnemonic == "PRLC":
                valid &= values > 0
            arrow_type = (
                _arrow_type_for(catalog) if catalog is not None else pa.float64()
            )
            add(name, _masked(values, arrow_type, valid))

        if repeated and mnemonic not in grouped:
            # Every subset of one template has the same occurrence count, so the
            # list column is a regular n x k block.
            stacked = np.stack([column_values(p) for p in positions], axis=1)
            width = stacked.shape[1]
            values = pa.array(
                np.ascontiguousarray(stacked).reshape(-1), type=pa.float64()
            )
            offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
            add(f"{name}_occurrences", pa.ListArray.from_arrays(offsets, values))

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(
            f"{root_sequence}: projection produced duplicate archive fields {duplicates}"
        )
    projected = pa.Table.from_arrays(arrays, names=names)

    # Drop rows with no usable time: they have no DA window, and the writer
    # rejects a day whose required da_window column contains nulls.
    if projected.num_rows:
        keep = ~np.isnat(obs_time_ns)
        if not keep.all():
            projected = projected.filter(pa.array(keep))
    return projected


# Decoded bytes per batch; bounds the decoder's peak memory.
TARGET_DECODED_BYTES = 64 * 2**20


def iter_projected_batches(
    path: str,
    cycle: datetime,
    *,
    source_stream: str = "satwnd",
):
    """Yield projected tables for one cycle, bounded in memory.

    Each batch is one (category, template, compression, table epoch) run and is
    labelled from its first source message.
    """
    declared = load_declared_schema()
    with earth2bufr.open_bufr(path) as handle:
        info = handle.info
        structures = _source_structures(path, handle)
        for batch in handle.iter_batches(target_decoded_bytes=TARGET_DECODED_BYTES):
            first_message = info.messages[
                batch.column("_source_message_index")[0].as_py()
            ]
            root = _root_of(first_message)
            structure = _structure_for_message(
                structures,
                root,
                info.table_epochs[first_message.table_epoch].embedded_table_sha256,
                first_message.descriptors,
            )
            table = pa.Table.from_batches([batch]).drop_columns(
                ["_source_message_index", "_source_subset_index"]
            )
            projected = project_table(
                table,
                root,
                cycle,
                structure["ncep_dump_subtype"],
                structure["active_elements"],
                structure["effective_occurrences"],
                source_stream=source_stream,
            )
            dropped = table.num_rows - projected.num_rows
            yield conform_to_declared(projected, declared), dropped


def load_declared_schema() -> pa.Schema:
    with pa.OSFile(str(DECLARED_SCHEMA_PATH), "rb") as source:
        return pa.ipc.open_stream(source).schema


def conform_to_declared(table: pa.Table, declared: pa.Schema) -> pa.Table:
    """Reindex a projected table onto the declared archive schema.

    Columns the template did not supply become nulls; an absent group is a null
    list, which costs almost nothing in Parquet, not a list of empty entries.
    """
    present = set(table.schema.names)
    # A projected column missing from the declared schema is an error, not
    # dropped.
    undeclared = sorted(present - set(declared.names))
    if undeclared:
        raise RuntimeError(
            f"projected columns absent from the declared schema: {undeclared}"
        )
    columns = []
    for field in declared:
        if field.name in present:
            column = _one_array(table.column(field.name))
            if not column.type.equals(field.type):
                raise TypeError(
                    f"{field.name} is {column.type}; the declared schema has {field.type}"
                )
            columns.append(column)
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=declared)

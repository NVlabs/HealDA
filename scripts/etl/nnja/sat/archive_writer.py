#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical Parquet writer for the NNJA sounder archive contract.

Spooling, one row group per DA window and atomic publication come from the
shared ``archive_common`` day writer. ``SounderArchiveDayWriter`` supplies the
sounder contract validator, footer provenance and the CrIS column encodings.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..archive_common.archive_writer import ArchiveDayWriter
from .archive_schema import (
    Channel,
    ChannelManifest,
    FileProvenance,
    SENSOR_SPECS,
    SourceStructure,
    archive_schema,
    bind_file_metadata,
    decode_message_header_manifest,
    source_file_structure_hashes,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _delta_encoded_columns(sensor: str, schema: pa.Schema) -> tuple[str, ...]:
    if sensor != "cris":
        return ()
    return tuple(
        name for name in schema.names if name.startswith("spectral_radiance_code__")
    )


class SounderArchiveDayWriter(ArchiveDayWriter):
    """Shared day writer with the sounder contract's hooks bound.

    Provenance is known only after streaming; ``bind_provenance`` puts it in
    the Parquet key-value footer.
    """

    def __init__(
        self,
        path: Path,
        *,
        sensor: str,
        schema: pa.Schema,
        scratch_dir: Path | None = None,
        require_complete_day: bool = False,
    ) -> None:
        self.require_complete_day = require_complete_day
        super().__init__(
            path,
            schema=schema,
            provenance=None,
            scratch_dir=scratch_dir,
            delta_encoded_columns=_delta_encoded_columns(sensor, schema),
        )

    def bind_metadata(self, schema: pa.Schema, provenance: FileProvenance) -> pa.Schema:
        return bind_file_metadata(schema, provenance)

    def bind_provenance(self, provenance: FileProvenance) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        provenance.validate()
        bound = bind_file_metadata(self.schema, provenance)
        self.closing_metadata = {
            key.decode(): value.decode() for key, value in bound.metadata.items()
        }
        self.provenance = provenance

    def validate_contract(self, path: Path) -> dict[str, Any]:
        if self.provenance is None:
            raise RuntimeError("actual-source provenance must be bound before close")
        if self._rows + self.provenance.quarantined_subsets != (
            self.provenance.source_subsets_encountered
        ):
            raise ValueError(
                "archive rows plus quarantined subsets do not match the source "
                "population"
            )
        return validate_parquet_contract(
            path, require_complete_day=self.require_complete_day
        )


def validate_parquet_contract(
    path: Path,
    *,
    require_complete_day: bool = False,
    expected_sensor: str | None = None,
    expected_day: date | None = None,
    expected_sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check self-description, structure hashes, and row-group invariants."""

    parquet = pq.ParquetFile(path)
    # Static contract metadata lives in the serialized Arrow schema. Dynamic
    # actual-source metadata may be attached to the Parquet file footer after
    # streaming has begun, so readers must merge both standard footer views.
    metadata = dict(parquet.schema_arrow.metadata or {})
    metadata.update(parquet.metadata.metadata or {})
    required = {
        b"nnja.archive.contract",
        b"nnja.archive.schema_id",
        b"nnja.archive.file_provenance",
        b"nnja.archive.source_structures",
        b"nnja.archive.source_structure_hashes",
    }
    missing = required - metadata.keys()
    if missing:
        raise AssertionError(f"missing archive metadata: {sorted(missing)}")

    contract = json.loads(metadata[b"nnja.archive.contract"])
    schema_id = json.loads(metadata[b"nnja.archive.schema_id"])
    if contract.get("schema_id") != schema_id:
        raise AssertionError("schema ID disagrees with the embedded contract")
    structure_records = json.loads(metadata[b"nnja.archive.source_structures"])
    structure_hashes = set()
    for record in structure_records:
        actual = hashlib.sha256(_canonical_json(record["structure"])).hexdigest()
        if actual != record["sha256"]:
            raise AssertionError("embedded source-structure hash mismatch")
        structure_hashes.add(actual)
    advertised = set(json.loads(metadata[b"nnja.archive.source_structure_hashes"]))
    if structure_hashes != advertised:
        raise AssertionError("source-structure hash index mismatch")
    provenance = json.loads(metadata[b"nnja.archive.file_provenance"])
    source_files = provenance["source_files"]
    source_references = []
    try:
        for source in source_files:
            source_references.append((source, source_file_structure_hashes(source)))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    referenced = {value for _, hashes in source_references for value in hashes}
    if referenced != structure_hashes:
        raise AssertionError("source files do not map exactly to embedded structures")
    sensor = contract["sensor"]
    manifest_payload = contract["channel_manifest"]
    manifest = ChannelManifest(
        sensor=manifest_payload["sensor"],
        release=manifest_payload["release"],
        channels=tuple(Channel(**channel) for channel in manifest_payload["channels"]),
    )
    expected_schemas = []
    for record in structure_records:
        structure = record["structure"]
        source_structure = SourceStructure(
            sensor=structure["sensor"],
            embedded_table_sha256=structure["embedded_table_sha256"],
            observation_descriptors=tuple(structure["observation_descriptors"]),
            active_sequences=tuple(structure["active_sequences"]),
            active_elements=tuple(structure["active_elements"]),
            effective_occurrences=tuple(structure["effective_occurrences"]),
            decoded_physical_schema=tuple(structure["decoded_physical_schema"]),
            format_version=structure["format_version"],
        )
        expected_schema = archive_schema(
            sensor,
            manifest,
            source_structure=source_structure,
            contract_version=str(contract["contract_version"]),
        )
        expected_schema_id = json.loads(
            expected_schema.metadata[b"nnja.archive.schema_id"]
        )
        if expected_schema_id != schema_id:
            raise AssertionError(
                "schema ID cannot be reproduced from every embedded structure"
            )
        expected_schemas.append(expected_schema)
    expected_schema = expected_schemas[0]
    actual_schema = parquet.schema_arrow
    if not expected_schema.remove_metadata().equals(actual_schema.remove_metadata()):
        raise AssertionError("actual Arrow fields do not match the embedded contract")
    for expected_field, actual_field in zip(
        expected_schema, actual_schema, strict=True
    ):
        if expected_field.metadata != actual_field.metadata:
            raise AssertionError(f"field metadata changed for {expected_field.name!r}")
    nested_columns = [
        field.name for field in actual_schema if pa.types.is_nested(field.type)
    ]
    if parquet.metadata.num_rows and nested_columns:
        for row_group in range(parquet.num_row_groups):
            probe = next(
                parquet.iter_batches(
                    batch_size=8,
                    row_groups=[row_group],
                    columns=nested_columns,
                ),
                None,
            )
            if probe is None or probe.num_rows == 0:
                raise AssertionError(
                    f"row group {row_group} could not decode nested archive fields"
                )
    utc_day = date.fromisoformat(provenance["utc_day"])
    if expected_sensor is not None and sensor != expected_sensor:
        raise AssertionError(f"expected sensor {expected_sensor}, found {sensor}")
    if expected_day is not None and utc_day != expected_day:
        raise AssertionError(f"expected day {expected_day}, found {utc_day}")
    structures_by_hash = {
        record["sha256"]: record["structure"] for record in structure_records
    }
    optional_fields = [
        field for field in SENSOR_SPECS[sensor].fields if field.source_optional
    ]
    missing_template_fields_by_source = {}
    for source, hashes in source_references:
        missing = set()
        referenced_templates = set()
        for structure_hash in hashes:
            structure = structures_by_hash[structure_hash]
            referenced_templates.add(tuple(structure["observation_descriptors"]))
            mnemonics = {
                str(occurrence["mnemonic"])
                for occurrence in structure["effective_occurrences"]
                if occurrence.get("mnemonic") is not None
            }
            for field in optional_fields:
                expected = (
                    (field.source_container,)
                    if field.source_container is not None
                    else field.source_mnemonics
                )
                if not any(mnemonic in mnemonics for mnemonic in expected):
                    missing.add(field.name)
        header_manifest = source.get("message_header_manifest")
        if header_manifest is not None:
            headers = decode_message_header_manifest(header_manifest)
            header_templates = {
                tuple(str(value) for value in descriptors)
                for descriptors in headers["descriptors"]
            }
            if header_templates != referenced_templates:
                raise AssertionError(
                    f"{source['object_key']} structure references do not match "
                    "message-header templates"
                )
        if missing:
            missing_template_fields_by_source[source["object_key"]] = sorted(missing)
    if (
        parquet.metadata.num_rows + provenance["quarantined_subsets"]
        != provenance["source_subsets_encountered"]
    ):
        raise AssertionError(
            "Parquet rows plus quarantine do not match source population"
        )
    quarantined_messages = provenance.get("quarantined_source_messages", ())
    quarantined_subsets = sum(
        int(record["subset_count"]) for record in quarantined_messages
    )
    if quarantined_subsets != int(provenance["quarantined_subsets"]):
        raise AssertionError("quarantine records disagree with quarantined population")
    if quarantined_messages:
        quarantine_sha256 = hashlib.sha256(
            _canonical_json(list(quarantined_messages))
        ).hexdigest()
        if quarantine_sha256 != provenance.get("quarantine_manifest_sha256"):
            raise AssertionError("quarantine records disagree with manifest SHA-256")
    if expected_sources is not None:
        # A source with no observation subsets is listed only as unusable.
        actual_identity = sorted(
            (
                {
                    "object_key": source["object_key"],
                    "cycle": source["cycle"],
                    "size_bytes": source["size_bytes"],
                }
                for source in (
                    *source_files,
                    *provenance.get("unusable_source_objects", ()),
                )
            ),
            key=lambda identity: identity["cycle"],
        )
        if actual_identity != list(expected_sources):
            raise AssertionError(
                "source object identity differs from the discovered inputs"
            )

    da_index = next(
        index
        for index in range(parquet.metadata.num_columns)
        if parquet.schema.column(index).path == "da_window"
    )
    windows = []
    for group_index in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(group_index).column(da_index).statistics
        if statistics is None or statistics.min != statistics.max:
            raise AssertionError(f"row group {group_index} crosses da_window")
        windows.append(statistics.min.isoformat())
    if len(windows) != len(set(windows)):
        raise AssertionError("more than one row group represents a da_window")
    if require_complete_day:
        if any(
            source.get("message_header_manifest") is None for source in source_files
        ):
            raise AssertionError(
                "complete archive is missing per-source message headers"
            )
        header_subsets = sum(
            sum(
                int(value)
                for value in decode_message_header_manifest(
                    source["message_header_manifest"]
                )["subset_count"]
            )
            for source in source_files
        )
        if header_subsets != provenance["source_subsets_encountered"]:
            raise AssertionError("message headers disagree with source population")
        expected_cycles = [
            datetime(
                utc_day.year,
                utc_day.month,
                utc_day.day,
                hour,
                tzinfo=timezone.utc,
            ).isoformat()
            for hour in (0, 6, 12, 18)
        ]
        missing_cycles = set(provenance.get("missing_cycles", ()))
        expected_cycles = [
            cycle
            for hour, cycle in zip((0, 6, 12, 18), expected_cycles, strict=True)
            if f"{hour:02d}Z" not in missing_cycles
        ]
        actual_cycles = [source["cycle"] for source in source_files]
        if actual_cycles != expected_cycles:
            raise AssertionError(
                "archive does not contain the ordered available source cycles"
            )
        expected_windows = [
            (datetime.fromisoformat(cycle) + offset).isoformat()
            for cycle in expected_cycles
            for offset in (timedelta(0), timedelta(hours=3))
        ]
        empty_windows = set(provenance.get("empty_windows", ()))
        expected_empty = {
            window for window in expected_windows if window not in windows
        }
        if empty_windows != expected_empty:
            raise AssertionError(
                f"empty-window provenance mismatch: expected "
                f"{sorted(expected_empty)}, found {sorted(empty_windows)}"
            )
        expected_populated = [
            window for window in expected_windows if window not in empty_windows
        ]
        if windows != expected_populated:
            raise AssertionError("archive DA windows are incomplete or out of order")
    return {
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "schema_id": schema_id,
        "sensor": sensor,
        "utc_day": utc_day.isoformat(),
        "missing_cycles": list(provenance.get("missing_cycles", ())),
        "empty_windows": list(provenance.get("empty_windows", ())),
        "unusable_source_objects": list(provenance.get("unusable_source_objects", ())),
        "quarantined_subsets": quarantined_subsets,
        "quarantined_source_messages": list(quarantined_messages),
        "missing_template_fields": sorted(
            {
                field
                for fields in missing_template_fields_by_source.values()
                for field in fields
            }
        ),
        "missing_template_fields_by_source": missing_template_fields_by_source,
        "source_files": source_files,
        "source_structure_hashes": sorted(structure_hashes),
        "windows": windows,
    }

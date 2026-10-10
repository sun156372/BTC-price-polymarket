"""Minimal, dependency-free Parquet metadata reader.

Why this exists
---------------
Knowing the real column names is the difference between a loader that runs and a
loader that crashes on line one. Parquet keeps its schema in a Thrift-encoded
footer, so the column names and physical types are recoverable **without
pyarrow, pandas, or any decompression** - we never touch the data pages.

The footer is read from the last few hundred KB of the file, so this is cheap
even on very large files.

What it does NOT do
-------------------
It does not decode column data (that needs a full Parquet reader: dictionary
decoding, RLE/bit-packing, page decompression). For actual values, install
``pyarrow``; this module is for schema discovery and row counts.

Implementation note
-------------------
Thrift compact protocol encodes a binary field as:
``<header byte> <varint length> <UTF-8 bytes>``. We look for the field-id bytes
immediately preceding a length-delimited UTF-8 run, which gives a reliable
signal for schema names (and for statistics keys like ``min_value``) while
rejecting random binary. This is deliberately conservative: a name is accepted
only if the run is printable and the preceding varint exactly matches its length.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Iterator

MAGIC = b"PAR1"

# parquet physical type enum (Type)
PHYSICAL_TYPES = {
    0: "BOOLEAN",
    1: "INT32",
    2: "INT64",
    3: "INT96",
    4: "FLOAT",
    5: "DOUBLE",
    6: "BYTE_ARRAY",
    7: "FIXED_LEN_BYTE_ARRAY",
}

# parquet ConvertedType enum (meaningful subset)
CONVERTED_TYPES = {
    0: "UTF8",
    1: "MAP",
    2: "MAP_KEY_VALUE",
    3: "LIST",
    4: "ENUM",
    5: "DECIMAL",
    6: "DATE",
    7: "TIME_MILLIS",
    8: "TIME_MICROS",
    9: "TIMESTAMP_MILLIS",
    10: "TIMESTAMP_MICROS",
    11: "UINT_8",
    12: "UINT_16",
    13: "UINT_32",
    14: "UINT_64",
    15: "INT_8",
    16: "INT_16",
    17: "INT_32",
    18: "INT_64",
    19: "JSON",
    20: "BSON",
    21: "INTERVAL",
}

# parquet LogicalType union field ids (LogicalType is a union of structs)
LOGICAL_TYPE_IDS = {
    1: "STRING",
    2: "MAP",
    3: "LIST",
    4: "ENUM",
    5: "DECIMAL",
    6: "DATE",
    7: "TIME",
    8: "TIMESTAMP",
    9: "INTEGER",
    10: "UNKNOWN",
    11: "JSON",
    12: "BSON",
    13: "UUID",
    14: "FLOAT16",
}

#: schema element fields we can recognise by name in the metadata
_SCHEMA_FIELDS = {"name", "type", "type_length", "repetition_type", "num_children",
                  "converted_type", "scale", "precision", "field_id", "logicalType"}


class ParquetMetadataError(RuntimeError):
    """Raised when a file is not a readable parquet file."""


@dataclass
class ParquetColumn:
    name: str
    physical_type: str = "UNKNOWN"
    converted_type: str | None = None
    logical_type: str | None = None
    repetition: str | None = None
    num_children: int = 0
    is_nested: bool = False

    @property
    def kind(self) -> str:
        """A friendly type name for reporting."""
        if self.logical_type:
            return self.logical_type
        if self.converted_type in {"UTF8", "JSON", "BSON", "ENUM"}:
            return "STRING"
        if self.converted_type in {"TIMESTAMP_MILLIS", "TIMESTAMP_MICROS", "DATE",
                                   "TIME_MILLIS", "TIME_MICROS"}:
            return self.converted_type
        if self.physical_type == "BYTE_ARRAY":
            # no logical annotation: likely a JSON/string blob such as a ladder
            return "BYTE_ARRAY(string?)"
        return self.physical_type

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "physical_type": self.physical_type,
            "converted_type": self.converted_type,
            "logical_type": self.logical_type,
            "kind": self.kind,
            "repetition": self.repetition,
            "num_children": self.num_children,
            "is_nested": self.is_nested,
        }


@dataclass
class ParquetMetadata:
    path: str
    size_bytes: int
    num_rows: int | None = None
    num_row_groups: int | None = None
    created_by: str = ""
    columns: list[ParquetColumn] = field(default_factory=list)
    stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "num_rows": self.num_rows,
            "num_row_groups": self.num_row_groups,
            "created_by": self.created_by,
            "columns": [c.as_dict() for c in self.columns],
            "stats": {k: list(v) for k, v in self.stats.items()},
            "notes": self.notes,
            "error": self.error,
        }


# ------------------------------------------------------- thrift compact decoding
# Thrift compact protocol:
#   field header = (delta << 4) | type ; delta 0 means an explicit zigzag field id
#   types: 1 BOOL_TRUE 2 BOOL_FALSE 3 BYTE 4 I16 5 I32 6 I64 7 DOUBLE
#          8 BINARY 9 LIST 10 SET 11 MAP 12 STRUCT
#   i16/i32/i64 are zigzag varints; DOUBLE is 8 little-endian bytes.
STRUCT = 12
LIST = 9
SET = 10
MAP = 11
BINARY = 8
DOUBLE = 7
BYTE = 3
BOOL_TRUE = 1
BOOL_FALSE = 2
INT_TYPES = {4, 5, 6}


@dataclass
class ThriftStruct:
    """A decoded Thrift struct. Fields are keyed by field id."""

    fields: dict[int, Any] = field(default_factory=dict)

    def get(self, key: int, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def __contains__(self, key: int) -> bool:
        return key in self.fields

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ThriftStruct({sorted(self.fields)})"


class CompactReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    # -- primitives -------------------------------------------------------
    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise ParquetMetadataError("unexpected end of metadata")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def varint(self) -> int:
        result = 0
        shift = 0
        while True:
            byte = self.byte()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 70:
                raise ParquetMetadataError("varint too long")

    def zigzag(self) -> int:
        value = self.varint()
        return (value >> 1) ^ -(value & 1)

    def binary(self) -> bytes:
        length = self.varint()
        if length < 0 or self.pos + length > len(self.data):
            raise ParquetMetadataError("binary length out of range")
        raw = self.data[self.pos : self.pos + length]
        self.pos += length
        return raw

    def double(self) -> float:
        if self.pos + 8 > len(self.data):
            raise ParquetMetadataError("truncated double")
        value = struct.unpack("<d", self.data[self.pos : self.pos + 8])[0]
        self.pos += 8
        return value

    # -- composites -------------------------------------------------------
    def value(self, type_id: int) -> Any:
        if type_id == BOOL_TRUE:
            return True
        if type_id == BOOL_FALSE:
            return False
        if type_id == BYTE:
            return self.byte()
        if type_id in INT_TYPES:
            return self.zigzag()
        if type_id == DOUBLE:
            return self.double()
        if type_id == BINARY:
            return self.binary()
        if type_id in (LIST, SET):
            return self.list_value()
        if type_id == MAP:
            return self.map_value()
        if type_id == STRUCT:
            return self.struct_value()
        raise ParquetMetadataError(f"unknown thrift type id {type_id}")

    def list_value(self) -> list[Any]:
        header = self.byte()
        size = (header >> 4) & 0x0F
        element_type = header & 0x0F
        if size == 15:
            size = self.varint()
        return [self.value(element_type) for _ in range(size)]

    def map_value(self) -> dict[Any, Any]:
        size = self.varint()
        if size == 0:
            return {}
        types = self.byte()
        key_type = (types >> 4) & 0x0F
        value_type = types & 0x0F
        return {self.value(key_type): self.value(value_type) for _ in range(size)}

    def struct_value(self) -> ThriftStruct:
        result = ThriftStruct()
        last_field_id = 0
        while True:
            header = self.byte()
            if header == 0:  # STOP
                return result
            delta = (header >> 4) & 0x0F
            type_id = header & 0x0F
            if delta == 0:
                field_id = self.zigzag()
            else:
                field_id = last_field_id + delta
            last_field_id = field_id
            result.fields[field_id] = self.value(type_id)


# ------------------------------------------------------- parquet FileMetaData
# FileMetaData field ids we care about
FMD_VERSION = 1
FMD_SCHEMA = 2
FMD_NUM_ROWS = 3
FMD_ROW_GROUPS = 4
FMD_CREATED_BY = 6

# SchemaElement field ids
SE_TYPE = 1
SE_TYPE_LENGTH = 2
SE_REPETITION = 3
SE_NAME = 4
SE_NUM_CHILDREN = 5
SE_CONVERTED_TYPE = 6
SE_SCALE = 7
SE_PRECISION = 8
SE_FIELD_ID = 9
SE_LOGICAL_TYPE = 10

# row group / column chunk field ids
RG_NUM_ROWS = 3
RG_COLUMNS = 1
CC_META_DATA = 3
CM_PATH_IN_SCHEMA = 1
CM_STATISTICS = 12
CM_NUM_VALUES = 5

REPETITION = {0: "REQUIRED", 1: "OPTIONAL", 2: "REPEATED"}


def _as_text(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _logical_type_name(node: Any) -> str | None:
    if not isinstance(node, ThriftStruct):
        return None
    for field_id in sorted(node.fields):
        name = LOGICAL_TYPE_IDS.get(field_id)
        if name:
            return name
    return None


def decode_file_metadata(footer: bytes) -> ThriftStruct:
    """Decode FileMetaData from the exact metadata block.

    The struct must start at byte 0 of the block. If it does not decode there, we
    refuse rather than probing into padding, because a mis-aligned decode happily
    produces a wrong-but-plausible schema (the most dangerous possible outcome).
    """
    reader = CompactReader(footer)
    meta = reader.struct_value()
    if FMD_SCHEMA not in meta.fields and FMD_NUM_ROWS not in meta.fields:
        raise ParquetMetadataError(
            "decoded struct has neither schema nor num_rows - not FileMetaData"
        )
    return meta


def parse_schema(meta: ThriftStruct) -> list[ParquetColumn]:
    """Flatten the schema tree into leaf columns in declaration order.

    The first SchemaElement is the root; elements are laid out depth-first and
    each node's ``num_children`` says how many elements follow it. A recursive
    walk is the only correct way to consume siblings - a naive loop assigns
    grandchildren to the wrong parents.
    """
    schema = meta.get(FMD_SCHEMA) or []
    if not isinstance(schema, list) or not schema:
        return []
    if not isinstance(schema[0], ThriftStruct):
        return []

    position = 0

    def walk(node: ThriftStruct, prefix: str) -> list[ParquetColumn]:
        nonlocal position
        name = _as_text(node.get(SE_NAME, ""))
        path = f"{prefix}{name}"
        children = node.get(SE_NUM_CHILDREN, 0) or 0

        if children > 0:
            collected: list[ParquetColumn] = []
            for _ in range(children):
                position += 1
                if position >= len(schema):
                    break
                child = schema[position]
                if isinstance(child, ThriftStruct):
                    collected.extend(walk(child, f"{path}."))
            return collected

        physical = PHYSICAL_TYPES.get(node.get(SE_TYPE), "UNKNOWN")
        converted_id = node.get(SE_CONVERTED_TYPE)
        converted = CONVERTED_TYPES.get(converted_id) if converted_id is not None else None
        logical = _logical_type_name(node.get(SE_LOGICAL_TYPE))
        repetition_id = node.get(SE_REPETITION)
        return [
            ParquetColumn(
                name=path,
                physical_type=physical,
                converted_type=converted,
                logical_type=logical,
                repetition=REPETITION.get(repetition_id) if repetition_id is not None else None,
                num_children=0,
                is_nested=bool(prefix),
            )
        ]

    root = schema[0]
    total_children = root.get(SE_NUM_CHILDREN, 0) or 0
    leaves: list[ParquetColumn] = []
    for _ in range(total_children):
        position += 1
        if position >= len(schema):
            break
        child = schema[position]
        if isinstance(child, ThriftStruct):
            leaves.extend(walk(child, ""))
    return leaves


def parse_statistics(meta: ThriftStruct) -> dict[str, tuple[float, float]]:
    """min/max per column from the first row group's column statistics."""
    ranges: dict[str, tuple[float, float]] = {}
    row_groups = meta.get(FMD_ROW_GROUPS) or []
    if not isinstance(row_groups, list) or not row_groups:
        return ranges
    first = row_groups[0]
    if not isinstance(first, ThriftStruct):
        return ranges
    chunks = first.get(RG_COLUMNS) or []
    if not isinstance(chunks, list):
        return ranges

    for chunk in chunks:
        if not isinstance(chunk, ThriftStruct):
            continue
        path = chunk.get(CM_PATH_IN_SCHEMA)
        column = _as_text(path[0]) if isinstance(path, list) and path else ""
        column_meta = chunk.get(CC_META_DATA)
        if not column or not isinstance(column_meta, ThriftStruct):
            continue
        stats = column_meta.get(CM_STATISTICS)
        if not isinstance(stats, ThriftStruct):
            continue
        # field ids: 1 max, 2 min, 5 max_value, 6 min_value (values may be binary)
        lower = _stat_number(stats.get(6, stats.get(2)))
        upper = _stat_number(stats.get(5, stats.get(1)))
        if lower is not None and upper is not None and lower <= upper:
            ranges[column] = (lower, upper)
    return ranges


def _stat_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bytes) and len(value) in (4, 8):
        try:
            if len(value) == 4:
                return float(struct.unpack("<i", value)[0])
            return float(struct.unpack("<d", value)[0])
        except struct.error:
            return None
    return None


def is_printable_utf8(raw: bytes) -> bool:
    """Accept only plausible identifiers/keys: printable, no NUL, short."""
    if not raw or len(raw) > 200:
        return False
    if b"\x00" in raw:
        return False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(ch.isprintable() or ch in " \t_-.:/" for ch in text)


def iter_length_prefixed_strings(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Yield (header_offset, length, raw) for every plausible length-prefixed string."""
    offset = 0
    limit = len(data)
    while offset < limit:
        # the length varint starts right after a 1-byte field header
        for header_size in (0, 1):
            start = offset + header_size
            if start >= limit:
                continue
            try:
                length, after = read_varint(data, start)
            except ParquetMetadataError:
                continue
            if length <= 0 or length > 200:
                continue
            end = after + length
            if end > limit:
                continue
            raw = data[after:end]
            if is_printable_utf8(raw):
                yield offset, length, raw
                break
        offset += 1


# ------------------------------------------------------------------- file read
def read_footer(path: str) -> bytes:
    """Return exactly the FileMetaData bytes (no leading padding).

    Layout: ``PAR1 <row groups> <FileMetaData thrift> <uint32 metadata_len> PAR1``
    so the metadata block is the last ``metadata_len`` bytes before the trailing
    8-byte trailer. Reading a wider window would hand the decoder padding bytes to
    misinterpret, which is how a decoder ends up reporting a plausible-looking but
    completely wrong schema.
    """
    size = os.path.getsize(path)
    if size < 12:
        raise ParquetMetadataError(f"file too small to be parquet: {size} bytes")
    with open(path, "rb") as fh:
        if fh.read(4) != MAGIC:
            raise ParquetMetadataError("missing leading PAR1 magic - not a parquet file")
        fh.seek(size - 8)
        tail = fh.read(8)
        if tail[4:] != MAGIC:
            raise ParquetMetadataError("missing trailing PAR1 magic - file is truncated")
        metadata_length = struct.unpack("<I", tail[:4])[0]
        if metadata_length <= 0 or metadata_length + 8 > size:
            raise ParquetMetadataError(f"implausible metadata length: {metadata_length}")
        fh.seek(size - 8 - metadata_length)
        return fh.read(metadata_length)


def extract_columns(footer: bytes) -> list[str]:
    """Column names via the real Thrift decode (kept for a stable API)."""
    try:
        return [c.name for c in parse_schema(decode_file_metadata(footer))]
    except ParquetMetadataError:
        return []


def read_parquet_metadata(path: str) -> ParquetMetadata:
    """Read schema, row count and row-group count with no third-party dependency."""
    meta = ParquetMetadata(path=path, size_bytes=os.path.getsize(path))
    try:
        footer = read_footer(path)
    except ParquetMetadataError as exc:
        meta.error = str(exc)
        return meta
    except OSError as exc:
        meta.error = f"cannot read file: {exc}"
        return meta

    try:
        file_meta = decode_file_metadata(footer)
    except ParquetMetadataError as exc:
        meta.error = f"could not decode FileMetaData: {exc}"
        return meta

    meta.num_rows = file_meta.get(FMD_NUM_ROWS)
    row_groups = file_meta.get(FMD_ROW_GROUPS)
    meta.num_row_groups = len(row_groups) if isinstance(row_groups, list) else None
    created = file_meta.get(FMD_CREATED_BY)
    meta.created_by = _as_text(created) if created is not None else ""

    meta.columns = parse_schema(file_meta)
    meta.stats = parse_statistics(file_meta)

    # self-check: a decoded schema must be self-consistent, otherwise say so
    if not meta.columns:
        meta.notes.append(
            "WARNING: decoded no columns - the footer may use an unsupported "
            "encoding; install pyarrow for an authoritative schema"
        )
    meta.notes.append(
        "schema decoded from the file footer without pyarrow (names/types only; "
        "install pyarrow to read row values)"
    )
    return meta


def summarize(path: str) -> dict[str, Any]:
    """Compact report used by the inspector."""
    meta = read_parquet_metadata(path)
    return meta.as_dict()

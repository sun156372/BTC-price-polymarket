"""Data audit / schema inspector for a large local dataset.

Purpose
-------
You cannot write a correct loader before you know the real column names. This
tool reports, per file:

* format, size, row count
* exact column names and dtypes
* a small sample of rows (never the whole file)
* for time-like columns: the detected epoch unit (s / ms / us / ns), min and max
  actually in the data

Design constraints (this is meant to run on a 1.88 GB folder on a laptop)
------------------------------------------------------------------------
* Parquet: reads only the **footer metadata** for row counts and schema, and at
  most one row group for the sample. It never loads a full file.
* CSV/JSONL: reads a bounded number of bytes for the sample, and counts rows in
  streaming chunks.
* Everything is capped by ``--max-files`` per directory, so a first run on a huge
  tree returns quickly.

Usage
-----
    python tools/inspect_dataset.py "D:\\polymarket_data"
    python tools/inspect_dataset.py "D:\\polymarket_data" --sample-rows 3 --out report.json
    python tools/inspect_dataset.py "D:\\polymarket_data" --max-files 5 --bytes-per-file 5mb

Output: a readable report on stdout plus a machine-readable JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------- helpers
TIME_HINTS = ("time", "ts", "date", "stamp", "epoch", "minute", "created", "updated")


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def parse_size(text: str) -> int:
    text = str(text).strip().lower()
    for suffix, factor in (("gb", 1024**3), ("mb", 1024**2), ("kb", 1024), ("b", 1)):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * factor)
            except ValueError:
                break
    return int(float(text))


def looks_like_time(name: str) -> bool:
    """Heuristic for time-like column names.

    Short names matter: exporters very often use ``t``, ``ts`` or ``T`` for the
    timestamp column, and a substring test for "time" misses all of them.
    """
    lowered = name.strip().lower()
    if lowered in {"t", "ts", "time", "timestamp", "datetime", "date", "epoch",
                   "open_time", "close_time", "start_ts", "end_ts", "t_ms", "t_s"}:
        return True
    return any(hint in lowered for hint in TIME_HINTS)


def detect_epoch_unit(values: Iterable[float]) -> tuple[str, float]:
    """Infer the epoch unit from magnitude. Returns (unit_name, divisor_to_seconds).

    Polymarket/Binance exports use seconds, milliseconds, microseconds or
    nanoseconds depending on the tool. Guessing wrong shifts your data by decades,
    so it is detected from the actual values rather than assumed.
    """
    candidates = [abs(float(v)) for v in values if v not in (None, "")]
    if not candidates:
        return "unknown", 1.0
    magnitude = max(candidates)
    if magnitude < 1e11:      # ~1973..5138 in seconds
        return "seconds", 1.0
    if magnitude < 1e14:      # milliseconds
        return "milliseconds", 1e3
    if magnitude < 1e17:      # microseconds
        return "microseconds", 1e6
    return "nanoseconds", 1e9


def to_epoch_seconds(value: Any, divisor: float) -> float | None:
    """Convert one timestamp cell to epoch seconds (float)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # ISO-8601 fallback
        if any(ch in text for ch in ("-", "T", ":")) and not text.replace(".", "").isdigit():
            from datetime import datetime, timezone

            try:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.timestamp()
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return None


def iso(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    from datetime import datetime, timezone

    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------- data model
@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_count: int | None = None
    epoch_unit: str | None = None
    min_value: Any = None
    max_value: Any = None
    min_iso: str = ""
    max_iso: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "epoch_unit": self.epoch_unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_utc": self.min_iso,
            "max_utc": self.max_iso,
        }


@dataclass
class FileReport:
    path: str
    rel_path: str
    format: str
    size_bytes: int
    row_count: int | None = None
    columns: list[ColumnInfo] = field(default_factory=list)
    sample: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "rel_path": self.rel_path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "size_human": human(self.size_bytes),
            "row_count": self.row_count,
            "columns": [c.as_dict() for c in self.columns],
            "sample": self.sample,
            "notes": self.notes,
            "error": self.error,
        }


# ------------------------------------------------------------- parquet (safe)
def inspect_parquet(path: str, rel: str, size: int, sample_rows: int) -> FileReport:
    report = FileReport(path=path, rel_path=rel, format="parquet", size_bytes=size)
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        report.error = "pyarrow not installed - run: pip install pyarrow"
        return report

    try:
        meta = pq.ParquetFile(path).metadata
        report.row_count = int(meta.num_rows)
        report.notes.append(f"{meta.num_row_groups} row group(s), created by {meta.created_by or 'unknown'}")
    except Exception as exc:  # noqa: BLE001
        report.error = f"cannot read parquet metadata: {type(exc).__name__}: {exc}"
        return report

    try:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        dtypes = [str(f.type) for f in pf.schema_arrow]
        for name, dtype in zip(names, dtypes):
            report.columns.append(ColumnInfo(name=name, dtype=dtype))

        if sample_rows > 0:
            # read at most the first row group, then head() it
            batch = next(pf.iter_batches(batch_size=min(sample_rows, 500)), None)
            if batch is not None:
                frame = batch.to_pylist()[:sample_rows]
                report.sample = [{k: _jsonable(v) for k, v in row.items()} for row in frame]
                _annotate_time_columns(report, frame)
    except Exception as exc:  # noqa: BLE001
        report.error = f"cannot read parquet schema/sample: {type(exc).__name__}: {exc}"
    return report


def _annotate_time_columns(report: FileReport, sample: list[dict]) -> None:
    """Detect epoch units from a sample and note them on the column list."""
    for column in report.columns:
        if not looks_like_time(column.name):
            continue
        values = [row.get(column.name) for row in sample]
        unit, divisor = detect_epoch_unit(values)
        column.epoch_unit = unit
        seconds = [s for s in (to_epoch_seconds(v, divisor) for v in values) if s is not None]
        if seconds:
            column.min_value = min(jsonable_number(v) for v in values if v is not None)
            column.max_value = max(jsonable_number(v) for v in values if v is not None)
            column.min_iso = iso(min(seconds))
            column.max_iso = iso(max(seconds))
            report.notes.append(
                f"column {column.name!r} looks like time -> unit={unit} "
                f"(divide by {divisor:g} for seconds)"
            )


def jsonable_number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value[:10]]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in list(value.items())[:10]}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# ----------------------------------------------------- csv / jsonl (streaming)
def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    counts = {d: head.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def inspect_csv(path: str, rel: str, size: int, sample_rows: int, bytes_cap: int) -> FileReport:
    import csv

    report = FileReport(path=path, rel_path=rel, format="csv", size_bytes=size)
    delimiter = sniff_delimiter(path)
    report.notes.append(f"delimiter={delimiter!r}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            names = reader.fieldnames or []
            report.columns = [ColumnInfo(name=n, dtype="str->?") for n in names]
            for index, row in enumerate(reader):
                if index >= sample_rows:
                    break
                report.sample.append({k: _jsonable(v) for k, v in row.items()})
        _annotate_time_columns(report, report.sample)
    except Exception as exc:  # noqa: BLE001
        report.error = f"cannot read csv: {type(exc).__name__}: {exc}"

    # streamed row count over the first `bytes_cap` bytes (bounded)
    try:
        read_bytes = 0
        lines = 0
        with open(path, "rb") as fh:
            while True:
                remaining = bytes_cap - read_bytes
                if remaining <= 0:
                    break
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                read_bytes += len(chunk)
                lines += chunk.count(b"\n")
        if read_bytes >= size:
            report.row_count = max(0, lines - 1)
            report.notes.append("row count is exact (file fully scanned)")
        else:
            estimate = int((lines - 1) / max(1, read_bytes) * size)
            report.row_count = estimate
            report.notes.append(
                f"row count ESTIMATED from first {human(read_bytes)} of {human(size)} "
                f"(scan capped by --bytes-per-file; use pandas chunksize for an exact count)"
            )
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"row count failed: {exc}")
    return report


def inspect_jsonl(path: str, rel: str, size: int, sample_rows: int, bytes_cap: int) -> FileReport:
    report = FileReport(path=path, rel_path=rel, format="jsonl", size_bytes=size)
    try:
        keys_seen: Counter = Counter()
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    keys_seen.update(obj.keys())
                if len(report.sample) < sample_rows:
                    report.sample.append(_jsonable(obj))
                if len(keys_seen) and len(report.sample) >= sample_rows and sum(keys_seen.values()) > sample_rows * 20:
                    break
        report.columns = [
            ColumnInfo(name=key, dtype="json") for key, _ in keys_seen.most_common()
        ]
        _annotate_time_columns(report, report.sample)
    except Exception as exc:  # noqa: BLE001
        report.error = f"cannot read jsonl: {type(exc).__name__}: {exc}"

    try:
        read_bytes = 0
        lines = 0
        with open(path, "rb") as fh:
            while True:
                remaining = bytes_cap - read_bytes
                if remaining <= 0:
                    break
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                read_bytes += len(chunk)
                lines += chunk.count(b"\n")
        if read_bytes >= size:
            report.row_count = lines
            report.notes.append("row count is exact (file fully scanned)")
        else:
            report.row_count = int(lines / max(1, read_bytes) * size)
            report.notes.append(
                f"row count ESTIMATED from first {human(read_bytes)} of {human(size)} "
                f"(scan capped by --bytes-per-file)"
            )
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"row count failed: {exc}")
    return report


# ------------------------------------------------------------------ discovery
SUPPORTED = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def discover(root: str, max_files_per_dir: int) -> list[tuple[str, str, int, str]]:
    """Return (abs_path, rel_path, size, format) found under root."""
    found: list[tuple[str, str, int, str]] = []
    per_dir: Counter = Counter()
    for current, dirs, files in os.walk(root):
        dirs.sort()
        rel_dir = os.path.relpath(current, root)
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            fmt = SUPPORTED.get(ext)
            if not fmt:
                continue
            if max_files_per_dir and per_dir[rel_dir] >= max_files_per_dir:
                continue
            path = os.path.join(current, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            rel = os.path.relpath(path, root)
            found.append((path, rel, size, fmt))
            per_dir[rel_dir] += 1
    return found


def directory_summary(root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        total = 0
        count = 0
        extensions: Counter = Counter()
        for current, _dirs, files in os.walk(path):
            for file_name in files:
                try:
                    total += os.path.getsize(os.path.join(current, file_name))
                except OSError:
                    continue
                count += 1
                extensions[os.path.splitext(file_name)[1].lower()] += 1
        rows.append(
            {
                "folder": name,
                "files": count,
                "size_human": human(total),
                "size_bytes": total,
                "extensions": dict(extensions.most_common(6)),
            }
        )
    return rows


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Inspect a large local dataset safely (metadata + small samples only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", help="dataset root folder")
    ap.add_argument("--sample-rows", type=int, default=3, help="rows to print per file")
    ap.add_argument("--max-files", type=int, default=4, help="max files per folder (0 = all)")
    ap.add_argument("--bytes-per-file", default="20mb", help="cap for csv/jsonl scanning")
    ap.add_argument("--out", default="", help="write the full JSON report here")
    ap.add_argument("--only", default="", help="comma-separated substrings to restrict paths")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: not a folder: {root}")
        print()
        print("If your data lives on another drive (e.g. D:\\), this process cannot see it.")
        print("Options: copy the folder to a path this session can read, or run this")
        print("script yourself in a terminal on the machine that has the drive.")
        return 2

    bytes_cap = parse_size(args.bytes_per_file)
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    print("=" * 100)
    print(f"DATASET AUDIT: {root}")
    print("=" * 100)

    folders = directory_summary(root)
    total_bytes = sum(f["size_bytes"] for f in folders)
    print(f"\nSubfolders: {len(folders)}   total size: {human(total_bytes)}")
    print(f"\n{'folder':<28}{'files':>7}{'size':>12}   extensions")
    print("-" * 100)
    for row in folders:
        print(f"{row['folder']:<28}{row['files']:>7}{row['size_human']:>12}   {row['extensions']}")

    targets = discover(root, args.max_files)
    if only:
        targets = [t for t in targets if any(key in t[1] for key in only)]
    print(f"\nFiles to inspect: {len(targets)} (capped at {args.max_files} per folder)")

    reports: list[FileReport] = []
    for path, rel, size, fmt in targets:
        print()
        print("=" * 100)
        print(f"FILE: {rel}")
        print(f"  format={fmt}  size={human(size)}")
        if fmt == "parquet":
            report = inspect_parquet(path, rel, size, args.sample_rows)
        elif fmt == "csv":
            report = inspect_csv(path, rel, size, args.sample_rows, bytes_cap)
        else:
            report = inspect_jsonl(path, rel, size, args.sample_rows, bytes_cap)
        reports.append(report)

        if report.error:
            print(f"  ERROR: {report.error}")
        print(f"  rows: {report.row_count if report.row_count is not None else 'unknown'}")
        for note in report.notes:
            print(f"  note: {note}")
        print(f"  columns ({len(report.columns)}):")
        for column in report.columns:
            extra = ""
            if column.epoch_unit:
                extra = f"   [time unit={column.epoch_unit}]"
            print(f"    - {column.name:<32} {column.dtype}{extra}")
        if report.sample:
            print(f"  sample rows ({len(report.sample)}):")
            for row in report.sample:
                rendered = json.dumps(row, ensure_ascii=False, default=str)
                print(f"    {rendered[:300]}")

    if args.out:
        payload = {
            "root": root,
            "total_size_bytes": total_bytes,
            "folders": folders,
            "files": [r.as_dict() for r in reports],
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        print()
        print(f"full JSON report written: {os.path.abspath(args.out)}")

    print()
    print("=" * 100)
    print("Next step: send me this report (or just the column lists + sample rows).")
    print("With the real names/dtypes I can write the loader and backtest against your")
    print("actual schema instead of guessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

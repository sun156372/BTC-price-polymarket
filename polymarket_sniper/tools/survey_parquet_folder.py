"""Survey every parquet file in a folder using the pure-Python footer reader.

Groups files by schema fingerprint so you can see at a glance which subfolders
share a shape, total rows, time span (from statistics) and row-group counts -
without pyarrow and without reading any column data.

    python tools/survey_parquet_folder.py "..\\data"
    python tools/survey_parquet_folder.py "D:\\polymarket_data\\orderbooks"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from tools import parquet_meta_pure as pq  # noqa: E402


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="folder containing .parquet files")
    ap.add_argument("--recursive", action="store_true", help="walk subfolders too")
    ap.add_argument("--limit", type=int, default=0, help="only inspect the first N files")
    ap.add_argument("--show-columns", action="store_true", help="print the full column list")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"ERROR: not a folder: {root}")
        return 2

    pattern = "**/*.parquet" if args.recursive else "*.parquet"
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=args.recursive))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no .parquet files found in {root}")
        return 1

    print("=" * 96)
    print(f"PARQUET SURVEY: {root}")
    print(f"files: {len(files)}")
    print("=" * 96)

    groups: dict[tuple, list] = defaultdict(list)
    failures: list[tuple[str, str]] = []

    for path in files:
        meta = pq.read_parquet_metadata(path)
        rel = os.path.relpath(path, root)
        if meta.error:
            failures.append((rel, meta.error))
            continue
        fingerprint = tuple((c.name, c.kind) for c in meta.columns)
        groups[fingerprint].append((rel, meta))

    for fingerprint, entries in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        names = [name for name, _ in fingerprint]
        total_rows = sum(m.num_rows or 0 for _rel, m in entries)
        total_bytes = sum(m.size_bytes for _rel, m in entries)
        print()
        print("-" * 96)
        print(f"SCHEMA GROUP: {len(entries)} file(s)   rows: {total_rows:,}   size: {human(total_bytes)}")
        print(f"  columns ({len(names)}): {', '.join(names)}")
        if args.show_columns:
            for name, kind in fingerprint:
                print(f"    - {name:<30} {kind}")
        print()
        print(f"  {'file':<44}{'rows':>14}{'rgs':>6}{'size':>10}  created_by")
        for rel, meta in entries[:15]:
            print(
                f"  {rel:<44}{(meta.num_rows or 0):>14,}"
                f"{(meta.num_row_groups or 0):>6}{human(meta.size_bytes):>10}  "
                f"{(meta.created_by or '')[:28]}"
            )
        if len(entries) > 15:
            print(f"  ... and {len(entries) - 15} more file(s) with the same schema")

    if failures:
        print()
        print("-" * 96)
        print(f"FAILED TO READ: {len(failures)} file(s)")
        for rel, error in failures[:10]:
            print(f"  {rel}: {error}")

    grand_rows = sum(
        m.num_rows or 0 for entries in groups.values() for _rel, m in entries
    )
    print()
    print("=" * 96)
    print(f"TOTAL rows across all readable files: {grand_rows:,}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

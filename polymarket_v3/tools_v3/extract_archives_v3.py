"""Extract the market_parquet_*.tar.gz archives into a per-day folder tree.

    python tools_v3/extract_archives_v3.py                # all archives found
    python tools_v3/extract_archives_v3.py --check        # report only, no writes

Archives are Hive-partitioned::

    dataset=<name>/date=YYYY-MM-DD/part-*.parquet

and are unpacked to ``<workspace>/micro_v3/<archive-stem>/dataset=<name>/date=...``
so a loader can glob ``dataset=polymarket_features_100ms/date=*/*.parquet`` without
caring which archive a day came from.

Re-running is safe: files already extracted at the same size are skipped.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(ROOT, "micro_v3")


def archives() -> list[str]:
    return sorted(glob.glob(os.path.join(ROOT, "market_parquet_*.tar.gz")))


def extract_one(path: str, *, apply: bool, dest: str = DEST,
                verbose: bool = True) -> dict:
    stem = os.path.basename(path).replace(".tar.gz", "")
    target_root = os.path.join(dest, stem)
    counts: dict[str, int] = {}
    skipped = 0
    written = 0
    total = 0

    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            total += 1
            dataset = member.name.split("/")[0]
            counts[dataset] = counts.get(dataset, 0) + 1
            if not apply:
                continue
            destination = os.path.join(target_root, member.name)
            if os.path.exists(destination) and os.path.getsize(destination) == member.size:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(destination, "wb") as handle:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            written += 1
    if verbose:
        print(f"  {stem}: {total} files ({written} written, {skipped} already present)")
        for dataset, count in sorted(counts.items()):
            print(f"      {dataset:<34} {count:>5} files")
    return {"stem": stem, "total": total, "written": written,
            "skipped": skipped, "datasets": counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--dest", default=DEST)
    args = parser.parse_args(argv)

    dest = args.dest
    found = archives()
    if not found:
        print(f"no market_parquet_*.tar.gz found in {ROOT}", file=sys.stderr)
        return 2

    print("=" * 90)
    print("EXTRACT HIGH-FREQUENCY ARCHIVES")
    print("=" * 90)
    print(f"source : {ROOT}")
    print(f"dest   : {dest}")
    print(f"mode   : {'CHECK ONLY' if args.check else 'EXTRACT'}")
    print(f"archives: {len(found)}")
    print()

    grand = 0
    for path in found:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"=== {os.path.basename(path)} ({size_mb:.0f} MB) ===")
        result = extract_one(path, apply=not args.check, dest=dest)
        grand += result["total"]
    print()
    print(f"total members: {grand}")
    if not args.check:
        print(f"extracted to: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a fast-loading cache from the raw 100ms feature files.

    python tools_v3/build_cache_v3.py                 # build (skips if current)
    python tools_v3/build_cache_v3.py --force

.. warning::

   **This tool is not usable as written and is kept only as a starting point.**
   It keys the cache on ``window_key``, which is corrupted in this dataset: it
   holds the literal string ``"et"`` (scraped from a slug suffix such as
   ``8pm-et``) rather than an epoch. The working path is ``engine_v3/hf.py``,
   which streams the per-market files keyed on ``market_id`` at roughly
   2 markets/second - about 3 minutes for all 1,148 markets, after which a single
   sweep covers the whole sample. That is slow enough to be annoying and fast
   enough not to matter, so the cache was never rebuilt.

Why this exists
---------------
Reading the raw ``polymarket_features_100ms`` layout takes ~13 minutes: every row
carries six JSON-encoded depth arrays (``up_side_bids``, ``up_bid_sizes``, ...)
which pyarrow hands back as strings that have to be ``json.loads``-ed per row.

That cost is paid once here. The cache flattens the depth arrays into fixed slots
(``up_bid_p0..p2``, ``up_bid_s0..s2``) and stores plain numerics, so later loads
are a straight columnar read with no per-row parsing.

Only the columns the strategy actually consumes are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3 import hf

#: how many depth levels to keep per side (top of book is what a small order eats)
DEPTH = 3

SCHEMA = pa.schema([
    ("window_key", pa.string()),
    ("ts", pa.float64()),
    ("slug", pa.string()),
    ("market_id", pa.string()),
    ("lifecycle_state", pa.string()),
    ("up_token", pa.string()),
    ("down_token", pa.string()),
    ("time_to_start_ms", pa.float64()),
    ("time_to_deadline_ms", pa.float64()),
    ("up_best_bid", pa.float32()),
    ("up_best_ask", pa.float32()),
    ("up_mid", pa.float32()),
    ("up_spread_bps", pa.float32()),
    ("down_best_bid", pa.float32()),
    ("down_best_ask", pa.float32()),
    ("down_mid", pa.float32()),
    ("down_spread_bps", pa.float32()),
    ("sum_best_ask", pa.float32()),
    ("sum_mid", pa.float32()),
    ("ask_notional", pa.float32()),
    ("bid_notional", pa.float32()),
    ("halt_flag", pa.bool_()),
] + [
    (f"{side}_{book}_p{i}", pa.float32())
    for side in ("up", "down") for book in ("bid", "ask") for i in range(DEPTH)
] + [
    (f"{side}_{book}_s{i}", pa.float32())
    for side in ("up", "down") for book in ("bid", "ask") for i in range(DEPTH)
])


def _levels(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value if v is not None]
    text = str(value).strip()
    if not text or text == "None":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [float(v) for v in parsed if v is not None] if isinstance(parsed, list) else []


def build(src_root: str, cache_path: str, *, force: bool = False) -> int:
    files = hf.day_files(src_root, hf.FEATURES)
    if not files:
        print(f"no feature files under {src_root}", file=sys.stderr)
        return 2

    newest_src = max(os.path.getmtime(f) for f in files)
    if not force and os.path.exists(cache_path) and os.path.getmtime(cache_path) > newest_src:
        print(f"cache is current: {cache_path}")
        return 0

    print("=" * 90)
    print("BUILD 100ms FEATURE CACHE")
    print("=" * 90)
    print(f"source files : {len(files)}")
    print(f"cache        : {cache_path}")
    print(f"depth kept   : {DEPTH} levels per side")
    print()

    writer = pq.ParquetWriter(cache_path, SCHEMA, compression="zstd")
    total = 0
    started = time.time()
    batch_rows: list[dict] = []
    BATCH = 50_000

    def flush() -> None:
        nonlocal total, batch_rows
        if not batch_rows:
            return
        table = pa.Table.from_pylist(batch_rows, schema=SCHEMA)
        writer.write_table(table)
        total += len(batch_rows)
        batch_rows = []

    for index, path in enumerate(files, start=1):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        cols = [c for c in (
            "timestamp_ms", "window_key", "slug", "market_id", "lifecycle_state",
            "up_token_id", "down_token_id", "time_to_start_ms", "time_to_deadline_ms",
            "up_best_bid", "up_best_ask", "up_midquote", "up_spread_bps",
            "down_best_bid", "down_best_ask", "down_midquote", "down_spread_bps",
            "sum_best_ask", "sum_midquote", "full_depth_ask_notional",
            "full_depth_bid_notional", "observed_halt_flag",
            "up_side_bids", "up_bid_sizes", "up_side_asks", "up_ask_sizes",
            "down_side_bids", "down_bid_sizes", "down_side_asks", "down_ask_sizes",
        ) if c in names]
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
            for row in batch.to_pylist():
                key = hf.normalize_key(row.get("window_key"))
                if not key:
                    continue
                record: dict = {
                    "window_key": key,
                    "ts": (hf._f(row.get("timestamp_ms"), 0.0) or 0.0) / 1000.0,
                    "slug": str(row.get("slug") or ""),
                    "market_id": str(row.get("market_id") or ""),
                    "lifecycle_state": str(row.get("lifecycle_state") or ""),
                    "up_token": str(row.get("up_token_id") or ""),
                    "down_token": str(row.get("down_token_id") or ""),
                    "time_to_start_ms": hf._f(row.get("time_to_start_ms"), 0.0),
                    "time_to_deadline_ms": hf._f(row.get("time_to_deadline_ms"), 0.0),
                    "up_best_bid": hf._f(row.get("up_best_bid")),
                    "up_best_ask": hf._f(row.get("up_best_ask")),
                    "up_mid": hf._f(row.get("up_midquote")),
                    "up_spread_bps": hf._f(row.get("up_spread_bps")),
                    "down_best_bid": hf._f(row.get("down_best_bid")),
                    "down_best_ask": hf._f(row.get("down_best_ask")),
                    "down_mid": hf._f(row.get("down_midquote")),
                    "down_spread_bps": hf._f(row.get("down_spread_bps")),
                    "sum_best_ask": hf._f(row.get("sum_best_ask")),
                    "sum_mid": hf._f(row.get("sum_midquote")),
                    "ask_notional": hf._f(row.get("full_depth_ask_notional")),
                    "bid_notional": hf._f(row.get("full_depth_bid_notional")),
                    "halt_flag": str(row.get("observed_halt_flag")).lower() == "true",
                }
                for side, bid_col, bid_sz, ask_col, ask_sz in (
                    ("up", "up_side_bids", "up_bid_sizes", "up_side_asks", "up_ask_sizes"),
                    ("down", "down_side_bids", "down_bid_sizes", "down_side_asks", "down_ask_sizes"),
                ):
                    bids = _levels(row.get(bid_col))
                    bsizes = _levels(row.get(bid_sz))
                    asks = _levels(row.get(ask_col))
                    asizes = _levels(row.get(ask_sz))
                    for i in range(DEPTH):
                        record[f"{side}_bid_p{i}"] = bids[i] if i < len(bids) else None
                        record[f"{side}_bid_s{i}"] = bsizes[i] if i < len(bsizes) else None
                        record[f"{side}_ask_p{i}"] = asks[i] if i < len(asks) else None
                        record[f"{side}_ask_s{i}"] = asizes[i] if i < len(asizes) else None
                batch_rows.append(record)
                if len(batch_rows) >= BATCH:
                    flush()
        if index % 100 == 0 or index == len(files):
            rate = total / max(1e-9, time.time() - started)
            print(f"  {index:>4}/{len(files)} files  {total:>10,} rows  "
                  f"{rate:>8.0f} rows/s  {time.time()-started:>6.0f}s")
    flush()
    writer.close()

    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print()
    print(f"rows written : {total:,}")
    print(f"cache size   : {size_mb:.0f} MB")
    print(f"elapsed      : {time.time()-started:.0f}s")
    return 0


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(root, "micro_v3"))
    parser.add_argument("--out", default=os.path.join(root, "micro_v3", "_cache_features.parquet"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return build(args.src, args.out, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())

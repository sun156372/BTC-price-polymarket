"""Build a SLIM feature cache so experiments stop re-parsing 850 MB of JSON.

    python tools_v3/build_slim_cache_v4.py            # build (skips if current)
    python tools_v3/build_slim_cache_v4.py --force

WHY THIS EXISTS
---------------
Every backtest so far streamed the raw ``polymarket_features_100ms`` files. Those
files carry per-tick JSON-ish depth columns that no experiment actually uses, and
each full load costs 30-50 minutes of wall clock. That cost is paid again on every
new question, which is why each round of this project took an hour.

This tool reads each market once and writes a compact parquet holding only:

* ``to_deadline_ms``      - the authoritative countdown to settlement
* ``ts``                  - wall clock, for spot alignment
* ``up_bid/up_ask/up_mid`` and the DOWN equivalents

It also writes the resolution table and a per-market index, and stores the ticks in
CHRONOLOGICAL order so downstream code cannot reintroduce the reverse-tape bug.

The output is ~100x smaller and loads in seconds. It is a cache: delete it and the
tools fall back to the raw files.

Deliberately NOT cached: anything derived from spot, because the spot feed is small
(one file per day) and joining it here would bake an assumption into the cache.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3 import hf  # noqa: E402

CACHE_NAME = "slim_ticks.parquet"
INDEX_NAME = "slim_index.parquet"
RESOLUTIONS_NAME = "slim_resolutions.parquet"

TICK_SCHEMA = pa.schema([
    ("market_id", pa.string()),
    ("day", pa.string()),
    ("slug", pa.string()),
    ("to_deadline_ms", pa.float64()),
    ("ts", pa.float64()),
    ("up_bid", pa.float64()),
    ("up_ask", pa.float64()),
    ("up_mid", pa.float64()),
    ("down_bid", pa.float64()),
    ("down_ask", pa.float64()),
    ("down_mid", pa.float64()),
])


def cache_dir(root: str) -> str:
    return os.path.join(root, "_cache")


def build(root: str, force: bool = False, progress_every: int = 1000) -> int:
    """Read only the six price columns per file and write the cache.

    This deliberately does NOT go through :func:`hf.iter_markets`. That loader
    materialises a ``Tick`` per row carrying every depth array (book levels and
    sizes) plus a sorted deadline index per market, which measures at ~3.4
    markets/s on 15,316 files. Selecting six columns straight out of parquet and
    emitting them skips all of it, and the point of a cache is to pay the parsing
    cost once rather than once per experiment.
    """
    out_dir = cache_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    ticks_path = os.path.join(out_dir, CACHE_NAME)
    index_path = os.path.join(out_dir, INDEX_NAME)
    res_path = os.path.join(out_dir, RESOLUTIONS_NAME)
    stamp_path = os.path.join(out_dir, "stamp.json")

    files = hf.day_files(root, hf.FEATURES)
    if not force and os.path.exists(stamp_path) and os.path.exists(ticks_path):
        try:
            old = json.load(io.open(stamp_path, encoding="utf-8"))
            if old.get("market_files") == len(files):
                print(f"cache is current ({len(files)} market files) - nothing to do")
                return 0
        except (ValueError, OSError):
            pass

    wanted = ["market_id", "slug", "timestamp_ms", "time_to_deadline_ms",
              "up_best_bid", "up_best_ask", "up_midquote",
              "down_best_bid", "down_best_ask", "down_midquote"]

    print(f"building slim cache from {len(files):,} market files (6 columns only)")
    print(f"  output: {ticks_path}")
    resolutions = hf.load_resolutions(root)
    writer = pq.ParquetWriter(ticks_path, TICK_SCHEMA, compression="zstd")
    index_rows: list[dict] = []
    total_ticks = 0
    skipped = 0
    started = time.time()

    for n, path in enumerate(files, start=1):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        cols = [c for c in wanted if c in names]
        if "market_id" not in cols or "time_to_deadline_ms" not in cols:
            skipped += 1
            continue
        table = pq.read_table(path, columns=cols)
        if table.num_rows == 0:
            skipped += 1
            continue
        data = table.to_pydict()
        market_id = str(data["market_id"][0] or "")
        slug = str((data.get("slug") or [""])[0] or "")
        horizon = hf.horizon_of(slug)
        deadlines = data["time_to_deadline_ms"]
        # resolution drives up_won and start_ts
        resolution = resolutions.get(market_id)
        up_won = None
        if resolution is not None:
            up_won = resolution.outcome_direction.upper() == "UP"

        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]
        # chronological order = descending countdown; the raw files are stored the
        # other way round, so sort once here and never worry about it downstream
        order = sorted(range(len(deadlines)),
                       key=lambda i: -(deadlines[i] if deadlines[i] is not None else -1e18))
        ordered = {k: [v[i] for i in order] for k, v in data.items()}

        def col(name):
            vals = ordered.get(name)
            return vals if vals else [None] * len(order)

        up_bid, up_ask, up_mid = col("up_best_bid"), col("up_best_ask"), col("up_midquote")
        dn_bid = col("down_best_bid")
        dn_ask, dn_mid = col("down_best_ask"), col("down_midquote")
        ts_raw = col("timestamp_ms")
        ts = [None if v is None else float(v) / 1000.0 for v in ts_raw]

        nrows = len(order)
        out = {
            "market_id": [market_id] * nrows,
            "day": [day] * nrows,
            "slug": [slug] * nrows,
            "to_deadline_ms": [None if v is None else float(v) for v in
                               ordered["time_to_deadline_ms"]],
            "ts": ts,
            "up_bid": [None if v is None else float(v) for v in up_bid],
            "up_ask": [None if v is None else float(v) for v in up_ask],
            "up_mid": [None if v is None else float(v) for v in up_mid],
            "down_bid": [None if v is None else float(v) for v in dn_bid],
            "down_ask": [None if v is None else float(v) for v in dn_ask],
            "down_mid": [None if v is None else float(v) for v in dn_mid],
        }
        writer.write_table(pa.table(out, schema=TICK_SCHEMA))
        total_ticks += nrows
        # start_ts uses the loader's own definition (the epoch in the slug) so the
        # cache cannot drift from hf.iter_markets
        start_ts = hf.start_of(slug)
        index_rows.append({
            "market_id": market_id, "day": day, "slug": slug, "horizon": horizon,
            "ticks": nrows, "start_ts": start_ts or 0.0,
            "up_won": None if up_won is None else int(up_won),
        })
        if n % progress_every == 0:
            rate = n / max(1e-9, time.time() - started)
            print(f"  {n:>6,}/{len(files):,} markets  {total_ticks:>11,} ticks  "
                  f"{rate:.1f} markets/s  {time.time()-started:.0f}s")

    writer.close()
    pq.write_table(pa.Table.from_pylist(index_rows), index_path, compression="zstd")

    res_rows = [{
        "market_id": r.market_id, "slug": r.slug, "horizon": r.horizon,
        "outcome_direction": r.outcome_direction, "up_token": r.up_token,
        "down_token": r.down_token, "winner_token": r.winner_token,
        "open_price": r.open_price, "close_price": r.close_price,
        "usable": int(r.usable),
    } for r in resolutions.values()]
    pq.write_table(pa.Table.from_pylist(res_rows), res_path, compression="zstd")

    stamp = {"market_files": len(files), "markets": len(index_rows),
             "ticks": total_ticks, "skipped": skipped,
             "built_utc": datetime.now(timezone.utc).isoformat()}
    io.open(stamp_path, "w", encoding="utf-8").write(json.dumps(stamp, indent=2))

    size_mb = os.path.getsize(ticks_path) / 1e6
    print()
    print(f"  markets {len(index_rows):,}   ticks {total_ticks:,}   "
          f"skipped {skipped}   size {size_mb:.1f} MB   "
          f"in {time.time()-started:.0f}s")
    print(f"  cache: {ticks_path}")
    return 0


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return build(args.root, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())

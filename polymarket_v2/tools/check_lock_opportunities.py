"""Answer one question from real data: does UP+DOWN ever get cheap enough to lock?

The lock condition is ``ask_up + ask_down < ~0.965`` (fees). This streams one day
of order books, pairs the UP and DOWN token per market, and reports the actual
distribution of the pair sum in the final minutes of each window - the only
region the sniper trades in.

    python tools/check_lock_opportunities.py --file "..\\data\\2026-03-02.parquet"
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)


def to_epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ladder_touch(value) -> tuple[float | None, float | None, float]:
    """(best_bid, best_ask, size_at_best_ask) from a BSON/JSON ladder cell."""
    if value in (None, ""):
        return None, None, 0.0
    data = value
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(bytes(data).decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None, None, 0.0
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None, None, 0.0
    if not isinstance(data, (list, tuple)):
        return None, None, 0.0
    best_bid = best_ask = None
    ask_size = 0.0
    for item in data:
        if isinstance(item, dict):
            price = item.get("price")
            size = item.get("size", 0.0)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price, size = item[0], item[1]
        else:
            continue
        try:
            price = float(price)
            size = float(size)
        except (TypeError, ValueError):
            continue
        if not 0.0 < price < 1.0:
            continue
        if best_bid is None or price > best_bid:
            best_bid = price
        if best_ask is None or price < best_ask:
            best_ask = price
            ask_size = size
    return best_bid, best_ask, ask_size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, help="one orderbooks parquet file")
    ap.add_argument("--last-seconds", type=float, default=180.0,
                    help="only consider snapshots this close to the window close")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    args = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("needs pyarrow: pip install pyarrow")
        return 2

    path = os.path.abspath(args.file)
    if not os.path.exists(path):
        print(f"file not found: {path}")
        return 2

    pf = pq.ParquetFile(path)
    print(f"reading {os.path.basename(path)}  ({pf.metadata.num_rows:,} rows)")
    print(f"restricting to the last {args.last_seconds:.0f}s of each window\n")

    # market -> token -> list[(ts, ask, ask_size)]
    books: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    read = 0
    for batch in pf.iter_batches(
        batch_size=50_000,
        columns=["timestamp", "market_id", "token_id", "ask_levels"],
    ):
        for row in batch.to_pylist():
            read += 1
            ts = to_epoch(row.get("timestamp"))
            if ts is None:
                continue
            _bid, ask, ask_size = ladder_touch(row.get("ask_levels"))
            if ask is None:
                continue
            market = str(row.get("market_id") or "")
            token = str(row.get("token_id") or "")
            books[market][token].append((ts, ask, ask_size))
        if args.max_rows and read >= args.max_rows:
            break

    print(f"rows read            : {read:,}")
    print(f"markets with books   : {len(books):,}")

    def window_start(market: str) -> int | None:
        try:
            return int(market.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return None

    sums: list[float] = []
    best_sums: list[float] = []
    per_market_best: list[tuple[str, float, float]] = []
    paired_markets = 0

    for market, tokens in books.items():
        if len(tokens) < 2:
            continue
        start = window_start(market)
        if start is None:
            continue
        close = start + 900  # this dataset is 15-minute windows

        # Build a per-server-second series of the best ask on each token, then pair
        # on the *same second*. Matching exact float timestamps instead would throw
        # away most rows (the two sides are captured a few hundred ms apart), and
        # summing the two sides' global minima would double-count the same moment.
        per_token: dict[str, dict[int, tuple[float, float]]] = {}
        for token, entries in tokens.items():
            series: dict[int, tuple[float, float]] = {}
            for ts, ask, size in entries:
                t_minus = close - ts
                if not 0 <= t_minus <= args.last_seconds:
                    continue
                second = int(ts)
                current = series.get(second)
                # keep the best (lowest) ask seen in that second
                if current is None or ask < current[0]:
                    series[second] = (ask, size)
            if series:
                per_token[token] = series

        if len(per_token) < 2:
            continue

        token_list = list(per_token)
        shared = set(per_token[token_list[0]])
        for token in token_list[1:]:
            shared &= set(per_token[token])
        if not shared:
            continue

        market_sums = []
        for second in shared:
            total = sum(per_token[token][second][0] for token in token_list)
            market_sums.append(total)
            sums.append(total)
        if market_sums:
            paired_markets += 1
            best = min(market_sums)
            best_sums.append(best)
            per_market_best.append((market, best, len(market_sums)))

    print(f"markets paired       : {paired_markets:,}")
    print(f"paired checkpoints   : {len(sums):,}")
    if not sums:
        print("\nno paired checkpoints found - widen --last-seconds")
        return 1

    ordered = sorted(sums)
    best_ordered = sorted(best_sums)

    def q(values: list[float], p: float) -> float:
        return values[min(len(values) - 1, int(p * len(values)))]

    print("\nUP+DOWN ask sum at paired checkpoints:")
    print(f"  min    {ordered[0]:.4f}")
    print(f"  p1     {q(ordered, 0.01):.4f}")
    print(f"  p5     {q(ordered, 0.05):.4f}")
    print(f"  p25    {q(ordered, 0.25):.4f}")
    print(f"  median {q(ordered, 0.50):.4f}")
    print(f"  p75    {q(ordered, 0.75):.4f}")
    print(f"  max    {ordered[-1]:.4f}")

    print("\nper-market BEST (lowest) sum reached inside the last"
          f" {args.last_seconds:.0f}s:")
    print(f"  min    {best_ordered[0]:.4f}")
    print(f"  p5     {q(best_ordered, 0.05):.4f}")
    print(f"  median {q(best_ordered, 0.50):.4f}")

    print("\nlock feasibility (break-even sum is ~0.9650-0.9687 after fees):")
    for threshold in (0.90, 0.95, 0.965, 0.98, 1.00):
        checkpoints = sum(1 for value in sums if value <= threshold)
        markets_hit = sum(1 for _m, best, _n in per_market_best if best <= threshold)
        print(f"  sum <= {threshold:.3f}: {checkpoints:>8,} checkpoints "
              f"({checkpoints / len(sums) * 100:5.2f}%)   "
              f"{markets_hit:>5,} markets "
              f"({markets_hit / paired_markets * 100:5.2f}% of paired)")

    print("\nlowest-sum markets (the best candidates that exist):")
    for market, best, count in sorted(per_market_best, key=lambda x: x[1])[:10]:
        print(f"  {market:<34} best_sum={best:.4f}  ({count} paired checkpoints)")

    lockable = sum(1 for _m, best, _n in per_market_best if best <= 0.965)
    print()
    print("=" * 78)
    if lockable == 0:
        print("VERDICT: not a single market in this file reached a lockable pair sum")
        print("         inside the final window. The 'buy both sides' arbitrage did")
        print("         not exist here - so tuning movePct/windowMin/sumTarget cannot")
        print("         create it. This is a strategy-viability result, not a tuning one.")
    else:
        print(f"VERDICT: {lockable} of {paired_markets} markets ({lockable / paired_markets * 100:.2f}%)")
        print("         reached a pair sum at or below the ~0.965 break-even, i.e. a")
        print("         genuine (if rare) lock existed. Check opportunities.csv for the")
        print("         per-market depth, because a thin lock is not fillable.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

DEFAULT_COEFFICIENT = 0.07

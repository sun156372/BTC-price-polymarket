"""Exact out-of-sample per-day numbers for the shipped cell.

The shipped cell is: follow, T-120, threshold 0.20, 4.5bps spot filter, hold to
settlement. This computes its real per-day pnl/share on the 10 unseen May days so
run_v4.py's outlook does not have to approximate them.

    python tools_v3/oos_daily_v4.py
"""
import bisect
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v4 as B  # noqa: E402
from engine_v3 import hf  # noqa: E402

ROOT = r"C:\Users\孙奕涵\Desktop\polymarket\micro_v3"
SPLIT = "2026-04-27"
CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
SIZE = 10.0


def load_spot(root):
    out = {}
    for path in hf.day_files(root, hf.EXTERNAL):
        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]
        tbl = pq.read_table(path, columns=["timestamp_ms", "midquote_price"])
        pairs = []
        for row in tbl.to_pylist():
            raw = row.get("midquote_price")
            if raw in (None, "", "None"):
                continue
            try:
                pairs.append((int(row["timestamp_ms"]) / 1000.0, float(raw)))
            except (TypeError, ValueError):
                continue
        pairs.sort()
        out[day] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def spot_at(index, ts):
    times, prices = index
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else prices[i]


def main() -> int:
    spot = load_spot(ROOT)
    res = hf.load_resolutions(ROOT)
    per_day: dict[str, list] = defaultdict(list)
    for market in hf.iter_markets(ROOT, horizons=["5m"], resolutions=res):
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        open_tick = market.first_tick()
        entry_tick = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if open_tick is None or entry_tick is None:
            continue
        picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        spot_open = spot_at(index, market.start_ts)
        spot_entry = spot_at(index, entry_tick.ts)
        if not spot_open or not spot_entry:
            continue
        if abs(spot_entry / spot_open - 1.0) < FLOOR:
            continue
        ask = entry_tick.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        shares = SIZE / ask
        cost = entry_tick.cost_to_buy(side, shares)
        if cost is None:
            continue
        entry_price = cost / shares
        if entry_price > 0.90:
            continue
        won = (side == "UP") == market.up_won
        fee = shares * (0.07 * entry_price * (1 - entry_price))
        per_day[market.day].append(((shares if won else 0.0) - shares * entry_price - fee) / shares)

    print("SHIPPED CELL out-of-sample, exact per-day: follow T-120 thr0.20 4.5bps")
    print(f"{'day':<12}{'n':>6}{'per-share':>12}{'period':>9}")
    oos, ins = [], []
    for day in sorted(per_day):
        vals = per_day[day]
        m = statistics.fmean(vals)
        period = "OUT" if day >= SPLIT else "in"
        (oos if period == "OUT" else ins).append((day, m, len(vals)))
        print(f"{day:<12}{len(vals):>6}{m:>+12.4f}{period:>9}")
    print()
    for label, rows in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        if not rows:
            continue
        vals = [m for _, m, _ in rows]
        n = sum(c for _, _, c in rows)
        print(f"{label}: {len(rows)} days, {n:,} trades")
        print(f"  mean {statistics.fmean(vals):+.4f}  std {statistics.pstdev(vals):.4f}  "
              f"positive {sum(1 for v in vals if v > 0)}/{len(vals)}")
    print()
    print("python tuples:")
    print("OOS_DAILY = (" + ", ".join(f"{m:.4f}" for _, m, _ in oos) + ")")
    print("IN_SAMPLE_DAILY = (" + ", ".join(f"{m:.4f}" for _, m, _ in ins) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

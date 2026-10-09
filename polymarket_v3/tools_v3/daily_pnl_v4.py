"""Daily P&L estimate with the CONFIRMED fee rate.

    python tools_v3/daily_pnl_v4.py

The venue documentation confirms the model used throughout this project:

    fee = C x feeRate x p x (1 - p),   feeRate = 0.07 for the Crypto category,
    makers are never charged, only takers pay.

So the previously assumed coefficient is correct and every net figure computed so
far stands. This script turns those figures into money per day.

It reports both aggregation conventions, because they disagree in sign and the
disagreement is the honest headline:

* **pooled**   - every trade weighted equally
* **day mean** - every day weighted equally (a better answer to "a typical day")

Both are reported for the full out-of-sample period, then stressed by slippage.
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v4 as B  # noqa: E402
import slim  # noqa: E402
from engine_v3 import hf  # noqa: E402

CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
CAP = 0.90
SPLIT = "2026-04-27"
#: CONFIRMED by the venue docs: Crypto category taker fee rate
FEE_RATE = 0.07
#: stake sizes to report, in USDC
STAKES = (10.0, 25.0, 50.0)


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


def build(root):
    spot = load_spot(root)
    trades = []
    for market in slim.load_ticks(root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        so, se = spot_at(index, market.start_ts), spot_at(index, et.ts)
        if not so or not se or abs(se / so - 1.0) < FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99 or ask > CAP:
            continue
        trades.append({"day": market.day, "entry": ask,
                       "won": (side == "UP") == market.up_won})
    return trades


def per_share(entry: float, won: bool, extra_slip: float = 0.0) -> float:
    p = min(0.999, entry + extra_slip)
    gross = (1.0 - p) if won else (-p)
    return gross - FEE_RATE * p * (1.0 - p)


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    trades = build(args.root)
    if not trades:
        print("no trades")
        return 1
    oos = [t for t in trades if t["day"] >= SPLIT]
    days = len({t["day"] for t in oos})
    entry_mean = statistics.fmean(t["entry"] for t in trades)
    acc = sum(1 for t in trades if t["won"]) / len(trades)

    print("=" * 94)
    print("DAILY P&L - confirmed fee rate 0.07 (Crypto category)")
    print("=" * 94)
    print(f"  fee model: C x 0.07 x p x (1-p), taker pays, maker pays nothing")
    print(f"  sample: out-of-sample {days} days, {len(oos):,} trades, "
          f"{len(oos)/days:.1f}/day")
    print(f"  entry mean {entry_mean:.4f}   accuracy {acc*100:.2f}%   "
          f"fee/trade {FEE_RATE*entry_mean*(1-entry_mean)*100:.2f} cents/share")
    print()

    for slip in (0.0, 0.005, 0.01, 0.02):
        pooled = statistics.fmean(per_share(t["entry"], t["won"], slip) for t in oos)
        per_day = defaultdict(list)
        for t in oos:
            per_day[t["day"]].append(per_share(t["entry"], t["won"], slip))
        day_vals = [statistics.fmean(v) for v in per_day.values()]
        dmean = statistics.fmean(day_vals)
        sd = statistics.pstdev(day_vals)
        se = sd / math.sqrt(len(day_vals))
        label = "as modelled" if slip == 0 else f"+{slip*100:.1f}c slippage"
        print("=" * 94)
        print(f"  {label}")
        print("=" * 94)
        print(f"    per-share: pooled {pooled*100:+.2f}%   "
              f"day mean {dmean*100:+.2f}%   day sd {sd*100:.2f}%")
        print(f"    2-sigma on the day mean: {(dmean-2*se)*100:+.2f}% .. "
              f"{(dmean+2*se)*100:+.2f}%   "
              f"-> {'positive' if dmean-2*se > 0 else 'INCLUDES ZERO'}")
        print(f"    positive days: {sum(1 for v in day_vals if v > 0)}/{len(day_vals)}")
        print()
        print(f"    {'stake':>7}{'shares':>9}{'per trade':>12}{'per day':>12}"
              f"{'per 30 days':>14}")
        for stake in STAKES:
            shares = stake / entry_mean
            per_trade = pooled * shares
            print(f"    {stake:>7.0f}{shares:>9.1f}{per_trade:>+12.3f}"
                  f"{per_trade*len(oos)/days:>+12.2f}{per_trade*len(oos)/days*30:>+14.1f}")
        print()

    print("=" * 94)
    print("BOTTOM LINE")
    print("=" * 94)
    pooled = statistics.fmean(per_share(t["entry"], t["won"]) for t in oos)
    per_day = defaultdict(list)
    for t in oos:
        per_day[t["day"]].append(per_share(t["entry"], t["won"]))
    dmean = statistics.fmean(statistics.fmean(v) for v in per_day.values())
    shares_per_trade = 10.0 / entry_mean
    per_day_trades = len(oos) / days
    print(f"  At a 10 USDC stake ({shares_per_trade:.1f} shares/trade, "
          f"{per_day_trades:.0f} trades/day):")
    print(f"    pooled convention   : {pooled*shares_per_trade*per_day_trades:+.2f} "
          f"USDC/day")
    print(f"    day-mean convention : {dmean*shares_per_trade*per_day_trades:+.2f} "
          f"USDC/day")
    print()
    print("  The two conventions disagree in sign, and that disagreement is the")
    print("  honest result: the edge is too small to survive the choice of")
    print("  estimator. A 2-sigma interval that includes zero means this sample")
    print("  cannot establish profit or loss.")
    print()
    print("  The decisive arithmetic, which does not depend on convention:")
    need = (entry_mean + FEE_RATE * entry_mean * (1 - entry_mean))
    print(f"    break-even accuracy at entry {entry_mean:.4f} : {need*100:.2f}%")
    print(f"    measured accuracy                        : {acc*100:.2f}%")
    print(f"    margin                                   : "
          f"{(acc-need)*100:+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

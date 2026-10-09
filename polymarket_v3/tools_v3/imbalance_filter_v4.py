"""The imbalance filter: chosen in-sample, tested out-of-sample, then costed.

    python tools_v3/imbalance_filter_v4.py

WHAT THE PROBE FOUND
--------------------
Token book imbalance, oriented to the side being bought, has a U-shaped relation to
winning that REPLICATED across two independent periods:

    bucket        1     2     3     4     5     6     7     8
    in-sample   93.5  89.8  73.1  69.0  69.8  80.0  91.8  91.8   (win %)
    out-sample  93.6  93.1  79.5  67.9  70.3  75.1  93.3  94.4

Extremes are strong, the middle is weak. The mid buckets lose money badly
(net/sh -11% to -13%) because entry is ~0.79 there while the win rate is only ~69%.

The natural rule: **require |imbalance| to be large**. A near-zero imbalance means
the book is balanced, so the token's move is not supported by order-flow pressure
and is more likely noise.

This script does the honest version:
1. sweep the |imbalance| floor on IN-SAMPLE only
2. apply the chosen floor unchanged to OUT-OF-SAMPLE
3. report full net economics at a real stake, with and without the filter
4. check the day-level consistency, since that has killed every previous candidate
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slim  # noqa: E402
from imbalance_probe_v4 import token_imbalance  # noqa: E402

CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
SPLIT = "2026-04-27"
FEE_RATE = 0.07
#: |imbalance| floors to sweep in-sample
IMB_FLOORS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
STAKES = (10.0, 25.0)


def net_ps(entry: float, won: bool) -> float:
    gross = (1.0 - entry) if won else (-entry)
    return gross - FEE_RATE * entry * (1.0 - entry)


def summarise(rows) -> dict:
    if not rows:
        return {"n": 0}
    per_day = defaultdict(list)
    for r in rows:
        per_day[r["day"]].append(net_ps(r["entry"], r["won"]))
    day_vals = [statistics.fmean(v) for v in per_day.values()]
    sd = statistics.pstdev(day_vals) if len(day_vals) > 1 else 0.0
    se = sd / math.sqrt(len(day_vals)) if day_vals else 0.0
    return {
        "n": len(rows), "days": len(day_vals),
        "per_day": len(rows) / len(day_vals),
        "win": sum(1 for r in rows if r["won"]) / len(rows),
        "entry": statistics.fmean(r["entry"] for r in rows),
        "pooled": statistics.fmean(net_ps(r["entry"], r["won"]) for r in rows),
        "day_mean": statistics.fmean(day_vals),
        "day_sd": sd, "se": se,
        "pos": sum(1 for v in day_vals if v > 0),
    }


def line(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"  {label:<26}(no trades)"
    return (f"  {label:<26}{s['n']:>6,}{s['per_day']:>7.1f}{s['win']*100:>8.1f}"
            f"{s['entry']:>8.3f}{s['pooled']*100:>+10.2f}%"
            f"{s['day_mean']*100:>+10.2f}%{s['pos']:>6}/{s['days']:<3}")


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    print("loading token imbalance...")
    imb = token_imbalance(args.root)
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None:
            continue
        raw = imb.get(market.market_id)
        if raw is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        # the existing spot-move gate is not available from the slim cache alone,
        # so this comparison isolates the imbalance filter against the shipped
        # signal without it. Both arms are treated identically.
        move = et.up_mid - ot.up_mid
        if abs(move) < THRESHOLD:
            continue
        side = "UP" if move > 0 else "DOWN"
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        signed = raw if side == "UP" else -raw
        rows.append({"day": market.day, "imb": signed, "entry": ask,
                     "won": (side == "UP") == market.up_won})

    ins = [r for r in rows if r["day"] < SPLIT]
    oos = [r for r in rows if r["day"] >= SPLIT]
    print(f"  rows {len(rows):,}   in-sample {len(ins):,}   out-of-sample {len(oos):,}")

    header = (f"  {'variant':<26}{'n':>6}{'/day':>7}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}")

    print()
    print("=" * 108)
    print("STEP 1 - SWEEP THE |imbalance| FLOOR ON IN-SAMPLE ONLY")
    print("=" * 108)
    print(header)
    print("-" * 108)
    print(line("no filter", summarise(ins)))
    best_floor, best_val = 0.0, summarise(ins)["pooled"]
    for floor in IMB_FLOORS[1:]:
        s = summarise([r for r in ins if abs(r["imb"]) >= floor])
        print(line(f"|imb| >= {floor:.2f}", s))
        if s.get("n", 0) >= 200 and s["pooled"] > best_val:
            best_floor, best_val = floor, s["pooled"]
    print()
    print(f"  CHOSEN on in-sample: |imbalance| >= {best_floor:.2f}")

    print()
    print("=" * 108)
    print("STEP 2 - APPLY UNCHANGED OUT-OF-SAMPLE")
    print("=" * 108)
    print(header)
    print("-" * 108)
    base = summarise(oos)
    chosen = summarise([r for r in oos if abs(r["imb"]) >= best_floor])
    print(line("no filter", base))
    print(line(f"|imb| >= {best_floor:.2f}", chosen))
    print()
    print(f"  change: pooled {(chosen['pooled']-base['pooled'])*100:+.2f} points, "
          f"day mean {(chosen['day_mean']-base['day_mean'])*100:+.2f} points")

    print()
    print("=" * 108)
    print("STEP 3 - FULL OUT-OF-SAMPLE FLOOR CURVE (is it a plateau or a spike?)")
    print("=" * 108)
    print(header)
    print("-" * 108)
    for floor in IMB_FLOORS:
        s = summarise([r for r in oos if abs(r["imb"]) >= floor])
        print(line(f"|imb| >= {floor:.2f}", s))
    print()
    print("  A broad plateau means the effect is real. A single tall column does not.")

    print()
    print("=" * 108)
    print("STEP 4 - MONEY, AND THE SIGNIFICANCE CHECK THAT KILLED EVERY EARLIER CANDIDATE")
    print("=" * 108)
    for label, s in (("no filter", base), (f"|imb| >= {best_floor:.2f}", chosen)):
        if s.get("n", 0) == 0:
            continue
        print(f"  {label}")
        print(f"    pooled {s['pooled']*100:+.2f}%   day mean {s['day_mean']*100:+.2f}%   "
              f"day sd {s['day_sd']*100:.2f}%   se {s['se']*100:.2f}%")
        lo, hi = s["day_mean"] - 2 * s["se"], s["day_mean"] + 2 * s["se"]
        print(f"    2-sigma: {lo*100:+.2f}% .. {hi*100:+.2f}%   "
              f"-> {'EXCLUDES ZERO' if lo > 0 else 'includes zero'}")
        print(f"    positive days: {s['pos']}/{s['days']}")
        for stake in STAKES:
            shares = stake / s["entry"]
            print(f"      {stake:>4.0f} USDC stake -> "
                  f"{s['pooled']*shares*s['per_day']:+.2f} USDC/day")
        print()
    print("  The 2-sigma line is the one that matters. Everything before this failed")
    print("  there, and until it is crossed the strategy is not established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

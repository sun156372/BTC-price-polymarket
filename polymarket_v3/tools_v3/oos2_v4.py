"""SECOND out-of-sample test - the one that actually decides.

    python tools_v3/oos2_v4.py

The first test (tools_v3/oos_v4.py) froze parameters on 2026-03-27..04-26 and
tested on the ten May days 05-01..05-17, giving +0.39%/share on 385 trades with a
2-sigma interval that still included zero.

That was one sample. A single out-of-sample pass can be luck. This runs the SECOND,
on days that did not exist when the parameters were frozen: 2026-05-18 onward.

Nothing is re-tuned here. The shipped configuration is used exactly as it stands:

    follow, T-120, threshold 0.20, spot-move floor 4.5bps, entry cap 0.90,
    hold to settlement

Two decisive questions:

1. Does the edge stay positive on a second independent sample?
2. Does the 2-sigma interval finally exclude zero, now that the two out-of-sample
   samples can be pooled?

This is the pre-registered test: the pass/fail rule is fixed before looking, so the
result cannot be reinterpreted afterwards.
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
from engine_v3 import hf  # noqa: E402
from engine_v3.signals import taker_fee_per_share  # noqa: E402

#: the SHIPPED configuration, frozen. Not to be adjusted in this file.
CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
CAP = 0.90
SIZE = 10.0

#: OOS-1 was trained on days before this date and tested on 05-01..05-17
OOS1_SPLIT = "2026-04-27"
OOS1_END = "2026-05-18"
#: OOS-2 is everything from here on
OOS2_START = "2026-05-18"


def load_spot(root: str) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
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


def spot_at(index, ts: float) -> float | None:
    times, prices = index
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else prices[i]


def collect(root: str) -> list[dict]:
    """Every trade the SHIPPED configuration would take, with its realised result."""
    spot = load_spot(root)
    res = hf.load_resolutions(root)
    rows: list[dict] = []
    for market in hf.iter_markets(root, horizons=["5m"], resolutions=res):
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
        if entry_price > CAP:
            continue
        won = (side == "UP") == market.up_won
        fee = shares * (0.07 * entry_price * (1 - entry_price))
        rows.append({
            "day": market.day,
            "entry": entry_price,
            "won": won,
            "per_share": ((shares if won else 0.0) - shares * entry_price - fee) / shares,
        })
    return rows


def daily(rows: list[dict]) -> dict[str, float]:
    per_day = defaultdict(list)
    for r in rows:
        per_day[r["day"]].append(r["per_share"])
    return {d: statistics.fmean(v) for d, v in per_day.items()}


def describe(label: str, rows: list[dict]) -> dict:
    if not rows:
        print(f"  {label}: no trades")
        return {}
    d = daily(rows)
    vals = list(d.values())
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    se = sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    print(f"  {label}")
    print(f"    days {len(vals)}   trades {len(rows):,}   "
          f"{len(rows)/len(vals):.1f}/day")
    print(f"    accuracy {sum(1 for r in rows if r['won'])/len(rows)*100:.1f}%   "
          f"entry {statistics.fmean(r['entry'] for r in rows):.4f}   "
          f"pooled per-share {statistics.fmean(r['per_share'] for r in rows)*100:+.2f}%")
    print(f"    day mean {mean*100:+.2f}%   sd {sd*100:.2f}%   se {se*100:.2f}%   "
          f"positive {sum(1 for v in vals if v > 0)}/{len(vals)}")
    if se > 0:
        lo, hi = mean - 2 * se, mean + 2 * se
        print(f"    2-sigma on the day mean: {lo*100:+.2f}% .. {hi*100:+.2f}%   "
              f"-> {'EXCLUDES zero' if lo > 0 else 'includes zero'}")
    return {"days": len(vals), "trades": len(rows), "mean": mean, "sd": sd,
            "se": se, "per_day": len(rows) / len(vals),
            "win": sum(1 for r in rows if r["won"]) / len(rows),
            "entry": statistics.fmean(r["entry"] for r in rows),
            "pooled": statistics.fmean(r["per_share"] for r in rows),
            "vals": vals, "daily": d}


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no trades collected")
        return 1
    days = sorted({r["day"] for r in rows})

    ins = [r for r in rows if r["day"] < OOS1_SPLIT]
    oos1 = [r for r in rows if OOS1_SPLIT <= r["day"] < OOS1_END]
    oos2 = [r for r in rows if r["day"] >= OOS2_START]

    print("=" * 96)
    print("SECOND OUT-OF-SAMPLE TEST - shipped configuration, frozen, not re-tuned")
    print("=" * 96)
    print(f"  follow | T-{CHECKPOINT:.0f} | threshold {THRESHOLD:.2f} | "
          f"spot floor {FLOOR*10_000:.1f}bps | cap {CAP:.2f} | hold to settlement")
    print(f"  data: {len(days)} days, {days[0]} .. {days[-1]}")
    print()

    print("=" * 96)
    print("THE THREE PERIODS")
    print("=" * 96)
    ins_res = describe("IN-SAMPLE (used to pick parameters)  03-27..04-26", ins)
    oos1_res = describe("OUT-OF-SAMPLE #1                      05-01..05-17", oos1)
    oos2_res = describe("OUT-OF-SAMPLE #2 (NEW, decisive)      05-18..", oos2)

    if not oos2:
        print()
        print("  No second out-of-sample days available yet.")
        return 1

    # ------------------------------------------------------------- day by day
    print()
    print("=" * 96)
    print("OUT-OF-SAMPLE #2, DAY BY DAY")
    print("=" * 96)
    print(f"  {'day':<12}{'n':>6}{'win%':>8}{'entry':>8}{'per-share':>12}")
    print("-" * 96)
    for day in sorted({r["day"] for r in oos2}):
        subset = [r for r in oos2 if r["day"] == day]
        print(f"  {day:<12}{len(subset):>6}"
              f"{sum(1 for r in subset if r['won'])/len(subset)*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in subset):>8.3f}"
              f"{statistics.fmean(r['per_share'] for r in subset)*100:>+11.2f}%")

    # ---------------------------------------------------------- pooled verdict
    print()
    print("=" * 96)
    print("PRE-REGISTERED VERDICT")
    print("=" * 96)
    pooled_oos = oos1 + oos2
    p = describe("POOLED out-of-sample #1 + #2", pooled_oos)
    print()
    if oos2_res and oos2_res["mean"] > 0:
        print("  OOS #2 is POSITIVE. The edge survived a second independent sample.")
    else:
        print("  OOS #2 is NOT positive. The edge did NOT survive a second")
        print("  independent sample.")
    print()
    if p and p["se"] > 0:
        lo = p["mean"] - 2 * p["se"]
        if lo > 0:
            print("  Pooled 2-sigma lower bound is ABOVE zero: the edge is now")
            print("  statistically established on unseen data.")
        else:
            print(f"  Pooled 2-sigma lower bound is {lo*100:+.2f}% - still at or below")
            print("  zero. Two samples have not established the edge.")
    print()
    print("  The rule was fixed before looking: a second positive sample with a")
    print("  pooled lower bound above zero is a pass; anything else is a fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

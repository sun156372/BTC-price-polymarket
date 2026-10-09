"""Where should the entry-price cap sit? Does the 0.90 cap cost money?

    python tools_v3/cap_optimal_v4.py

WHY THIS MATTERS
----------------
The cap sweep in the accuracy investigation showed something uncomfortable:

    cap 0.80 -> accuracy 76.1%, mean entry 0.756
    cap 0.90 -> accuracy 82.6%, mean entry 0.819
    no cap   -> accuracy 86.4%, mean entry 0.857
    (the trades REJECTED by the 0.90 cap win 94.9% of the time)

So the expensive trades the cap throws away are the MOST accurate ones. The cap was
justified as "above break-even is negative EV", which is true for a single trade at
a fixed accuracy - but the accuracy is not fixed, it rises with the price. If the
relationship is steep enough, a loose cap can beat a tight one, and the shipped 0.90
may simply be wrong.

WHAT IS SWEPT
-------------
The cap, from 0.70 to no-cap. For each, the net per-share result after the
CONFIRMED 0.07 crypto taker fee, reported on in-sample and out-of-sample separately.

The honest selection rule, applied before looking at the answer:

* choose the cap on IN-SAMPLE net per-share only
* report that cap on OUT-OF-SAMPLE
* also report the out-of-sample curve in full, because a broad plateau is evidence
  and a single spike is not

The unconstrained optimum on the full sample is also shown, but labelled as what it
is - a fitted number, not a result.
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
SPLIT = "2026-04-27"
FEE_RATE = 0.07
#: caps to sweep; 1.0 means "no cap"
CAPS = (0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86,
        0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00)


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


def collect(root):
    """Every candidate trade, with NO cap applied - the cap is a sweep variable."""
    spot = load_spot(root)
    rows = []
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
        if ask is None or not 0.01 < ask < 0.99:
            continue
        rows.append({"day": market.day, "entry": ask,
                     "won": (side == "UP") == market.up_won})
    return rows


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
        "n": len(rows),
        "per_day": len(rows) / len(per_day),
        "win": sum(1 for r in rows if r["won"]) / len(rows),
        "entry": statistics.fmean(r["entry"] for r in rows),
        "pooled": statistics.fmean(net_ps(r["entry"], r["won"]) for r in rows),
        "day_mean": statistics.fmean(day_vals),
        "day_sd": sd,
        "se": se,
        "pos": sum(1 for v in day_vals if v > 0),
        "days": len(day_vals),
    }


def fmt_row(label: str, s: dict, base: float | None = None) -> str:
    if s.get("n", 0) == 0:
        return f"  {label:<12}(no trades)"
    delta = "" if base is None else f"{s['pooled']-base:>+9.2f}%"
    return (f"  {label:<12}{s['n']:>6,}{s['per_day']:>8.1f}{s['win']*100:>8.1f}"
            f"{s['entry']:>8.3f}{s['pooled']*100:>+10.2f}%"
            f"{s['day_mean']*100:>+10.2f}%{s['pos']:>6}/{s['days']:<3}{delta}")


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no trades")
        return 1
    ins = [r for r in rows if r["day"] < SPLIT]
    oos = [r for r in rows if r["day"] >= SPLIT]
    print("=" * 104)
    print("ENTRY-PRICE CAP SWEEP - is 0.90 costing money?")
    print("=" * 104)
    print(f"  gates: T-{CHECKPOINT:.0f} | thr {THRESHOLD:.2f} | "
          f"spot floor {FLOOR*10_000:.1f}bps | fee rate {FEE_RATE} (confirmed)")
    print(f"  candidate trades before any cap: {len(rows):,}   "
          f"in-sample {len(ins):,}   out-of-sample {len(oos):,}")
    print(f"  net per share is AFTER the {FEE_RATE} crypto taker fee.")
    print()

    header = (f"  {'cap':<12}{'n':>6}{'/day':>8}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}{'vs base':>10}")

    for label, subset in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        print("=" * 104)
        print(f"{label}  ({len({r['day'] for r in subset})} days)")
        print("=" * 104)
        print(header)
        print("-" * 104)
        capped = [r for r in subset if r["entry"] <= 0.90]
        base = summarise(capped)["pooled"]
        print(fmt_row("no cap", summarise(subset), base))
        for cap in CAPS:
            kept = [r for r in subset if r["entry"] <= cap]
            print(fmt_row(f"<= {cap:.2f}", summarise(kept), base))
        print()

    # ------------------------------------------------ honest selection
    print("=" * 104)
    print("SELECTION - cap chosen on IN-SAMPLE only, then applied out-of-sample")
    print("=" * 104)
    best_cap, best_val = None, None
    for cap in CAPS:
        kept = [r for r in ins if r["entry"] <= cap]
        s = summarise(kept)
        if s.get("n", 0) < 200:
            continue
        if best_val is None or s["pooled"] > best_val:
            best_cap, best_val = cap, s["pooled"]
    print(f"  in-sample optimum: cap {best_cap:.2f}  ({best_val*100:+.2f}% pooled)")
    chosen = summarise([r for r in oos if r["entry"] <= best_cap])
    shipped = summarise([r for r in oos if r["entry"] <= 0.90])
    print(f"  applied out-of-sample:")
    print(header)
    print("-" * 104)
    print(fmt_row(f"<= {0.90:.2f} (shipped)", shipped))
    print(fmt_row(f"<= {best_cap:.2f} (chosen)", chosen))
    print()
    print(f"  change from moving the cap: "
          f"{(chosen['pooled']-shipped['pooled'])*100:+.2f} points per share")
    print()
    print("  Read the SHAPE, not the peak. A broad plateau across neighbouring caps")
    print("  means the effect is real; a single tall column means it is noise.")

    # ------------------------------------------------- unconstrained optimum
    print()
    print("=" * 104)
    print("FITTED OPTIMUM (for reference only - this number is not a result)")
    print("=" * 104)
    allbest_cap, allbest = None, None
    for cap in CAPS:
        s = summarise([r for r in rows if r["entry"] <= cap])
        if s.get("n", 0) < 300:
            continue
        if allbest is None or s["pooled"] > allbest:
            allbest_cap, allbest = cap, s["pooled"]
    print(f"  best cap over ALL 37 days: {allbest_cap:.2f} "
          f"({allbest*100:+.2f}% pooled)")
    print(f"  this is the maximum of 16 cells on one dataset - expect it to")
    print(f"  overstate. The out-of-sample answer above is the one to use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

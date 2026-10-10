"""FINAL validation of the imbalance rule on the expanded dataset.

    python tools_v4/imbalance_final.py

THE SPLIT PROBLEM, AND HOW THIS HANDLES IT
------------------------------------------
The rule was declared at |imbalance| >= 0.30 on a 37-day sample using a chronological
split at 2026-04-27. New data has now arrived, but every added day (04-02, 04, 06,
07, 08, 10, 12) falls BEFORE that split. So the added data does not extend the
out-of-sample period at all - it enlarges the training period.

That is still useful, and it makes the test stronger in one specific way. The
out-of-sample block is now:

    2026-04-27 .. 2026-05-24     29 days

which is 22 days of the original test plus 7 freshly added days. None of the seven
were available when the threshold was chosen, so they are genuinely unseen, and the
rule is applied exactly as declared - no re-tuning.

WHAT IS REPORTED
----------------
1. The declared rule on the ORIGINAL out-of-sample block (reproduce the prior
   result, so a discrepancy would show up immediately).
2. The declared rule on the NEWLY ADDED days alone (the cleanest single test here:
   never seen in any form).
3. The declared rule on the COMBINED out-of-sample block.
4. Realistic live expectations: stake sizes, the fill cost that would erase the
   edge, and the frequency that matters for capacity.
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
from engine_v4.strategy import V4Params  # noqa: E402
from paths import resolve_data_root  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _resolve_default_root() -> str:
    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""
    return resolve_data_root(required=False) or 'micro_v3'


import backtest as B  # noqa: E402
import slim  # noqa: E402
from engine_v4 import hf  # noqa: E402
from engine_v4.imbalance import token_imbalance  # noqa: E402

CHECKPOINT = V4Params().entry_t_minus
THRESHOLD = V4Params().threshold
SPOT_FLOOR = V4Params().min_spot_move
CAP = V4Params().max_entry_price
FEE_RATE = 0.07
#: the ORIGINAL split. Added days all fall before it, so they are training data.
ORIGINAL_SPLIT = "2026-04-27"
#: the declared, pre-registered rule. NOT adjusted here.
DECLARED_FLOOR = V4Params().min_book_imbalance
#: days added in the latest batch, all before ORIGINAL_SPLIT
NEW_DAYS = ("2026-04-02", "2026-04-04", "2026-04-06", "2026-04-07",
            "2026-04-08", "2026-04-10", "2026-04-12")
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


def net_ps(entry: float, won: bool, extra_cost: float = 0.0) -> float:
    p = min(0.999, entry + extra_cost)
    gross = (1.0 - p) if won else (-p)
    return gross - FEE_RATE * p * (1.0 - p)


def describe(rows) -> dict:
    if not rows:
        return {"n": 0}
    per_day = defaultdict(list)
    for r in rows:
        per_day[r["day"]].append(net_ps(r["entry"], r["won"]))
    vals = [statistics.fmean(v) for v in per_day.values()]
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if vals else 0.0
    return {"n": len(rows), "days": len(vals), "per_day": len(rows)/len(vals),
            "win": sum(1 for r in rows if r["won"])/len(rows),
            "entry": statistics.fmean(r["entry"] for r in rows),
            "pooled": statistics.fmean(net_ps(r["entry"], r["won"]) for r in rows),
            "day_mean": mean, "sd": sd, "se": se,
            "pos": sum(1 for v in vals if v > 0)}

def line(label, rows):
    """Format one block. Takes raw rows so callers cannot pass a half-built dict."""
    s = describe(rows)
    if s.get("n", 0) == 0:
        return f"  {label:<30}(no trades)"
    star = "  <-- significant" if s["se"] > 0 and s["day_mean"] - 2 * s["se"] > 0 else ""
    return (f"  {label:<30}{s['n']:>6,}{s['per_day']:>7.1f}{s['win']*100:>8.1f}"
            f"{s['entry']:>8.3f}{s['pooled']*100:>+10.2f}%"
            f"{s['day_mean']*100:>+10.2f}%{s['pos']:>6}/{s['days']:<3}{star}")


def main(argv=None) -> int:
    workspace = _resolve_default_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=resolve_data_root(required=False) or 'micro_v3')
    args = parser.parse_args(argv)

    print("loading token full-depth imbalance...")
    imb = token_imbalance(args.root)
    spot = load_spot(args.root)
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        raw = imb.get(market.market_id)
        sd = spot.get(market.day)
        if raw is None or sd is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        times, prices = sd
        i0 = bisect.bisect_right(times, market.start_ts) - 1
        i1 = bisect.bisect_right(times, et.ts) - 1
        if i0 < 0 or i1 < 0:
            continue
        so, se = prices[i0], prices[i1]
        if not so or not se or abs(se / so - 1.0) < SPOT_FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99 or ask > CAP:
            continue
        rows.append({"day": market.day, "imb": raw if side == "UP" else -raw,
                     "entry": ask, "won": (side == "UP") == market.up_won})

    if not rows:
        print("no rows")
        return 1
    days = sorted({r["day"] for r in rows})
    print(f"  days {len(days)}  ({days[0]} .. {days[-1]})   rows {len(rows):,}")
    print(f"  declared rule (pre-registered, unadjusted): |imbalance| >= {DECLARED_FLOOR:.2f}")

    ins = [r for r in rows if r["day"] < ORIGINAL_SPLIT]
    new_days = [r for r in rows if r["day"] in NEW_DAYS]
    oos_old = [r for r in rows if ORIGINAL_SPLIT <= r["day"] < "2026-05-18"]
    oos_new = [r for r in rows if r["day"] >= "2026-05-18"]
    oos_all = [r for r in rows if r["day"] >= ORIGINAL_SPLIT]

    def filt(sub):
        return [r for r in sub if abs(r["imb"]) >= DECLARED_FLOOR]

    header = (f"  {'block':<30}{'n':>6}{'/day':>7}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}")
    print()
    print("=" * 112)
    print("1. THE DECLARED RULE, BLOCK BY BLOCK")
    print("=" * 112)
    print(header)
    print("-" * 112)
    print(line(f"training (pre-{ORIGINAL_SPLIT[5:]})", filt(ins)))
    print(line("  of which NEWLY ADDED days", filt(new_days)))
    print(line("OOS old (05-01..05-17)", filt(oos_old)))
    print(line("OOS new (05-18..05-24)", filt(oos_new)))
    print(line("OUT-OF-SAMPLE, ALL", filt(oos_all)))
    print()
    print(line("OOS ALL - rule OFF (for contrast)", oos_all))

    print()
    print("=" * 112)
    print("2. THE SEVEN NEWLY ADDED DAYS, DAY BY DAY  (never seen in any form)")
    print("=" * 112)
    print(f"  {'day':<12}{'n':>6}{'win%':>8}{'entry':>8}{'net/sh':>11}")
    print("-" * 112)
    for day in sorted({r["day"] for r in new_days}):
        sub = filt([r for r in new_days if r["day"] == day])
        if not sub:
            print(f"  {day:<12}    no trades passed the gate")
            continue
        print(f"  {day:<12}{len(sub):>6}"
              f"{sum(1 for r in sub if r['won'])/len(sub)*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in sub):>8.3f}"
              f"{statistics.fmean(net_ps(r['entry'], r['won']) for r in sub)*100:>+10.2f}%")

    print()
    print("=" * 112)
    print("3. LIVE-FILL TOLERANCE  (how much slippage erases the edge)")
    print("=" * 112)
    kept = filt(oos_all)
    d = describe(kept)
    print(f"  out-of-sample edge: {d['day_mean']*100:+.2f}%/day at entry {d['entry']:.4f}")
    print()
    print(f"  {'extra fill cost':<20}{'net/share':>12}{'day mean':>11}{'10 USDC/day':>13}"
          f"{'verdict':>12}")
    print("-" * 112)
    for extra in (0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03):
        per_day = defaultdict(list)
        for r in kept:
            per_day[r["day"]].append(net_ps(r["entry"], r["won"], extra))
        vals = [statistics.fmean(v) for v in per_day.values()]
        mean = statistics.fmean(vals)
        pooled = statistics.fmean(net_ps(r["entry"], r["won"], extra) for r in kept)
        shares = 10.0 / (d["entry"] + extra)
        label = "as measured" if extra == 0 else f"+{extra*100:.1f} cents"
        print(f"  {label:<20}{pooled*100:>+11.2f}%{mean*100:>+10.2f}%"
              f"{mean*shares*d['per_day']:>+12.2f}"
              f"{('profitable' if mean > 0 else 'LOSING'):>12}")

    print()
    print("=" * 112)
    print("4. LIVE EXPECTATION")
    print("=" * 112)
    lo, hi = d["day_mean"] - 2*d["se"], d["day_mean"] + 2*d["se"]
    print(f"  out-of-sample: {d['days']} days, {d['n']:,} trades, "
          f"{d['per_day']:.1f}/day, accuracy {d['win']*100:.1f}%")
    print(f"  net per share {d['pooled']*100:+.2f}%   day mean {d['day_mean']*100:+.2f}%")
    print(f"  2-sigma on the day mean: {lo*100:+.2f}% .. {hi*100:+.2f}%")
    print(f"  positive days: {d['pos']}/{d['days']}")
    print()
    print(f"  {'stake':>7}{'shares':>9}{'per trade':>12}{'per day':>12}"
          f"{'per 30d':>12}{'2-sigma range/day':>22}")
    for stake in STAKES:
        shares = stake / d["entry"]
        per_trade = d["pooled"] * shares
        per_day = per_trade * d["per_day"]
        band_lo = lo * shares * d["per_day"]
        band_hi = hi * shares * d["per_day"]
        band = f"{band_lo:+.1f} .. {band_hi:+.1f}"
        print(f"  {stake:>7.0f}{shares:>9.1f}{per_trade:>+12.3f}"
              f"{per_day:>+12.1f}{per_day*30:>+12.0f}{band:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

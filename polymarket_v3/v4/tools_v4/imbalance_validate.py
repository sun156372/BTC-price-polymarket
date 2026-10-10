"""Validate the imbalance filter properly, on the SHIPPED gates, with a fixed rule.

    python tools_v4/imbalance_validate.py

The first pass found |imbalance| >= 0.40 gives +7.25% per share out-of-sample with
22/22 positive days and a 2-sigma interval that excludes zero. That is the first
such result in this project, which is exactly why it needs harder checks before it
is believed. Three specific worries:

1. **It was not tested on the shipped strategy.** The first pass dropped the
   spot-move gate and used the slim cache's touch price, so it measured a slightly
   different thing. This reruns on the FULL shipped gate set: threshold 0.20, the
   4.5bps spot floor, no price cap, hold to settlement.

2. **0.40 was chosen as a maximum, which invites overfitting.** The plateau spans
   0.20-0.60, so the fix is a PRE-REGISTERED threshold rather than the argmax. This
   declares the rule up front - |imbalance| >= 0.30, a round number in the middle of
   the plateau - and reports it without further adjustment.

3. **The imbalance may just be a proxy for realised volatility or for how far price
   has already moved.** If so, it adds nothing and the gate would have captured it.
   This tests that by splitting imbalance against the magnitude of the token move.
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

def _resolve_default_root() -> str:
    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""
    return resolve_data_root(required=False) or 'micro_v3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as B  # noqa: E402
import slim  # noqa: E402
from engine_v4 import hf  # noqa: E402
from engine_v4.imbalance import token_imbalance  # noqa: E402

CHECKPOINT = V4Params().entry_t_minus
THRESHOLD = V4Params().threshold
SPOT_FLOOR = V4Params().min_spot_move
CAP = V4Params().max_entry_price
SPLIT = "2026-04-27"
FEE_RATE = 0.07
#: PRE-REGISTERED before looking at the out-of-sample result. Round number in the
#: middle of the plateau found in-sample. Must NOT be adjusted afterwards.
DECLARED_FLOOR = V4Params().min_book_imbalance


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


def net_ps(entry: float, won: bool) -> float:
    gross = (1.0 - entry) if won else (-entry)
    return gross - FEE_RATE * entry * (1.0 - entry)


def main(argv=None) -> int:
    workspace = _resolve_default_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=resolve_data_root(required=False) or 'micro_v3')
    args = parser.parse_args(argv)

    spot = load_spot(args.root)
    imb = token_imbalance(args.root)
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        sd = spot.get(market.day)
        raw = imb.get(market.market_id)
        if sd is None or raw is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        idx = sd[0]
        i0 = bisect.bisect_right(idx, market.start_ts) - 1
        i1 = bisect.bisect_right(idx, et.ts) - 1
        if i0 < 0 or i1 < 0:
            continue
        so, se = sd[1][i0], sd[1][i1]
        if not so or not se or abs(se / so - 1.0) < SPOT_FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99 or ask > CAP:
            continue
        rows.append({
            "day": market.day,
            "imb": raw if side == "UP" else -raw,
            "abs_move": abs(et.up_mid - ot.up_mid),
            "entry": ask,
            "won": (side == "UP") == market.up_won,
        })
    if not rows:
        print("no rows")
        return 1
    ins = [r for r in rows if r["day"] < SPLIT]
    oos = [r for r in rows if r["day"] >= SPLIT]
    print("=" * 100)
    print("IMBALANCE FILTER ON THE FULL SHIPPED GATES")
    print("=" * 100)
    print(f"  threshold {THRESHOLD} | spot floor {SPOT_FLOOR*1e4:.1f}bps | "
          f"no price cap | hold to settlement | fee {FEE_RATE}")
    print(f"  PRE-REGISTERED rule: |imbalance| >= {DECLARED_FLOOR:.2f}")
    print(f"  rows {len(rows):,}   in-sample {len(ins):,} / "
          f"{len({r['day'] for r in ins})}d   "
          f"out-of-sample {len(oos):,} / {len({r['day'] for r in oos})}d")

    def report(label, subset):
        if not subset:
            print(f"  {label:<34}(none)")
            return None
        per_day = defaultdict(list)
        for r in subset:
            per_day[r["day"]].append(net_ps(r["entry"], r["won"]))
        vals = [statistics.fmean(v) for v in per_day.values()]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        se = sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {label:<34}{len(subset):>6,}{len(subset)/len(vals):>7.1f}"
              f"{sum(1 for r in subset if r['won'])/len(subset)*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in subset):>8.3f}"
              f"{statistics.fmean(net_ps(r['entry'], r['won']) for r in subset)*100:>+10.2f}%"
              f"{mean*100:>+10.2f}%{sum(1 for v in vals if v > 0):>6}/{len(vals):<3}")
        if se > 0:
            lo, hi = mean - 2 * se, mean + 2 * se
            print(f"  {'':<34}2-sigma {lo*100:+.2f}% .. {hi*100:+.2f}%  -> "
                  f"{'EXCLUDES ZERO' if lo > 0 else 'includes zero'}")
        return {"mean": mean, "se": se, "n": len(subset), "per_day": len(subset)/len(vals),
                "entry": statistics.fmean(r["entry"] for r in subset)}

    header = (f"  {'variant':<34}{'n':>6}{'/day':>7}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}")
    print()
    print("=" * 100)
    print("1. IN-SAMPLE")
    print("=" * 100)
    print(header)
    print("-" * 100)
    report("no filter", ins)
    report(f"|imb| >= {DECLARED_FLOOR:.2f}", [r for r in ins if abs(r["imb"]) >= DECLARED_FLOOR])

    print()
    print("=" * 100)
    print("2. OUT-OF-SAMPLE (the declared rule, untouched)")
    print("=" * 100)
    print(header)
    print("-" * 100)
    base = report("no filter", oos)
    kept = [r for r in oos if abs(r["imb"]) >= DECLARED_FLOOR]
    filt = report(f"|imb| >= {DECLARED_FLOOR:.2f}", kept)

    print()
    print("=" * 100)
    print("3. IS IT JUST A PROXY?  imbalance vs move size, crossed")
    print("=" * 100)
    print("  If imbalance only mattered through the size of the move already made, then")
    print("  within a fixed move band the imbalance split would show no effect.")
    print()
    print(f"  {'move band':<16}{'|imb|<0.30':>22}{'|imb|>=0.30':>22}")
    print(f"  {'':<16}{'n':>8}{'net/sh':>14}{'n':>8}{'net/sh':>14}")
    print("-" * 100)
    for lo, hi in ((0.20, 0.30), (0.30, 0.40), (0.40, 0.55), (0.55, 0.75), (0.75, 2.0)):
        band = [r for r in oos if lo <= r["abs_move"] < hi]
        low = [r for r in band if abs(r["imb"]) < DECLARED_FLOOR]
        high = [r for r in band if abs(r["imb"]) >= DECLARED_FLOOR]
        cells = []
        for sub in (low, high):
            if len(sub) < 20:
                cells.append(f"{len(sub):>8}{'-':>14}")
            else:
                cells.append(f"{len(sub):>8}"
                             f"{statistics.fmean(net_ps(r['entry'], r['won']) for r in sub)*100:>+13.2f}%")
        print(f"  {f'{lo:.2f}-{hi:.2f}':<16}" + "".join(cells))
    print()
    print("  If the right column beats the left in EVERY band, imbalance is an")
    print("  independent effect. If the gap vanishes inside bands, it was a proxy.")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    if filt:
        per_day = defaultdict(list)
        for r in kept:
            per_day[r["day"]].append(net_ps(r["entry"], r["won"]))
        vals = [statistics.fmean(v) for v in per_day.values()]
        mean = statistics.fmean(vals)
        se = statistics.pstdev(vals) / math.sqrt(len(vals))
        print(f"  out-of-sample: {mean*100:+.2f}%/day  ({sum(1 for v in vals if v>0)}/"
              f"{len(vals)} days positive)")
        print(f"  2-sigma: {(mean-2*se)*100:+.2f}% .. {(mean+2*se)*100:+.2f}%")
        shares = 10.0 / filt["entry"]
        print(f"  at a 10 USDC stake: {mean*shares*filt['per_day']:+.2f} USDC/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Is the shipped configuration still positive on the full 37-day sample?

    python tools_v4/baseline_check.py

The A/B run reported the shipped cell at -0.54% pooled / -0.24% day-mean on
out-of-sample. Earlier runs reported +0.39% and +0.89%. Those cannot all describe
the same thing, so this reconciles them.

The suspects, all checkable:

1. **Day mean vs pooled per share.** They are different estimators: pooled weights
   every trade equally, the day mean weights every DAY equally. A few heavy days
   move them apart.
2. **Fee sign.** ``per_share`` is net of the taker fee, which at a 0.82 entry is
   ~1.1 points. Gross and net differ by more than the whole edge.
3. **The sample changed.** OOS grew from 17 to 22 days; the newest days may simply
   be worse.
4. **Entry method.** This check walks the real book; the A/B script used the touch
   ask because the slim cache omits depth.

Each is reported separately so the discrepancy is explained rather than averaged.
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
from engine_v4.signals import taker_fee_per_share  # noqa: E402

CHECKPOINT = V4Params().entry_t_minus
THRESHOLD = V4Params().threshold
FLOOR = 0.00045
CAP = V4Params().max_entry_price
SIZE = 10.0
SPLIT = "2026-04-27"


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


def main(argv=None) -> int:
    workspace = _resolve_default_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=resolve_data_root(required=False) or 'micro_v3')
    args = parser.parse_args(argv)

    spot = load_spot(args.root)
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
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
        so, se = spot_at(index, market.start_ts), spot_at(index, entry_tick.ts)
        if not so or not se:
            continue
        if abs(se / so - 1.0) < FLOOR:
            continue
        ask = entry_tick.ask(side)
        if ask is None or not 0.01 < ask < 0.99 or ask > CAP:
            continue
        shares = SIZE / ask
        won = (side == "UP") == market.up_won
        fee_per_share = taker_fee_per_share(ask)
        gross = (1.0 - ask) if won else (-ask)
        rows.append({
            "day": market.day, "entry": ask, "won": won,
            "gross": gross, "net": gross - fee_per_share,
        })

    if not rows:
        print("no trades")
        return 1
    print("=" * 96)
    print("BASELINE RECONCILIATION - shipped configuration, full sample")
    print("=" * 96)
    print(f"  gates: T-{CHECKPOINT:.0f} | thr {THRESHOLD:.2f} | "
          f"spot floor {FLOOR*10_000:.1f}bps | cap {CAP:.2f}")
    print(f"  trades {len(rows):,}   days {len({r['day'] for r in rows})}")
    print(f"  entry {statistics.fmean(r['entry'] for r in rows):.4f}   "
          f"accuracy {sum(1 for r in rows if r['won'])/len(rows)*100:.1f}%   "
          f"fee/trade {statistics.fmean(taker_fee_per_share(r['entry']) for r in rows):.4f}")
    print()

    header = (f"  {'period':<18}{'days':>6}{'n':>7}{'/day':>7}{'win%':>8}"
              f"{'entry':>8}{'GROSS ps':>11}{'NET ps':>10}{'net day':>10}{'days+':>9}")
    print(header)
    print("-" * 96)
    # The three out-of-sample slices are reported separately as well as pooled,
    # because the whole point is the progression across independent samples.
    periods = (
        ("in-sample", [r for r in rows if r["day"] < SPLIT]),
        ("OOS #1 05-01..17", [r for r in rows
                              if SPLIT <= r["day"] < "2026-05-18"]),
        ("OOS #2 05-18..24", [r for r in rows if r["day"] >= "2026-05-18"]),
        ("OOS pooled", [r for r in rows if r["day"] >= SPLIT]),
        ("ALL", rows),
    )
    for label, subset in periods:
        if not subset:
            continue
        per_day = defaultdict(list)
        for r in subset:
            per_day[r["day"]].append(r)
        day_net = [statistics.fmean(x["net"] for x in v) for v in per_day.values()]
        print(f"  {label:<18}{len(per_day):>6}{len(subset):>7}"
              f"{len(subset)/len(per_day):>7.1f}"
              f"{sum(1 for r in subset if r['won'])/len(subset)*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in subset):>8.3f}"
              f"{statistics.fmean(r['gross'] for r in subset)*100:>+10.2f}%"
              f"{statistics.fmean(r['net'] for r in subset)*100:>+9.2f}%"
              f"{statistics.fmean(day_net)*100:>+9.2f}%"
              f"{sum(1 for v in day_net if v > 0):>6}/{len(day_net):<3}")

    print()
    print("=" * 96)
    print("PER-DAY, OUT-OF-SAMPLE, NET")
    print("=" * 96)
    oos = [r for r in rows if r["day"] >= SPLIT]
    per_day = defaultdict(list)
    for r in oos:
        per_day[r["day"]].append(r)
    print(f"  {'day':<12}{'n':>6}{'win%':>8}{'entry':>8}{'net/sh':>10}")
    for day in sorted(per_day):
        v = per_day[day]
        print(f"  {day:<12}{len(v):>6}"
              f"{sum(1 for x in v if x['won'])/len(v)*100:>8.1f}"
              f"{statistics.fmean(x['entry'] for x in v):>8.3f}"
              f"{statistics.fmean(x['net'] for x in v)*100:>+9.2f}%")

    day_net = [statistics.fmean(x["net"] for x in v) for v in per_day.values()]
    print()
    n = len(day_net)
    mean = statistics.fmean(day_net)
    sd = statistics.pstdev(day_net)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    print("=" * 96)
    print("OUT-OF-SAMPLE SIGNIFICANCE")
    print("=" * 96)
    print(f"  days {n}   day mean (net) {mean*100:+.2f}%   sd {sd*100:.2f}%   "
          f"se {se*100:.2f}%")
    if se > 0:
        print(f"  2-sigma: {(mean-2*se)*100:+.2f}% .. {(mean+2*se)*100:+.2f}%   "
              f"-> {'EXCLUDES zero' if mean-2*se > 0 else 'includes zero'}")
    print(f"  pooled net per-share {statistics.fmean(r['net'] for r in oos)*100:+.2f}%")
    print()
    print("  If the day mean and the pooled number disagree, the pooled figure is")
    print("  being driven by a few heavy days. The day mean is the more honest")
    print("  summary of 'what does a typical day return'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

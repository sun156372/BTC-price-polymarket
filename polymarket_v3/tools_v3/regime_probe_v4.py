"""Did the market itself also change, or was the decay purely selection bias?

    python tools_v3/regime_probe_v4.py

The selection-bias demo (tools_v3/selection_bias_demo.py) shows the reported edge
MUST fall as days accumulate even if the market never changes. That is the main
explanation. This probe asks the separate question: did anything about the market
also move?

Read-only regime indicators, none of which involve the strategy's parameters:

* spread width      - is trading more expensive now?
* entry price at a fixed checkpoint - is the same signal more crowded now?
* |move| distribution - did volatility change (which would change how often the
  threshold trips)?

If those are flat across the period, there is no evidence of a market change and
the whole decay is attributable to the estimator.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v4 as B  # noqa: E402
from engine_v3 import hf  # noqa: E402


def main(argv=None) -> int:
    # tools_v3 -> polymarket_v3 -> workspace root, which holds micro_v3
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    parser.add_argument("--limit", type=int, default=0,
                        help="cap markets processed (0 = all)")
    args = parser.parse_args(argv)
    root = args.root

    res = hf.load_resolutions(root)
    if os.environ.get("REGIME_DEBUG"):
        print(f"[debug] data root = {root}")
        print(f"[debug] resolutions = {len(res)}")
        print(f"[debug] feature files = {len(hf.day_files(root, hf.FEATURES))}")
    # regime indicators, keyed by day
    rows: dict[str, list[dict]] = defaultdict(list)
    seen = 0
    for market in hf.iter_markets(root, horizons=["5m"], resolutions=res):
        seen += 1
        if market.up_won is None:
            continue
        open_tick = market.first_tick()
        if open_tick is None:
            continue
        entry_tick = market.tick_at_t_minus(120.0, tolerance_s=1.5)
        if entry_tick is None:
            continue
        picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, 0.0, "follow")
        if picked is None:
            continue
        side, move = picked
        ask, bid = entry_tick.ask(side), entry_tick.bid(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        won = (side == "UP") == market.up_won
        rows[market.day].append({
            "abs_move": abs(move),
            "up_mid": entry_tick.up_mid,
            "spread": None if bid is None else ask - bid,
            "won": won,
        })
        if args.limit and sum(len(v) for v in rows.values()) >= args.limit:
            break

    days = sorted(rows)
    if os.environ.get("REGIME_DEBUG"):
        print(f"[debug] markets seen = {seen}, usable rows = "
              f"{sum(len(v) for v in rows.values())}, days = {len(days)}")
    if not days:
        print("no data")
        return 1

    print("=" * 92)
    print("REGIME PROBE - did the market change?  (read-only indicators)")
    print("=" * 92)
    print(f"  days: {len(days)}  ({days[0]} .. {days[-1]})")
    print()
    print(f"  {'day':<12}{'n':>6}{'spread(c)':>11}{'Up mid':>9}"
          f"{'|move| med':>12}{'|move| p90':>12}{'follow win%':>13}")
    print("-" * 92)
    recs = []
    for day in days:
        subset = rows[day]
        moves = sorted(r["abs_move"] for r in subset)
        spreads = [r["spread"] for r in subset if r["spread"] is not None]
        rec = {
            "day": day,
            "spread": statistics.median(spreads) if spreads else 0.0,
            "mid": statistics.fmean(r["up_mid"] for r in subset),
            "med": moves[len(moves) // 2],
            "p90": moves[9 * len(moves) // 10],
            "win": sum(1 for r in subset if r["won"]) / len(subset),
        }
        recs.append(rec)
        print(f"  {day:<12}{len(subset):>6}{rec['spread']*100:>11.2f}{rec['mid']:>9.3f}"
              f"{rec['med']:>12.3f}{rec['p90']:>12.3f}{rec['win']*100:>13.1f}")

    print()
    print("=" * 92)
    print("TREND - first half vs second half")
    print("=" * 92)
    half = len(recs) // 2
    early, late = recs[:half], recs[half:]
    print(f"  early {early[0]['day']}..{early[-1]['day']}   "
          f"late {late[0]['day']}..{late[-1]['day']}")
    print()
    print(f"  {'indicator':<22}{'early':>12}{'late':>12}{'change':>12}{'verdict':>14}")
    print("-" * 92)
    for key, label, fmt in (("spread", "spread (cents)", "{:.3f}"),
                            ("mid", "Up mid", "{:.3f}"),
                            ("med", "|move| median", "{:.3f}"),
                            ("p90", "|move| p90", "{:.3f}"),
                            ("win", "follow win rate", "{:.4f}")):
        e = statistics.fmean(r[key] for r in early)
        l = statistics.fmean(r[key] for r in late)
        delta = l - e
        # a rough significance read, using the day-level spread
        sd = statistics.pstdev([r[key] for r in recs])
        se = sd / (len(recs) ** 0.5) if sd else 0.0
        significant = abs(delta) > 2 * se if se else False
        verdict = "MOVED" if significant else "flat"
        scale = 100.0 if key in ("spread", "win") else 1.0
        print(f"  {label:<22}{e*scale:>12.4f}{l*scale:>12.4f}"
              f"{delta*scale:>+12.4f}{verdict:>14}")
    print()
    print("  If every indicator reads 'flat', there is no evidence the market")
    print("  changed, and the decay in the reported edge is fully explained by")
    print("  selection bias (see tools_v3/selection_bias_demo.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Why does the edge shrink as more days are added?

Four competing explanations, each with a measurable signature:

A. **The market adapted.** The edge was real and has been arbitraged away. Signature:
   accuracy falls over time WHILE the entry price stays put.
B. **Prices rose.** The signal still works but you pay more for it. Signature:
   entry price rises over time while accuracy holds.
C. **Regime / volatility change.** The filter or threshold depends on volatility,
   which moved. Signature: spot-move distribution shifts.
D. **It was never there - multiple testing.** The parameter was picked from 1,260
   cells, so the "best" cell is upward-biased, and widening the sample shrinks it
   toward zero mechanically. Signature: the effect shrinks toward ~0 as n grows,
   with no companion change in accuracy or price, and the whole surface degrades
   together rather than one cell decaying.

The decompositions below separate accuracy from price, and look at the whole
parameter surface rather than the one chosen cell, because only the surface can
distinguish D from A and B.

    python tools_v3/decay_analysis_v4.py
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
from engine_v3.signals import taker_fee_per_share  # noqa: E402

#: split the 24 days into four chronological blocks of six
BLOCKS = 4
CHECKPOINTS = (240.0, 180.0, 120.0, 60.0)
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
SIZE = 10.0


def collect(root: str) -> list[dict]:
    res = hf.load_resolutions(root)
    rows: list[dict] = []
    for market in hf.iter_markets(root, horizons=["5m"], resolutions=res):
        if market.up_won is None:
            continue
        open_tick = market.first_tick()
        if open_tick is None:
            continue
        for checkpoint in CHECKPOINTS:
            entry_tick = market.tick_at_t_minus(checkpoint, tolerance_s=1.5)
            if entry_tick is None:
                continue
            picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, 0.0, "follow")
            if picked is None:
                continue
            side, move = picked
            ask = entry_tick.ask(side)
            bid = entry_tick.bid(side)
            if ask is None or not 0.01 < ask < 0.99:
                continue
            shares = SIZE / ask
            cost = entry_tick.cost_to_buy(side, shares)
            if cost is None:
                continue
            entry_price = cost / shares
            won = (side == "UP") == market.up_won
            fee = shares * (0.07 * entry_price * (1 - entry_price))
            rows.append({
                "day": market.day,
                "checkpoint": checkpoint,
                "abs_move": abs(move),
                "entry": entry_price,
                "spread": None if bid is None else ask - bid,
                "won": won,
                "per_share": ((shares if won else 0.0) - shares * entry_price - fee) / shares,
            })
    return rows


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(root, "micro_v3"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no data")
        return 1
    days = sorted({r["day"] for r in rows})
    per_block = max(1, len(days) // BLOCKS)
    blocks = [days[i * per_block:(i + 1) * per_block] for i in range(BLOCKS)]
    blocks[-1] = days[(BLOCKS - 1) * per_block:]
    labels = [f"{b[0][5:]}~{b[-1][5:]}" for b in blocks if b]

    print("=" * 100)
    print(f"DECAY ANALYSIS - {len(days)} days, {len(rows):,} candidate trades")
    print("=" * 100)
    print(f"  blocks: {labels}")

    # ---------------------------------------------------- 1. regime indicators
    print()
    print("=" * 100)
    print("1. DID THE MARKET ITSELF CHANGE?  (regime indicators, no strategy involved)")
    print("=" * 100)
    print(f"  {'block':<16}{'days':>6}{'spread(c)':>11}{'entry@T120':>12}"
          f"{'|move| med':>12}{'|move| p90':>12}{'accuracy':>10}")
    print("-" * 100)
    for block, label in zip([b for b in blocks if b], labels):
        subset = [r for r in rows if r["day"] in block and r["checkpoint"] == 120.0]
        if not subset:
            continue
        moves = sorted(r["abs_move"] for r in subset)
        spreads = [r["spread"] for r in subset if r["spread"] is not None]
        print(f"  {label:<16}{len(block):>6}"
              f"{statistics.median(spreads)*100 if spreads else 0:>11.2f}"
              f"{statistics.fmean(r['entry'] for r in subset):>12.3f}"
              f"{moves[len(moves)//2]:>12.3f}{moves[9*len(moves)//10]:>12.3f}"
              f"{sum(1 for r in subset if r['won'])/len(subset)*100:>10.1f}")

    # -------------------------------- 2. accuracy vs price decomposition, by cell
    print()
    print("=" * 100)
    print("2. ACCURACY OR PRICE?  the same cell, block by block")
    print("=" * 100)
    print("  If accuracy falls and price holds -> the market adapted (A).")
    print("  If price rises and accuracy holds -> you are paying more (B).")
    print()
    for checkpoint in (240.0, 120.0):
        for threshold in (0.15, 0.20):
            print(f"  --- T-{checkpoint:.0f}  threshold {threshold:.2f} ---")
            print(f"  {'block':<16}{'n':>7}{'win%':>8}{'entry':>8}{'break-even':>12}"
                  f"{'per-share':>11}")
            for block, label in zip([b for b in blocks if b], labels):
                subset = [r for r in rows if r["day"] in block
                          and r["checkpoint"] == checkpoint
                          and r["abs_move"] >= threshold]
                if len(subset) < 20:
                    continue
                win = sum(1 for r in subset if r["won"]) / len(subset)
                entry = statistics.fmean(r["entry"] for r in subset)
                be = entry + taker_fee_per_share(entry)
                ps = statistics.fmean(r["per_share"] for r in subset)
                print(f"  {label:<16}{len(subset):>7}{win*100:>8.1f}{entry:>8.3f}"
                      f"{be:>12.4f}{ps:>+11.4f}")
            print()

    # ------------------------------------------ 3. does the WHOLE surface degrade?
    print("=" * 100)
    print("3. DOES THE WHOLE PARAMETER SURFACE DEGRADE TOGETHER?")
    print("=" * 100)
    print("  Explanation D (multiple testing) predicts every cell loses roughly")
    print("  together as n grows. Adaptation (A) predicts the surface SHIFTS - some")
    print("  cells improve while others decay.")
    print()
    half = len(days) // 2
    early, late = set(days[:half]), set(days[half:])
    print(f"  early = {days[0]}..{days[half-1]}   late = {days[half]}..{days[-1]}")
    print()
    print(f"  {'cell':<22}{'early n':>9}{'early ps':>10}{'late n':>8}{'late ps':>10}"
          f"{'change':>10}")
    print("-" * 100)
    changes = []
    for checkpoint in CHECKPOINTS:
        for threshold in THRESHOLDS:
            e = [r for r in rows if r["day"] in early and r["checkpoint"] == checkpoint
                 and r["abs_move"] >= threshold]
            l = [r for r in rows if r["day"] in late and r["checkpoint"] == checkpoint
                 and r["abs_move"] >= threshold]
            if len(e) < 30 or len(l) < 30:
                continue
            eps = statistics.fmean(r["per_share"] for r in e)
            lps = statistics.fmean(r["per_share"] for r in l)
            changes.append(lps - eps)
            print(f"  T-{checkpoint:.0f} thr{threshold:.2f}{'':<8}{len(e):>9}{eps:>+10.4f}"
                  f"{len(l):>8}{lps:>+10.4f}{lps-eps:>+10.4f}")
    if changes:
        print()
        print(f"  cells measured      : {len(changes)}")
        print(f"  cells that worsened : {sum(1 for c in changes if c < 0)}/{len(changes)}")
        print(f"  mean change         : {statistics.fmean(changes):+.4f}/share")
        print()
        if all(c < 0 for c in changes):
            print("  EVERY cell worsened. That points at a market-wide cause (A or B)")
            print("  or at mechanical shrinkage from multiple testing (D), not at one")
            print("  cell decaying because its specific pattern stopped working.")
        else:
            print("  Some cells improved. That is consistent with the surface SHIFTING")
            print("  (adaptation) rather than uniformly shrinking.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

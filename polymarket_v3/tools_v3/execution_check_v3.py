"""V3 execution reality check against the real order book.

    python tools_v3/execution_check_v3.py

Why this is the most important test available
---------------------------------------------
V3's entire conclusion rested on an unknown: **what does it actually cost to get
filled?** The aggregate 9-month dataset had no prices, so V3 assumed a
conservative 15-cent cost and concluded "do not go live". One day of real L2 data
can answer that question directly.

What is measurable here
-----------------------
* the real spread distribution (3,889 snapshots, all 288 markets)
* real depth at the touch, and the true cost of taking a given size
* whether 90.5% of quotes sit at the 1-cent tick, i.e. whether the old 15-cent
  assumption was ever realistic

What is NOT measurable here
---------------------------
Token attribution. ``snapshots.parquet`` carries ``market_id`` but not ``token_id``,
and the row order within a market is not (Up, Down): same-instant pairs sum to 1.0
only 8.9% of the time, and treating odd/even rows as Up scores 0.495/0.509. So this
tool measures the *cost structure of the book*, not which side a given quote is on.
For cost and depth that is sufficient: a 1-cent spread costs 1 cent on either side.

Also: one day. Signal tuning needs months; this file has 24 hours. Frequency and
cost are answerable, edge is not.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3 import micro
from engine_v3.signals import taker_fee_per_share

#: the cost tier V3 assumed in its own analysis, for comparison
V3_CONSERVATIVE_CENTS = 15.0
V3_REALISTIC_CENTS = 1.0

#: order sizes to price off the real book, in USDC at ~0.50 (so ~2x shares)
SIZE_USDC = (10.0, 25.0, 50.0, 100.0, 250.0)


def run(root: str, out_dir: str) -> int:
    markets_path = os.path.join(root, "markets.parquet")
    snapshots_path = os.path.join(root, "snapshots.parquet")
    trades_path = os.path.join(root, "trades.parquet")
    for path in (markets_path, snapshots_path):
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            return 2

    markets = micro.load_markets(markets_path)
    cov = micro.coverage(markets)

    print("=" * 100)
    print("V3 EXECUTION REALITY CHECK - measured order book, not a model")
    print("=" * 100)
    print(f"markets        : {cov['markets']}  ({cov['start'][:16]} -> {cov['end'][:16]} UTC)")
    print(f"coverage       : {cov['days']:.1f} day   ({cov['markets_per_day']:.0f} markets/day)")
    print(f"Up base rate   : {cov['up_rate']:.4f}")
    print()
    print("NOTE: one day only. This file can price execution; it cannot establish edge.")

    # ------------------------------------------------------------- spreads
    grouped = micro.load_snapshots(snapshots_path, {m.market_id: m for m in markets})
    quotes = [q for rows in grouped.values() for q in rows]
    spreads = [q.spread for q in quotes if q.spread is not None]
    stats = micro.summarise(spreads)
    tick = markets[0].tick_size

    print()
    print("=" * 100)
    print("1. SPREAD DISTRIBUTION  (the assumption V3 could not verify)")
    print("=" * 100)
    print(f"  quotes                : {stats['n']:,}")
    print(f"  tick size             : {tick:.2f}")
    print(f"  min / p25 / median    : {stats['min']:.3f} / {stats['p25']:.3f} / {stats['median']:.3f}")
    print(f"  p75 / p90 / max       : {stats['p75']:.3f} / {stats['p90']:.3f} / {stats['max']:.3f}")
    print(f"  mean                  : {stats['mean']:.4f}")
    for threshold in (0.01, 0.02, 0.05, 0.10, 0.50):
        share = sum(1 for s in spreads if s <= threshold + 1e-9) / len(spreads)
        print(f"  spread <= {threshold:.2f}        : {share*100:5.1f}%")
    at_tick = sum(1 for s in spreads if s <= tick + 1e-9) / len(spreads)

    print()
    print("=" * 100)
    print("2. VERDICT ON V3's COST ASSUMPTION")
    print("=" * 100)
    print(f"  V3 assumed  : {V3_CONSERVATIVE_CENTS:.0f} cents (conservative tier), "
          f"{V3_REALISTIC_CENTS:.0f} cent (realistic tier)")
    print(f"  measured    : {stats['median']*100:.1f} cents median, "
          f"{at_tick*100:.1f}% of quotes at the {tick*100:.0f}-cent tick")
    if stats["median"] <= V3_REALISTIC_CENTS / 100.0 + 1e-9:
        print()
        print("  THE CONSERVATIVE TIER WAS NEVER REALISTIC.")
        print(f"  {at_tick*100:.1f}% of quotes sit at the {tick*100:.0f}-cent tick size, so a "
              f"{V3_CONSERVATIVE_CENTS:.0f}-cent")
        print("  cost assumption described a dead book, not this market. The realistic tier")
        print("  (1 cent slippage) is the correct planning assumption on this evidence.")
    else:
        print()
        print(f"  The conservative tier is closer to reality than expected "
              f"(median {stats['median']*100:.1f} cents).")

    # -------------------------------------------------------------- depth
    print()
    print("=" * 100)
    print("3. DEPTH: real cost to take a given size off the visible book")
    print("=" * 100)
    print("  'extra cost' is what walking the book adds on top of the best ask.")
    print("  'fillable' counts quotes that can absorb the size at all; the rest are")
    print("  one-sided snapshots with no asks, not a depth failure.")
    print()
    print(f"  {'size':>8}{'fillable':>10}{'extra cost (median)':>21}{'p90':>9}{'max':>9}")
    print("-" * 100)
    rows = []
    for usdc in SIZE_USDC:
        filled = 0
        slips = []
        for quote in quotes:
            if not quote.best_ask or not quote.asks:
                continue
            shares = usdc / quote.best_ask
            cost = quote.cost_to_buy(shares)
            if cost is None:
                continue
            filled += 1
            slips.append(cost / shares - quote.best_ask)
        fill_rate = filled / len(quotes)
        stats_slip = micro.summarise(slips)
        rows.append({"size_usdc": usdc, "fill_rate": fill_rate,
                     "median_slip": stats_slip.get("median", 0.0),
                     "p90_slip": stats_slip.get("p90", 0.0),
                     "max_slip": stats_slip.get("max", 0.0)})
        print(f"  {usdc:>8.0f}{fill_rate*100:>9.1f}%"
              f"{stats_slip.get('median', 0.0)*100:>20.2f}c"
              f"{stats_slip.get('p90', 0.0)*100:>8.2f}c"
              f"{stats_slip.get('max', 0.0)*100:>8.2f}c")

    print()
    print("  A median extra cost of ~0 cents means the touch is deep enough for these")
    print("  sizes: the cost of taking is the SPREAD, not the walk. So V3's slippage")
    print("  limit (3 cents) is not the binding constraint - the entry PRICE CAP is.")

    # ------------------------------------------------------- entry at T-240
    print()
    print("=" * 100)
    print("4. WHAT DOES AN ENTRY AT V3's DECISION POINT (T-240) ACTUALLY COST?")
    print("=" * 100)
    print("  Only quotes INSIDE the window count. Markets are back-to-back, so a naive")
    print("  offset filter sweeps in the previous market's closing snapshots, which sit")
    print("  at 0.01/0.99 and would make the entry price look absurd.")
    print()
    print(f"  {'time into window':>18}{'quotes':>9}{'median ask':>12}{'p10 ask':>9}"
          f"{'p90 ask':>9}{'median spread':>15}")
    print("-" * 100)
    offset_rows = []
    buckets = ((0, 60, "0-60s"), (60, 120, "60-120s"), (120, 180, "120-180s"),
               (180, 240, "180-240s"), (240, 300, "240-300s"))
    for low, high, label in buckets:
        picked = []
        for market in markets:
            for quote in grouped.get(market.market_id, []):
                into = quote.ts - market.open_ts
                if low <= into < high and quote.best_ask:
                    picked.append(quote)
        if not picked:
            continue
        stats_ask = micro.summarise([q.best_ask for q in picked])
        stats_sp = micro.summarise([q.spread for q in picked if q.spread is not None])
        offset_rows.append({"window": label, "n": len(picked),
                            "median_ask": stats_ask["median"],
                            "median_spread": stats_sp.get("median", 0.0)})
        print(f"  {label:>18}{len(picked):>9,}{stats_ask['median']:>12.3f}"
              f"{stats_ask['p10']:>9.3f}{stats_ask['p90']:>9.3f}"
              f"{stats_sp.get('median', 0.0):>15.3f}")

    print()
    print("  V3 decides at T-240, i.e. 60s into the window. Read the 60-120s row: that")
    print("  is the price actually available at the decision moment. A median ask near")
    print("  0.50 there is what makes the 0.535 cap a live constraint rather than a")
    print("  theoretical one - and it is the number V3 previously had to model.")
    print()
    print("  Caveat that matters: snapshots are sparse (median ~13 per market), so the")
    print("  60-120s bucket is thin. This prices the book on ONE day; it is not a")
    print("  distribution over a year.")

    # ------------------------------------------------------- trades sanity
    depth_summary = {}
    if os.path.exists(trades_path):
        prints = micro.load_prints(trades_path, {m.market_id for m in markets})
        all_prints = [p for v in prints.values() for p in v]
        prices = [p.price for p in all_prints]
        sizes = [p.size for p in all_prints]
        pstats = micro.summarise(prices)
        sstats = micro.summarise(sizes)
        per_market = [len(v) for v in prints.values()]
        print()
        print("=" * 100)
        print("4. REAL PRINTS (for context; not attributable to a side)")
        print("=" * 100)
        print(f"  prints            : {len(all_prints):,} across {len(prints)} markets")
        print(f"  prints per market : min {min(per_market)} median "
              f"{int(statistics.median(per_market))} max {max(per_market)}")
        print(f"  price             : p10 {pstats['p10']:.3f} median {pstats['median']:.3f} "
              f"p90 {pstats['p90']:.3f}")
        print(f"  size              : p10 {sstats['p10']:.2f} median {sstats['median']:.2f} "
              f"p90 {sstats['p90']:.2f} USDC")
        notional = [p.price * p.size for p in all_prints]
        nstats = micro.summarise(notional)
        print(f"  notional/print    : median {nstats['median']:.2f} USDC, "
              f"p90 {nstats['p90']:.2f}")
        print(f"  -> a 10 USDC order is about {10.0/nstats['median']:.1f}x the median print "
              f"({nstats['median']:.2f} USDC) and about "
              f"{10.0/nstats['p90']:.2f}x the p90 print ({nstats['p90']:.2f} USDC).")
        depth_summary = {"prints": len(all_prints),
                         "median_notional": nstats["median"]}

    # ------------------------------------------------------- break-even use
    print()
    print("=" * 100)
    print("5. WHAT THIS DOES TO THE ENTRY-PRICE DISCUSSION")
    print("=" * 100)
    measured_spread = stats["median"]
    print("  Break-even accuracy at entry p is p + 0.07*p*(1-p). With the measured")
    print(f"  {measured_spread*100:.1f}-cent spread, a taker buying the touch pays roughly "
          f"mid + {measured_spread*100/2:.1f}c:")
    print()
    print(f"  {'mid':>7}{'pay':>8}{'accuracy needed':>17}{'measured 8-fold':>17}{'verdict':>14}")
    print("-" * 100)
    measured_accuracy = 0.5523
    for mid_cents in (49, 50, 51, 52, 53):
        mid = mid_cents / 100.0
        pay = mid + measured_spread / 2.0
        needed = pay + taker_fee_per_share(pay)
        verdict = "positive EV" if measured_accuracy > needed else "NEGATIVE EV"
        print(f"  {mid:>7.3f}{pay:>8.4f}{needed:>17.4f}{measured_accuracy:>17.4f}"
              f"{verdict:>14}")
    print()
    print("  The measured accuracy (0.5523, nine months, 8 folds) clears the bar for")
    print("  entries up to about 0.535. On this one day of book data that price looks")
    print("  reachable, because the spread is 1 cent rather than the 15 assumed.")
    print()
    print("  BUT: one day of book data does not prove the spread is always 1 cent, and")
    print("  the strategy's edge was measured on a DIFFERENT time period. The honest")
    print("  read is 'the cost objection is weakened, not eliminated'.")

    # ------------------------------------------------------------- outputs
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "v3_execution_check.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["markets", cov["markets"]])
        writer.writerow(["days", round(cov["days"], 3)])
        writer.writerow(["quotes", stats["n"]])
        writer.writerow(["spread_median", round(stats["median"], 5)])
        writer.writerow(["spread_p90", round(stats["p90"], 5)])
        writer.writerow(["spread_max", round(stats["max"], 5)])
        writer.writerow(["share_at_tick", round(at_tick, 5)])
        writer.writerow(["tick_size", tick])
        for row in rows:
            writer.writerow([f"depth_{row['size_usdc']:.0f}usdc_fill_rate",
                             round(row["fill_rate"], 5)])
            writer.writerow([f"depth_{row['size_usdc']:.0f}usdc_median_slip",
                             round(row["median_slip"], 6)])
        for key, value in depth_summary.items():
            writer.writerow([f"prints_{key}", value])
    print()
    print(f"report : {path}")
    return 0


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(root),
                        help="folder holding markets/snapshots/trades parquet")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(root), "output", "v3"))
    args = parser.parse_args(argv)
    return run(args.root, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

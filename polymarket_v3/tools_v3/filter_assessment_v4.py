"""Is the price-deviation filter NECESSARY? A four-way test.

    python tools_v3/filter_assessment_v4.py

The filter asks: has the underlying BTC actually moved since the window opened?
If not, the token's move is not backed by anything and the quote is noise.

Four questions decide whether it earns its place:

1. **Is it real?** Does the filter improve the pooled edge AND the day-level hit
   rate, or just one of them?
2. **Is it robust?** Earlier work showed the improvement was NOT monotone in the
   floor (2bps worse than 0, 5bps best, 8bps worse again). A non-monotone response
   is the signature of fitting noise, so the curve is mapped properly here.
3. **Is it redundant?** If a plain spread filter achieves the same thing, the spot
   feed is unnecessary complexity. Both are tested side by side.
4. **Is it lookahead-free?** The filter reads the spot price at the decision
   timestamp. If the spot series carries any forward information, the filter looks
   good for the wrong reason. Re-running with a delayed spot read tests that.

Answering these decides whether the filter stays, and it is written so the same
script can say "remove it".
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

CHECKPOINT = 240.0
THRESHOLD = 0.15
MODE = "follow"
SIZE = 10.0
FEE = 0.07
#: spot-move floors to map, in basis points
FLOORS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
#: spread caps to test as the redundant alternative, in cents
SPREAD_CAPS = (float("inf"), 3.0, 2.0, 1.5, 1.0)
#: how late the spot read is allowed to be, to test for lookahead (seconds)
LAG_CHOICES = (0.0, 1.0, 2.0, 5.0)


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
    """Every trade, with the fields the filters need and the outcome."""
    spot = load_spot(root)
    resolutions = hf.load_resolutions(root)
    rows: list[dict] = []
    for market in hf.iter_markets(root, horizons=["5m"], resolutions=resolutions):
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        entry_tick = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        open_tick = market.first_tick()
        if entry_tick is None or open_tick is None:
            continue
        picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, THRESHOLD, MODE)
        if picked is None:
            continue
        side, _ = picked
        ask = entry_tick.ask(side)
        bid = entry_tick.bid(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        shares = SIZE / ask
        cost = entry_tick.cost_to_buy(side, shares)
        if cost is None:
            continue
        entry_price = cost / shares

        spot_open = spot_at(index, market.start_ts)
        # spot read at several lags, to detect lookahead
        moves = {}
        for lag in LAG_CHOICES:
            later = spot_at(index, entry_tick.ts - lag)
            moves[lag] = None if (not spot_open or not later) else abs(later / spot_open - 1.0)

        won = (side == "UP") == market.up_won
        fee = shares * (FEE * entry_price * (1 - entry_price))
        rows.append({
            "day": market.day,
            "spot_move": moves.get(0.0),
            "moves_by_lag": moves,
            "spread": None if (bid is None) else ask - bid,
            "entry": entry_price,
            "won": won,
            "per_share": ((shares if won else 0.0) - shares * entry_price - fee) / shares,
        })
    return rows


def report(rows: list[dict]) -> None:
    days = sorted({r["day"] for r in rows})
    print(f"trades {len(rows):,} over {len(days)} days: "
          f"{days[0]} .. {days[-1]}")

    def stats(subset: list[dict]) -> tuple[float, float, int, int]:
        if not subset:
            return 0.0, 0.0, 0, 0
        per_day = defaultdict(list)
        for r in subset:
            per_day[r["day"]].append(r["per_share"])
        pooled = statistics.fmean(r["per_share"] for r in subset)
        dmean = statistics.fmean(statistics.fmean(v) for v in per_day.values())
        return pooled, dmean, len(subset), sum(1 for v in per_day.values()
                                               if statistics.fmean(v) > 0)

    base_pooled, base_dmean, base_n, base_pos = stats(rows)
    print(f"\nBASELINE: n={base_n:,}  pooled {base_pooled:+.4f}/share  "
          f"day-mean {base_dmean:+.4f}  days+ {base_pos}/{len(days)}")

    # ---------------------------------------------------------------- Q2 curve
    print()
    print("=" * 100)
    print("Q2. IS THE RESPONSE MONOTONE IN THE SPOT-MOVE FLOOR?")
    print("=" * 100)
    print("  A monotone curve means a real effect; a bump means a fitted one.")
    print()
    print(f"  {'floor':>7}{'kept':>7}{'kept%':>8}{'win%':>8}{'entry':>8}"
          f"{'pooled':>10}{'day-mean':>11}{'days+':>9}")
    print("-" * 100)
    curve = []
    for bps in FLOORS:
        floor = bps / 10_000.0
        kept = [r for r in rows if r["spot_move"] is not None and r["spot_move"] >= floor]
        if len(kept) < 40:
            continue
        pooled, dmean, n, pos = stats(kept)
        curve.append((bps, pooled, dmean, n, pos / len(days)))
        print(f"  {bps:>5.0f}bp{n:>7,}{n/len(rows)*100:>7.1f}%"
              f"{sum(1 for r in kept if r['won'])/n*100:>8.1f}"
              f"{statistics.fmean(r['entry'] for r in kept):>8.3f}"
              f"{pooled:>+10.4f}{dmean:>+11.4f}{pos:>5}/{len(days)}")
    monotone = all(curve[i][1] <= curve[i + 1][1] + 1e-9
                   for i in range(len(curve) - 1)) if len(curve) > 1 else False
    print()
    print(f"  monotone increasing in pooled pnl/share: {monotone}")
    if not monotone:
        best = max(curve, key=lambda c: c[1])
        print(f"  best floor is an interior point: {best[0]:.0f} bps "
              f"({best[1]:+.4f}/share). Interior optima on a noisy curve are the")
        print("  classic overfit signature - the gain may not survive new data.")

    # ------------------------------------------------------------ Q3 redundancy
    print()
    print("=" * 100)
    print("Q3. IS IT REDUNDANT?  spot-move floor vs a plain spread cap")
    print("=" * 100)
    print("  If the spread cap does the same job, the spot feed is unnecessary.")
    print()
    print(f"  {'filter':<34}{'kept':>7}{'win%':>8}{'pooled':>10}"
          f"{'day-mean':>11}{'days+':>9}")
    print("-" * 100)
    spread_known = [r for r in rows if r["spread"] is not None]
    print(f"  {'none':<34}{len(rows):>7,}"
          f"{sum(1 for r in rows if r['won'])/len(rows)*100:>8.1f}"
          f"{base_pooled:>+10.4f}{base_dmean:>+11.4f}{base_pos:>5}/{len(days)}")
    for cap in SPREAD_CAPS[1:]:
        kept = [r for r in spread_known if r["spread"] <= cap / 100.0 + 1e-9]
        if len(kept) < 40:
            continue
        pooled, dmean, n, pos = stats(kept)
        print(f"  {'spread <= ' + format(cap, '.1f') + 'c':<34}{n:>7,}"
              f"{sum(1 for r in kept if r['won'])/n*100:>8.1f}"
              f"{pooled:>+10.4f}{dmean:>+11.4f}{pos:>5}/{len(days)}")
    for bps in (5.0, 8.0):
        floor = bps / 10_000.0
        kept = [r for r in rows if r["spot_move"] is not None and r["spot_move"] >= floor]
        if len(kept) < 40:
            continue
        pooled, dmean, n, pos = stats(kept)
        print(f"  {'spot >= ' + format(bps, '.0f') + 'bps':<34}{n:>7,}"
              f"{sum(1 for r in kept if r['won'])/n*100:>8.1f}"
              f"{pooled:>+10.4f}{dmean:>+11.4f}{pos:>5}/{len(days)}")
    # both together
    both = [r for r in spread_known
            if r["spot_move"] is not None and r["spot_move"] >= 0.0005
            and r["spread"] <= 0.02 + 1e-9]
    if len(both) >= 40:
        pooled, dmean, n, pos = stats(both)
        print(f"  {'spot >= 5bps AND spread <= 2.0c':<34}{n:>7,}"
              f"{sum(1 for r in both if r['won'])/n*100:>8.1f}"
              f"{pooled:>+10.4f}{dmean:>+11.4f}{pos:>5}/{len(days)}")

    # ------------------------------------------------------------- Q4 lookahead
    print()
    print("=" * 100)
    print("Q4. IS THE FILTER LOOKAHEAD-FREE?  delay the spot read and re-test")
    print("=" * 100)
    print("  The filter reads spot at the decision timestamp. If delaying that read")
    print("  destroys the improvement, the filter was reading the future.")
    print()
    print(f"  {'spot read lag':>16}{'kept':>7}{'win%':>8}{'pooled':>10}"
          f"{'day-mean':>11}{'days+':>9}")
    print("-" * 100)
    for lag in LAG_CHOICES:
        kept = [r for r in rows
                if r["moves_by_lag"].get(lag) is not None
                and r["moves_by_lag"][lag] >= 0.0005]
        if len(kept) < 40:
            continue
        pooled, dmean, n, pos = stats(kept)
        print(f"  {format(lag, '.1f') + 's':>16}{n:>7,}"
              f"{sum(1 for r in kept if r['won'])/n*100:>8.1f}"
              f"{pooled:>+10.4f}{dmean:>+11.4f}{pos:>5}/{len(days)}")

    # ---------------------------------------------------------------- verdict
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  The decision is made on the numbers above, not on the idea. See the")
    print("  printed flags: monotonicity, whether the spread cap already covers it,")
    print("  and whether the gain survives a delayed spot read.")


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(root, "micro_v3"))
    args = parser.parse_args(argv)
    rows = collect(args.root)
    if not rows:
        print("no trades collected")
        return 1
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

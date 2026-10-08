"""Out-of-sample test: freeze parameters on the early period, predict the late one.

    python tools_v3/oos_v4.py

Everything so far selected its parameters using all the available days, which is
exactly how a backtest flatters itself. This tool does the honest thing:

* **IN-SAMPLE**  = the first 14 days (2026-03-27 .. 2026-04-26). Parameters are
  chosen here: entry checkpoint, move threshold, spot-move floor.
* **OUT-OF-SAMPLE** = every day after that (May). The frozen parameters are applied
  with no adjustment.

If the edge is real it should survive. If it was fitted to those 14 days it will
not. Either answer is useful; only one of them can be discovered this way.

The sweep is deliberately small. Searching a wide grid on the in-sample period and
then reporting the best out-of-sample cell would reintroduce the same bias through
the back door, so the in-sample choice is a single grid and the out-of-sample run
is one evaluation per cell, reported in full rather than cherry-picked.
"""
from __future__ import annotations

import argparse
import bisect
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v4 as B  # noqa: E402
from engine_v3 import hf  # noqa: E402

#: the split. Anything before SPLIT is in-sample; on/after is out-of-sample.
SPLIT_DATE = "2026-04-27"
#: candidate entry checkpoints (seconds before close)
CHECKPOINTS = (240.0, 120.0)
#: candidate move thresholds
THRESHOLDS = (0.15, 0.20)
#: candidate spot-move floors in bps (0 = filter off)
FLOORS = (0.0, 4.5)
SIZE = 10.0
MODE = "follow"


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
    spot = load_spot(root)
    resolutions = hf.load_resolutions(root)
    rows: list[dict] = []
    for market in hf.iter_markets(root, horizons=["5m"], resolutions=resolutions):
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        open_tick = market.first_tick()
        if open_tick is None:
            continue
        spot_open = spot_at(index, market.start_ts)
        for checkpoint in CHECKPOINTS:
            entry_tick = market.tick_at_t_minus(checkpoint, tolerance_s=1.5)
            if entry_tick is None:
                continue
            picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, 0.0, MODE)
            if picked is None:
                continue
            side, move = picked
            ask = entry_tick.ask(side)
            if ask is None or not 0.01 < ask < 0.99:
                continue
            shares = SIZE / ask
            cost = entry_tick.cost_to_buy(side, shares)
            if cost is None:
                continue
            entry_price = cost / shares
            won = (side == "UP") == market.up_won
            fee = shares * (0.07 * entry_price * (1 - entry_price))
            spot_entry = spot_at(index, entry_tick.ts)
            rows.append({
                "day": market.day,
                "checkpoint": checkpoint,
                "abs_move": abs(move),
                "spot_move": (None if (not spot_open or not spot_entry)
                              else abs(spot_entry / spot_open - 1.0)),
                "entry": entry_price,
                "won": won,
                "per_share": ((shares if won else 0.0) - shares * entry_price - fee) / shares,
            })
    return rows


def evaluate(rows: list[dict], checkpoint: float, threshold: float,
             floor: float) -> dict:
    kept = [r for r in rows
            if r["checkpoint"] == checkpoint
            and r["abs_move"] >= threshold
            and (floor <= 0 or (r["spot_move"] is not None and r["spot_move"] >= floor / 10_000.0))]
    if not kept:
        return {"n": 0, "per_share": 0.0, "win": 0.0, "pos_days": 0, "days": 0,
                "entry": 0.0, "per_day": 0.0}
    per_day = defaultdict(list)
    for r in kept:
        per_day[r["day"]].append(r["per_share"])
    return {
        "n": len(kept),
        "per_share": statistics.fmean(r["per_share"] for r in kept),
        "day_mean": statistics.fmean(statistics.fmean(v) for v in per_day.values()),
        "win": sum(1 for r in kept if r["won"]) / len(kept),
        "entry": statistics.fmean(r["entry"] for r in kept),
        "pos_days": sum(1 for v in per_day.values() if statistics.fmean(v) > 0),
        "days": len(per_day),
        "per_day": len(kept) / len(per_day) if per_day else 0.0,
    }


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(root, "micro_v3"))
    parser.add_argument("--split", default=SPLIT_DATE)
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no trades collected")
        return 1

    in_sample = [r for r in rows if r["day"] < args.split]
    out_sample = [r for r in rows if r["day"] >= args.split]
    days_in = sorted({r["day"] for r in in_sample})
    days_out = sorted({r["day"] for r in out_sample})

    print("=" * 104)
    print("OUT-OF-SAMPLE TEST - parameters frozen on the early period")
    print("=" * 104)
    print(f"in-sample     : {len(days_in)} days, {days_in[0]} .. {days_in[-1]}")
    print(f"out-of-sample : {len(days_out)} days, "
          + (f"{days_out[0]} .. {days_out[-1]}" if days_out else "NONE AVAILABLE"))
    if not days_out:
        print("\nNo out-of-sample days found. Cannot run the test.")
        return 1

    # ---------------------------------------------------- choose in-sample
    print()
    print("=" * 104)
    print("STEP 1 - IN-SAMPLE SELECTION (this is where fitting happens)")
    print("=" * 104)
    print(f"  {'T-k':>6}{'thr':>6}{'floor':>8}{'n':>7}{'/day':>7}{'win%':>8}"
          f"{'entry':>8}{'per-share':>11}{'days+':>9}")
    print("-" * 104)
    grid = []
    for checkpoint in CHECKPOINTS:
        for threshold in THRESHOLDS:
            for floor in FLOORS:
                res = evaluate(in_sample, checkpoint, threshold, floor)
                if res["n"] < 50:
                    continue
                grid.append((checkpoint, threshold, floor, res))
                print(f"  {checkpoint:>6.0f}{threshold:>6.2f}{floor:>8.1f}{res['n']:>7,}"
                      f"{res['per_day']:>7.1f}{res['win']*100:>8.1f}{res['entry']:>8.3f}"
                      f"{res['per_share']:>+11.4f}{res['pos_days']:>5}/{res['days']}")
    if not grid:
        print("  nothing met the minimum trade count")
        return 1
    best = max(grid, key=lambda g: g[3]["per_share"])
    cp, thr, floor = best[0], best[1], best[2]
    print()
    print(f"  CHOSEN on in-sample only: T-{cp:.0f}, threshold {thr:.2f}, "
          f"spot floor {floor:.1f} bps")
    print(f"  in-sample result: {best[3]['per_share']:+.4f}/share, "
          f"{best[3]['pos_days']}/{best[3]['days']} days positive")

    # ------------------------------------------------------ apply unchanged
    print()
    print("=" * 104)
    print("STEP 2 - OUT-OF-SAMPLE (frozen parameters, no adjustment)")
    print("=" * 104)
    print(f"  {'T-k':>6}{'thr':>6}{'floor':>8}{'n':>7}{'/day':>7}{'win%':>8}"
          f"{'entry':>8}{'per-share':>11}{'days+':>9}{'vs in-sample':>14}")
    print("-" * 104)
    for checkpoint, threshold, f in ((cp, thr, floor),):
        res = evaluate(out_sample, checkpoint, threshold, f)
        if res["n"] == 0:
            print("  no out-of-sample trades")
            continue
        delta = res["per_share"] - best[3]["per_share"]
        print(f"  {checkpoint:>6.0f}{threshold:>6.2f}{f:>8.1f}{res['n']:>7,}"
              f"{res['per_day']:>7.1f}{res['win']*100:>8.1f}{res['entry']:>8.3f}"
              f"{res['per_share']:>+11.4f}{res['pos_days']:>5}/{res['days']}"
              f"{delta:>+14.4f}")

    # full grid out-of-sample, reported in full (no cherry-picking)
    print()
    print("  The same grid evaluated out-of-sample, for completeness:")
    print(f"  {'T-k':>6}{'thr':>6}{'floor':>8}{'n':>7}{'win%':>8}{'per-share':>11}{'days+':>9}")
    print("-" * 104)
    for checkpoint in CHECKPOINTS:
        for threshold in THRESHOLDS:
            for f in FLOORS:
                res = evaluate(out_sample, checkpoint, threshold, f)
                if res["n"] < 20:
                    continue
                print(f"  {checkpoint:>6.0f}{threshold:>6.2f}{f:>8.1f}{res['n']:>7,}"
                      f"{res['win']*100:>8.1f}{res['per_share']:>+11.4f}"
                      f"{res['pos_days']:>5}/{res['days']}")

    # ----------------------------------------------------- day-by-day detail
    print()
    print("=" * 104)
    print("STEP 3 - DAY BY DAY, FROZEN PARAMETERS")
    print("=" * 104)
    print(f"  {'day':<12}{'period':>10}{'n':>6}{'win%':>8}{'entry':>8}{'per-share':>11}")
    print("-" * 104)
    for day in days_in + days_out:
        subset = [r for r in rows if r["day"] == day and r["checkpoint"] == cp
                  and r["abs_move"] >= thr
                  and (floor <= 0 or (r["spot_move"] is not None
                                      and r["spot_move"] >= floor / 10_000.0))]
        if not subset:
            continue
        period = "in-sample" if day < args.split else "OUT"
        ps = statistics.fmean(r["per_share"] for r in subset)
        win = sum(1 for r in subset if r["won"]) / len(subset) * 100
        entry = statistics.fmean(r["entry"] for r in subset)
        print(f"  {day:<12}{period:>10}{len(subset):>6}{win:>8.1f}{entry:>8.3f}{ps:>+11.4f}")

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    ins = evaluate(in_sample, cp, thr, floor)
    oos = evaluate(out_sample, cp, thr, floor)
    print(f"  in-sample  : {ins['per_share']:+.4f}/share on {ins['n']:,} trades, "
          f"{ins['pos_days']}/{ins['days']} days positive")
    print(f"  out-of-sample: {oos['per_share']:+.4f}/share on {oos['n']:,} trades, "
          f"{oos['pos_days']}/{oos['days']} days positive")
    if oos["per_share"] > 0:
        print()
        print("  The frozen parameters stayed positive out of sample. That is the")
        print("  first genuine evidence in this whole project that the edge is not")
        print("  purely an artefact of the fitting period.")
    else:
        print()
        print("  The frozen parameters went NEGATIVE out of sample. On this evidence")
        print("  the in-sample edge did not transfer, and no amount of further")
        print("  parameter tuning on the same period should be trusted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the 1200-row parameter grid for the shadow (multi-account) engine.

Grid dimensions
---------------
time            entry checkpoint, seconds before window close
threshold       minimum spot move over the lookback, in PERCENT
tp / sl         take profit / stop loss, in probability points
min_gap         minimum seconds between entries
dynamic_tp_sl   when true, tp/sl are scaled by realised ATR instead of fixed
hedge_enabled   when true, a stop is answered by hedging instead of closing
hedge_ratio     fraction of the position to hedge (only used when hedging)

Why the grid was rebuilt
------------------------
The previous grid pinned ``time`` to a single value (15s), so it could not test
entry timing at all - the one dimension the research showed matters most
(direction accuracy rises monotonically with the lookback window and entry
timing). It also had no notion of hedging or dynamic exits. This version covers
all five original dimensions plus the two new ones.

    python scripts/make_param_grid.py                 # 1200 rows -> data/
    python scripts/make_param_grid.py --count 3000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: entry checkpoints. T-30 and later were unverifiable on the parquet history
#: (no prints), but they are included here so the live shadow run can measure
#: them - which is the whole point of running a shadow book.
TIMES = [5, 15, 30, 60, 90, 120, 180, 240]
#: spot-move thresholds in percent. 0.01% is deliberately far looser than the
#: 0.05% used before; the "too conservative, never fills" complaint is answered
#: partly here.
THRESHOLDS = [round(0.01 * i, 3) for i in range(1, 13)]          # 0.01 .. 0.12
TAKE_PROFITS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
STOP_LOSSES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
MIN_GAPS = [0, 15, 30, 60, 120]
DYNAMIC_FLAGS = [False, True]
HEDGE_OPTIONS = [(False, 0.0), (True, 0.3), (True, 0.5), (True, 0.75), (True, 1.0)]

FIELDNAMES = [
    "combo_id", "time", "threshold", "tp", "sl", "min_gap",
    "dynamic_tp_sl", "hedge_enabled", "hedge_ratio",
    "risk_reward", "source",
]


def full_grid() -> list[tuple]:
    return list(itertools.product(
        TIMES, THRESHOLDS, TAKE_PROFITS, STOP_LOSSES, MIN_GAPS,
        DYNAMIC_FLAGS, HEDGE_OPTIONS,
    ))


def stratified_sample(grid: list[tuple], count: int, rng: random.Random) -> list[tuple]:
    """Sample so every (time, threshold) cell is represented.

    A blind random draw over 67k combinations would leave cells empty and make the
    per-parameter comparison meaningless; stratifying keeps the sweep balanced.
    """
    if count >= len(grid):
        return list(grid)
    buckets: dict[tuple, list[tuple]] = {}
    for row in grid:
        buckets.setdefault((row[0], row[1]), []).append(row)

    keys = sorted(buckets)
    picked: list[tuple] = []
    per_bucket = max(1, count // len(keys))
    for key in keys:
        pool = buckets[key]
        picked.extend(rng.sample(pool, min(per_bucket, len(pool))))
    if len(picked) > count:
        picked = rng.sample(picked, count)

    # top up with unique leftovers so the output is exactly `count` rows
    seen = set(picked)
    leftovers = [row for row in grid if row not in seen]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, count - len(picked))])
    rng.shuffle(picked)
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "param_combinations.csv"))
    args = ap.parse_args(argv)

    grid = full_grid()
    rng = random.Random(args.seed)
    rows = stratified_sample(grid, args.count, rng)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDNAMES)
        for index, (time_s, thr, tp, sl, gap, dyn, (hedge_on, ratio)) in enumerate(
            sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3], r[4])), start=1
        ):
            writer.writerow([
                f"p{index:05d}", time_s, thr, tp, sl, gap,
                "true" if dyn else "false",
                "true" if hedge_on else "false",
                ratio,
                round(tp / sl, 4) if sl else "",
                "grid_v2",
            ])

    hedged = sum(1 for r in rows if r[6][0])
    dynamic = sum(1 for r in rows if r[5])
    print(f"full grid        : {len(grid):,} combinations")
    print(f"written          : {len(rows):,} rows")
    print(f"  with hedging   : {hedged:,} ({hedged / len(rows) * 100:.0f}%)")
    print(f"  dynamic TP/SL  : {dynamic:,} ({dynamic / len(rows) * 100:.0f}%)")
    print(f"  entry times    : {TIMES}")
    print(f"  thresholds     : {THRESHOLDS}")
    print(f"file             : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

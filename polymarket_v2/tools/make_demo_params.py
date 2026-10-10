"""Build a small parameter file pinned to the offset a dataset actually carries.

Why this exists
---------------
The 1200-combination grid in ``data/param_combinations.csv`` sweeps T-30s..T-240s.
The public dataset currently in ``../data`` only carries a real price at T-15s
(near close), so *no* row of the grid can trade against it. Rather than mutate the
user's grid, this generator emits a separate file for the offsets that do have
prices, so the ranking path can still be exercised end to end.

    python tools/make_demo_params.py --offsets 15 --count 1200
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offsets", default="15", help="T-minus offsets that have prices")
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "output", "demo_param_combinations.csv"),
    )
    args = ap.parse_args()

    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    thresholds = [round(0.01 * i, 4) for i in range(1, 11)]
    tps = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    sls = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    gaps = [0, 30, 60, 90, 120]

    full = list(itertools.product(offsets, thresholds, tps, sls, gaps))
    rng = random.Random(args.seed)
    if args.count >= len(full):
        sample = full
    else:
        # stratified on (offset, threshold) so every cell is represented
        buckets: dict[tuple, list] = {}
        for row in full:
            buckets.setdefault((row[0], row[1]), []).append(row)
        sample = []
        per_bucket = max(1, args.count // len(buckets))
        for key in sorted(buckets):
            pool = buckets[key]
            sample.extend(rng.sample(pool, min(per_bucket, len(pool))))
        rng.shuffle(sample)
        sample = sample[: args.count]

    sample.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["combo_id", "time", "threshold", "tp", "sl", "min_gap", "risk_reward", "source"])
        for index, (time_s, thr, tp, sl, gap) in enumerate(sample, start=1):
            writer.writerow(
                [f"d{index:05d}", time_s, thr, tp, sl, gap, round(tp / sl, 4), "demo_offsets"]
            )

    print(f"grid size        : {len(full)}")
    print(f"combinations out : {len(sample)}")
    print(f"offsets          : {offsets}")
    print(f"file             : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

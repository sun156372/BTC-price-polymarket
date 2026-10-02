"""Generate 1000+ parameter combinations for the BTC 5-minute simulation.

Dimensions (per the spec):
  --time       observation time before window close, in seconds
  --threshold  BTC price offset threshold, in percent
  --tp         take-profit offset
  --sl         stop-loss offset
  --min-gap    minimum interval between entries, in seconds

Outputs (into --outdir, default ./params):
  param_combinations.csv   every combination, one row each
  param_combinations.json  same content as JSON
  commands.txt             one ready-to-paste command line per combination
  commands.sh              bash array / loop form for batch execution
  grid_definition.json     the full grid + sampling metadata (reproducible)

The sampling is a stratified random sample so every observation time and every
threshold value is represented, instead of a blind random draw.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from itertools import product

# ---------------------------------------------------------------- dimensions
OBSERVATION_TIMES = list(range(240, 29, -30))  # 240,210,...,30  -> 8 values
EXTRA_OBSERVATION_TIMES = [300, 270, 15]       # optional wider net
THRESHOLDS = [round(0.01 * i, 4) for i in range(1, 11)]  # 0.01..0.10 -> 10
TAKE_PROFITS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]      # 6
STOP_LOSSES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]       # 6
MIN_GAPS = [0, 30, 60, 90, 120]                          # 5

FIELDNAMES = [
    "combo_id",
    "time",
    "threshold",
    "tp",
    "sl",
    "min_gap",
    "risk_reward",
    "source",
]


def build_grid(times: list[int]) -> list[dict]:
    rows = []
    for time_s, threshold, tp, sl, gap in product(
        times, THRESHOLDS, TAKE_PROFITS, STOP_LOSSES, MIN_GAPS
    ):
        rows.append(
            {
                "time": time_s,
                "threshold": threshold,
                "tp": tp,
                "sl": sl,
                "min_gap": gap,
            }
        )
    return rows


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Random sample that guarantees coverage of every (time, threshold) cell."""
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["time"], row["threshold"]), []).append(row)

    keys = sorted(buckets)
    picked: list[dict] = []
    per_bucket = max(1, n // len(keys))
    for key in keys:
        pool = buckets[key]
        take = min(per_bucket, len(pool))
        picked.extend(rng.sample(pool, take))

    # top up to exactly n with unique leftovers
    seen = {(r["time"], r["threshold"], r["tp"], r["sl"], r["min_gap"]) for r in picked}
    leftovers = [
        r
        for r in rows
        if (r["time"], r["threshold"], r["tp"], r["sl"], r["min_gap"]) not in seen
    ]
    rng.shuffle(leftovers)
    need = max(0, n - len(picked))
    picked.extend(leftovers[:need])
    rng.shuffle(picked)
    return picked


def fmt(value: float) -> str:
    """0.05 -> '0.05', 0.10 -> '0.1' is ugly; keep the grid's own text."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def command_line(row: dict) -> str:
    return (
        f"--time {row['time']} --threshold {fmt(row['threshold'])} "
        f"--tp {fmt(row['tp'])} --sl {fmt(row['sl'])} --min-gap {row['min_gap']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate parameter combinations.")
    ap.add_argument("--count", type=int, default=1200, help="how many combinations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params"))
    ap.add_argument("--include-full", action="store_true", help="also emit the raw full grid as CSV")
    ap.add_argument("--prefix", default="", help="command prefix, e.g. 'python sim.py' or './sim'")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    times = list(OBSERVATION_TIMES)
    full_grid = build_grid(times)
    sample = stratified_sample(full_grid, args.count, rng)

    rows = []
    for i, row in enumerate(sample, start=1):
        rr = round(row["tp"] / row["sl"], 4) if row["sl"] else ""
        rows.append(
            {
                "combo_id": f"c{i:05d}",
                "time": row["time"],
                "threshold": row["threshold"],
                "tp": row["tp"],
                "sl": row["sl"],
                "min_gap": row["min_gap"],
                "risk_reward": rr,
                "source": "stratified_sample",
            }
        )

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "param_combinations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.outdir, "param_combinations.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "count": len(rows),
                "seed": args.seed,
                "dimensions": {
                    "time": times,
                    "threshold": THRESHOLDS,
                    "tp": TAKE_PROFITS,
                    "sl": STOP_LOSSES,
                    "min_gap": MIN_GAPS,
                },
                "combinations": rows,
            },
            fh,
            indent=2,
        )

    prefix = (args.prefix + " ") if args.prefix else ""
    txt_path = os.path.join(args.outdir, "commands.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{prefix}{command_line(row)}\n")

    sh_path = os.path.join(args.outdir, "commands.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("# Generated parameter sweep. Run one line at a time, or loop them.\n")
        fh.write(f"# {len(rows)} combinations\n\n")
        fh.write(f"CMD={json.dumps(args.prefix or './sim')}\n\n")
        fh.write("run() {\n")
        fh.write('  echo "==> $CMD $*"\n')
        fh.write('  $CMD "$@"\n')
        fh.write("}\n\n")
        fh.write("while read -r line; do\n")
        fh.write('  [ -z "$line" ] && continue\n')
        fh.write("  run $line\n")
        fh.write(f"done < <(grep -v '^#' {os.path.basename(txt_path)})\n")
        fh.write("\n# --- explicit list (same order as commands.txt) ---\n")
        for row in rows:
            fh.write(f"# run {command_line(row)}\n")

    grid_path = os.path.join(args.outdir, "grid_definition.json")
    with open(grid_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "observation_times_used": times,
                "observation_times_extra_optional": EXTRA_OBSERVATION_TIMES,
                "thresholds_pct": THRESHOLDS,
                "take_profits": TAKE_PROFITS,
                "stop_losses": STOP_LOSSES,
                "min_gaps_s": MIN_GAPS,
                "full_grid_size": len(full_grid),
                "sampled": len(rows),
                "seed": args.seed,
                "cli_flags": {
                    "--time": "seconds before close when the entry is evaluated",
                    "--threshold": "BTC price offset vs the window open, in percent",
                    "--tp": "take-profit offset",
                    "--sl": "stop-loss offset",
                    "--min-gap": "minimum seconds between entries",
                },
            },
            fh,
            indent=2,
        )

    if args.include_full:
        full_path = os.path.join(args.outdir, "param_combinations_full_grid.csv")
        with open(full_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            for i, row in enumerate(full_grid, start=1):
                rr = round(row["tp"] / row["sl"], 4) if row["sl"] else ""
                writer.writerow(
                    {
                        "combo_id": f"g{i:06d}",
                        "time": row["time"],
                        "threshold": row["threshold"],
                        "tp": row["tp"],
                        "sl": row["sl"],
                        "min_gap": row["min_gap"],
                        "risk_reward": rr,
                        "source": "full_grid",
                    }
                )

    print(f"combinations generated : {len(rows)}")
    print(f"full grid size         : {len(full_grid)}")
    print(f"CSV                    : {csv_path}")
    print(f"JSON                   : {json_path}")
    print(f"commands (per combo)   : {txt_path}")
    print(f"bash runner            : {sh_path}")
    print(f"grid definition        : {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

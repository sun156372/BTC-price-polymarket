"""Turn the parameter combination files into ready-to-paste command lines.

Reads params/param_combinations.csv (or .json) and writes:

  commands.txt      one command line per combination
  commands_args.txt just the arguments (no program name)
  commands.csv      combo_id + the argument string + the parsed values
  commands.sh       bash runner that loops over every combination
  commands.jsonl    one JSON object per line (easy to feed to xargs / a queue)

Usage
-----
  python scripts/build_commands.py                          # default prefix
  python scripts/build_commands.py --prefix "./sim"
  python scripts/build_commands.py --prefix "python sim.py" --flag-style long
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(value: str) -> str:
    """0.1000 -> 0.1 ; keep integers intact."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def load_rows(params_dir: str) -> list[dict]:
    csv_path = os.path.join(params_dir, "param_combinations.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    json_path = os.path.join(params_dir, "param_combinations.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data["combinations"] if isinstance(data, dict) else data
    raise SystemExit(f"no parameter file found in {params_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CLI command lists from the parameter grid.")
    ap.add_argument("--params-dir", default=os.path.join(ROOT, "params"))
    ap.add_argument("--outdir", default="", help="defaults to the params dir")
    ap.add_argument("--prefix", default="./sim", help="program/command that runs one simulation")
    ap.add_argument("--time-flag", default="--time")
    ap.add_argument("--threshold-flag", default="--threshold")
    ap.add_argument("--tp-flag", default="--tp")
    ap.add_argument("--sl-flag", default="--sl")
    ap.add_argument("--min-gap-flag", default="--min-gap")
    ap.add_argument("--symbol", default="", help="optional extra flag, e.g. --symbol BTCUSDT")
    ap.add_argument("--extra", default="", help="extra raw args appended to every line")
    args = ap.parse_args()

    outdir = args.outdir or args.params_dir
    os.makedirs(outdir, exist_ok=True)
    rows = load_rows(args.params_dir)

    def args_for(row: dict) -> list[str]:
        pieces = [
            args.time_flag, str(row["time"]),
            args.threshold_flag, fmt(row["threshold"]),
            args.tp_flag, fmt(row["tp"]),
            args.sl_flag, fmt(row["sl"]),
            args.min_gap_flag, str(row["min_gap"]),
        ]
        if args.symbol:
            pieces += ["--symbol", args.symbol]
        if args.extra:
            pieces += shlex.split(args.extra)
        return pieces

    commands = []
    for row in rows:
        pieces = args_for(row)
        argline = " ".join(pieces)
        full = f"{args.prefix} {argline}".strip() if args.prefix else argline
        commands.append({"combo_id": row.get("combo_id", ""), "row": row, "args": argline, "full": full})

    txt_path = os.path.join(outdir, "commands.txt")
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
        for item in commands:
            fh.write(item["full"] + "\n")

    args_path = os.path.join(outdir, "commands_args.txt")
    with open(args_path, "w", encoding="utf-8", newline="\n") as fh:
        for item in commands:
            fh.write(item["args"] + "\n")

    csv_path = os.path.join(outdir, "commands.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["combo_id", "time", "threshold", "tp", "sl", "min_gap", "args", "command"])
        for item in commands:
            row = item["row"]
            writer.writerow([
                item["combo_id"], row["time"], row["threshold"], row["tp"],
                row["sl"], row["min_gap"], item["args"], item["full"],
            ])

    jsonl_path = os.path.join(outdir, "commands.jsonl")
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        for item in commands:
            fh.write(json.dumps({
                "combo_id": item["combo_id"],
                "time": int(item["row"]["time"]),
                "threshold": float(item["row"]["threshold"]),
                "tp": float(item["row"]["tp"]),
                "sl": float(item["row"]["sl"]),
                "min_gap": int(item["row"]["min_gap"]),
                "args": item["args"],
                "command": item["full"],
            }) + "\n")

    sh_path = os.path.join(outdir, "commands.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write(f"# {len(commands)} combinations\n")
        fh.write("# Runs every parameter set in sequence. Edit CMD to match your simulator.\n\n")
        fh.write(f"CMD={shlex.quote(args.prefix or './sim')}\n")
        fh.write('LOG="${LOG:-sweep_results.log}"\n\n')
        fh.write('while IFS= read -r line; do\n')
        fh.write('  [ -z "$line" ] && continue\n')
        fh.write('  echo "==> $CMD $line" | tee -a "$LOG"\n')
        fh.write('  $CMD $line 2>&1 | tee -a "$LOG"\n')
        fh.write(f'done < "{os.path.basename(args_path)}"\n')

    print(f"combinations           : {len(commands)}")
    print(f"full commands          : {txt_path}")
    print(f"arguments only         : {args_path}")
    print(f"CSV with parsed values : {csv_path}")
    print(f"JSONL                  : {jsonl_path}")
    print(f"bash loop runner       : {sh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

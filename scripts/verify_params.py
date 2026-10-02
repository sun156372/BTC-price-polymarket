"""Validate the generated parameter and command files."""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "params")
checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))


with open(os.path.join(PARAMS, "param_combinations.csv"), encoding="utf-8") as fh:
    combos = list(csv.DictReader(fh))
check("param CSV has 1200 rows", len(combos) == 1200, str(len(combos)))

keys = {(c["time"], c["threshold"], c["tp"], c["sl"], c["min_gap"]) for c in combos}
check("all combinations unique", len(keys) == len(combos), f"{len(keys)} unique of {len(combos)}")

check("time values are on the 30s grid",
      all(int(c["time"]) % 30 == 0 and 30 <= int(c["time"]) <= 240 for c in combos),
      str(sorted({int(c['time']) for c in combos})))
check("threshold values within 0.01-0.10",
      all(0.01 <= float(c["threshold"]) <= 0.10 for c in combos))
check("tp/sl values positive",
      all(float(c["tp"]) > 0 and float(c["sl"]) > 0 for c in combos))

# every observation time and every threshold must appear (stratified coverage)
times = {int(c["time"]) for c in combos}
thresholds = {float(c["threshold"]) for c in combos}
check("all 8 observation times covered", times == set(range(30, 241, 30)), str(sorted(times)))
check("all 10 thresholds covered",
      thresholds == {round(0.01 * i, 4) for i in range(1, 11)}, str(sorted(thresholds)))

with open(os.path.join(PARAMS, "commands.txt"), encoding="utf-8") as fh:
    lines = [line.strip() for line in fh if line.strip()]
check("commands.txt has 1200 lines", len(lines) == 1200, str(len(lines)))

pattern = re.compile(
    r"^\./sim --time \d+ --threshold [\d.]+ --tp [\d.]+ --sl [\d.]+ --min-gap \d+$"
)
bad = [line for line in lines if not pattern.match(line)]
check("every command line matches the expected flag format", not bad,
      bad[0] if bad else "")

check("commands.txt matches the CSV rows", len(lines) == len(combos))

with open(os.path.join(PARAMS, "commands.jsonl"), encoding="utf-8") as fh:
    jsonl = [json.loads(line) for line in fh if line.strip()]
check("commands.jsonl has 1200 records", len(jsonl) == 1200, str(len(jsonl)))
check("jsonl records carry parsed numeric params",
      all(isinstance(r["time"], int) and isinstance(r["threshold"], float) for r in jsonl))

with open(os.path.join(PARAMS, "grid_definition.json"), encoding="utf-8") as fh:
    grid = json.load(fh)
check("grid definition reports the full grid size", grid["full_grid_size"] == 14400,
      str(grid["full_grid_size"]))

with open(os.path.join(PARAMS, "commands.sh"), encoding="utf-8") as fh:
    sh = fh.read()
check("bash runner reads commands_args.txt", "commands_args.txt" in sh)

with open(os.path.join(PARAMS, "param_combinations_full_grid.csv"), encoding="utf-8") as fh:
    full = list(csv.DictReader(fh))
check("optional full grid CSV has 14400 rows", len(full) == 14400, str(len(full)))

print("sample lines from commands.txt:")
for line in lines[:5]:
    print("  " + line)
print()

failed = 0
for name, ok, detail in checks:
    if not ok:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))

print()
print(f"checks: {len(checks) - failed}/{len(checks)} passed")
sys.exit(0 if failed == 0 else 1)

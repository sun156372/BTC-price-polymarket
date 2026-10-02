#!/usr/bin/env bash
# 1200 combinations
# Runs every parameter set in sequence. Edit CMD to match your simulator.

CMD=./sim
LOG="${LOG:-sweep_results.log}"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "==> $CMD $line" | tee -a "$LOG"
  $CMD $line 2>&1 | tee -a "$LOG"
done < "commands_args.txt"

#!/usr/bin/env bash
# One-command setup + scrape for the Polymarket BTC 5-minute dataset.
# Safe to re-run: responses are cached in ./.cache, so an interrupted scrape resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

COUNT="${COUNT:-500}"      # how many markets
WORKERS="${WORKERS:-6}"    # parallel market workers
RPS="${RPS:-6}"            # requests per second (lower this if you see HTTP 429)

echo "== python =="
"$PY" --version

echo
echo "== dependencies =="
if "$PY" -c "import requests" 2>/dev/null; then
  echo "requests already installed"
else
  echo "installing requests (falls back to stdlib urllib if this fails)"
  "$PY" -m pip install --user requests || echo "WARN: pip install failed; the script will use urllib"
fi

echo
echo "== scrape: $COUNT markets =="
"$PY" scripts/scrape_btc5m.py --count "$COUNT" --workers "$WORKERS" --rps "$RPS"

echo
echo "== done =="
ls -la data/

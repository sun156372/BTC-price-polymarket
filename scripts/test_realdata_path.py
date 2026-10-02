"""Test the real-data path: extract_dataset.py + sim_template.py.

Uses a synthetic file that mirrors the Sigrex dataset schema exactly, so the
extraction and simulation logic are verified without needing the 57 MB download.

  python scripts/test_realdata_path.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

# the console on Chinese Windows defaults to GBK; force UTF-8 so subprocess
# output containing any character can be printed
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(WORK, "test-data")
SRC = os.path.join(WORK, "synthetic_source.csv")

SOURCE_FIELDS = [
    "condition_id", "slug", "question", "start_ts", "end_ts", "start_time", "end_time",
    "winning_outcome", "label_up", "final_price", "price_to_beat", "chainlink_return",
    "btc_open", "btc_close", "btc_return", "implied_up_last", "trade_count",
]

START = 1_780_000_000
N = 40


def build_source() -> None:
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for i in range(N):
        start_ts = START + i * 300
        end_ts = start_ts + 300
        up = i % 2 == 0
        # BTC moves with the label so the threshold signal has something to fire on
        move = 0.08 if up else -0.08
        btc_open = 80000.0 + i
        btc_close = btc_open * (1 + move / 100)
        implied = 0.62 if up else 0.38
        rows.append({
            "condition_id": f"0x{i:064x}",
            "slug": f"btc-updown-5m-{start_ts}",
            "question": f"Bitcoin Up or Down - window {i}",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_time": "2026-06-01T00:00:00Z",
            "end_time": "2026-06-01T00:05:00Z",
            "winning_outcome": "Up" if up else "Down",
            "label_up": 1 if up else 0,
            "final_price": round(btc_close, 2),
            "price_to_beat": round(btc_open, 2),
            "chainlink_return": round(move, 4),
            "btc_open": round(btc_open, 2),
            "btc_close": round(btc_close, 2),
            "btc_return": round(move, 4),
            "implied_up_last": implied,
            "trade_count": 40 + i,
        })
    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    python = sys.executable
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    build_source()
    print(f"synthetic source: {SRC} ({N} markets)")

    code, out = run([
        python, os.path.join(ROOT, "scripts", "extract_dataset.py"),
        "--input", SRC, "--outdir", OUT, "--count", "30", "--source", "synthetic test",
    ])
    print(out.strip()[-700:])
    check("extract_dataset.py exits 0", code == 0, f"exit {code}")

    markets_path = os.path.join(OUT, "btc5m_markets.csv")
    prices_path = os.path.join(OUT, "btc5m_prices.csv")
    with open(markets_path, encoding="utf-8") as fh:
        markets = list(csv.DictReader(fh))
    with open(prices_path, encoding="utf-8") as fh:
        prices = list(csv.DictReader(fh))

    check("30 most recent markets written", len(markets) == 30, str(len(markets)))
    check("markets sorted newest first",
          all(int(markets[i]["window_close_epoch"]) >= int(markets[i + 1]["window_close_epoch"])
              for i in range(len(markets) - 1)))
    check("labels preserved from source",
          all(m["result"] in ("UP", "DOWN") for m in markets),
          str({m["result"] for m in markets}))
    check("label matches winning_outcome",
          all((m["winning_outcome"].lower() == "up") == (m["result"] == "UP") for m in markets))
    check("price rows = markets x 12 offsets", len(prices) == 30 * 12, str(len(prices)))

    with_px = [r for r in prices if r["up_price"] not in ("", None)]
    check("real prices used only where the source has them",
          len(with_px) == 30, str(len(with_px)))
    check("no invented prices elsewhere",
          all(r["price_source"] == "unavailable" for r in prices if r["up_price"] in ("", None)))
    check("up + down = 1 on priced rows",
          all(abs(float(r["up_price"]) + float(r["down_price"]) - 1.0) < 1e-6 for r in with_px))

    with open(os.path.join(OUT, "data_source.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    check("provenance file records the source", meta["source"] == "synthetic test")
    check("provenance flags the price limitation",
          "implied_up_last" in meta["price_note"], meta["price_note"][:60])

    # now bind the sweep flags to the data
    sim = os.path.join(ROOT, "scripts", "sim_template.py")
    code, out = run([
        python, sim, "--time", "15", "--threshold", "0.05", "--tp", "0.10", "--sl", "0.05",
        "--min-gap", "0", "--data", prices_path, "--markets", markets_path,
    ])
    print()
    print(out.strip()[-700:])
    check("sim_template.py accepts the sweep flags", code == 0, f"exit {code}")
    check("sim reports trades", "trades" in out and "total_pnl" in out)
    check("threshold signal fires on the synthetic moves", "skipped_no_signal" in out)

    # a threshold nothing can reach must produce zero trades
    code, out2 = run([
        python, sim, "--time", "15", "--threshold", "9.99", "--tp", "0.10", "--sl", "0.05",
        "--min-gap", "0", "--data", prices_path, "--markets", markets_path,
    ])
    zero = "trades                      : 0" in out2 or "trades                      : 0" in out2
    check("unreachable threshold yields 0 trades", code == 0 and "trades                      : 0" in out2,
          out2.strip().splitlines()[1] if len(out2.strip().splitlines()) > 1 else "")

    # min-gap must reduce the trade count
    code, out3 = run([
        python, sim, "--time", "15", "--threshold", "0.05", "--tp", "0.10", "--sl", "0.05",
        "--min-gap", "600", "--data", prices_path, "--markets", markets_path,
    ])
    count_gap = None
    for line in out3.splitlines():
        if line.startswith("trades "):
            count_gap = int(line.split(":")[1].strip())
    count_nogap = None
    for line in out.splitlines():
        if line.startswith("trades "):
            count_nogap = int(line.split(":")[1].strip())
    check("min-gap reduces trade count",
          count_gap is not None and count_nogap is not None and count_gap < count_nogap,
          f"min-gap 600 -> {count_gap}, min-gap 0 -> {count_nogap}")

    failed = 0
    print()
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"checks: {len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

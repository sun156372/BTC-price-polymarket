"""Validate the generated dataset in data/ (works for mirror-sourced and scraped data).

  python scripts/verify_dataset.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load(name: str):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    markets = load("btc5m_markets.csv")
    prices = load("btc5m_prices.csv")
    if markets is None or prices is None:
        print("data/btc5m_markets.csv or data/btc5m_prices.csv missing - run extract_dataset.py first")
        return 2

    check("market rows >= 500", len(markets) >= 500, str(len(markets)))

    results = {m["result"] for m in markets}
    check("results are UP/DOWN", results <= {"UP", "DOWN"}, str(results))
    balance = {r: sum(1 for m in markets if m["result"] == r) for r in ("UP", "DOWN")}
    up_share = balance["UP"] / len(markets) if markets else 0
    check("label balance is plausible (0.35-0.65 Up)",
          0.35 <= up_share <= 0.65, f"Up share {up_share:.3f}")

    dupes = len(markets) - len({m["market_id"] for m in markets})
    check("no duplicate market ids", dupes == 0, f"{dupes} duplicates")

    bad_time = []
    for m in markets:
        try:
            start = int(m["window_start_epoch"])
            close = int(m["window_close_epoch"])
        except (TypeError, ValueError):
            bad_time.append(m["market_id"])
            continue
        if close - start != 300:
            bad_time.append(m["market_id"])
    check("every window is exactly 300s", not bad_time, f"{len(bad_time)} off, e.g. {bad_time[:2]}")

    bad_iso = []
    for m in markets:
        try:
            epoch = int(m["window_close_epoch"])
            expected = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if m["window_close_utc"] != expected:
                bad_iso.append(m["slug"])
        except (TypeError, ValueError):
            bad_iso.append(m["slug"])
    check("close timestamps are consistent UTC", not bad_iso, f"{len(bad_iso)} bad")

    close_epochs = [int(m["window_close_epoch"]) for m in markets]
    check("markets are contiguous 5-minute windows",
          len(set(close_epochs)) == len(close_epochs) and (max(close_epochs) - min(close_epochs)) // 300 + 1 == len(close_epochs),
          f"span {(max(close_epochs) - min(close_epochs)) // 300 + 1} windows for {len(close_epochs)} markets")
    check("epoch seconds are on the 300s grid",
          all(e % 300 == 0 for e in close_epochs), "some not multiples of 300")

    market_ids = {m["market_id"] for m in markets}
    ghost = {p["market_id"] for p in prices} - market_ids
    check("price rows reference known markets", not ghost, f"{len(ghost)} unknown ids")

    priced = [r for r in prices if r.get("up_price") not in ("", None)]
    if priced:
        sums = [abs(float(r["up_price"]) + float(r["down_price"]) - 1.0) for r in priced]
        check("up + down = 1 where priced", max(sums) < 0.02, f"max deviation {max(sums):.4f}")

    if "btc_offset_pct" in (prices[0] if prices else {}):
        offsets = [float(r["btc_offset_pct"]) for r in prices if r.get("btc_offset_pct") not in ("", None)]
        check("btc_offset_pct present", bool(offsets), f"{len(offsets)} values")

    per_market = {}
    for row in prices:
        per_market[row["market_id"]] = per_market.get(row["market_id"], 0) + 1
    check("each market has every observation offset",
          len(set(per_market.values())) == 1 if per_market else False,
          f"row counts per market: {sorted(set(per_market.values()))}")

    meta_path = os.path.join(DATA, "data_source.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        check("provenance recorded", bool(meta.get("source")), str(meta.get("source"))[:60])
        covered = sum(1 for v in (meta.get("price_coverage_per_offset") or {}).values() if v)
        if meta.get("price_coverage_per_offset"):
            print(f"note: real prices present at {covered} of "
                  f"{len(meta['price_coverage_per_offset'])} observation offsets")
    else:
        check("provenance recorded", False, "data_source.json missing")

    print()
    print(f"markets : {len(markets)}   price rows: {len(prices)}")
    print(f"period  : {markets[-1]['window_close_utc']} -> {markets[0]['window_close_utc']}")
    print(f"balance : UP {balance['UP']} / DOWN {balance['DOWN']}  (Up share {up_share:.3f})")
    print(f"priced rows: {len(priced)} of {len(prices)}")
    print()

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"checks: {len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

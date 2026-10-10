"""Self-test for core/dataset_loader.py (no third-party packages needed).

    python tools/test_dataset_loader.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from core import dataset_loader as dl  # noqa: E402

WORK = os.path.join(PROJECT, ".work", "loader-fixture")


def build() -> dict[str, str]:
    os.makedirs(WORK, exist_ok=True)
    paths: dict[str, str] = {}

    # ---- crypto_prices: 1s bars, epoch SECONDS, 'open_time' naming
    spot_path = os.path.join(WORK, "spot.csv")
    with open(spot_path, "w", encoding="utf-8") as fh:
        fh.write("symbol,open_time,open,high,low,close,volume\n")
        base = 1_780_000_000
        for i in range(600):
            price = 80000.0 + i * 0.5
            fh.write(f"BTCUSDT,{base + i},{price:.2f},{price+5:.2f},{price-5:.2f},{price:.2f},10\n")
    paths["spot"] = spot_path

    # ---- orderbooks: epoch MILLISECONDS, ladder columns, 't' naming for time
    book_path = os.path.join(WORK, "book.jsonl")
    with open(book_path, "w", encoding="utf-8") as fh:
        base = 1_780_000_000
        window_start = (base // 300) * 300
        for i in range(0, 400, 2):
            ts_ms = (base + i) * 1000
            for outcome, bid, ask in (("Up", 0.40, 0.42), ("Down", 0.58, 0.60)):
                fh.write(json.dumps({
                    "t": ts_ms,
                    "market_slug": f"btc-updown-5m-{window_start}",
                    "outcome": outcome,
                    "bids": [[bid, 100.0], [bid - 0.01, 250.0]],
                    "asks": [[ask, 120.0], [ask + 0.01, 300.0]],
                }) + "\n")
    paths["book"] = book_path

    # ---- a tiny csv orderbook with DIRECT top-of-book columns, microseconds
    direct_path = os.path.join(WORK, "book_direct.csv")
    with open(direct_path, "w", encoding="utf-8") as fh:
        fh.write("timestamp_us,slug,outcome,best_bid,bid_size,best_ask,ask_size\n")
        base = 1_780_000_000
        for i in range(50):
            fh.write(f"{base*1_000_000 + i*1_000_000},btc-updown-15m-1780000000,Up,0.45,80,0.47,90\n")
    paths["direct"] = direct_path
    return paths


def main() -> int:
    paths = build()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---- column detection
    mapping = dl.detect_mapping(["t", "market_slug", "outcome", "bids", "asks"], kind="book")
    check("detects short 't' as timestamp", mapping.timestamp == "t", str(mapping.timestamp))
    check("detects ladder columns", mapping.bids == "bids" and mapping.asks == "asks")
    check("detects slug", mapping.slug == "market_slug", str(mapping.slug))

    mapping2 = dl.detect_mapping(
        ["open_time", "open", "high", "low", "close", "volume"], kind="spot"
    )
    check("detects spot close column", mapping2.close == "close", str(mapping2.close))
    check("detects open_time", mapping2.timestamp == "open_time", str(mapping2.timestamp))

    # ---- epoch units
    check("seconds detected", dl.detect_epoch_divisor([1_780_000_000])[0] == "seconds")
    check("milliseconds detected", dl.detect_epoch_divisor([1_780_000_000_000])[0] == "milliseconds")
    check("microseconds detected", dl.detect_epoch_divisor([1_780_000_000_000_000])[0] == "microseconds")
    check(
        "ms normalises to the right second",
        abs((dl.to_epoch_seconds(1_780_000_000_000, 1e3) or 0) - 1_780_000_000) < 1e-6,
    )
    check("ISO-8601 accepted", (dl.to_epoch_seconds("2026-06-01T00:00:00Z") or 0) > 1.7e9)

    # ---- ladder parsing: all the shapes seen in real exports
    check("ladder: list of pairs", dl.parse_ladder([[0.5, 10], [0.49, 20]]) == [(0.5, 10.0), (0.49, 20.0)])
    check("ladder: json string", dl.parse_ladder('[[0.5,10]]') == [(0.5, 10.0)])
    check("ladder: list of dicts", dl.parse_ladder([{"price": 0.5, "size": 10}]) == [(0.5, 10.0)])
    check("ladder: price->size map", dl.parse_ladder({"0.52": 100}) == [(0.52, 100.0)])
    check("ladder: empty is safe", dl.parse_ladder(None) == [] and dl.parse_ladder("") == [])

    # ---- slug window parsing
    check("slug epoch extracted", dl.slug_window_start("btc-updown-5m-1780319100") == 1780319100)
    check("5m window inferred", dl.infer_window_seconds("btc-updown-5m-1780319100") == 300)
    check("15m window inferred", dl.infer_window_seconds("btc-updown-15m-1780319100") == 900)
    check("non-numeric slug returns None", dl.slug_window_start("bitcoin-up-or-down-june-1") is None)

    # ---- spot series + as-of semantics (the critical anti-lookahead property)
    series = dl.build_spot_series(paths["spot"], symbol_column="symbol", symbol_value="BTCUSDT")
    check("spot series loaded", len(series) == 600, str(len(series)))
    price, ts = series.price_asof(1_780_000_010)
    check("as-of returns the exact tick", price is not None and abs(price - 80005.0) < 1e-9, str(price))
    price_between, ts_between = series.price_asof(1_780_000_010.5)
    check("as-of between ticks uses the PRIOR tick (no lookahead)",
          price_between is not None and abs(price_between - 80005.0) < 1e-9, str(price_between))
    check("as-of before the series returns None", series.price_asof(1_000.0) == (None, None))
    change = series.change_pct(1_780_000_010, lookback_s=3.0)
    check("3s change computed", change is not None, str(change))
    check("3s change magnitude sane", change is not None and 0 < change < 1, str(change))

    # ---- streaming reader honours the chunk cap (memory safety)
    chunks = list(dl.iter_rows(paths["book"], chunk_size=100))
    check("jsonl streamed in chunks", len(chunks) > 1, str(len(chunks)))
    check("chunk size respected", all(len(c) <= 100 for c, _ in chunks))
    total = sum(len(c) for c, _ in chunks)
    check("total rows correct", total == 400, str(total))
    capped = list(dl.iter_rows(paths["book"], chunk_size=100, max_rows=50))
    check("max_rows caps the read", sum(len(c) for c, _ in capped) == 50)

    # ---- alignment end to end
    rows = []
    for aligned in dl.align_book_with_spot(
        paths["book"], series, lookback_s=3.0, default_window_s=300, chunk_size=100
    ):
        rows.extend(aligned)
    check("alignment produced rows", len(rows) > 0, str(len(rows)))
    first = rows[0]
    check("midpoint computed", first.midpoint is not None, str(first.midpoint))
    check("midpoint is the mid of the ladder top",
          first.midpoint is not None and abs(first.midpoint - 0.41) < 1e-9, str(first.midpoint))
    check("best_bid from ladder", first.best_bid == 0.40, str(first.best_bid))
    check("spot attached", first.spot_price is not None, str(first.spot_price))
    check("seconds_to_close derived from slug",
          first.seconds_to_close is not None and first.seconds_to_close > 0,
          str(first.seconds_to_close))
    check("window_close derived", first.window_close is not None, str(first.window_close))
    check("both outcomes present", {r.outcome for r in rows} == {"Up", "Down"}, str({r.outcome for r in rows}))

    # no row may use a spot price from the future
    violations = [r for r in rows if r.spot_ts is not None and r.spot_ts > r.timestamp]
    check("no look-ahead spot joins", not violations, f"{len(violations)} violations")

    # ---- direct top-of-book csv, microsecond timestamps
    series2 = series
    rows2 = []
    for aligned in dl.align_book_with_spot(paths["direct"], series2, default_window_s=900):
        rows2.extend(aligned)
    check("direct bid/ask columns parsed", len(rows2) == 50, str(len(rows2)))
    check("direct best_bid read", rows2[0].best_bid == 0.45, str(rows2[0].best_bid))
    check("microsecond timestamps normalised",
          abs(rows2[0].timestamp - 1_780_000_000) < 1.001, str(rows2[0].timestamp))
    check("15m window inferred in alignment",
          rows2[0].seconds_to_close is not None and rows2[0].seconds_to_close > 800,
          str(rows2[0].seconds_to_close))

    # ---- outcome filtering
    up_only = []
    for aligned in dl.align_book_with_spot(paths["book"], series, outcomes=["up"], default_window_s=300):
        up_only.extend(aligned)
    check("outcome filter works", {r.outcome for r in up_only} == {"Up"}, str({r.outcome for r in up_only}))

    # ---- results labelling + csv writing
    joint_path = os.path.join(WORK, "joint.csv")
    written = dl.write_joint_csv(joint_path, rows)
    check("joint csv written", written == len(rows) and os.path.exists(joint_path), str(written))
    with open(joint_path, encoding="utf-8") as fh:
        head = fh.readline().strip()
    check("joint csv has the expected header", "best_bid" in head and "seconds_to_close" in head, head[:80])

    # ---- clear error when a file has no usable book columns
    bad_path = os.path.join(WORK, "bad.csv")
    with open(bad_path, "w", encoding="utf-8") as fh:
        fh.write("alpha,beta\n1,2\n")
    try:
        list(dl.align_book_with_spot(bad_path, series))
        check("unusable book raises a clear error", False, "no exception")
    except dl.DataError as exc:
        check("unusable book raises a clear error", "timestamp" in str(exc) or "orderbook" in str(exc), str(exc)[:80])

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

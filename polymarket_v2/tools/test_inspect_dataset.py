"""Self-test for tools/inspect_dataset.py.

Builds a tiny dataset that mimics the user's layout (orderbooks / crypto_prices /
trades / price_history) with realistic timestamp units and payload shapes, then
runs the inspector over it and asserts the audit found what it should.

    python tools/test_inspect_dataset.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from tools import inspect_dataset as insp  # noqa: E402

WORK = os.path.join(PROJECT, ".work", "audit-fixture")


def build_fixture() -> str:
    os.makedirs(WORK, exist_ok=True)
    for sub in ("orderbooks", "crypto_prices", "trades", "price_history"):
        os.makedirs(os.path.join(WORK, sub), exist_ok=True)

    # --- ordersbooks as parquet with a milliseconds timestamp (the common case)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "timestamp_ms": pa.array([1_780_000_000_000 + i * 1000 for i in range(50)], pa.int64()),
                "market_slug": pa.array(["btc-updown-15m-1780000000"] * 50),
                "outcome": pa.array(["Up", "Down"] * 25),
                "best_bid": pa.array([0.48 + i * 0.001 for i in range(50)], pa.float64()),
                "best_ask": pa.array([0.52 + i * 0.001 for i in range(50)], pa.float64()),
                "bids": pa.array([json.dumps([[0.48, 100.0]])] * 50),
                "asks": pa.array([json.dumps([[0.52, 120.0]])] * 50),
            }
        )
        pq.write_table(table, os.path.join(WORK, "orderbooks", "btc_15m_2026-03.parquet"))
        parquet_ok = True
    except ImportError:
        parquet_ok = False

    # --- crypto_prices as csv with ISO-ish plus epoch seconds
    with open(os.path.join(WORK, "crypto_prices", "btcusdt_1s.csv"), "w", encoding="utf-8") as fh:
        fh.write("open_time,open,high,low,close,volume\n")
        for i in range(120):
            fh.write(f"{1_780_000_000 + i},80000.{i},80010.{i},79990.{i},80005.{i},12.5\n")

    # --- trades as jsonl
    with open(os.path.join(WORK, "trades", "trades.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(json.dumps({
                "timestamp": 1_780_000_000 + i,
                "slug": "btc-updown-15m-1780000000",
                "outcome": "Up" if i % 2 == 0 else "Down",
                "price": 0.4 + i * 0.005,
                "size": 25.0,
                "side": "BUY",
            }) + "\n")

    # --- price_history as csv (microseconds timestamp, to test unit detection)
    with open(os.path.join(WORK, "price_history", "history.csv"), "w", encoding="utf-8") as fh:
        fh.write("t,p,outcome\n")
        for i in range(60):
            fh.write(f"{1_780_000_000_000_000 + i * 1_000_000},0.5,Up\n")

    return WORK if parquet_ok else WORK


def main() -> int:
    root = build_fixture()
    print(f"fixture built at {root}")
    print()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---- discovery + folder summary
    folders = insp.directory_summary(root)
    names = {f["folder"] for f in folders}
    check("finds all four subfolders", names == {"orderbooks", "crypto_prices", "trades", "price_history"}, str(names))

    targets = insp.discover(root, max_files_per_dir=4)
    formats = {t[3] for t in targets}
    check("discovers parquet/csv/jsonl", {"csv", "jsonl"} <= formats, str(formats))

    # ---- timestamp unit detection (the highest-risk piece)
    check("epoch seconds detected", insp.detect_epoch_unit([1_780_000_000])[0] == "seconds")
    check("epoch milliseconds detected", insp.detect_epoch_unit([1_780_000_000_000])[0] == "milliseconds")
    check("epoch microseconds detected", insp.detect_epoch_unit([1_780_000_000_000_000])[0] == "microseconds")
    check("epoch nanoseconds detected", insp.detect_epoch_unit([1_780_000_000_000_000_000])[0] == "nanoseconds")

    unit, divisor = insp.detect_epoch_unit([1_780_000_000_000])
    seconds = insp.to_epoch_seconds(1_780_000_000_000, divisor)
    check("ms -> seconds conversion", abs((seconds or 0) - 1_780_000_000) < 1e-6, f"{seconds}")
    iso_text = insp.iso(seconds)
    check("iso formatting sane", iso_text.startswith("2026-"), iso_text)

    # ISO string input must also work
    iso_seconds = insp.to_epoch_seconds("2026-06-01T00:00:00Z", 1.0)
    check("ISO-8601 parsed", iso_seconds is not None and iso_seconds > 1.7e9, str(iso_seconds))

    # ---- csv inspection
    csv_path = os.path.join(root, "crypto_prices", "btcusdt_1s.csv")
    report = insp.inspect_csv(csv_path, "crypto_prices/btcusdt_1s.csv", os.path.getsize(csv_path), 3, 5_000_000)
    check("csv columns read", [c.name for c in report.columns][:3] == ["open_time", "open", "high"], str([c.name for c in report.columns]))
    check("csv row count exact for small file", report.row_count == 120, str(report.row_count))
    check("csv sample captured", len(report.sample) == 3, str(len(report.sample)))
    check("csv time column flagged", any(c.epoch_unit == "seconds" for c in report.columns), str([(c.name, c.epoch_unit) for c in report.columns]))

    # ---- jsonl inspection
    jsonl_path = os.path.join(root, "trades", "trades.jsonl")
    report = insp.inspect_jsonl(jsonl_path, "trades/trades.jsonl", os.path.getsize(jsonl_path), 2, 5_000_000)
    check("jsonl keys discovered", {"timestamp", "slug", "price"} <= {c.name for c in report.columns}, str([c.name for c in report.columns]))
    check("jsonl row count", report.row_count == 30, str(report.row_count))

    # ---- microseconds detection via a real file
    micro_path = os.path.join(root, "price_history", "history.csv")
    report = insp.inspect_csv(micro_path, "price_history/history.csv", os.path.getsize(micro_path), 2, 5_000_000)
    micro_cols = {c.name: c.epoch_unit for c in report.columns}
    check("microsecond column detected in a real file", micro_cols.get("t") == "microseconds", str(micro_cols))

    # ---- parquet (only if pyarrow is present)
    try:
        import pyarrow  # noqa: F401

        pq_path = os.path.join(root, "orderbooks", "btc_15m_2026-03.parquet")
        report = insp.inspect_parquet(pq_path, "orderbooks/btc_15m_2026-03.parquet", os.path.getsize(pq_path), 3)
        check("parquet row count from metadata", report.row_count == 50, str(report.row_count))
        check("parquet schema read", "best_bid" in [c.name for c in report.columns], str([c.name for c in report.columns]))
        check("parquet ms timestamp detected", any(c.epoch_unit == "milliseconds" for c in report.columns),
              str([(c.name, c.epoch_unit) for c in report.columns]))
        check("parquet sample readable", len(report.sample) == 3, str(len(report.sample)))
        check("parquet inspected without loading full file", report.error == "", report.error)
    except ImportError:
        print("note: pyarrow not installed - parquet checks skipped")

    # ---- the memory-safety contract: byte cap respected
    big_csv = os.path.join(root, "crypto_prices", "big.csv")
    with open(big_csv, "w", encoding="utf-8") as fh:
        fh.write("t,p\n")
        for i in range(50_000):
            fh.write(f"{1_780_000_000 + i},0.5\n")
    report = insp.inspect_csv(big_csv, "crypto_prices/big.csv", os.path.getsize(big_csv), 2, 4096)
    check("row count estimated when capped", any("ESTIMATED" in n for n in report.notes), str(report.notes))
    check("still returns a sample when capped", len(report.sample) == 2, str(len(report.sample)))

    # ---- graceful failure on a missing folder
    missing = insp.main([os.path.join(root, "does-not-exist")])
    check("missing folder exits cleanly", missing == 2, str(missing))

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

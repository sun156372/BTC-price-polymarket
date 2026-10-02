"""Turn a public Polymarket BTC 5-minute dataset into the project's dataset format.

Input  (default): work/sigrex_btc5m.csv  (Sigrex/polymarket_btc_up_down_5m_251218_260909)
Output (default): data/

  btc5m_markets.csv   one row per market: window times, label, resolution prices, BTC spot
  btc5m_prices.csv    one row per market x observation point (up_price / down_price / result)
  data_source.json    provenance + per-offset coverage counts (so gaps are explicit)
  templates/btc5m_sim_template.py   skeleton that consumes the two CSVs with --time/--threshold/--tp/--sl/--min-gap

Honest limitation: this public dataset records only `implied_up_last` (the Up-token price
near close), not a full intra-window series. So the price grid carries the real label and
real window times, but only one real price point per market; other offsets are emitted
empty rather than fabricated. For real per-offset prices run scripts/scrape_btc5m.py
against the live Polymarket API on a server with egress.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OFFSETS = [300, 270, 240, 210, 180, 150, 120, 90, 60, 45, 30, 15]

MARKET_FIELDS = [
    "market_id", "slug", "question",
    "window_start_utc", "window_close_utc", "window_start_epoch", "window_close_epoch",
    "result", "label_up", "winning_outcome",
    "final_price", "price_to_beat", "chainlink_return",
    "btc_open", "btc_close", "btc_return",
    "implied_up_last", "trade_count", "source",
]

PRICE_FIELDS = [
    "market_id", "slug", "window_close_epoch", "window_close_utc",
    "t_minus_s", "up_price", "down_price", "result", "price_source",
]

# column aliases -> canonical name
ALIASES = {
    "condition_id": "market_id",
    "market_id": "market_id",
    "id": "market_id",
    "slug": "slug",
    "question": "question",
    "start_ts": "start_ts",
    "end_ts": "end_ts",
    "start_time": "start_time",
    "end_time": "end_time",
    "winning_outcome": "winning_outcome",
    "label_up": "label_up",
    "final_price": "final_price",
    "price_to_beat": "price_to_beat",
    "chainlink_return": "chainlink_return",
    "btc_open": "btc_open",
    "btc_close": "btc_close",
    "btc_return": "btc_return",
    "implied_up_last": "implied_up_last",
    "trade_count": "trade_count",
}


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_epoch(value):
    if value in (None, "", "null"):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def as_float(value):
    if value in (None, "", "null", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_header(fieldnames: list[str]) -> dict[str, str]:
    mapping = {}
    for name in fieldnames or []:
        key = name.strip().lower()
        if key in ALIASES:
            mapping[ALIASES[key]] = name
    return mapping


def normalize_result(row: dict, cols: dict) -> str:
    """Prefer explicit label columns; fall back to comparing resolution prices."""
    label_up = str(row.get(cols.get("label_up", ""), "")).strip().lower()
    if label_up in ("1", "true", "up", "yes"):
        return "UP"
    if label_up in ("0", "false", "down", "no"):
        return "DOWN"

    winner = str(row.get(cols.get("winning_outcome", ""), "")).strip().lower()
    if winner in ("up", "yes"):
        return "UP"
    if winner in ("down", "no"):
        return "DOWN"

    final = as_float(row.get(cols.get("final_price", "")))
    beat = as_float(row.get(cols.get("price_to_beat", "")))
    if final is not None and beat is not None:
        return "UP" if final >= beat else "DOWN"
    return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(ROOT, "work", "sigrex_btc5m.csv"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--count", type=int, default=500, help="how many markets (most recent first)")
    ap.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    ap.add_argument("--source", default="Sigrex/polymarket_btc_up_down_5m_251218_260909 (hf-mirror.com)")
    ap.add_argument("--templates-dir", default=os.path.join(ROOT, "templates"))
    args = ap.parse_args()

    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()}, reverse=True)
    if not os.path.exists(args.input):
        print(f"input not found: {args.input}")
        return 2

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = canonical_header(reader.fieldnames or [])
        raw_rows = list(reader)

    print(f"input rows : {len(raw_rows)}")
    print(f"columns    : {len(reader.fieldnames or [])} (recognised {len(cols)})")
    if "market_id" not in cols or "end_ts" not in cols:
        print("could not identify market_id / end_ts columns - aborting")
        return 2

    markets = []
    for row in raw_rows:
        end_ts = parse_epoch(row.get(cols["end_ts"]))
        if end_ts is None:
            end_ts = parse_epoch(row.get(cols.get("end_time", "")))
        start_ts = parse_epoch(row.get(cols.get("start_ts", "")))
        if start_ts is None:
            start_ts = parse_epoch(row.get(cols.get("start_time", "")))
        if start_ts is None and end_ts is not None:
            start_ts = end_ts - 300
        if end_ts is None:
            continue

        result = normalize_result(row, cols)
        markets.append({
            "market_id": str(row.get(cols["market_id"], "")).strip(),
            "slug": str(row.get(cols.get("slug", ""), "")).strip(),
            "question": str(row.get(cols.get("question", ""), "")).strip(),
            "window_start_epoch": start_ts,
            "window_close_epoch": end_ts,
            "window_start_utc": iso(start_ts) if start_ts else "",
            "window_close_utc": iso(end_ts),
            "result": result,
            "label_up": row.get(cols.get("label_up", ""), ""),
            "winning_outcome": row.get(cols.get("winning_outcome", ""), ""),
            "final_price": row.get(cols.get("final_price", ""), ""),
            "price_to_beat": row.get(cols.get("price_to_beat", ""), ""),
            "chainlink_return": row.get(cols.get("chainlink_return", ""), ""),
            "btc_open": row.get(cols.get("btc_open", ""), ""),
            "btc_close": row.get(cols.get("btc_close", ""), ""),
            "btc_return": row.get(cols.get("btc_return", ""), ""),
            "implied_up_last": row.get(cols.get("implied_up_last", ""), ""),
            "trade_count": row.get(cols.get("trade_count", ""), ""),
            "source": args.source,
        })

    markets.sort(key=lambda item: item["window_close_epoch"], reverse=True)
    selected = markets[: args.count]
    print(f"markets    : {len(markets)} resolved, taking the {len(selected)} most recent")

    if selected:
        print(f"period     : {selected[-1]['window_close_utc']} -> {selected[0]['window_close_utc']}")

    os.makedirs(args.outdir, exist_ok=True)
    markets_path = os.path.join(args.outdir, "btc5m_markets.csv")
    with open(markets_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MARKET_FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    # price grid: the dataset has one real Up price per market (near close), so we
    # place it at the closest offset and leave the others empty instead of inventing data.
    price_rows = []
    coverage = {offset: 0 for offset in offsets}
    near_close = min(offsets) if offsets else 15
    for market in selected:
        up_last = as_float(market["implied_up_last"])
        for offset in offsets:
            up_price = down_price = ""
            price_source = "unavailable"
            if up_last is not None and offset == near_close:
                up_price = round(up_last, 6)
                down_price = round(1.0 - up_last, 6)
                price_source = "implied_up_last"
                coverage[offset] += 1
            price_rows.append({
                "market_id": market["market_id"],
                "slug": market["slug"],
                "window_close_epoch": market["window_close_epoch"],
                "window_close_utc": market["window_close_utc"],
                "t_minus_s": offset,
                "up_price": up_price,
                "down_price": down_price,
                "result": market["result"],
                "price_source": price_source,
            })

    prices_path = os.path.join(args.outdir, "btc5m_prices.csv")
    with open(prices_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRICE_FIELDS)
        writer.writeheader()
        writer.writerows(price_rows)

    up_count = sum(1 for m in selected if m["result"] == "UP")
    down_count = sum(1 for m in selected if m["result"] == "DOWN")
    unknown = sum(1 for m in selected if m["result"] == "UNKNOWN")
    resolved_px = sum(1 for m in selected if as_float(m["final_price"]) is not None)

    meta = {
        "source": args.source,
        "mirror": "https://hf-mirror.com",
        "input_file": os.path.relpath(args.input, ROOT),
        "markets_total_in_source": len(markets),
        "markets_written": len(selected),
        "period_utc": [selected[-1]["window_close_utc"], selected[0]["window_close_utc"]] if selected else [],
        "result_balance": {"UP": up_count, "DOWN": down_count, "UNKNOWN": unknown},
        "chainlink_price_coverage": resolved_px,
        "observation_offsets_s": offsets,
        "price_coverage_per_offset": coverage,
        "price_note": (
            "Public dataset exposes a single Up-token price per market (implied_up_last, "
            "near close). Only that offset carries a real price; other offsets are blank "
            "rather than synthesised. Run scripts/scrape_btc5m.py on a host with egress "
            "for true per-offset UP/DOWN prices."
        ),
        "lookahead_warning": (
            "implied_up_last is a NEAR-CLOSE price, so a strategy that reads it at a "
            "large t_minus_s value is using information that did not exist yet. Any "
            "win rate measured on this file alone is inflated by that lookahead. Treat "
            "results at --time 15 as the only unbiased ones here, and use "
            "scripts/scrape_btc5m.py or the 100ms book archives for multi-offset tests."
        ),
        "known_limitations": [
            "implied-probability coverage in the source is ~84%",
            "closed markets return only 2-30 price points from the public API",
            "no historical order book is available from Polymarket",
        ],
    }
    meta_path = os.path.join(args.outdir, "data_source.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # simulation template that consumes both CSVs
    os.makedirs(args.templates_dir, exist_ok=True)
    template_src = os.path.join(ROOT, "scripts", "sim_template.py")
    if os.path.exists(template_src):
        with open(template_src, encoding="utf-8") as fh:
            body = fh.read()
        with open(os.path.join(args.templates_dir, "btc5m_sim_template.py"), "w", encoding="utf-8") as fh:
            fh.write(body)

    print()
    print(f"UP / DOWN  : {up_count} / {down_count} (unknown {unknown})")
    print(f"chainlink  : {resolved_px}/{len(selected)} markets carry resolution prices")
    print(f"markets CSV: {markets_path}")
    print(f"prices CSV : {prices_path}")
    print(f"metadata   : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

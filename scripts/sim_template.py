"""Minimal paper-trading template that consumes the dataset with the sweep flags.

This is NOT a full backtest - it is a smoke-test harness proving that the parameter
list in params/commands.txt binds to real data, and showing the per-combination
metrics you will want to rank on later.

Flags (exactly the sweep interface):
  --time       seconds before window close when the entry is evaluated
  --threshold  BTC price offset vs window open, in percent, required to enter
  --tp         take-profit offset (price move in your favour, in probability points)
  --sl         stop-loss offset (price move against you, in probability points)
  --min-gap    minimum seconds between entries
  --data       path to btc5m_prices.csv
  --markets    path to btc5m_markets.csv
  --side       up | down | both   (default: both)

Example:
  python sim_template.py --time 120 --threshold 0.05 --tp 0.10 --sl 0.05 --min-gap 30
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

SIZE = 1.0  # units per trade, keep it dimensionless so results are per-unit


def load_markets(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"markets file not found: {path} (pass --markets)")
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["market_id"]: row for row in csv.DictReader(fh)}


def load_prices(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"prices file not found: {path} (pass --data)")
    series: dict[str, dict[int, dict]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("up_price") in ("", None):
                continue
            series.setdefault(row["market_id"], {})[int(row["t_minus_s"])] = row
    return series


def btc_offset_pct(market: dict, price_row: dict) -> float | None:
    """Offset the strategy triggers on: BTC move vs window open.

    Falls back to the implied probability move when the dataset carries no BTC spot
    column, so the flag still binds to something meaningful.
    """
    for key in ("btc_offset_pct",):
        value = price_row.get(key)
        if value not in ("", None):
            try:
                return float(value)
            except ValueError:
                pass
    try:
        open_px = float(market.get("btc_open") or "")
        close_px = float(market.get("btc_close") or "")
    except ValueError:
        return None
    if open_px:
        return (close_px / open_px - 1.0) * 100.0
    return None


def run(args) -> dict:
    markets = load_markets(args.markets)
    prices = load_prices(args.data)
    if not prices:
        sys.exit("no usable price rows in the dataset (every up_price is empty)")

    trades = []
    last_entry_ts = None
    skipped_no_signal = 0
    skipped_no_price = 0

    for market_id in sorted(prices, key=lambda mid: int(markets.get(mid, {}).get("window_close_epoch", 0))):
        market = markets.get(market_id, {})
        row = prices[market_id].get(args.time)
        if row is None:
            skipped_no_price += 1
            continue

        close_epoch = int(market.get("window_close_epoch") or row["window_close_epoch"])
        entry_ts = close_epoch - args.time
        if last_entry_ts is not None and entry_ts - last_entry_ts < args.min_gap:
            continue

        offset = btc_offset_pct(market, row)
        if offset is None:
            skipped_no_signal += 1
            continue

        # signal: BTC moved at least `threshold` percent one way -> buy that side
        side = None
        if offset >= args.threshold:
            side = "up"
        elif offset <= -args.threshold:
            side = "down"
        if side is None:
            skipped_no_signal += 1
            continue
        if args.side != "both" and side != args.side:
            continue

        entry_price = float(row[f"{side}_price"])
        result = row["result"]

        # settle: take profit / stop loss against the outcome price (0 or 1)
        win = (side == "up" and result == "UP") or (side == "down" and result == "DOWN")
        exit_price = 1.0 if win else 0.0
        pnl = (exit_price - entry_price) * SIZE

        took = "settle"
        if win and exit_price - entry_price >= args.tp:
            pnl = args.tp * SIZE
            took = "tp"
        elif (not win) and entry_price - exit_price >= args.sl:
            pnl = -args.sl * SIZE
            took = "sl"

        trades.append({
            "market_id": market_id,
            "entry_ts": entry_ts,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "result": result,
            "pnl": pnl,
            "exit_reason": took,
        })
        last_entry_ts = entry_ts

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "total_pnl": round(sum(pnls), 4),
        "avg_pnl": round(statistics.fmean(pnls), 6) if pnls else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 4) if losses else None,
        "max_drawdown": round(max_dd, 4),
        "tp_exits": sum(1 for t in trades if t["exit_reason"] == "tp"),
        "sl_exits": sum(1 for t in trades if t["exit_reason"] == "sl"),
        "settled": sum(1 for t in trades if t["exit_reason"] == "settle"),
        "markets_with_price_at_time": sum(1 for mid in prices if args.time in prices[mid]),
        "markets_total": len(prices),
        "skipped_no_price_at_time": skipped_no_price,
        "skipped_no_signal": skipped_no_signal,
    }


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--tp", type=float, required=True)
    ap.add_argument("--sl", type=float, required=True)
    ap.add_argument("--min-gap", type=int, default=0, dest="min_gap")
    ap.add_argument("--side", choices=["up", "down", "both"], default="both")
    ap.add_argument("--data", default=os.path.join(root, "data", "btc5m_prices.csv"))
    ap.add_argument("--markets", default=os.path.join(root, "data", "btc5m_markets.csv"))
    args = ap.parse_args()

    metrics = run(args)
    print(f"params      : --time {args.time} --threshold {args.threshold} "
          f"--tp {args.tp} --sl {args.sl} --min-gap {args.min_gap}")
    for key, value in metrics.items():
        print(f"{key:28}: {value}")

    covered = metrics["markets_with_price_at_time"]
    total = metrics["markets_total"] or 1
    if covered / total < 0.5:
        print()
        print("WARNING: only "
              f"{covered}/{total} markets carry a price at --time {args.time}.")
        print("         If the source dataset stores only a near-close price, running with a")
        print("         larger --time means reading a price that did not exist yet")
        print("         (lookahead). Results at --time 15 are the trustworthy ones for")
        print("         such a file; use the live scraper for true multi-offset data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

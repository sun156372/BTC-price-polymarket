"""Directional end-of-window sniper: TP/SL and win-rate study on REAL trades.

What this measures (and why it is trustworthy)
---------------------------------------------
Every price used here is a **real executed trade price** from your `trades`
parquet files, not a modelled book. Your order books are unusable for this - 99.77%
of snapshots carry spreads >= 0.50 - so a backtest built on them would be fiction.
Trades print at a sane distribution (median 0.53), so they are the honest source.

Join
----
``all.parquet``        market metadata: slug, start_time, end_time, asset
``all (1).parquet``    outcomes, but keyed by numeric id -> matched on
                       (asset, start_time, end_time)
``trades/*.parquet``   executed prices (+ side/size where present)

Strategy under test
-------------------
At ``T - entry_s``: read the spot move from the window open. If |move| >=
``move_pct``, buy the side the spot favours, at the prevailing trade price. Then
exit by one of three rules, and report all three separately:

* ``settle`` - hold to settlement (payoff 1.0 win / 0.0 lose)
* ``tp``     - exit if price ever trades at or above entry + tp
* ``sl``     - exit if price ever trades at or below entry - sl (checked first,
               which is the conservative assumption)

Live-vs-backtest gaps modelled explicitly
-----------------------------------------
1. **Fees** - real taker fee ``shares * 0.07 * p * (1-p)`` on both legs.
2. **Adverse fill** - entry pays the trade price *plus* ``entry_slippage_bps``;
   exits receive the trigger price *minus* ``exit_slippage_bps``. A backtest that
   fills at the exact trigger price is the classic way to overstate a TP/SL edge.
3. **TP reachability** - how many trades actually print after entry at all. In the
   last seconds of a window there is often nothing left to trade against, so "TP
   hit" is frequently fiction. This is reported as ``traded_after_entry``.
4. **No-trade windows** - markets with no trade at the entry offset are skipped,
   never interpolated.

    python tools/backtest_directional.py --trades "..\\data\\2026-03-01.parquet"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from utils.fee_calculator import taker_fee_usdc, DEFAULT_COEFFICIENT  # noqa: E402


def to_epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class MarketMeta:
    slug: str
    asset: str
    start_ts: float | None
    end_ts: float | None
    outcome: str = ""      # "UP" / "DOWN" / ""

    @property
    def window_s(self) -> float | None:
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts


def load_markets(data_dir: str) -> dict[tuple, MarketMeta]:
    """Union of the two resolution-style files, keyed by (asset, start, end).

    The two exports disagree on identifiers - one keys markets by the slug
    (``btc-updown-15m-...``) but carries no outcome, the other carries the
    outcome but only a numeric id. Neither alone can label a trade, so both are
    merged on the window they describe: ``(asset, start_time, end_time)``.
    """
    import pyarrow.parquet as pq

    markets: dict[tuple, MarketMeta] = {}

    def key(asset: str, start: float | None, end: float | None) -> tuple:
        return (asset.strip().upper(), round(start or 0.0), round(end or 0.0))

    for name in ("all (1).parquet", "all.parquet"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        pf = pq.ParquetFile(path)
        columns = [c for c in ("market_id", "asset", "start_time", "end_time",
                              "outcome", "slug") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=50_000, columns=columns):
            for row in batch.to_pylist():
                asset = str(row.get("asset") or "")
                start = to_epoch(row.get("start_time"))
                end = to_epoch(row.get("end_time"))
                if end is None:
                    continue
                k = key(asset, start, end)
                # a slug looks like '<asset>-updown-<tf>-<epoch>'; a numeric id does not
                raw_slug = str(row.get("slug") or "")
                raw_id = str(row.get("market_id") or "")
                slug = raw_slug or (raw_id if "-updown-" in raw_id else "")
                outcome = str(row.get("outcome") or "").strip().upper()

                existing = markets.get(k)
                if existing is None:
                    markets[k] = MarketMeta(slug=slug, asset=asset, start_ts=start,
                                            end_ts=end, outcome=outcome)
                else:
                    # merge: whichever row supplies the missing piece wins
                    if not existing.outcome and outcome:
                        existing.outcome = outcome
                    if not existing.slug and slug:
                        existing.slug = slug
    return markets


def load_spot(data_dir: str) -> dict[str, list[tuple[float, float]]]:
    """asset -> sorted [(ts, close)] from crypto_prices exports."""
    import pyarrow.parquet as pq
    import glob

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in glob.glob(os.path.join(data_dir, "*.parquet")):
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if not {"timestamp", "asset", "close"} <= set(names):
            continue
        for batch in pf.iter_batches(batch_size=100_000, columns=["timestamp", "asset", "close"]):
            for row in batch.to_pylist():
                ts = to_epoch(row.get("timestamp"))
                close = row.get("close")
                if ts is None or close is None:
                    continue
                series[str(row.get("asset") or "")].append((ts, float(close)))
    for asset in series:
        series[asset].sort(key=lambda pair: pair[0])
    return series


def spot_asof(series, ts, max_lag=10.0):
    import bisect

    if not series:
        return None
    index = bisect.bisect_right(series, (ts, float("inf"))) - 1
    if index < 0:
        return None
    spot_ts, price = series[index]
    if ts - spot_ts > max_lag:
        return None
    return price


def load_trades(path: str):
    """market_slug -> {token_id: [(ts, price)]}; plus a global time index."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    columns = [c for c in ("timestamp", "market_id", "token_id", "price", "side", "size")
               if c in pf.schema_arrow.names]
    by_market: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for batch in pf.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            ts = to_epoch(row.get("timestamp"))
            price = row.get("price")
            if ts is None or price is None:
                continue
            market = str(row.get("market_id") or "")
            token = str(row.get("token_id") or "")
            by_market[market][token].append((ts, float(price)))
    for market in by_market:
        for token in by_market[market]:
            by_market[market][token].sort(key=lambda pair: pair[0])
    return by_market, columns


@dataclass
class TradeOutcome:
    slug: str
    asset: str
    side: str
    t_minus: float
    spot_move_pct: float
    entry_price: float
    settle_price: float
    tp_price: float
    sl_price: float
    exit_settle: float
    exit_tp: float
    exit_sl: float
    reason_tp: str
    reason_sl: str
    pnl_settle: float
    pnl_tp: float
    pnl_sl: float
    pnl_both: float
    traded_after_entry: int
    won: bool
    direction_correct: bool
    entry_staleness_s: float
    reason_both: str = "open_at_settle"
    exit_both: float = 0.0

    def as_row(self) -> dict:
        return {
            "slug": self.slug, "asset": self.asset, "side": self.side,
            "t_minus": self.t_minus,
            "spot_move_pct": round(self.spot_move_pct, 5),
            "entry_price": round(self.entry_price, 5),
            "tp_price": round(self.tp_price, 5), "sl_price": round(self.sl_price, 5),
            "won": self.won,
            "direction_correct": self.direction_correct,
            "entry_staleness_s": round(self.entry_staleness_s, 1),
            "exit_settle": round(self.exit_settle, 5),
            "exit_tp": round(self.exit_tp, 5), "reason_tp": self.reason_tp,
            "exit_sl": round(self.exit_sl, 5), "reason_sl": self.reason_sl,
            "exit_both": round(self.exit_both, 5), "reason_both": self.reason_both,
            "pnl_settle": round(self.pnl_settle, 6),
            "pnl_tp": round(self.pnl_tp, 6),
            "pnl_sl": round(self.pnl_sl, 6),
            "pnl_both": round(self.pnl_both, 6),
            "traded_after_entry": self.traded_after_entry,
        }


def simulate(
    trades_by_market, markets, spot, *,
    entry_seconds: list[float],
    move_pct: float = 0.05,
    tp: float = 0.10,
    sl: float = 0.05,
    notional: float = 100.0,
    coefficient: float = DEFAULT_COEFFICIENT,
    entry_slippage_bps: float = 0.0,
    exit_slippage_bps: float = 0.0,
    lookback_s: float = 60.0,
    verbose_every: int = 0,
):
    """Run the directional sniper over every market/token pair we can price."""
    meta_by_slug: dict[str, MarketMeta] = {}
    for meta in markets.values():
        if meta.slug and meta.outcome:
            meta_by_slug.setdefault(meta.slug, meta)

    outcomes: list[TradeOutcome] = []
    considered = skipped_meta = skipped_entry = 0

    for slug, tokens in trades_by_market.items():
        meta = meta_by_slug.get(slug)
        if meta is None or meta.end_ts is None or not meta.outcome:
            skipped_meta += 1
            continue
        if len(tokens) < 2:
            continue
        considered += 1

        # which token is UP? the winner's price converges to 1 by the close
        late: dict[str, float] = {}
        for token, series in tokens.items():
            tail = [p for ts, p in series if meta.end_ts - ts <= 120]
            if tail:
                late[token] = statistics.fmean(tail)
        if len(late) < 2:
            continue
        ranked = sorted(late.items(), key=lambda kv: kv[1], reverse=True)
        (top_token, top_price), (bottom_token, bottom_price) = ranked[0], ranked[1]
        if top_price - bottom_price < 0.05:
            continue  # ambiguous

        outcome_up = meta.outcome.startswith("UP")
        up_token = top_token if outcome_up else bottom_token
        down_token = bottom_token if outcome_up else top_token

        spot_series = spot.get(meta.asset) or []

        for entry_s in entry_seconds:
            target = meta.end_ts - entry_s
            # spot move into the entry moment
            now_price = spot_asof(spot_series, target)
            past_price = spot_asof(spot_series, target - lookback_s)
            if now_price is None or not past_price:
                continue
            move = (now_price / past_price - 1.0) * 100.0
            if abs(move) < move_pct:
                continue

            side = "UP" if move > 0 else "DOWN"
            token = up_token if side == "UP" else down_token
            series = tokens.get(token) or []
            # prevailing price = last trade at or before the entry moment
            before = [(ts, p) for ts, p in series if ts <= target]
            if not before:
                skipped_entry += 1
                continue
            entry_ts, entry_price = before[-1]
            if not 0.02 < entry_price < 0.98:
                skipped_entry += 1
                continue

            after = [(ts, p) for ts, p in series if ts > target]
            entry_fill = entry_price * (1 + entry_slippage_bps / 10_000.0)

            shares = notional / entry_fill
            entry_fee = taker_fee_usdc(shares, entry_fill, coefficient=coefficient)

            settle_price = 1.0 if (side == "UP") == outcome_up else 0.0
            tp_price = min(0.999, entry_price + tp)
            sl_price = max(0.001, entry_price - sl)

            def exit_fee_for(exit_price: float, reason: str) -> float:
                """Fees are only paid on a TRADE. Settlement is not a trade, and the
                payoff price (0.0 or 1.0) is outside the fee curve's domain."""
                if reason == "open_at_settle":
                    return 0.0
                clamped = min(0.999, max(0.001, exit_price))
                return taker_fee_usdc(shares, clamped, coefficient=coefficient)

            # ---- settle branch: hold to expiry, no second trade
            pnl_settle = shares * settle_price - notional - entry_fee

            # ---- TP-only branch (rule: exit at TP if reached, else settle)
            reason_tp = "open_at_settle"
            exit_tp = settle_price
            for ts, price in after:
                if price >= tp_price:
                    exit_tp = tp_price * (1 - exit_slippage_bps / 10_000.0)
                    reason_tp = "tp_hit"
                    break
            pnl_tp = (shares * (exit_tp - entry_fill) - entry_fee
                      - exit_fee_for(exit_tp, reason_tp))

            # ---- SL-only branch
            reason_sl = "open_at_settle"
            exit_sl = settle_price
            for ts, price in after:
                if price <= sl_price:
                    exit_sl = sl_price * (1 - exit_slippage_bps / 10_000.0)
                    reason_sl = "sl_hit"
                    break
            pnl_sl = (shares * (exit_sl - entry_fill) - entry_fee
                      - exit_fee_for(exit_sl, reason_sl))

            # ---- COMBINED branch: both orders live, first touch wins
            # This is the only branch that answers the real question, because a
            # live OCO pair exits on whichever level the price reaches first.
            # Ties inside one trade print are resolved as SL (conservative).
            reason_both = "open_at_settle"
            exit_both = settle_price
            for ts, price in after:
                if price <= sl_price:
                    exit_both = sl_price * (1 - exit_slippage_bps / 10_000.0)
                    reason_both = "sl_hit"
                    break
                if price >= tp_price:
                    exit_both = tp_price * (1 - exit_slippage_bps / 10_000.0)
                    reason_both = "tp_hit"
                    break
            pnl_both = (shares * (exit_both - entry_fill) - entry_fee
                        - exit_fee_for(exit_both, reason_both))

            outcomes.append(TradeOutcome(
                slug=slug, asset=meta.asset, side=side, t_minus=entry_s,
                spot_move_pct=move, entry_price=entry_price,
                settle_price=settle_price, tp_price=tp_price, sl_price=sl_price,
                exit_settle=settle_price, exit_tp=exit_tp, exit_sl=exit_sl,
                reason_tp=reason_tp, reason_sl=reason_sl,
                pnl_settle=pnl_settle, pnl_tp=pnl_tp, pnl_sl=pnl_sl,
                pnl_both=pnl_both, reason_both=reason_both,
                exit_both=exit_both,
                traded_after_entry=len(after), won=pnl_settle > 0,
                direction_correct=((side == "UP") == outcome_up),
                entry_staleness_s=target - entry_ts,
            ))

    return outcomes, {"considered": considered, "skipped_meta": skipped_meta,
                      "skipped_entry": skipped_entry}


def summarise(outcomes: list[TradeOutcome]) -> dict:
    if not outcomes:
        return {"trades": 0}

    def block(pnls: list[float]) -> dict:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        equity = peak = dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = max(dd, peak - equity)
        return {
            "trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "net_pnl": sum(pnls),
            "avg_pnl": statistics.fmean(pnls),
            "avg_win": statistics.fmean(wins) if wins else 0.0,
            "avg_loss": statistics.fmean(losses) if losses else 0.0,
            "profit_factor": (sum(wins) / abs(sum(losses))) if losses else None,
            "max_drawdown": dd,
        }

    after = [o.traded_after_entry for o in outcomes]
    staleness = [o.entry_staleness_s for o in outcomes]
    return {
        "trades": len(outcomes),
        "settle": block([o.pnl_settle for o in outcomes]),
        "tp": block([o.pnl_tp for o in outcomes]),
        "sl": block([o.pnl_sl for o in outcomes]),
        "both": block([o.pnl_both for o in outcomes]),
        "tp_hit_rate": sum(1 for o in outcomes if o.reason_tp == "tp_hit") / len(outcomes),
        "sl_hit_rate": sum(1 for o in outcomes if o.reason_sl == "sl_hit") / len(outcomes),
        "both_tp_rate": sum(1 for o in outcomes if o.reason_both == "tp_hit") / len(outcomes),
        "both_sl_rate": sum(1 for o in outcomes if o.reason_both == "sl_hit") / len(outcomes),
        # NOTE: these are two different things and conflating them is how a
        # backtest invents an edge. `direction_correct` is signal accuracy;
        # `settle_win_rate` is the fraction of PROFITABLE trades, which is
        # inflated whenever the average entry price is below 0.50.
        "direction_hit_rate": sum(1 for o in outcomes if o.direction_correct) / len(outcomes),
        "settle_win_rate": sum(1 for o in outcomes if o.won) / len(outcomes),
        "avg_entry_price": statistics.fmean(o.entry_price for o in outcomes),
        "breakeven_hit_rate": statistics.fmean(o.entry_price for o in outcomes),
        "avg_trades_after_entry": statistics.fmean(after),
        "no_trade_after_entry": sum(1 for x in after if x == 0) / len(outcomes),
        "avg_entry_staleness_s": statistics.fmean(staleness),
        "p90_entry_staleness_s": sorted(staleness)[int(0.9 * len(staleness))],
        "avg_spot_move": statistics.fmean(o.spot_move_pct for o in outcomes),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(PROJECT, "..", "data"))
    ap.add_argument("--trades", default="", help="trades parquet (default: biggest in --data)")
    ap.add_argument("--entries", default="120,60,30,15,5", help="T-minus checkpoints to test")
    ap.add_argument("--move-pct", type=float, default=0.05, help="spot move needed, in percent")
    ap.add_argument("--tp", type=float, default=0.10)
    ap.add_argument("--sl", type=float, default=0.05)
    ap.add_argument("--entry-slip-bps", type=float, default=0.0)
    ap.add_argument("--exit-slip-bps", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(PROJECT, "output", "directional"))
    args = ap.parse_args(argv)

    import glob
    import pyarrow.parquet as pq

    data_dir = os.path.abspath(args.data)
    trades_path = args.trades
    if not trades_path:
        # prefer a file that actually has side/size (the richest trade export)
        candidates = []
        for path in glob.glob(os.path.join(data_dir, "*.parquet")):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if {"timestamp", "price", "market_id"} <= names:
                size = os.path.getsize(path)
                rich = "side" in names
                candidates.append((rich, size, path))
        if not candidates:
            print("no trades parquet found")
            return 2
        candidates.sort(reverse=True)
        trades_path = candidates[0][2]

    print("=" * 96)
    print("DIRECTIONAL SNIPER - TP/SL AND WIN-RATE STUDY (real trade prices)")
    print("=" * 96)
    print(f"trades file : {os.path.basename(trades_path)}")
    print(f"data dir    : {data_dir}")

    markets = load_markets(data_dir)
    with_outcome = sum(1 for m in markets.values() if m.outcome)
    print(f"markets     : {len(markets):,} ({with_outcome:,} with an outcome)")
    if not with_outcome:
        print("no outcomes found - cannot label anything")
        return 1

    print("\nloading spot prices ...")
    spot = load_spot(data_dir)
    print(f"  {len(spot)} assets, {sum(len(v) for v in spot.values()):,} bars")

    print("loading trades ...")
    trades_by_market, columns = load_trades(trades_path)
    total_trades = sum(len(s) for t in trades_by_market.values() for s in t.values())
    print(f"  {total_trades:,} trades across {len(trades_by_market):,} markets")
    print(f"  columns: {columns}")

    entries = [float(x) for x in str(args.entries).split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    def run(label: str, **kw) -> dict:
        outcomes, stats = simulate(
            trades_by_market, markets, spot, entry_seconds=entries,
            move_pct=kw.pop("move_pct", args.move_pct),
            tp=kw.pop("tp", args.tp), sl=kw.pop("sl", args.sl),
            entry_slippage_bps=kw.pop("entry_slippage_bps", args.entry_slip_bps),
            exit_slippage_bps=kw.pop("exit_slippage_bps", args.exit_slip_bps),
            **kw,
        )
        summary = summarise(outcomes)
        summary["label"] = label
        summary["considered_markets"] = stats["considered"]
        return summary, outcomes

    print("\n[1/4] ideal fills (no slippage) ...")
    base, base_outcomes = run("ideal")
    print("[2/4] live-realistic fills (entry +50bps, exit -50bps, adverse) ...")
    live, _ = run("live", entry_slippage_bps=50.0, exit_slippage_bps=50.0)

    if not base_outcomes:
        print("\nno trades were generated - the entry filter is too strict for this file.")
        print("try a smaller --move-pct, e.g. --move-pct 0.01")
        return 1

    with open(os.path.join(args.out, "trades.csv"), "w", newline="", encoding="utf-8") as fh:
        rows = [o.as_row() for o in base_outcomes]
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[3/4] entry-timing sweep ...")
    timing = []
    for entry_s in entries:
        subset = [o for o in base_outcomes if o.t_minus == entry_s]
        if not subset:
            continue
        s = summarise(subset)
        timing.append({
            "t_minus": entry_s,
            "trades": s["trades"],
            "direction_hit_rate": s["direction_hit_rate"],
            "breakeven_hit_rate": s["breakeven_hit_rate"],
            "settle_win_rate": s["settle_win_rate"],
            "settle_avg_pnl": s["settle"]["avg_pnl"],
            "tp_hit_rate": s["tp_hit_rate"],
            "tp_avg_pnl": s["tp"]["avg_pnl"],
            "sl_hit_rate": s["sl_hit_rate"],
            "sl_avg_pnl": s["sl"]["avg_pnl"],
            "avg_price": s["avg_entry_price"],
            "avg_staleness_s": s["avg_entry_staleness_s"],
            "no_trade_after_entry": s["no_trade_after_entry"],
        })
    with open(os.path.join(args.out, "timing.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(timing[0].keys()))
        writer.writeheader()
        writer.writerows(timing)

    print("[4/4] TP/SL grid ...")
    grid = []
    for tp in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        for sl in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
            summary, _ = run(f"tp{tp}_sl{sl}", tp=tp, sl=sl)
            if not summary["trades"]:
                continue
            grid.append({
                "tp": tp, "sl": sl, "trades": summary["trades"],
                "tp_hit_rate": summary["tp_hit_rate"],
                "sl_hit_rate": summary["sl_hit_rate"],
                "tp_net_pnl": summary["tp"]["net_pnl"],
                "tp_avg_pnl": summary["tp"]["avg_pnl"],
                "tp_win_rate": summary["tp"]["win_rate"],
                "tp_profit_factor": summary["tp"]["profit_factor"],
                "sl_net_pnl": summary["sl"]["net_pnl"],
                "sl_avg_pnl": summary["sl"]["avg_pnl"],
                "sl_win_rate": summary["sl"]["win_rate"],
                "settle_net_pnl": summary["settle"]["net_pnl"],
                "settle_win_rate": summary["settle_win_rate"],
                # the branch that actually answers "what if I run both orders"
                "both_net_pnl": summary["both"]["net_pnl"],
                "both_avg_pnl": summary["both"]["avg_pnl"],
                "both_win_rate": summary["both"]["win_rate"],
                "both_tp_rate": summary["both_tp_rate"],
                "both_sl_rate": summary["both_sl_rate"],
                "both_profit_factor": summary["both"]["profit_factor"],
                "both_max_dd": summary["both"]["max_drawdown"],
            })
    with open(os.path.join(args.out, "tpsl_grid.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grid[0].keys()))
        writer.writeheader()
        writer.writerows(grid)

    # ---- report
    def show(title: str, s: dict) -> None:
        edge = s["direction_hit_rate"] - s["breakeven_hit_rate"]
        print(f"\n{title}")
        print(f"  trades            : {s['trades']:,}")
        print(f"  DIRECTION hit rate: {s['direction_hit_rate']*100:6.2f}%   "
              f"(break-even needs {s['breakeven_hit_rate']*100:.2f}% at avg entry "
              f"{s['avg_entry_price']:.4f})   edge {edge*100:+.2f} pts")
        print(f"  profitable trades : {s['settle_win_rate']*100:6.2f}%   "
              f"avg P&L {s['settle']['avg_pnl']:+.4f}   net {s['settle']['net_pnl']:+.2f}")
        print(f"  with TP {args.tp:<5} win rate: {s['tp']['win_rate']*100:6.2f}%   "
              f"avg P&L {s['tp']['avg_pnl']:+.4f}   net {s['tp']['net_pnl']:+.2f}   "
              f"TP hit {s['tp_hit_rate']*100:.1f}%")
        print(f"  with SL {args.sl:<5} win rate: {s['sl']['win_rate']*100:6.2f}%   "
              f"avg P&L {s['sl']['avg_pnl']:+.4f}   net {s['sl']['net_pnl']:+.2f}   "
              f"SL hit {s['sl_hit_rate']*100:.1f}%")
        print(f"  BOTH orders live (first touch wins): "
              f"win {s['both']['win_rate']*100:6.2f}%   "
              f"avg P&L {s['both']['avg_pnl']:+.4f}   net {s['both']['net_pnl']:+.2f}   "
              f"TP {s['both_tp_rate']*100:.1f}% / SL {s['both_sl_rate']*100:.1f}%   "
              f"maxDD {s['both']['max_drawdown']:,.0f}")
        print(f"  hold to settlement (no TP/SL)^   : "
              f"win {s['settle']['win_rate']*100:6.2f}%   "
              f"avg P&L {s['settle']['avg_pnl']:+.4f}   net {s['settle']['net_pnl']:+.2f}   "
              f"maxDD {s['settle']['max_drawdown']:,.0f}")
        print(f"  entry staleness   : avg {s['avg_entry_staleness_s']:.1f}s   "
              f"p90 {s['p90_entry_staleness_s']:.1f}s   "
              f"(how old the last trade was when we 'filled')")
        print(f"  trades after entry: avg {s['avg_trades_after_entry']:.1f}   "
              f"windows with NONE after entry: {s['no_trade_after_entry']*100:.1f}%")

    show("IDEAL FILLS", base)
    show("LIVE-REALISTIC FILLS (entry +50bps, exit -50bps)", live)

    print("\nentry-timing sweep (ideal fills):")
    print(f"  {'T-':>5}{'trades':>8}{'DIR hit%':>10}{'b/e%':>8}{'edge pts':>10}"
          f"{'settle avg':>12}{'TP hit%':>9}{'SL hit%':>9}{'stale s':>9}{'none after':>12}")
    for row in timing:
        edge_pts = (row['direction_hit_rate'] - row['breakeven_hit_rate']) * 100
        print(f"  {row['t_minus']:>5.0f}{row['trades']:>8,}"
              f"{row['direction_hit_rate']*100:>9.2f}%{row['breakeven_hit_rate']*100:>7.2f}%"
              f"{edge_pts:>10.2f}{row['settle_avg_pnl']:>12.4f}"
              f"{row['tp_hit_rate']*100:>8.1f}%{row['sl_hit_rate']*100:>8.1f}%"
              f"{row['avg_staleness_s']:>9.1f}{row['no_trade_after_entry']*100:>11.1f}%")

    best = sorted(grid, key=lambda r: -r["both_net_pnl"])[:10]
    print("\nbest TP/SL combinations by net P&L of the COMBINED (OCO) rule:")
    print(f"  {'TP':>6}{'SL':>6}{'trades':>8}{'BOTH net':>11}{'BOTH avg':>10}{'BOTH win%':>11}"
          f"{'TP hit%':>9}{'SL hit%':>9}{'maxDD':>10}{'hold net':>11}")
    for row in best:
        print(f"  {row['tp']:>6.2f}{row['sl']:>6.2f}{row['trades']:>8,}"
              f"{row['both_net_pnl']:>11.2f}{row['both_avg_pnl']:>10.4f}"
              f"{row['both_win_rate']*100:>10.1f}%{row['both_tp_rate']*100:>8.1f}%"
              f"{row['both_sl_rate']*100:>8.1f}%{row['both_max_dd']:>10,.0f}"
              f"{row['settle_net_pnl']:>11.2f}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"ideal": base, "live": live, "timing": timing,
                   "grid_by_combined": sorted(grid, key=lambda r: -r["both_net_pnl"])[:50],
                   "grid_by_tp": sorted(grid, key=lambda r: -r["tp_net_pnl"])[:50]},
                  fh, indent=2, default=str)

    print(f"\noutputs: {args.out}")
    for name in ("trades.csv", "timing.csv", "tpsl_grid.csv", "summary.json"):
        p = os.path.join(args.out, name)
        if os.path.exists(p):
            print(f"  {name:<16} {os.path.getsize(p)/1024:>10,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

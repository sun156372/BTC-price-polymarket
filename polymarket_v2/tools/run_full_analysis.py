"""End-of-window sniper analysis, written for THIS dataset's real schema.

Your exports (all written by DuckDB v1.4.4) have exactly these columns:

  orderbooks/    timestamp, asset, market_id, condition_id, token_id, question,
                 best_bid, best_ask, spread, mid_price, bid_levels, ask_levels
  crypto_prices/ timestamp, asset, open, high, low, close, volume, trade_count
  trades/        timestamp, asset, market_id, condition_id, token_id, side,
                 price, size, tx_hash, maker, source
  price_history/ timestamp, asset, market_id, token_id, price
  resolution/    market_id, asset, condition_id, question, start_time, end_time,
                 resolved_at, outcome          <- ground truth, 28,016 markets

What it answers
---------------
1. Does the "panic dump + hedge for UP+DOWN <= sumTarget" trade actually occur,
   how often, and at which T-minus offsets?
2. When it occurs, does the *cheap* side win? (If not, the "free" 5c is really the
   market pricing a losing outcome, and the strategy is picking up pennies in
   front of a steamroller.)
3. Which entry offset maximises net P&L **after** the real taker fee
   ``shares * 0.07 * p * (1-p)`` and depth-based slippage?
4. How do movePct / windowMin / sumTarget change the outcome - and what are the
   best values on this data?

Run
---
    pip install pyarrow                      # needed to read parquet VALUES
    python tools/run_full_analysis.py --root "..\\data" --out output\\analysis

Memory: streams row groups, keeps only per-market rollups. Safe on 1.88 GB.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from dataset_loader import parse_ladder  # noqa: E402
    break_even_move,
    taker_fee_usdc,
    DEFAULT_COEFFICIENT,
)

#: candidate T-minus checkpoints for the entry-timing sweep
DEFAULT_OFFSETS = [240, 180, 120, 90, 60, 45, 30, 20, 15, 10, 5]
#: tolerance (seconds) when matching an orderbook snapshot to a checkpoint
OFFSET_TOLERANCE = 3.0


# ---------------------------------------------------------------- data access
def require_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq
    except ImportError:
        print("=" * 96)
        print("This analysis needs to read parquet VALUES, which requires pyarrow.")
        print()
        print("    pip install pyarrow")
        print()
        print("(Your files were written by DuckDB, so if you have duckdb installed you")
        print(" can instead export once:  duckdb -c \"COPY (SELECT * FROM")
        print(" read_parquet('orderbooks/*.parquet')) TO 'orderbooks.csv' (HEADER)\"")
        print("=" * 96)
        raise SystemExit(2)


def to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_parquet_rows(path: str, columns: Sequence[str], batch_size: int = 100_000) -> Iterator[list[dict]]:
    pq = require_pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        yield batch.to_pylist()


def find_files(root: str, folder_hint: str, required: Sequence[str]) -> list[str]:
    """Find parquet files whose schema contains all ``required`` columns."""
    pq = require_pyarrow()
    matches: list[str] = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)):
        try:
            names = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:  # noqa: BLE001
            continue
        if all(name in names for name in required):
            if folder_hint and folder_hint.lower() not in path.lower():
                # still accept if the schema is unambiguous
                pass
            matches.append(path)
    return matches


# --------------------------------------------------------------------- ground truth
@dataclass
class MarketInfo:
    market_id: str
    asset: str = ""
    start_ts: float | None = None
    end_ts: float | None = None
    resolved_ts: float | None = None
    outcome: str = ""
    question: str = ""


def load_resolution(root: str) -> dict[str, MarketInfo]:
    files = find_files(root, "resolution", ["market_id", "outcome", "end_time"])
    if not files:
        # the single 'all (1).parquet' style file
        files = find_files(root, "", ["market_id", "outcome", "end_time", "start_time"])
    markets: dict[str, MarketInfo] = {}
    for path in files:
        for rows in iter_parquet_rows(
            path,
            ["market_id", "asset", "start_time", "end_time", "resolved_at", "outcome", "question"]
            if "question" in _columns_of(path)
            else ["market_id", "asset", "start_time", "end_time", "resolved_at", "outcome"],
        ):
            for row in rows:
                market_id = str(row.get("market_id") or "")
                if not market_id:
                    continue
                markets[market_id] = MarketInfo(
                    market_id=market_id,
                    asset=str(row.get("asset") or ""),
                    start_ts=to_epoch(row.get("start_time")),
                    end_ts=to_epoch(row.get("end_time")),
                    resolved_ts=to_epoch(row.get("resolved_at")),
                    outcome=str(row.get("outcome") or "").strip().upper(),
                    question=str(row.get("question") or ""),
                )
    return markets


_COLUMN_CACHE: dict[str, list[str]] = {}


def _columns_of(path: str) -> list[str]:
    if path not in _COLUMN_CACHE:
        pq = require_pyarrow()
        _COLUMN_CACHE[path] = list(pq.ParquetFile(path).schema_arrow.names)
    return _COLUMN_CACHE[path]


# ------------------------------------------------------------- market rollups
@dataclass
class Snapshot:
    """One orderbook observation, reduced to what the strategy needs."""

    ts: float
    t_minus: float
    token_id: str
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    bid_depth_notional: float
    ask_depth_notional: float
    ask_vwap_small: float | None
    ask_vwap_big: float | None


@dataclass
class MarketSeries:
    market_id: str
    asset: str = ""
    by_token: dict[str, list[Snapshot]] = field(default_factory=lambda: defaultdict(list))
    first_ts: float | None = None
    last_ts: float | None = None

    def add(self, snapshot: Snapshot) -> None:
        self.by_token[snapshot.token_id].append(snapshot)
        self.first_ts = snapshot.ts if self.first_ts is None else min(self.first_ts, snapshot.ts)
        self.last_ts = snapshot.ts if self.last_ts is None else max(self.last_ts, snapshot.ts)


def collect_orderbooks(
    root: str,
    markets: dict[str, MarketInfo],
    *,
    probe_notional: float = 100.0,
    big_notional: float = 500.0,
    progress: bool = True,
) -> dict[str, MarketSeries]:
    """Stream every orderbook file into per-market series.

    Only markets present in ``resolution`` are kept, and each snapshot is stamped
    with ``t_minus`` (seconds to window close) so the timing sweep is a lookup
    rather than a re-scan.
    """
    files = find_files(root, "orderbooks", ["best_bid", "best_ask", "token_id", "timestamp"])
    if not files:
        raise SystemExit("no orderbook files found (need best_bid/best_ask/token_id/timestamp)")

    series: dict[str, MarketSeries] = {}
    kept = skipped_no_market = skipped_no_book = 0

    for path in files:
        if progress:
            print(f"  reading {os.path.basename(path)} ...", flush=True)
        for rows in iter_parquet_rows(
            path,
            ["timestamp", "market_id", "token_id", "best_bid", "best_ask", "bid_levels", "ask_levels"],
        ):
            for row in rows:
                market_id = str(row.get("market_id") or "")
                info = markets.get(market_id)
                if info is None or info.end_ts is None:
                    skipped_no_market += 1
                    continue
                ts = to_epoch(row.get("timestamp"))
                if ts is None:
                    continue

                bid = row.get("best_bid")
                ask = row.get("best_ask")
                if bid is None and ask is None:
                    bid_ladder = parse_ladder(row.get("bid_levels"))
                    ask_ladder = parse_ladder(row.get("ask_levels"))
                    bid = max(bid_ladder, key=lambda p: p[0])[0] if bid_ladder else None
                    ask = min(ask_ladder, key=lambda p: p[0])[0] if ask_ladder else None
                if bid is None and ask is None:
                    skipped_no_book += 1
                    continue
                if bid is not None and not 0.0 < bid < 1.0:
                    bid = None
                if ask is not None and not 0.0 < ask < 1.0:
                    ask = None
                if bid is None and ask is None:
                    skipped_no_book += 1
                    continue

                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else (bid or ask)

                bid_ladder = parse_ladder(row.get("bid_levels"))
                ask_ladder = parse_ladder(row.get("ask_levels"))
                ask_ladder.sort(key=lambda pair: pair[0])
                bid_ladder.sort(key=lambda pair: pair[0], reverse=True)

                bid_notional = sum(p * s for p, s in bid_ladder)
                ask_notional = sum(p * s for p, s in ask_ladder)
                _, _, small_vwap, _ = walk_ladder(ask_ladder, probe_notional / max(1e-9, ask or 0.5)) if ask_ladder else (0, 0, 0.0, False)
                _, _, big_vwap, _ = walk_ladder(ask_ladder, big_notional / max(1e-9, ask or 0.5)) if ask_ladder else (0, 0, 0.0, False)

                entry = series.get(market_id)
                if entry is None:
                    entry = MarketSeries(market_id=market_id, asset=info.asset)
                    series[market_id] = entry
                entry.add(
                    Snapshot(
                        ts=ts,
                        t_minus=info.end_ts - ts,
                        token_id=str(row.get("token_id") or ""),
                        best_bid=float(bid) if bid is not None else None,
                        best_ask=float(ask) if ask is not None else None,
                        mid=float(mid) if mid is not None else None,
                        bid_depth_notional=bid_notional,
                        ask_depth_notional=ask_notional,
                        ask_vwap_small=float(small_vwap) or None,
                        ask_vwap_big=float(big_vwap) or None,
                    )
                )
                kept += 1

    if progress:
        print(
            f"  kept {kept:,} snapshots across {len(series):,} markets "
            f"(skipped {skipped_no_market:,} without resolution, {skipped_no_book:,} without a book)"
        )
    return series


def identify_up_token(entry: MarketSeries, info: MarketInfo) -> tuple[str | None, str | None, str]:
    """Work out which token_id is UP and which is DOWN.

    The orderbook export carries no outcome label, so the side is inferred from
    final prices: the winning token trades near 1.0 at the close. When the two
    tokens cannot be told apart (both near 0.5, or no late snapshots) the market
    is skipped rather than guessed at.
    """
    if len(entry.by_token) < 2:
        return None, None, "only one token present"
    late: dict[str, float] = {}
    for token, snapshots in entry.by_token.items():
        tail = [s for s in snapshots if s.t_minus <= 60 and s.mid is not None]
        if tail:
            late[token] = statistics.fmean([s.mid for s in tail])
    if len(late) < 2:
        return None, None, "no late prices for both tokens"
    ranked = sorted(late.items(), key=lambda kv: kv[1], reverse=True)
    (top_token, top_price), (bottom_token, bottom_price) = ranked[0], ranked[1]
    if top_price - bottom_price < 0.05:
        return None, None, f"tokens too close to call ({top_price:.3f} vs {bottom_price:.3f})"
    outcome_up = info.outcome.startswith("UP") or info.outcome in {"UP", "YES", "1", "TRUE"}
    implied_up = top_token if outcome_up else bottom_token
    implied_down = bottom_token if outcome_up else top_token
    return implied_up, implied_down, ""


# --------------------------------------------------------------- strategy core
@dataclass
class Params:
    """The tunables the user asked to optimize, plus the hedge target."""

    move_pct: float = 0.15        # panic-dump threshold, in percent
    window_min: float = 2.0       # look-back window to measure the dump, minutes
    sum_target: float = 0.95      # buy both sides when up+down <= this
    entry_t_minus: float = 30.0   # seconds before close to act
    taker: bool = True


@dataclass
class Opportunity:
    market_id: str
    asset: str
    t_minus: float
    up_token: str
    down_token: str
    up_ask: float | None
    down_ask: float | None
    sum_cost: float | None
    up_mid: float | None
    down_mid: float | None
    spot_move_pct: float | None
    outcome_up: bool
    winner: str
    cheap_side: str
    cheap_won: bool


@dataclass
class SnapshotFeatureRow:
    """One checkpoint row per market - the raw material for the timing sweep."""

    market_id: str
    asset: str
    t_minus: float
    up_mid: float | None
    down_mid: float | None
    up_ask: float | None
    down_ask: float | None
    sum_ask: float | None
    sum_mid: float | None
    up_spread: float | None
    down_spread: float | None
    up_ask_depth: float
    down_ask_depth: float
    spot_price: float | None
    spot_move_3s: float | None
    spot_move_60s: float | None
    spot_move_window: float | None
    outcome_up: bool


def load_spot_series(root: str) -> dict[str, list[tuple[float, float]]]:
    """asset -> sorted [(ts, close)] for the crypto_prices exports."""
    files = find_files(root, "crypto_prices", ["timestamp", "close"])
    if not files:
        files = find_files(root, "", ["timestamp", "asset", "open", "high", "low", "close", "volume"])
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in files:
        columns = ["timestamp", "asset", "close"]
        for rows in iter_parquet_rows(path, columns):
            for row in rows:
                ts = to_epoch(row.get("timestamp"))
                close = row.get("close")
                asset = str(row.get("asset") or "")
                if ts is None or close is None:
                    continue
                try:
                    series[asset].append((ts, float(close)))
                except (TypeError, ValueError):
                    continue
    for asset in series:
        series[asset].sort(key=lambda pair: pair[0])
    return series


def spot_asof(series: Sequence[tuple[float, float]], ts: float, max_lag: float = 5.0) -> tuple[float | None, float | None]:
    """Backward-looking lookup; never uses a future price."""
    import bisect

    if not series:
        return None, None
    index = bisect.bisect_right(series, (ts, math.inf)) - 1
    if index < 0:
        return None, None
    spot_ts, price = series[index]
    if ts - spot_ts > max_lag:
        return None, spot_ts
    return price, spot_ts


def build_checkpoint_rows(
    series: dict[str, MarketSeries],
    markets: dict[str, MarketInfo],
    spot: dict[str, list[tuple[float, float]]],
    offsets: Sequence[float],
    *,
    tolerance: float = OFFSET_TOLERANCE,
    progress: bool = True,
    keep_opportunities: list[Opportunity] | None = None,
) -> list[SnapshotFeatureRow]:
    """One row per (market, checkpoint) with everything the sweep needs."""
    rows: list[SnapshotFeatureRow] = []
    processed = 0
    ambiguous = 0
    for market_id, entry in series.items():
        info = markets.get(market_id)
        if info is None or info.end_ts is None:
            continue
        up_token, down_token, _reason = identify_up_token(entry, info)
        if not up_token or not down_token:
            ambiguous += 1
            continue
        outcome_up = info.outcome.startswith("UP") or info.outcome in {"UP", "YES", "1", "TRUE"}

        up_snaps = sorted(entry.by_token[up_token], key=lambda s: s.ts)
        down_snaps = sorted(entry.by_token[down_token], key=lambda s: s.ts)
        if not up_snaps and not down_snaps:
            continue
        spot_series = spot.get(entry.asset) or spot.get("") or []

        for offset in offsets:
            target = info.end_ts - offset
            up = _nearest(up_snaps, target, tolerance)
            down = _nearest(down_snaps, target, tolerance)
            if up is None and down is None:
                continue

            spot_price, _spot_ts = spot_asof(spot_series, target)
            move_3s = _spot_change(spot_series, target, 3.0)
            move_60s = _spot_change(spot_series, target, 60.0)
            move_window = _spot_change(spot_series, target, 120.0)

            sum_ask = (
                (up.best_ask or up.mid) + (down.best_ask or down.mid)
                if up and down and (up.best_ask or up.mid) and (down.best_ask or down.mid)
                else None
            )
            sum_mid = (
                up.mid + down.mid if up and down and up.mid and down.mid else None
            )

            rows.append(
                SnapshotFeatureRow(
                    market_id=market_id,
                    asset=entry.asset,
                    t_minus=offset,
                    up_mid=up.mid if up else None,
                    down_mid=down.mid if down else None,
                    up_ask=up.best_ask if up else None,
                    down_ask=down.best_ask if down else None,
                    sum_ask=sum_ask,
                    sum_mid=sum_mid,
                    up_spread=(up.best_ask - up.best_bid)
                    if up and up.best_ask is not None and up.best_bid is not None
                    else None,
                    down_spread=(down.best_ask - down.best_bid)
                    if down and down.best_ask is not None and down.best_bid is not None
                    else None,
                    up_ask_depth=up.ask_depth_notional if up else 0.0,
                    down_ask_depth=down.ask_depth_notional if down else 0.0,
                    spot_price=spot_price,
                    spot_move_3s=move_3s,
                    spot_move_60s=move_60s,
                    spot_move_window=move_window,
                    outcome_up=outcome_up,
                )
            )
            if keep_opportunities is not None and sum_ask is not None and sum_ask <= 1.0001:
                cheap = "UP" if (up.mid or 1.0) < (down.mid or 1.0) else "DOWN"
                keep_opportunities.append(
                    Opportunity(
                        market_id=market_id,
                        asset=entry.asset,
                        t_minus=offset,
                        up_token=up_token,
                        down_token=down_token,
                        up_ask=up.best_ask if up else None,
                        down_ask=down.best_ask if down else None,
                        sum_cost=sum_ask,
                        up_mid=up.mid if up else None,
                        down_mid=down.mid if down else None,
                        spot_move_pct=move_window,
                        outcome_up=outcome_up,
                        winner="UP" if outcome_up else "DOWN",
                        cheap_side=cheap,
                        cheap_won=(cheap == "UP") == outcome_up,
                    )
                )
        processed += 1
        if progress and processed % 500 == 0:
            print(f"    ... {processed:,} markets processed", flush=True)

    if progress:
        print(f"  checkpoints: {len(rows):,} rows over {processed:,} markets ({ambiguous:,} skipped as ambiguous)")
    return rows


def _nearest(snapshots: Sequence[Snapshot], target: float, tolerance: float) -> Snapshot | None:
    best = None
    best_gap = None
    for snapshot in snapshots:
        gap = abs(snapshot.ts - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = snapshot, gap
        elif snapshot.ts > target and best_gap is not None and gap > best_gap:
            break
    if best is None or best_gap is None or best_gap > tolerance:
        return None
    return best


def _spot_change(series: Sequence[tuple[float, float]], ts: float, lookback: float) -> float | None:
    now_price, _ = spot_asof(series, ts)
    past_price, _ = spot_asof(series, ts - lookback)
    if now_price is None or not past_price:
        return None
    return (now_price / past_price - 1.0) * 100.0


# ------------------------------------------------------------------ backtest
def hedge_pnl(
    up_ask: float,
    down_ask: float,
    *,
    notional: float = 100.0,
    coefficient: float = DEFAULT_COEFFICIENT,
    slippage_bps: float = 0.0,
    mode: str = "equal_notional",
    units: float | None = None,
) -> dict:
    """Buy both sides and hold to settlement, paying real taker fees.

    **The distinction that decides whether this strategy makes or loses money:**

    ``equal_notional`` (what the described bot currently does)
        Spend the same USDC on each side: ``shares = notional / ask``. The payout
        is ``shares * 1.0`` on the winning side only, so P&L depends entirely on
        **which side wins**. On a 0.30/0.60 pair the cheap side pays 125.6 while
        the expensive side pays only 41.0 against a ~200.6 cost - so one branch
        loses ~41. This is a directional bet dressed up as a hedge. There is no
        lock, and "UP+DOWN <= 0.95" does not change that.

    ``lock`` (real arbitrage)
        Buy **equal share counts** on both sides, so both settlement branches pay
        the same ``units``. Cost is ``units * (up_ask + down_ask)`` plus fees, so
        the trade is risk-free **iff**::

            units * (ask_up + ask_down) + fees < units
            <=>  ask_up + ask_down + fee_per_unit < 1

        The profit is therefore ``units * (1 - sum_ask - fee_per_unit)`` and scales
        linearly with size - bounded in practice by the depth on the expensive
        side and by the fee curve, which peaks at p = 0.5.

    ``pnl_worst`` is always the worse of the two settlement branches: the honest
    number for anyone who cannot predict the winner.
    """
    slip = slippage_bps / 10_000.0
    up_price = up_ask * (1 + slip)
    down_price = down_ask * (1 + slip)
    if up_price <= 0 or down_price <= 0:
        raise ValueError("ask prices must be positive")

    if mode == "equal_notional":
        up_shares = notional / up_price
        down_shares = notional / down_price
    elif mode == "lock":
        if units is None:
            # size the lock so the expensive leg costs `notional`, which is what a
            # real order would be depth-limited by
            units = notional / max(up_price, down_price)
        up_shares = down_shares = float(units)
    else:
        raise ValueError(f"unknown hedge mode: {mode!r}")

    up_fee = taker_fee_usdc(up_shares, up_price, coefficient=coefficient)
    down_fee = taker_fee_usdc(down_shares, down_price, coefficient=coefficient)
    cost = up_shares * up_price + down_shares * down_price + up_fee + down_fee

    pnl_if_up = up_shares - cost      # UP settles at 1.0, DOWN at 0.0
    pnl_if_down = down_shares - cost  # DOWN settles at 1.0, UP at 0.0
    return {
        "mode": mode,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "sum_ask": up_ask + down_ask,
        "up_shares": up_shares,
        "down_shares": down_shares,
        "fees": up_fee + down_fee,
        "fee_per_unit": (up_fee + down_fee) / max(1e-12, max(up_shares, down_shares)),
        "cost": cost,
        "pnl_if_up": pnl_if_up,
        "pnl_if_down": pnl_if_down,
        "pnl_worst": min(pnl_if_up, pnl_if_down),
        "pnl_best": max(pnl_if_up, pnl_if_down),
        "guaranteed_profit": min(pnl_if_up, pnl_if_down) > 0,
    }


def lock_break_even_sum(up_ask: float, down_ask: float, coefficient: float = DEFAULT_COEFFICIENT) -> float:
    """The largest ``up_ask + down_ask`` that still locks a profit after fees.

    Solves ``sum + fee_per_unit < 1`` where the fee per unit is
    ``coefficient * p * (1-p)`` summed over both legs. This is the number the
    ``sumTarget`` parameter should actually be set from - not guessed.
    """
    fee_per_unit = coefficient * (
        up_ask * (1 - up_ask) + down_ask * (1 - down_ask)
    )
    return 1.0 - fee_per_unit


def sweep_timing(rows: Sequence[SnapshotFeatureRow], params: Params) -> list[dict]:
    """For every checkpoint, how both hedge constructions perform after fees.

    Reports ``equal_notional`` (the described bot: directional, can lose) next to
    ``share_ratio`` (true lock: profitable only when the pair is cheap enough).
    """
    by_offset: dict[float, list[SnapshotFeatureRow]] = defaultdict(list)
    for row in rows:
        by_offset[row.t_minus].append(row)

    results = []
    for offset in sorted(by_offset, reverse=True):
        group = by_offset[offset]
        candidates = [r for r in group if r.sum_ask is not None and r.sum_ask <= params.sum_target]
        if not candidates:
            results.append({
                "t_minus": offset, "markets": len(group), "triggered": 0,
                "trigger_rate": 0.0, "avg_sum_ask": None,
                "equal_notional_avg_worst": 0.0, "equal_notional_avg_best": 0.0,
                "lock_avg_pnl": 0.0, "lock_win_rate": 0.0, "lock_guaranteed": 0,
                "net_pnl": 0.0, "avg_pnl": 0.0, "win_rate": 0.0, "worst": None,
            })
            continue

        worst_pnls: list[float] = []
        best_pnls: list[float] = []
        lock_pnls: list[float] = []
        guaranteed = 0
        for row in candidates:
            # slippage proxy: thin books cost more to cross
            slip_bps = 0.0
            if row.up_ask_depth and row.down_ask_depth:
                thin = min(row.up_ask_depth, row.down_ask_depth)
                slip_bps = max(0.0, 60.0 * math.exp(-thin / 200.0))
            equal = hedge_pnl(
                row.up_ask or 0.0, row.down_ask or 0.0,
                slippage_bps=slip_bps, mode="equal_notional",
            )
            lock = hedge_pnl(
                row.up_ask or 0.0, row.down_ask or 0.0,
                slippage_bps=slip_bps, mode="lock",
            )
            worst_pnls.append(equal["pnl_worst"])
            best_pnls.append(equal["pnl_best"])
            lock_pnls.append(lock["pnl_worst"])
            if lock["guaranteed_profit"]:
                guaranteed += 1

        results.append({
            "t_minus": offset,
            "markets": len(group),
            "triggered": len(candidates),
            "trigger_rate": len(candidates) / len(group),
            "avg_sum_ask": statistics.fmean([r.sum_ask for r in candidates if r.sum_ask]),
            "equal_notional_avg_worst": statistics.fmean(worst_pnls),
            "equal_notional_avg_best": statistics.fmean(best_pnls),
            "lock_avg_pnl": statistics.fmean(lock_pnls),
            "lock_win_rate": sum(1 for p in lock_pnls if p > 0) / len(lock_pnls),
            "lock_guaranteed": guaranteed,
            "net_pnl": sum(lock_pnls),
            "avg_pnl": statistics.fmean(lock_pnls),
            "win_rate": sum(1 for p in lock_pnls if p > 0) / len(lock_pnls),
            "worst": min(lock_pnls),
        })
    return results


def sweep_params(
    rows: Sequence[SnapshotFeatureRow], offsets: Sequence[float]
) -> list[dict]:
    """Grid over movePct x windowMin x sumTarget at the best-ish entry times."""
    moves = [0.0, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5]
    windows = [0.0, 0.5, 1.0, 2.0, 3.0]
    sums = [0.90, 0.93, 0.95, 0.97, 0.99]
    out: list[dict] = []

    indexed: dict[tuple[float, float], list[SnapshotFeatureRow]] = defaultdict(list)
    for row in rows:
        indexed[(row.t_minus, row.spot_move_window if row.spot_move_window is not None else 0.0)].append(row)

    for offset in offsets:
        group = [r for r in rows if r.t_minus == offset]
        if not group:
            continue
        for move in moves:
            for window in windows:
                for target in sums:
                    selected = []
                    for row in group:
                        if row.sum_ask is None or row.sum_ask > target:
                            continue
                        move_value = row.spot_move_window if window >= 1.0 else (
                            row.spot_move_60s if window >= 0.5 else row.spot_move_3s
                        )
                        if move > 0 and (move_value is None or abs(move_value) < move):
                            continue
                        selected.append(row)
                    if not selected:
                        continue
                    pnls = []
                    for row in selected:
                        result = hedge_pnl(row.up_ask or 0.0, row.down_ask or 0.0)
                        pnls.append(result["pnl_worst"])
                    out.append({
                        "t_minus": offset,
                        "move_pct": move,
                        "window_min": window,
                        "sum_target": target,
                        "trades": len(selected),
                        "net_pnl": sum(pnls),
                        "avg_pnl": statistics.fmean(pnls),
                        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    })
    return out


# --------------------------------------------------------------------- report
def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return path
    columns = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_report(
    markets: dict[str, MarketInfo],
    series: dict[str, MarketSeries],
    rows: Sequence[SnapshotFeatureRow],
    timing: Sequence[dict],
    best_params: Sequence[dict],
    min_sum: float,
) -> None:
    print()
    print("=" * 96)
    print("FINDINGS")
    print("=" * 96)

    assets = Counter(info.asset for info in markets.values())
    outcomes = Counter(info.outcome for info in markets.values())
    spans = [info.end_ts for info in markets.values() if info.end_ts]
    print(f"\nmarkets with resolution : {len(markets):,}")
    if spans:
        print(f"window closes span      : {datetime.fromtimestamp(min(spans), tz=timezone.utc):%Y-%m-%d} "
              f"-> {datetime.fromtimestamp(max(spans), tz=timezone.utc):%Y-%m-%d}")
    print(f"asset mix               : {dict(assets.most_common(8))}")
    print(f"outcome mix             : {dict(outcomes.most_common(6))}")
    print(f"markets with order books: {len(series):,}")
    print(f"checkpoint rows         : {len(rows):,}")

    # --- the headline: does UP+DOWN ever get cheap?
    sums = [r.sum_ask for r in rows if r.sum_ask is not None]
    if sums:
        sums_sorted = sorted(sums)
        def pct(p: float) -> float:
            return sums_sorted[min(len(sums_sorted) - 1, int(p * len(sums_sorted)))]
        print(f"\nUP+DOWN (ask) distribution across {len(sums):,} checkpoints:")
        print(f"  min {sums_sorted[0]:.4f}   p1 {pct(0.01):.4f}   p5 {pct(0.05):.4f}   "
              f"median {pct(0.50):.4f}   p95 {pct(0.95):.4f}   max {sums_sorted[-1]:.4f}")
        for target in (0.90, 0.93, 0.95, 0.97, 0.99, 1.00):
            hits = sum(1 for value in sums if value <= target)
            print(f"  share of checkpoints with sum <= {target:.2f}: {hits / len(sums) * 100:6.2f}%  ({hits:,})")

    # --- fee reality check
    print("\nfee reality check (taker, p=0.5 -> worst case):")
    for price in (0.10, 0.30, 0.50, 0.70, 0.90):
        print(f"  p={price:.2f}: fee {taker_fee_usdc(100, price) / 100:.4%} of notional per side, "
              f"break-even move {break_even_move(price):.4f}")

    # --- timing sweep
    print("\nentry-timing sweep (buy both sides whenever sum_ask <= 0.95, fees + depth slippage included):")
    print("  equal_notional = what the described bot does (directional, can lose)")
    print("  share_ratio    = true lock, profitable only if the pair is cheap enough")
    print(f"  {'T-minus':>8}{'mkts':>6}{'trig':>7}{'rate':>7}{'avg sum':>9}"
          f"{'eq worst':>10}{'eq best':>10}{'lock avg':>10}{'lock win%':>10}{'locked':>8}")
    for row in timing:
        if not row["triggered"]:
            continue
        print(f"  {row['t_minus']:>8.0f}{row['markets']:>6,}{row['triggered']:>7,}"
              f"{row['trigger_rate'] * 100:>6.1f}%{row['avg_sum_ask']:>9.4f}"
              f"{row['equal_notional_avg_worst']:>10.2f}{row['equal_notional_avg_best']:>10.2f}"
              f"{row['lock_avg_pnl']:>10.4f}{row['lock_win_rate'] * 100:>9.1f}%"
              f"{row['lock_guaranteed']:>8,}")

    print(f"\nlowest UP+DOWN ask seen anywhere: {min_sum:.4f}")

    if best_params:
        top = sorted(best_params, key=lambda r: (r["net_pnl"], r["trades"]), reverse=True)[:12]
        print("\nbest parameter combinations by net P&L (min 20 trades):")
        print(f"  {'T-':>5}{'move%':>8}{'window':>8}{'sum<=':>7}{'trades':>8}{'net P&L':>11}{'avg':>9}{'win%':>7}")
        for row in top:
            if row["trades"] < 20:
                continue
            print(f"  {row['t_minus']:>5.0f}{row['move_pct']:>8.2f}{row['window_min']:>8.1f}"
                  f"{row['sum_target']:>7.2f}{row['trades']:>8,}{row['net_pnl']:>11.2f}"
                  f"{row['avg_pnl']:>9.3f}{row['win_rate'] * 100:>6.1f}%")


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(PROJECT, "..", "data"),
                    help="folder holding the parquet exports (default: workspace data/)")
    ap.add_argument("--out", default=os.path.join(PROJECT, "output", "analysis"))
    ap.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    ap.add_argument("--max-markets", type=int, default=0, help="cap markets for a quick trial")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    offsets = [float(x) for x in str(args.offsets).split(",") if x.strip()]
    require_pyarrow()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 96)
    print("END-OF-WINDOW SNIPER ANALYSIS")
    print(f"root: {root}")
    print("=" * 96)

    print("\n[1/5] loading resolution (ground truth)...")
    markets = load_resolution(root)
    print(f"  {len(markets):,} markets with outcomes")
    if not markets:
        print("  ERROR: no resolution rows found - cannot label anything")
        return 1

    print("\n[2/5] loading spot prices...")
    spot = load_spot_series(root)
    print(f"  {len(spot)} asset series, "
          f"{sum(len(v) for v in spot.values()):,} bars")

    print("\n[3/5] streaming order books...")
    series = collect_orderbooks(root, markets)
    if args.max_markets:
        keys = list(series)[: args.max_markets]
        series = {k: series[k] for k in keys}
        print(f"  capped to {len(series):,} markets")

    print("\n[4/5] building checkpoints...")
    opportunities: list[Opportunity] = []
    rows = build_checkpoint_rows(series, markets, spot, offsets, keep_opportunities=opportunities)

    print("\n[5/5] sweeping entry timing and parameters...")
    timing = sweep_timing(rows, Params())
    best_params = sweep_params(rows, offsets)

    min_sum = min((r.sum_ask for r in rows if r.sum_ask is not None), default=float("nan"))
    print_report(markets, series, rows, timing, best_params, min_sum)

    write_csv(os.path.join(args.out, "timing_sweep.csv"), timing)
    write_csv(os.path.join(args.out, "param_sweep.csv"), best_params)
    write_csv(
        os.path.join(args.out, "checkpoints.csv"),
        [r.__dict__ for r in rows[:200_000]],
    )
    if opportunities:
        write_csv(
            os.path.join(args.out, "opportunities.csv"),
            [o.__dict__ for o in opportunities[:200_000]],
        )
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "markets": len(markets),
                "markets_with_books": len(series),
                "checkpoints": len(rows),
                "min_sum_ask": min_sum,
                "timing": timing,
                "best_params": sorted(best_params, key=lambda r: r["net_pnl"], reverse=True)[:50],
            },
            fh,
            indent=2,
            default=str,
        )

    print()
    print("=" * 96)
    print(f"outputs written to: {args.out}")
    for name in ("timing_sweep.csv", "param_sweep.csv", "checkpoints.csv",
                 "opportunities.csv", "summary.json"):
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            print(f"  {name:<22} {os.path.getsize(path) / 1024:>12,.0f} KB")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

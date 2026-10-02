"""Module 2a - backtest engine with real Polymarket fees and VWAP slippage.

What this engine models
-----------------------
* **Entry** - at ``T - observation_time`` the strategy looks at the market's
  implied probability and the underlying move, and decides whether to take UP or
  DOWN. The fill is priced by walking a book (:class:`SyntheticBookModel` builds
  a reproducible ladder when only a midpoint is available), so slippage is a
  modelled quantity, not an assumption.
* **Fees** - every fill charges the documented Polymarket taker fee
  ``shares * 0.07 * p * (1 - p)`` via :mod:`utils.fee_calculator`; maker mode
  credits the rebate share instead.
* **Exit** - take-profit / stop-loss are resolved against the *real settled
  result* of the window, which is the only honest way to backtest a binary:
  the payoff is 1.0 for the winning side and 0.0 for the loser, so "TP hit" means
  the winning side's price reached the target before settlement.

Honesty notes
-------------
* Only observation offsets that actually carry prices are traded. Missing prices
  are skipped, never interpolated - a silent fill at an invented price would
  inflate every metric downstream.
* When the source data has no order book, `SyntheticBookModel` is used and the
  report says so (``book_source: "synthetic"``). Slippage numbers from a
  synthetic book are indicative of *method*, not of that market's real depth.
* ``max_holding`` / settlement: a binary position held to settlement needs no
  exit liquidity, which is exactly why the sniper design works at T-30s.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from core.execution import ExecutionEngine, ExecutionError, estimate_execution
from core.market_data import BookSnapshot, UP, DOWN
from utils.fee_calculator import net_fee_usdc
from utils.logger import get_logger, write_csv, write_json
from utils.vwap_calculator import Level

log = get_logger("backtest")

WINNING_PAYOFF = 1.0
LOSING_PAYOFF = 0.0


class BacktestError(RuntimeError):
    """Raised when the inputs cannot produce a meaningful backtest."""


# ------------------------------------------------------------------ param set
@dataclass(frozen=True)
class ParamSet:
    """One row of ``param_combinations.csv``."""

    combo_id: str
    time_s: int
    threshold_pct: float
    take_profit: float
    stop_loss: float
    min_gap_s: int
    risk_reward: float | None = None
    source: str = ""

    @property
    def label(self) -> str:
        return (
            f"{self.combo_id}:T{self.time_s}:thr{self.threshold_pct:g}:"
            f"tp{self.take_profit:g}:sl{self.stop_loss:g}:gap{self.min_gap_s}"
        )

    def as_dict(self) -> dict:
        return {
            "combo_id": self.combo_id,
            "time_s": self.time_s,
            "threshold_pct": self.threshold_pct,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "min_gap_s": self.min_gap_s,
            "risk_reward": self.risk_reward,
            "source": self.source,
        }


def load_param_sets(path: str, limit: int | None = None) -> list[ParamSet]:
    """Read the 1200-combination grid (or any file with the same columns)."""
    if not os.path.exists(path):
        raise BacktestError(f"parameter file not found: {path}")
    param_sets: list[ParamSet] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"time", "threshold", "tp", "sl"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BacktestError(f"{path} is missing required columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            param_sets.append(
                ParamSet(
                    combo_id=str(row.get("combo_id") or f"c{index + 1:05d}"),
                    time_s=int(float(row["time"])),
                    threshold_pct=float(row["threshold"]),
                    take_profit=float(row["tp"]),
                    stop_loss=float(row["sl"]),
                    min_gap_s=int(float(row.get("min_gap") or 0)),
                    risk_reward=(
                        float(row["risk_reward"])
                        if row.get("risk_reward") not in (None, "", "None")
                        else None
                    ),
                    source=str(row.get("source") or ""),
                )
            )
    if not param_sets:
        raise BacktestError(f"no parameter rows found in {path}")
    return param_sets


# ---------------------------------------------------------------- observation
@dataclass
class MarketObservation:
    """One tradeable observation: a market plus its price at one offset."""

    market_id: str
    slug: str
    window_start_epoch: float
    window_close_epoch: float
    t_minus_s: int
    up_price: float | None
    down_price: float | None
    result: str
    btc_open: float | None = None
    btc_close: float | None = None
    btc_spot: float | None = None
    btc_offset_pct: float | None = None
    trade_count: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def observation_epoch(self) -> float:
        return self.window_close_epoch - self.t_minus_s

    @property
    def up_won(self) -> bool:
        return str(self.result).upper() == "UP"

    def price_of(self, side: str) -> float | None:
        return self.up_price if side == UP else self.down_price

    def outcome_price(self, side: str) -> float:
        """Settlement value of one share of ``side`` (1.0 win / 0.0 lose)."""
        won = (side == UP and self.up_won) or (side == DOWN and not self.up_won)
        return WINNING_PAYOFF if won else LOSING_PAYOFF

    def implied_probability(self, side: str) -> float | None:
        price = self.price_of(side)
        if price is None:
            return None
        return float(price)

    def underlying_move_pct(self) -> float | None:
        """BTC move over the window, in percent (the signal the strategy reads)."""
        if self.btc_offset_pct is not None:
            return float(self.btc_offset_pct)
        if self.btc_open and self.btc_close:
            try:
                return (float(self.btc_close) / float(self.btc_open) - 1.0) * 100.0
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def load_observations_from_prices_csv(
    path: str,
    *,
    markets_csv: str | None = None,
    offsets: Sequence[int] | None = None,
    require_price: bool = True,
    limit_markets: int | None = None,
) -> list[MarketObservation]:
    """Load observations from ``btc5m_prices.csv`` (+ optional markets sidecar).

    Rows whose price is missing are dropped when ``require_price`` - the engine
    will not invent a fill price.
    """
    if not os.path.exists(path):
        raise BacktestError(f"prices file not found: {path}")

    market_rows: dict[str, dict] = {}
    if markets_csv and os.path.exists(markets_csv):
        with open(markets_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                market_rows[str(row.get("market_id"))] = row

    wanted = {int(o) for o in offsets} if offsets else None
    observations: list[MarketObservation] = []
    seen_markets: list[str] = []

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                offset = int(float(row["t_minus_s"]))
            except (KeyError, TypeError, ValueError):
                continue
            if wanted is not None and offset not in wanted:
                continue
            up_price = _as_float(row.get("up_price"))
            down_price = _as_float(row.get("down_price"))
            if require_price and up_price is None and down_price is None:
                continue
            if up_price is None and down_price is not None:
                up_price = 1.0 - down_price
            if down_price is None and up_price is not None:
                down_price = 1.0 - up_price

            market_id = str(row.get("market_id") or row.get("slug") or "")
            sidecar = market_rows.get(market_id, {})
            if market_id and market_id not in seen_markets:
                seen_markets.append(market_id)
            if limit_markets is not None and market_id not in seen_markets[:limit_markets]:
                continue

            observations.append(
                MarketObservation(
                    market_id=market_id,
                    slug=str(row.get("slug") or sidecar.get("slug") or ""),
                    window_start_epoch=float(sidecar.get("window_start_epoch") or 0.0),
                    window_close_epoch=float(
                        row.get("window_close_epoch")
                        or sidecar.get("window_close_epoch")
                        or 0.0
                    ),
                    t_minus_s=offset,
                    up_price=up_price,
                    down_price=down_price,
                    result=str(row.get("result") or sidecar.get("result") or "UNKNOWN"),
                    btc_open=_as_float(sidecar.get("btc_open")),
                    btc_close=_as_float(sidecar.get("btc_close")),
                    btc_spot=_as_float(row.get("btc_spot")),
                    btc_offset_pct=_as_float(row.get("btc_offset_pct")),
                    trade_count=_as_float(sidecar.get("trade_count")),
                    raw={"price_row": row, "market_row": sidecar},
                )
            )

    if not observations:
        raise BacktestError(
            f"no usable observations in {path} "
            f"(offsets={sorted(wanted) if wanted else 'all'}, require_price={require_price})"
        )
    return observations


def load_observations_from_raw_jsonl(
    path: str, *, offsets: Sequence[int], tolerance_s: int = 12
) -> list[MarketObservation]:
    """Load from the scraper's ``btc5m_raw.jsonl`` (full per-second series).

    This is the richer source: real prices at every offset, because the raw file
    keeps the whole price history rather than the 12-point grid.
    """
    import json

    if not os.path.exists(path):
        raise BacktestError(f"raw jsonl not found: {path}")
    wanted = sorted({int(o) for o in offsets}, reverse=True)
    observations: list[MarketObservation] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            close_epoch = _as_float(record.get("window_close_epoch"))
            if not close_epoch:
                continue
            up_hist = [(float(t), float(p)) for t, p in (record.get("up_history") or [])]
            down_hist = [(float(t), float(p)) for t, p in (record.get("down_history") or [])]
            btc_hist = [(float(t), float(p)) for t, p in (record.get("btc_history") or [])]

            def nearest(series: Sequence[tuple[float, float]], target: float):
                best = None
                best_gap = None
                for ts, value in series:
                    gap = abs(ts - target)
                    if best_gap is None or gap < best_gap:
                        best, best_gap = value, gap
                if best is None or best_gap is None or best_gap > tolerance_s:
                    return None
                return best

            for offset in wanted:
                target = close_epoch - offset
                up_price = nearest(up_hist, target)
                down_price = nearest(down_hist, target)
                if up_price is None and down_price is None:
                    continue
                if up_price is None and down_price is not None:
                    up_price = 1.0 - down_price
                if down_price is None and up_price is not None:
                    down_price = 1.0 - up_price
                btc_spot = nearest(btc_hist, target)
                btc_open = nearest(btc_hist, float(record.get("window_start_epoch") or 0.0))
                observations.append(
                    MarketObservation(
                        market_id=str(record.get("slug") or ""),
                        slug=str(record.get("slug") or ""),
                        window_start_epoch=float(record.get("window_start_epoch") or 0.0),
                        window_close_epoch=float(close_epoch),
                        t_minus_s=offset,
                        up_price=up_price,
                        down_price=down_price,
                        result=str(record.get("result") or "UNKNOWN"),
                        btc_open=btc_open,
                        btc_close=btc_spot,
                        btc_spot=btc_spot,
                        btc_offset_pct=(
                            (btc_spot / btc_open - 1.0) * 100.0
                            if btc_spot and btc_open
                            else None
                        ),
                    )
                )
    if not observations:
        raise BacktestError(f"no observations parsed from {path}")
    return observations


# -------------------------------------------------------------- book synthesis
def coerce_midpoint(midpoint: float | None, tick_size: float = 0.001) -> float:
    """Clamp any price into a band where a two-sided book can exist.

    Near 0 or 1 the touch prices would fall outside (0, 1) and the book would
    collapse to one side; clamping keeps extreme-probability markets tradeable
    instead of raising.
    """
    floor = max(tick_size, 0.001)
    ceiling = min(1.0 - tick_size, 0.999)
    if midpoint is None:
        return 0.5
    try:
        value = float(midpoint)
    except (TypeError, ValueError):
        return 0.5
    if value != value:  # NaN
        return 0.5
    return float(min(ceiling, max(floor, value)))


@dataclass
class SyntheticBookModel:
    """Builds a reproducible book ladder from a midpoint.

    Used when the dataset has prices but no depth. Sizes are deterministic in the
    midpoint, so a given observation always prices identically - important for
    comparing 1200 parameter sets against each other.
    """

    tick_size: float = 0.001
    half_spread_ticks: float = 2.0
    levels: int = 5
    base_size: float = 40.0
    size_decay: float = 0.75
    jitter: float = 0.15

    def build(
        self,
        midpoint: float,
        *,
        t_minus_s: int = 0,
        seed_key: str = "",
        obi: float = 0.0,
        time_decay: bool = True,
    ) -> BookSnapshot:
        """Return a two-sided snapshot around ``midpoint``.

        ``time_decay`` thins the book as the window closes, which is what real
        5-minute books do - and it is what makes a naive T-10s snipe expensive.
        """
        center = coerce_midpoint(midpoint, self.tick_size)
        half_spread = max(self.tick_size / 2.0, self.half_spread_ticks * self.tick_size / 2.0)
        best_bid = max(self.tick_size, center - half_spread)
        best_ask = min(1.0 - self.tick_size, center + half_spread)
        if best_bid >= best_ask:
            # squeeze the quote to the smallest legal width around the centre
            best_bid = max(self.tick_size, center - self.tick_size / 2.0)
            best_ask = min(1.0 - self.tick_size, best_bid + self.tick_size)

        thin = 1.0
        if time_decay:
            # liquidity roughly halves over the last minute
            thin = 0.5 + 0.5 * min(1.0, max(0.0, t_minus_s / 60.0))

        bids: list[Level] = []
        asks: list[Level] = []
        for level in range(self.levels):
            step = level * self.tick_size
            wobble = 1.0 + self.jitter * math.sin(hash((seed_key, level)) % 997 / 997.0 * math.pi)
            size = max(self.tick_size, self.base_size * (self.size_decay**level) * thin * wobble)
            bid_px = best_bid - step
            ask_px = best_ask + step
            # OBI shifts visible size between the sides
            bid_size = max(self.tick_size, size * (1.0 + obi * 0.5))
            ask_size = max(self.tick_size, size * (1.0 - obi * 0.5))
            # keep levels strictly inside (0, 1) but allow the full range, so a
            # market quoted at 0.998 still has a valid ask above it
            if bid_px > 0.0:
                bids.append((round(bid_px, 6), round(bid_size, 4)))
            if ask_px < 1.0:
                asks.append((round(ask_px, 6), round(ask_size, 4)))

        if not bids or not asks:
            raise BacktestError(
                f"synthetic book collapsed to one side at midpoint {midpoint!r}; "
                "check tick_size and half_spread_ticks"
            )

        return BookSnapshot(
            token_id=f"synthetic:{seed_key}",
            side=UP,
            bids=tuple(bids),
            asks=tuple(asks),
            timestamp=0.0,
            source="synthetic",
        )


# --------------------------------------------------------------------- trades
@dataclass
class SimTrade:
    """One simulated round trip."""

    market_id: str
    slug: str
    side: str
    t_minus_s: int
    entry_ts: float
    entry_price: float
    entry_vwap: float
    shares: float
    notional_usdc: float
    entry_fee_usdc: float
    entry_slippage_bps: float
    exit_price: float
    exit_fee_usdc: float
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    result: str
    won: bool
    implied_probability: float
    underlying_move_pct: float | None
    obi: float
    liquidity_score: float
    hedge_shares: float = 0.0
    hedge_pnl: float = 0.0
    combo_id: str = ""

    @property
    def fees_usdc(self) -> float:
        return self.entry_fee_usdc + self.exit_fee_usdc

    @property
    def edge_per_share(self) -> float:
        return (self.exit_price - self.entry_vwap) if self.side == UP else (self.exit_price - self.entry_vwap)

    def as_dict(self) -> dict:
        return {
            "combo_id": self.combo_id,
            "market_id": self.market_id,
            "slug": self.slug,
            "side": self.side,
            "t_minus_s": self.t_minus_s,
            "entry_ts": self.entry_ts,
            "entry_price": round(self.entry_price, 6),
            "entry_vwap": round(self.entry_vwap, 6),
            "shares": round(self.shares, 6),
            "notional_usdc": round(self.notional_usdc, 6),
            "entry_fee_usdc": round(self.entry_fee_usdc, 8),
            "entry_slippage_bps": round(self.entry_slippage_bps, 3),
            "exit_price": round(self.exit_price, 6),
            "exit_fee_usdc": round(self.exit_fee_usdc, 8),
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 6),
            "net_pnl": round(self.net_pnl, 6),
            "fees_usdc": round(self.fees_usdc, 8),
            "result": self.result,
            "won": self.won,
            "implied_probability": round(self.implied_probability, 6),
            "underlying_move_pct": self.underlying_move_pct,
            "obi": round(self.obi, 4),
            "liquidity_score": round(self.liquidity_score, 6),
            "hedge_shares": round(self.hedge_shares, 6),
            "hedge_pnl": round(self.hedge_pnl, 6),
        }


@dataclass
class BacktestMetrics:
    """Aggregate performance of one parameter set."""

    combo_id: str = ""
    label: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage_usdc: float = 0.0
    net_pnl: float = 0.0
    avg_net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float | None = None
    expectancy_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    avg_slippage_bps: float = 0.0
    avg_hold_seconds: float = 0.0
    tp_exits: int = 0
    sl_exits: int = 0
    settled_exits: int = 0
    hedged_trades: int = 0
    markets: int = 0
    skipped_no_price: int = 0
    skipped_no_signal: int = 0
    skipped_min_gap: int = 0
    turnover_usdc: float = 0.0
    fee_drag_pct_of_gross: float = 0.0
    book_source: str = "synthetic"

    def as_dict(self) -> dict:
        return {
            "combo_id": self.combo_id,
            "label": self.label,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "gross_pnl": round(self.gross_pnl, 6),
            "total_fees": round(self.total_fees, 6),
            "total_slippage_usdc": round(self.total_slippage_usdc, 6),
            "net_pnl": round(self.net_pnl, 6),
            "avg_net_pnl": round(self.avg_net_pnl, 6),
            "avg_win": round(self.avg_win, 6),
            "avg_loss": round(self.avg_loss, 6),
            "profit_factor": (
                None if self.profit_factor is None else round(self.profit_factor, 4)
            ),
            "expectancy_per_trade": round(self.expectancy_per_trade, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "sharpe": None if self.sharpe is None else round(self.sharpe, 4),
            "sortino": None if self.sortino is None else round(self.sortino, 4),
            "avg_slippage_bps": round(self.avg_slippage_bps, 3),
            "avg_hold_seconds": round(self.avg_hold_seconds, 1),
            "tp_exits": self.tp_exits,
            "sl_exits": self.sl_exits,
            "settled_exits": self.settled_exits,
            "hedged_trades": self.hedged_trades,
            "markets": self.markets,
            "skipped_no_price": self.skipped_no_price,
            "skipped_no_signal": self.skipped_no_signal,
            "skipped_min_gap": self.skipped_min_gap,
            "turnover_usdc": round(self.turnover_usdc, 6),
            "fee_drag_pct_of_gross": round(self.fee_drag_pct_of_gross, 4),
            "book_source": self.book_source,
        }


@dataclass
class BacktestResult:
    """Metrics plus the trade log for one parameter set."""

    params: ParamSet
    metrics: BacktestMetrics
    trades: list[SimTrade] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "params": self.params.as_dict(),
            "metrics": self.metrics.as_dict(),
            "diagnostics": self.diagnostics,
        }

    def trades_as_rows(self) -> list[dict]:
        return [trade.as_dict() for trade in self.trades]


# -------------------------------------------------------------------- engine
class BacktestEngine:
    """Runs one parameter set over a set of observations."""

    def __init__(
        self,
        *,
        execution: ExecutionEngine | None = None,
        book_model: SyntheticBookModel | None = None,
        notional_per_trade: float = 100.0,
        equity: float = 10_000.0,
        risk_fraction: float = 0.02,
        maker_probability: float = 0.0,
        fee_coefficient: float = 0.07,
        fee_curve: str = "documented",
        maker_rebate_share: float = 0.25,
        entry_side_mode: str = "both",
        spot_confirm_pct: float = 0.0,
        probability_gap_min: float = 0.0,
        probability_gap_max: float = 1.0,
        max_entry_price: float = 0.95,
        min_entry_price: float = 0.05,
        book_source: str = "synthetic",
    ) -> None:
        self.execution = execution or ExecutionEngine(
            fee_coefficient=fee_coefficient, fee_curve=fee_curve
        )
        self.book_model = book_model or SyntheticBookModel()
        self.notional_per_trade = float(notional_per_trade)
        self.equity = float(equity)
        self.risk_fraction = float(risk_fraction)
        self.maker_probability = float(maker_probability)
        self.fee_coefficient = float(fee_coefficient)
        self.fee_curve = fee_curve
        self.maker_rebate_share = float(maker_rebate_share)
        self.entry_side_mode = entry_side_mode
        self.spot_confirm_pct = float(spot_confirm_pct)
        self.probability_gap_min = float(probability_gap_min)
        self.probability_gap_max = float(probability_gap_max)
        self.max_entry_price = float(max_entry_price)
        self.min_entry_price = float(min_entry_price)
        self.book_source = book_source

    # -- signal -----------------------------------------------------------
    def decide_side(
        self, observation: MarketObservation, threshold_pct: float = 0.0
    ) -> tuple[str | None, str]:
        """Which side to buy, and why.

        The sniper thesis: the underlying has already moved (``underlying_move``
        beyond ``threshold_pct``) but the market's implied probability has not
        caught up. We buy the side the underlying favours, provided it is still
        cheap enough that the fee leaves edge on the table.
        """
        move = observation.underlying_move_pct()
        if move is None:
            return None, "no underlying move available"

        if abs(move) < threshold_pct:
            return None, (
                f"underlying move {move:+.4f}% below threshold {threshold_pct:.4f}%"
            )

        if move > 0:
            side = UP
        elif move < 0:
            side = DOWN
        else:
            return None, "flat underlying"

        if self.entry_side_mode != "both" and side != self.entry_side_mode:
            return None, f"side {side} filtered out by entry_side_mode"

        implied = observation.implied_probability(side)
        if implied is None:
            return None, "no market price for the chosen side"

        # Same entry band the live strategy enforces (strategy.min/max_entry_price):
        # buying at 0.99 to win 0.01 is not a snipe, it is a fee donation, and it is
        # exactly what a near-close price snapshot would otherwise let through.
        if not self.min_entry_price <= implied <= self.max_entry_price:
            return None, (
                f"implied probability {implied:.4f} outside entry bounds "
                f"[{self.min_entry_price}, {self.max_entry_price}]"
            )

        # "caught up" test: how far is the market from certain?
        gap = 1.0 - implied
        if gap < self.probability_gap_min:
            return None, f"market already priced it ({implied:.4f}); gap {gap:.4f} too small"
        if gap > self.probability_gap_max:
            return None, (
                f"gap {gap:.4f} beyond {self.probability_gap_max} - market disagrees "
                "with the underlying, not a lag"
            )
        return side, f"underlying {move:+.4f}%, implied {implied:.4f}, gap {gap:.4f}"

    # -- run --------------------------------------------------------------
    def run(
        self,
        params: ParamSet,
        observations: Sequence[MarketObservation],
        *,
        on_trade: Callable[[SimTrade], None] | None = None,
    ) -> BacktestResult:
        """Backtest one parameter set."""
        selected = [o for o in observations if o.t_minus_s == params.time_s]
        metrics = BacktestMetrics(combo_id=params.combo_id, label=params.label, book_source=self.book_source)

        if not selected:
            metrics.skipped_no_price = len(observations)
            return BacktestResult(
                params=params,
                metrics=metrics,
                diagnostics={"reason": f"no observations at T-{params.time_s}s"},
            )

        selected.sort(key=lambda o: o.window_close_epoch)
        trades: list[SimTrade] = []
        last_entry_epoch: float | None = None
        markets_seen: set[str] = set()

        for observation in selected:
            markets_seen.add(observation.market_id)
            entry_epoch = observation.observation_epoch

            if last_entry_epoch is not None and (entry_epoch - last_entry_epoch) < params.min_gap_s:
                metrics.skipped_min_gap += 1
                continue

            side, reason = self.decide_side(observation, params.threshold_pct)
            if side is None:
                metrics.skipped_no_signal += 1
                continue

            traded = self._simulate_trade(params, observation, side, reason)
            if traded is None:
                metrics.skipped_no_price += 1
                continue

            trades.append(traded)
            last_entry_epoch = entry_epoch
            if on_trade is not None:
                on_trade(traded)

        self._aggregate(metrics, trades, markets_seen)
        return BacktestResult(
            params=params,
            metrics=metrics,
            trades=trades,
            diagnostics={
                "observations_at_offset": len(selected),
                "markets": len(markets_seen),
                "book_source": self.book_source,
            },
        )

    def _simulate_trade(
        self,
        params: ParamSet,
        observation: MarketObservation,
        side: str,
        signal_reason: str,
    ) -> SimTrade | None:
        price = observation.price_of(side)
        if price is None or not 0.0 < price < 1.0:
            return None

        # market context: OBI and liquidity come from the same book we price against
        move = observation.underlying_move_pct() or 0.0
        obi = max(-1.0, min(1.0, move / max(1e-9, params.threshold_pct * 2.0)))
        book = self.book_model.build(
            price,
            t_minus_s=observation.t_minus_s,
            seed_key=observation.market_id,
            obi=obi,
        )
        snapshot = book

        target_notional = min(
            self.notional_per_trade,
            self.equity * self.risk_fraction,
        )
        shares = target_notional / price
        is_maker = self.maker_probability > 0.0 and (
            hash((observation.market_id, params.combo_id)) % 1000
        ) < int(self.maker_probability * 1000)

        try:
            estimate = estimate_execution(
                snapshot,
                "buy",
                shares,
                is_taker=not is_maker,
                slippage_tolerance=self.execution.slippage_tolerance,
                fee_coefficient=self.fee_coefficient,
                fee_curve=self.fee_curve,
                maker_rebate_share=self.maker_rebate_share,
                reference_price=price,
            )
        except ExecutionError:
            return None

        if estimate.filled_shares <= 0:
            return None

        filled_shares = estimate.filled_shares
        entry_vwap = estimate.vwap
        entry_fee = estimate.fee_usdc
        # slippage cost in USDC vs the quoted price we aimed at
        slippage_usdc = max(0.0, (entry_vwap - price) * filled_shares)

        # --- exit resolution -------------------------------------------------
        outcome = observation.outcome_price(side)
        move_from_entry = outcome - entry_vwap
        if move_from_entry >= params.take_profit:
            exit_price = entry_vwap + params.take_profit
            exit_reason = "take_profit"
        elif move_from_entry <= -params.stop_loss:
            exit_price = max(0.0, entry_vwap - params.stop_loss)
            exit_reason = "stop_loss"
        else:
            exit_price = outcome
            exit_reason = "settled"

        exit_fee = net_fee_usdc(
            filled_shares,
            exit_price if 0.0 < exit_price < 1.0 else min(0.999, max(0.001, entry_vwap)),
            is_taker=True,
            coefficient=self.fee_coefficient,
            curve=self.fee_curve,  # type: ignore[arg-type]
            rebate_share=self.maker_rebate_share,
        ) if exit_reason != "settled" else 0.0

        gross = (exit_price - entry_vwap) * filled_shares
        net = gross - entry_fee - exit_fee
        won = net > 0

        side_snapshot = snapshot
        bid_depth = side_snapshot.depth("bid")
        ask_depth = side_snapshot.depth("ask")
        liquidity = (bid_depth.liquidity_score + ask_depth.liquidity_score) / 2.0

        return SimTrade(
            market_id=observation.market_id,
            slug=observation.slug,
            side=side,
            t_minus_s=observation.t_minus_s,
            entry_ts=observation.observation_epoch,
            entry_price=price,
            entry_vwap=entry_vwap,
            shares=filled_shares,
            notional_usdc=estimate.notional_usdc,
            entry_fee_usdc=entry_fee,
            entry_slippage_bps=estimate.slippage_bps,
            exit_price=exit_price,
            exit_fee_usdc=exit_fee,
            exit_reason=exit_reason,
            gross_pnl=gross,
            net_pnl=net,
            result=observation.result,
            won=won,
            implied_probability=price,
            underlying_move_pct=observation.underlying_move_pct(),
            obi=obi,
            liquidity_score=liquidity,
            combo_id=params.combo_id,
        )

    # -- metrics ----------------------------------------------------------
    @staticmethod
    def _aggregate(
        metrics: BacktestMetrics, trades: Sequence[SimTrade], markets: set[str]
    ) -> None:
        metrics.trades = len(trades)
        metrics.markets = len(markets)
        if not trades:
            return

        pnls = [t.net_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics.wins = len(wins)
        metrics.losses = len(losses)
        metrics.win_rate = len(wins) / len(trades)
        metrics.gross_pnl = sum(t.gross_pnl for t in trades)
        metrics.total_fees = sum(t.fees_usdc for t in trades)
        metrics.total_slippage_usdc = sum(
            max(0.0, (t.entry_vwap - t.entry_price) * t.shares) for t in trades
        )
        metrics.net_pnl = sum(pnls)
        metrics.avg_net_pnl = statistics.fmean(pnls)
        metrics.avg_win = statistics.fmean(wins) if wins else 0.0
        metrics.avg_loss = statistics.fmean(losses) if losses else 0.0
        metrics.profit_factor = (
            sum(wins) / abs(sum(losses)) if losses else (None if not wins else float("inf"))
        )
        metrics.expectancy_per_trade = metrics.avg_net_pnl
        metrics.avg_slippage_bps = statistics.fmean([t.entry_slippage_bps for t in trades])
        metrics.avg_hold_seconds = statistics.fmean(
            [max(0.0, t.t_minus_s) for t in trades]
        )
        metrics.tp_exits = sum(1 for t in trades if t.exit_reason == "take_profit")
        metrics.sl_exits = sum(1 for t in trades if t.exit_reason == "stop_loss")
        metrics.settled_exits = sum(1 for t in trades if t.exit_reason == "settled")
        metrics.hedged_trades = sum(1 for t in trades if t.hedge_shares > 0)
        metrics.turnover_usdc = sum(t.notional_usdc for t in trades)
        metrics.fee_drag_pct_of_gross = (
            metrics.total_fees / metrics.gross_pnl * 100.0 if metrics.gross_pnl else 0.0
        )

        # drawdown on the realised P&L curve
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        metrics.max_drawdown = max_dd

        if len(pnls) > 1:
            std = statistics.stdev(pnls)
            mean = statistics.fmean(pnls)
            metrics.sharpe = (mean / std * math.sqrt(len(pnls))) if std > 0 else None
            downside = [p for p in pnls if p < 0]
            if downside:
                dstd = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                metrics.sortino = (mean / dstd * math.sqrt(len(pnls))) if dstd > 0 else None


# ------------------------------------------------------------------- running
def run_backtest(
    params: ParamSet,
    observations: Sequence[MarketObservation],
    **engine_kwargs: Any,
) -> BacktestResult:
    engine = BacktestEngine(**engine_kwargs)
    return engine.run(params, observations)


def compare_fee_curves(
    trades: Sequence[SimTrade], *, coefficient: float = 0.07
) -> dict:
    """What the same fills would cost under taker vs maker assumptions."""
    taker = sum(t.fees_usdc for t in trades)
    notional = sum(t.notional_usdc for t in trades)
    return {
        "trades": len(trades),
        "taker_fees_usdc": round(taker, 6),
        "maker_fees_usdc": round(
            sum(
                net_fee_usdc(
                    t.shares,
                    t.entry_vwap,
                    is_taker=False,
                    coefficient=coefficient,
                )
                for t in trades
            ),
            6,
        ),
        "notional_usdc": round(notional, 6),
        "taker_fee_bps_of_notional": round(taker / notional * 10_000.0, 4) if notional else 0.0,
    }


def write_backtest_outputs(
    result: BacktestResult, output_dir: str, *, write_trades: bool = True
) -> dict[str, str]:
    """Persist one backtest: metrics JSON + trade log CSV."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}
    stem = result.params.combo_id or "backtest"
    paths["metrics"] = write_json(
        os.path.join(output_dir, f"{stem}_metrics.json"), result.as_dict()
    )
    if write_trades and result.trades:
        paths["trades"] = write_csv(
            os.path.join(output_dir, f"{stem}_trades.csv"), result.trades_as_rows()
        )
    return paths

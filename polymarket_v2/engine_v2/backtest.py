"""Backtester: replay real trade prints through the live decision path.

Design rules that keep the numbers honest
-----------------------------------------
1. **Every price is a real executed print.** No modelled book, no midpoints. The
   order books in this dataset carry spreads >= 0.50 in 99.77% of snapshots, so a
   book-based backtest here would be fiction.
2. **The entry fill is the last print at or before the decision moment**, and its
   age is passed to the staleness guard. Using a later print would be lookahead.
3. **Exits are sequential, first touch wins** - the way a live OCO pair behaves.
   Ties inside one print resolve to the stop (``stop_loss_wins_ties``).
4. **Settlement is not a trade**: holding to expiry costs no exit fee and needs no
   liquidity, which is the whole reason this strategy can run late in a window.
5. **Direction accuracy and profitability are reported separately.** They are
   different quantities: with a 0.52 average entry, profits need ~52% accuracy, so
   reporting only one of them is how a backtest invents an edge.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .broker import BrokerError, OrderRequest, OrderSide, PaperBroker, Position
from .config import EngineConfig
from .dataio import Dataset, Market, MarketTrades, classify_tokens, utc
from .guards import GuardChain, Quote
from .mathx import MathError, shares_for_notional, taker_fee
from .signals import Reason, Side, Signal, SignalDetector


class BacktestError(RuntimeError):
    """Raised when the run cannot produce a meaningful result."""


@dataclass
class Trade:
    """One completed round trip."""

    slug: str
    asset: str
    side: str
    entry_ts: float
    entry_price: float
    entry_fee: float
    shares: float
    entry_staleness_s: float
    move_pct: float
    signal_strength: float
    exit_ts: float
    exit_price: float
    exit_fee: float
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    direction_correct: bool
    outcome: str

    @property
    def fees_usdc(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def return_pct(self) -> float:
        cost = self.shares * self.entry_price + self.entry_fee
        return (self.net_pnl / cost * 100.0) if cost else 0.0

    def as_row(self) -> dict:
        return {
            "slug": self.slug,
            "asset": self.asset,
            "side": self.side,
            "outcome": self.outcome,
            "direction_correct": self.direction_correct,
            "entry_utc": utc(self.entry_ts),
            "entry_price": round(self.entry_price, 6),
            "entry_fee": round(self.entry_fee, 8),
            "shares": round(self.shares, 6),
            "entry_staleness_s": round(self.entry_staleness_s, 2),
            "move_pct": round(self.move_pct, 6),
            "signal_strength": round(self.signal_strength, 4),
            "exit_utc": utc(self.exit_ts),
            "exit_price": round(self.exit_price, 6),
            "exit_fee": round(self.exit_fee, 8),
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 6),
            "net_pnl": round(self.net_pnl, 6),
            "return_pct": round(self.return_pct, 4),
        }


@dataclass
class Metrics:
    """Aggregate performance, with the two accuracy measures kept apart."""

    label: str = ""
    trades: int = 0
    direction_hits: int = 0
    wins: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    avg_net_pnl: float = 0.0
    avg_entry_price: float = 0.0
    profit_factor: float | None = None
    max_drawdown: float = 0.0
    sharpe: float | None = None
    take_profit_exits: int = 0
    stop_loss_exits: int = 0
    settled_exits: int = 0
    avg_entry_staleness_s: float = 0.0
    p90_entry_staleness_s: float = 0.0
    avg_holding_s: float = 0.0
    rejected_entries: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    signal_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def direction_accuracy(self) -> float:
        return self.direction_hits / self.trades if self.trades else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def break_even_accuracy(self) -> float:
        """Hit rate needed just to break even at the average entry price."""
        return self.avg_entry_price

    @property
    def edge_points(self) -> float:
        """Accuracy minus break-even, in percentage points. The real headline."""
        return (self.direction_accuracy - self.break_even_accuracy) * 100.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "trades": self.trades,
            "direction_accuracy_pct": round(self.direction_accuracy * 100, 2),
            "break_even_accuracy_pct": round(self.break_even_accuracy * 100, 2),
            "edge_points": round(self.edge_points, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "avg_entry_price": round(self.avg_entry_price, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "fees": round(self.fees, 4),
            "net_pnl": round(self.net_pnl, 4),
            "avg_net_pnl": round(self.avg_net_pnl, 4),
            "profit_factor": None if self.profit_factor is None else round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe": None if self.sharpe is None else round(self.sharpe, 4),
            "take_profit_exits": self.take_profit_exits,
            "stop_loss_exits": self.stop_loss_exits,
            "settled_exits": self.settled_exits,
            "avg_entry_staleness_s": round(self.avg_entry_staleness_s, 2),
            "p90_entry_staleness_s": round(self.p90_entry_staleness_s, 2),
            "avg_holding_s": round(self.avg_holding_s, 1),
            "rejected_entries": self.rejected_entries,
            "rejection_reasons": dict(self.rejection_reasons),
            "signal_reasons": dict(self.signal_reasons),
        }


@dataclass
class BacktestResult:
    metrics: Metrics
    trades: list[Trade] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"metrics": self.metrics.as_dict(),
                "trades": [t.as_row() for t in self.trades]}


def compute_metrics(trades: Sequence[Trade], *, label: str = "",
                    rejected: dict[str, int] | None = None,
                    signal_reasons: dict[str, int] | None = None) -> Metrics:
    metrics = Metrics(label=label, rejected_entries=sum((rejected or {}).values()),
                      rejection_reasons=dict(rejected or {}),
                      signal_reasons=dict(signal_reasons or {}))
    metrics.trades = len(trades)
    if not trades:
        return metrics

    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    metrics.direction_hits = sum(1 for t in trades if t.direction_correct)
    metrics.wins = len(wins)
    metrics.gross_pnl = sum(t.gross_pnl for t in trades)
    metrics.fees = sum(t.fees_usdc for t in trades)
    metrics.net_pnl = sum(pnls)
    metrics.avg_net_pnl = statistics.fmean(pnls)
    metrics.avg_entry_price = statistics.fmean(t.entry_price for t in trades)
    metrics.profit_factor = (sum(wins) / abs(sum(losses))) if losses else None
    metrics.take_profit_exits = sum(1 for t in trades if t.exit_reason == "take_profit")
    metrics.stop_loss_exits = sum(1 for t in trades if t.exit_reason == "stop_loss")
    metrics.settled_exits = sum(1 for t in trades if t.exit_reason == "settlement")
    staleness = sorted(t.entry_staleness_s for t in trades)
    metrics.avg_entry_staleness_s = statistics.fmean(staleness)
    metrics.p90_entry_staleness_s = staleness[min(len(staleness) - 1, int(0.9 * len(staleness)))]
    metrics.avg_holding_s = statistics.fmean(t.exit_ts - t.entry_ts for t in trades)

    equity = peak = drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    metrics.max_drawdown = drawdown

    if len(pnls) > 1:
        std = statistics.stdev(pnls)
        if std > 0:
            metrics.sharpe = statistics.fmean(pnls) / std * math.sqrt(len(pnls))
    return metrics


@dataclass
class Engine:
    """Runs the strategy over historical prints."""

    config: EngineConfig
    detector: SignalDetector | None = None
    guards: GuardChain | None = None
    broker: PaperBroker | None = None
    spot: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal_cfg = self.config.section("signal")
        guard_cfg = self.config.section("guards")
        if self.detector is None:
            self.detector = SignalDetector(
                lookback_s=signal_cfg.lookback_s,
                min_move_pct=signal_cfg.min_move_pct,
                max_spot_staleness_s=signal_cfg.max_spot_staleness_s,
                min_window_elapsed_s=guard_cfg.min_window_elapsed_s,
                one_entry_per_window=signal_cfg.one_entry_per_window,
            )
        if self.guards is None:
            self.guards = GuardChain.from_config(guard_cfg)
        if self.broker is None:
            self.broker = PaperBroker(coefficient=self.config.section("fees").taker_coefficient)

    # -- one window -------------------------------------------------------
    def trade_window(
        self,
        market: Market,
        trades: MarketTrades,
        tokens: tuple[str, str],
    ) -> tuple[Trade | None, str]:
        """Attempt one trade in ``market``. Returns (trade, refusal_reason)."""
        signal_cfg = self.config.section("signal")
        exits = self.config.section("exits")
        sizing = self.config.section("sizing")

        entry_ts = market.t_minus_to_epoch(signal_cfg.entry_t_minus_s)
        up_token, down_token = tokens

        # direction, from spot only, using data available at entry_ts
        spot = self.spot.get(market.asset)
        if spot is None:
            return None, Reason.NO_SPOT.value
        signal: Signal = self.detector.evaluate(
            spot, ts=entry_ts, window_start_ts=market.start_ts, already_traded=False
        )
        if not signal.is_actionable:
            return None, signal.reason.value

        token = up_token if signal.side is Side.UP else down_token

        # the ONLY fill we may assume: the last real print at or before entry_ts
        last = trades.last_at_or_before(token, entry_ts)
        if last is None:
            return None, "no_print_before_entry"
        print_ts, print_price = last
        quote = Quote(ts=print_ts, price=print_price)

        notional = float(sizing.notional_usdc)
        equity_cap = float(sizing.starting_equity) * float(sizing.max_position_pct_equity) / 100.0
        notional = min(notional, equity_cap) if equity_cap > 0 else notional
        try:
            shares = shares_for_notional(notional, print_price)
        except MathError:
            return None, "bad_entry_price"

        allowed, decisions = self.guards.check(
            signal, quote, now=entry_ts, shares=shares,
            signal_price=print_price if print_price else None,
        )
        if not allowed:
            refusal = next((d.reason for d in decisions if not d.allowed), "guard")
            return None, f"guard:{refusal}"

        try:
            entry_fill = self.broker.submit(
                OrderRequest(slug=market.slug, token_id=token, side=OrderSide.BUY,
                             shares=shares, limit_price=print_price, reason=signal.detail),
                at_price=print_price,
            )
        except BrokerError as exc:
            return None, f"broker:{exc}"

        tp_price = min(0.999, entry_fill.price + float(exits.take_profit))
        sl_price = max(0.001, entry_fill.price - float(exits.stop_loss))

        exit_price, exit_ts, exit_reason, exit_fee = self._resolve_exit(
            market=market, trades=trades, token=token,
            after_ts=entry_ts, tp_price=tp_price, sl_price=sl_price,
            shares=shares, side=signal.side,
        )

        gross = shares * (exit_price - entry_fill.price)
        net = gross - entry_fill.fee_usdc - exit_fee
        return (
            Trade(
                slug=market.slug, asset=market.asset, side=signal.side.value,
                entry_ts=entry_ts, entry_price=entry_fill.price,
                entry_fee=entry_fill.fee_usdc, shares=shares,
                entry_staleness_s=entry_ts - print_ts, move_pct=signal.move_pct or 0.0,
                signal_strength=signal.strength, exit_ts=exit_ts, exit_price=exit_price,
                exit_fee=exit_fee, exit_reason=exit_reason, gross_pnl=gross, net_pnl=net,
                direction_correct=((signal.side is Side.UP) == market.up_won),
                outcome=market.outcome,
            ),
            "",
        )

    def _resolve_exit(
        self,
        *,
        market: Market,
        trades: MarketTrades,
        token: str,
        after_ts: float,
        tp_price: float,
        sl_price: float,
        shares: float,
        side: Side,
    ) -> tuple[float, float, str, float]:
        """Sequential first-touch: walk prints after entry, first level wins."""
        exits = self.config.section("exits")
        coefficient = self.config.section("fees").taker_coefficient
        stop_wins_ties = bool(exits.get("stop_loss_wins_ties", True))

        for ts, price in trades.prints_after(token, after_ts):
            hit_stop = price <= sl_price
            hit_target = price >= tp_price
            if hit_stop and hit_target:
                hit_target = not stop_wins_ties
                hit_stop = stop_wins_ties
            if hit_stop:
                fill_price = sl_price
                fee = taker_fee(shares, fill_price, coefficient=coefficient)
                return fill_price, ts, "stop_loss", fee
            if hit_target:
                fill_price = tp_price
                fee = taker_fee(shares, fill_price, coefficient=coefficient)
                return fill_price, ts, "take_profit", fee

        # neither level touched -> settle. Settlement is not a trade: no fee.
        settle_price = 1.0 if ((side is Side.UP) == market.up_won) else 0.0
        return settle_price, market.end_ts, "settlement", 0.0

    # -- whole dataset ----------------------------------------------------
    def run(
        self,
        *,
        dataset: Dataset,
        trades_file: str = "auto",
        assets: Sequence[str] | None = None,
        max_markets: int = 0,
        progress_every: int = 250,
        logger=None,
    ) -> BacktestResult:
        """Replay the dataset, returning trades and metrics."""
        markets = dataset.markets()
        if not markets:
            raise BacktestError(
                "no labelled markets found - check that both the slug file and the "
                "outcome file are present in the data directory"
            )
        self.spot = self.spot or dataset.spot()

        path = dataset.pick_trades_file(trades_file)
        if logger:
            logger.info("trades file: %s", os.path.basename(path))
        trades_by_slug = dataset.load_trades(path, max_markets=max_markets)

        wanted = {a.strip().upper() for a in (assets or []) if a.strip()}
        refused: dict[str, int] = {}
        signal_reasons: dict[str, int] = {}
        completed: list[Trade] = []
        processed = 0

        for slug, market_trades in trades_by_slug.items():
            market = markets.get(slug)
            if market is None:
                continue
            if wanted and market.asset not in wanted:
                continue
            tokens = classify_tokens(market_trades, market)
            if tokens is None:
                refused["unlabelled_tokens"] = refused.get("unlabelled_tokens", 0) + 1
                continue

            trade, refusal = self.trade_window(market, market_trades, tokens)
            processed += 1
            if trade is not None:
                completed.append(trade)
            elif refusal:
                if refusal.startswith("guard:"):
                    key = refusal
                elif refusal in {r.value for r in Reason}:
                    key = f"signal:{refusal}"
                    signal_reasons[refusal] = signal_reasons.get(refusal, 0) + 1
                else:
                    key = refusal
                refused[key] = refused.get(key, 0) + 1

            if logger and progress_every and processed % progress_every == 0:
                logger.info("processed %d markets, %d trades", processed, len(completed))

        completed.sort(key=lambda t: t.entry_ts)
        metrics = compute_metrics(completed, label=f"T-{self.config.section('signal').entry_t_minus_s:.0f}s",
                                  rejected=refused, signal_reasons=signal_reasons)
        return BacktestResult(metrics=metrics, trades=completed)


def run_backtest(
    config: EngineConfig,
    *,
    data_dir: str | None = None,
    trades_file: str = "auto",
    assets: Sequence[str] | None = None,
    adverse_bps: float | None = None,
    label: str = "",
    logger=None,
) -> tuple[BacktestResult, Dataset]:
    """Convenience wrapper: build a dataset, run the engine, return both."""
    data_cfg = config.section("data")
    backtest_cfg = config.section("backtest")
    dataset = Dataset(
        data_dir or config.resolve(data_cfg.data_dir),
        assets=assets or data_cfg.assets,
        batch_size=data_cfg.batch_size,
    )
    engine = Engine(config)
    if adverse_bps is not None:
        engine.broker = PaperBroker(
            coefficient=config.section("fees").taker_coefficient, adverse_bps=adverse_bps
        )
    result = engine.run(
        dataset=dataset,
        trades_file=trades_file if trades_file != "auto" else backtest_cfg.trades_file,
        assets=assets,
        max_markets=backtest_cfg.max_markets,
        logger=logger,
    )
    if label:
        result.metrics.label = label
    return result, dataset


def write_outputs(result: BacktestResult, output_dir: str, *, stem: str = "backtest") -> dict[str, str]:
    """Persist metrics JSON + trade log CSV."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(result.metrics.as_dict(), fh, indent=2)
    paths["metrics"] = metrics_path
    if result.trades:
        trades_path = os.path.join(output_dir, f"{stem}_trades.csv")
        rows = [t.as_row() for t in result.trades]
        with open(trades_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths["trades"] = trades_path
    return paths

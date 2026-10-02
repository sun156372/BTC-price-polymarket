"""Main strategy - late-window sniper.

The thesis in one paragraph
---------------------------
Bitcoin moves; Polymarket's 5-minute "Up or Down" book does not always reprice
instantly. Inside the final seconds of a window the outcome is nearly decided by
the underlying, so when the spot price has clearly crossed the window's
reference and the book is *still* offering the winning side below fair value,
buying it is a positive-expectancy trade - provided fees, slippage and the risk
of a late reversal are all accounted for. That is what this module orchestrates:

``market_data`` -> read the book, its imbalance and its depth
``snapshot``    -> compare Binance spot against implied probability
``execution``   -> price the fill, choose maker/taker, split if large
``risk_control``-> dynamic stops, reverse hedge when the signal flips

The strategy object is deliberately I/O-free: feed it a
:class:`~core.market_data.MarketDataHub` and a reference price, and it returns
decisions. Live wiring (a real WebSocket transport, a real broker) is the
caller's job, which keeps this file testable and keeps accidental live trading
impossible.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from config.loader import Config
from core.backtest_engine import MarketObservation
from core.execution import (
    ExecutionEngine,
    ExecutionError,
    OrderPlan,
    SlippageToleranceExceeded,
    estimate_execution,
)
from core.market_data import DOWN, UP, BookSnapshot, MarketDataHub, SpotTick
from core.risk_control import (
    Action,
    Position,
    RiskController,
    RiskDecision,
    RiskLimits,
    opposite_side,
)
from utils.logger import get_logger

log = get_logger("sniper")


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class SignalQuality(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


@dataclass
class SniperDecision:
    """Everything the strategy concluded on one tick."""

    action: str
    reason: str
    side: str | None = None
    shares: float = 0.0
    seconds_to_close: float | None = None
    fair_probability: float | None = None
    market_probability: float | None = None
    edge: float | None = None
    signal_strength: float = 0.0
    signal_quality: str = SignalQuality.NONE.value
    spot_price: float | None = None
    reference_price: float | None = None
    underlying_move_pct: float | None = None
    obi: float = 0.0
    plan: OrderPlan | None = None
    risk: RiskDecision | None = None
    data_healthy: bool = True
    blockers: tuple[str, ...] = ()

    @property
    def is_entry(self) -> bool:
        return self.action == "enter"

    @property
    def is_hedge(self) -> bool:
        return self.action in {"hedge", "hedge_topup"}

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "side": self.side,
            "shares": round(self.shares, 8),
            "seconds_to_close": self.seconds_to_close,
            "fair_probability": self.fair_probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "signal_strength": round(self.signal_strength, 4),
            "signal_quality": self.signal_quality,
            "spot_price": self.spot_price,
            "reference_price": self.reference_price,
            "underlying_move_pct": self.underlying_move_pct,
            "obi": round(self.obi, 4),
            "data_healthy": self.data_healthy,
            "blockers": list(self.blockers),
            "plan": self.plan.as_dict() if self.plan else None,
            "risk": self.risk.as_dict() if self.risk else None,
        }


@dataclass
class StrategyStats:
    ticks: int = 0
    entries: int = 0
    hedges: int = 0
    stops: int = 0
    take_profits: int = 0
    blocked_data: int = 0
    blocked_gap: int = 0
    blocked_price_band: int = 0
    blocked_execution: int = 0

    def as_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "entries": self.entries,
            "hedges": self.hedges,
            "stops": self.stops,
            "take_profits": self.take_profits,
            "blocked_data": self.blocked_data,
            "blocked_gap": self.blocked_gap,
            "blocked_price_band": self.blocked_price_band,
            "blocked_execution": self.blocked_execution,
        }


def implied_move_probability(
    move_pct: float,
    *,
    volatility_pct: float,
    seconds_to_close: float,
    scale: float = 1.0,
) -> float:
    """Fair probability the move holds to the close.

    A logistic in the size of the move relative to what is still left to move::

        z = move_pct / max(vol_floor, volatility_pct * sqrt(seconds/60)) * scale

    With no time left, any move is decisive (probability -> 1). With lots of time
    left, the same move is worth much less. This is the honest core of the
    strategy: it is the *only* place a "fair value" is produced, and it depends
    entirely on the measured move and volatility you feed it.
    """
    remaining = max(0.0, seconds_to_close)
    time_factor = max(1e-6, (remaining / 60.0) ** 0.5)
    denom = max(1e-4, volatility_pct * time_factor)
    z = (move_pct / denom) * scale
    # numeric guard against overflow in exp
    if z > 60:
        return 1.0
    if z < -60:
        return 0.0
    return 1.0 / (1.0 + pow(2.718281828459045, -z))


class SniperStrategy:
    """Orchestrates market data, execution and risk into trade decisions."""

    def __init__(
        self,
        config: Config,
        *,
        mode: str | Mode = Mode.BACKTEST,
        execution: ExecutionEngine | None = None,
        risk: RiskController | None = None,
        broker: Callable[[OrderPlan], Any] | None = None,
    ) -> None:
        self.config = config
        self.mode = Mode(mode) if not isinstance(mode, Mode) else mode
        s = config.section("strategy")

        self.scan_window_s = float(s.scan_window_seconds)
        self.observation_time_s = int(s.observation_time_s)
        self.threshold_pct = float(s.threshold_pct)
        self.take_profit = float(s.take_profit)
        self.stop_loss = float(s.stop_loss)
        self.min_entry_gap_s = float(s.min_entry_gap_s)
        self.side_mode = str(s.entry_side_mode)
        self.spot_confirm_pct = float(s.spot_confirm_pct)
        self.gap_min = float(s.probability_gap_min)
        self.gap_max = float(s.probability_gap_max)
        self.max_entry_price = float(s.max_entry_price)
        self.min_entry_price = float(s.min_entry_price)

        e = config.section("execution")
        self.execution = execution or ExecutionEngine(
            slippage_tolerance=float(e.slippage_tolerance),
            max_slippage_bps=float(e.max_slippage_bps),
            maker_first=bool(e.maker_first),
            maker_timeout_s=float(e.maker_timeout_s),
            split_threshold_notional=float(e.split_order_threshold_notional),
            participation_rate=float(e.split_participation_rate),
            min_children=int(e.min_split_children),
            max_children=int(e.max_split_children),
            tick_size=float(e.tick_size),
            price_improve_ticks=int(e.passive_price_improve_ticks),
            fee_coefficient=float(config.section("fees").taker_fee_coefficient),
            fee_curve=str(config.section("backtest").fee_curve),
            maker_rebate_share=float(config.section("fees").maker_rebate_share),
        )
        self.order_notional = float(e.default_order_notional)

        r = config.section("risk")
        self.risk = risk or RiskController(
            RiskLimits(
                max_position_notional=float(r.max_position_notional),
                max_window_notional=float(r.max_window_notional),
                stop_loss_prob=float(r.stop_loss_prob),
                trailing_stop_prob=float(r.trailing_stop_prob),
                obi_extreme_abs=float(r.obi_extreme_abs),
                obi_flip_abs=float(r.obi_flip_abs),
                volatility_stop_mult=float(r.volatility_stop_mult),
                hedge_enabled=bool(r.hedge_enabled),
                hedge_ratio=float(r.hedge_ratio),
                max_hedge_ratio=float(r.max_hedge_ratio),
                hedge_min_signal_strength=float(r.hedge_min_signal_strength),
            ),
            take_profit_prob=self.take_profit,
        )
        self.broker = broker
        self.stats = StrategyStats()
        self.last_entry_ts: dict[str, float] = {}
        self.decisions: list[SniperDecision] = []
        self.orders_sent: list[dict] = []

    # -- helpers ----------------------------------------------------------
    def in_scan_window(self, seconds_to_close: float | None) -> bool:
        return seconds_to_close is not None and 0.0 <= seconds_to_close <= self.scan_window_s

    def apply_param_set(self, param_set: Any) -> None:
        """Rebind the tunables from one ``param_combinations.csv`` row.

        This is the bridge between the 1200-combination grid and live/paper
        behaviour: the same numbers the sweep ranks can be applied here.
        """
        self.observation_time_s = int(param_set.time_s)
        self.threshold_pct = float(param_set.threshold_pct)
        self.take_profit = float(param_set.take_profit)
        self.stop_loss = float(param_set.stop_loss)
        self.min_entry_gap_s = float(param_set.min_gap_s)
        self.risk.take_profit_prob = float(param_set.take_profit)

    # -- core -------------------------------------------------------------
    def evaluate(
        self,
        hub: MarketDataHub,
        *,
        reference_price: float | None = None,
        window_open_price: float | None = None,
        now: float | None = None,
        volatility_pct: float | None = None,
    ) -> SniperDecision:
        """One decision cycle. Pure: reads the hub, returns a decision."""
        moment = now if now is not None else time.time()
        self.stats.ticks += 1
        blockers: list[str] = []

        report = hub.validate(moment)
        up_snapshot = hub.snapshot(UP)
        down_snapshot = hub.snapshot(DOWN)
        seconds_to_close = up_snapshot.distance_to_close(moment)
        obi = up_snapshot.obi(hub.depth)

        spot_series = hub.spot.price if hub.spot else reference_price
        ref = window_open_price
        move_pct = None
        if spot_series is not None and ref:
            move_pct = (spot_series / ref - 1.0) * 100.0

        vol = volatility_pct
        if vol is None:
            realized = hub.realized_volatility(UP)
            vol = None if realized is None else realized * 100.0

        # ---- 1) risk first: an open position may need defending -------------
        risk_decision = self._evaluate_risk(
            hub,
            up_snapshot=up_snapshot,
            down_snapshot=down_snapshot,
            move_pct=move_pct,
            vol_pct=vol,
            seconds_to_close=seconds_to_close,
            moment=moment,
        )
        if risk_decision is not None and risk_decision.action is not Action.HOLD:
            decision = self._decision_from_risk(
                risk_decision, hub, seconds_to_close, move_pct, obi, spot_series, ref, report.healthy
            )
            self.decisions.append(decision)
            return decision

        # ---- 2) data quality gate -------------------------------------------
        if not report.healthy:
            self.stats.blocked_data += 1
            blockers.extend(report.reasons)
            decision = SniperDecision(
                action="hold",
                reason="market data failed cross-validation",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
                data_healthy=False,
                blockers=tuple(blockers),
            )
            self.decisions.append(decision)
            return decision

        # ---- 3) only trade inside the endgame window ------------------------
        if not self.in_scan_window(seconds_to_close):
            decision = SniperDecision(
                action="hold",
                reason=(
                    f"outside scan window (T-{seconds_to_close:.1f}s, "
                    f"window T-{self.scan_window_s:.0f}s)"
                    if seconds_to_close is not None
                    else "window close time unknown"
                ),
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        # ---- 4) signal ------------------------------------------------------
        if move_pct is None or not ref:
            decision = SniperDecision(
                action="hold",
                reason="no reference price for the window yet",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                data_healthy=report.healthy,
            )
            self.decisions.append(decision)
            return decision

        side = UP if move_pct > 0 else DOWN if move_pct < 0 else None
        if side is None:
            decision = SniperDecision(
                action="hold",
                reason="underlying flat; no direction to snipe",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        if self.side_mode != "both" and side != self.side_mode:
            decision = SniperDecision(
                action="hold",
                reason=f"side {side} excluded by entry_side_mode={self.side_mode}",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        if abs(move_pct) < self.threshold_pct:
            self.stats.blocked_gap += 1
            decision = SniperDecision(
                action="hold",
                reason=f"move {move_pct:+.4f}% below threshold {self.threshold_pct:.4f}%",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        if abs(move_pct) < self.spot_confirm_pct:
            decision = SniperDecision(
                action="hold",
                reason=f"spot confirmation {abs(move_pct):.4f}% below {self.spot_confirm_pct:.4f}%",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        snapshot = up_snapshot if side == UP else down_snapshot
        market_prob = hub.implied_probability(side)
        if market_prob is None:
            decision = SniperDecision(
                action="hold",
                reason=f"no market price for {side}",
                seconds_to_close=seconds_to_close,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        fair = implied_move_probability(
            abs(move_pct),
            volatility_pct=vol if vol else max(self.threshold_pct, 0.02),
            seconds_to_close=seconds_to_close or 0.0,
        )
        fair_probability = fair if side == UP else 1.0 - fair
        edge = fair_probability - market_prob

        signal_strength = self._signal_strength(edge, move_pct, obi, side)
        quality = (
            SignalQuality.STRONG.value
            if signal_strength >= 0.55
            else SignalQuality.WEAK.value
            if signal_strength >= 0.3
            else SignalQuality.NONE.value
        )

        if not self.min_entry_price <= market_prob <= self.max_entry_price:
            self.stats.blocked_price_band += 1
            decision = SniperDecision(
                action="hold",
                reason=(
                    f"{side} price {market_prob:.4f} outside entry band "
                    f"[{self.min_entry_price}, {self.max_entry_price}]"
                ),
                side=side,
                seconds_to_close=seconds_to_close,
                fair_probability=fair_probability,
                market_probability=market_prob,
                edge=edge,
                signal_strength=signal_strength,
                signal_quality=quality,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        if edge < self.gap_min:
            self.stats.blocked_gap += 1
            decision = SniperDecision(
                action="hold",
                reason=(
                    f"no lag to exploit: fair {fair_probability:.4f} vs market "
                    f"{market_prob:.4f} (edge {edge:+.4f} < {self.gap_min:.4f})"
                ),
                side=side,
                seconds_to_close=seconds_to_close,
                fair_probability=fair_probability,
                market_probability=market_prob,
                edge=edge,
                signal_strength=signal_strength,
                signal_quality=quality,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        if edge > self.gap_max:
            self.stats.blocked_gap += 1
            decision = SniperDecision(
                action="hold",
                reason=(
                    f"edge {edge:+.4f} beyond {self.gap_max:.4f}: market is not "
                    "lagging, it disagrees - do not take the other side of informed flow"
                ),
                side=side,
                seconds_to_close=seconds_to_close,
                fair_probability=fair_probability,
                market_probability=market_prob,
                edge=edge,
                signal_strength=signal_strength,
                signal_quality=quality,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        # ---- 5) entry gating on repetition ---------------------------------
        last = self.last_entry_ts.get(side)
        if last is not None and (moment - last) < self.min_entry_gap_s:
            self.stats.blocked_gap += 1
            decision = SniperDecision(
                action="hold",
                reason=(
                    f"min entry gap: {(moment - last):.1f}s since last {side} entry "
                    f"(need {self.min_entry_gap_s:.0f}s)"
                ),
                side=side,
                seconds_to_close=seconds_to_close,
                fair_probability=fair_probability,
                market_probability=market_prob,
                edge=edge,
                signal_strength=signal_strength,
                signal_quality=quality,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
            )
            self.decisions.append(decision)
            return decision

        # ---- 6) execution planning ------------------------------------------
        plan = self._plan_entry(
            hub,
            side,
            snapshot=snapshot,
            seconds_to_close=seconds_to_close,
            volatility=vol,
            edge=edge,
        )
        if plan is None or not plan.executable:
            self.stats.blocked_execution += 1
            reason = plan.blocked_reason if plan else "execution planning failed"
            decision = SniperDecision(
                action="hold",
                reason=f"execution blocked: {reason}",
                side=side,
                seconds_to_close=seconds_to_close,
                fair_probability=fair_probability,
                market_probability=market_prob,
                edge=edge,
                signal_strength=signal_strength,
                signal_quality=quality,
                obi=obi,
                spot_price=spot_series,
                reference_price=ref,
                underlying_move_pct=move_pct,
                plan=plan,
                blockers=(reason,) if reason else (),
            )
            self.decisions.append(decision)
            return decision

        # ---- 7) accepted -> record and hand to the broker --------------------
        self.stats.entries += 1
        self.last_entry_ts[side] = moment
        decision = SniperDecision(
            action="enter",
            reason=(
                f"lag detected: underlying {move_pct:+.4f}%, fair {fair_probability:.4f} "
                f"vs market {market_prob:.4f} (edge {edge:+.4f}), {seconds_to_close:.1f}s left"
            ),
            side=side,
            shares=plan.filled_shares,
            seconds_to_close=seconds_to_close,
            fair_probability=fair_probability,
            market_probability=market_prob,
            edge=edge,
            signal_strength=signal_strength,
            signal_quality=quality,
            obi=obi,
            spot_price=spot_series,
            reference_price=ref,
            underlying_move_pct=move_pct,
            plan=plan,
        )
        self._dispatch(plan)
        self.decisions.append(decision)
        return decision

    # -- risk -------------------------------------------------------------
    def _evaluate_risk(
        self,
        hub: MarketDataHub,
        *,
        up_snapshot: BookSnapshot,
        down_snapshot: BookSnapshot,
        move_pct: float | None,
        vol_pct: float | None,
        seconds_to_close: float | None,
        moment: float,
    ) -> RiskDecision | None:
        if not self.risk.positions:
            return None
        signal_side = None
        if move_pct is not None and move_pct != 0:
            signal_side = UP if move_pct > 0 else DOWN
        strength = 0.0
        if move_pct is not None:
            strength = min(1.0, abs(move_pct) / max(1e-9, self.threshold_pct * 3.0))

        decisions = []
        for side in list(self.risk.positions):
            snapshot = up_snapshot if side == UP else down_snapshot
            opposite = down_snapshot if side == UP else up_snapshot
            decisions.append(
                self.risk.evaluate(
                    side,
                    snapshot=snapshot,
                    opposite_snapshot=opposite,
                    volatility=vol_pct,
                    seconds_to_close=seconds_to_close,
                    signal_strength=strength,
                    signal_side=signal_side,
                )
            )
        for decision in decisions:
            if decision.action is not Action.HOLD:
                return decision
        return decisions[0] if decisions else None

    def _decision_from_risk(
        self,
        risk_decision: RiskDecision,
        hub: MarketDataHub,
        seconds_to_close: float | None,
        move_pct: float | None,
        obi: float,
        spot: float | None,
        ref: float | None,
        healthy: bool,
    ) -> SniperDecision:
        if risk_decision.action in {Action.HEDGE, Action.HEDGE_TOPUP}:
            self.stats.hedges += 1
        elif risk_decision.action is Action.STOP_LOSS:
            self.stats.stops += 1
        elif risk_decision.action is Action.TAKE_PROFIT:
            self.stats.take_profits += 1

        plan = None
        if risk_decision.is_hedge and risk_decision.side and risk_decision.shares > 0:
            plan = self._plan_entry(
                hub,
                risk_decision.side,
                seconds_to_close=seconds_to_close,
                volatility=None,
                edge=0.0,
                shares_override=risk_decision.shares,
            )
            if plan is not None:
                self._dispatch(plan)
        return SniperDecision(
            action=risk_decision.action.value,
            reason=risk_decision.reason,
            side=risk_decision.side,
            shares=risk_decision.shares,
            seconds_to_close=seconds_to_close,
            signal_strength=risk_decision.signal_strength,
            obi=obi,
            spot_price=spot,
            reference_price=ref,
            underlying_move_pct=move_pct,
            risk=risk_decision,
            plan=plan,
            data_healthy=healthy,
        )

    # -- execution --------------------------------------------------------
    def _plan_entry(
        self,
        hub: MarketDataHub | None,
        side: str,
        *,
        snapshot: BookSnapshot | None = None,
        seconds_to_close: float | None,
        volatility: float | None,
        edge: float,
        shares_override: float | None = None,
    ) -> OrderPlan | None:
        """Price an order for ``side``. Either pass a snapshot or a hub to read it from."""
        if snapshot is None and hub is not None:
            snapshot = hub.snapshot(side)
        if snapshot is None:
            return None
        urgency = 0.0
        if seconds_to_close is not None:
            urgency = max(0.0, min(1.0, 1.0 - (seconds_to_close / max(1.0, self.scan_window_s))))
        try:
            if shares_override:
                return self.execution.plan_order(
                    snapshot,
                    "buy",
                    shares=shares_override,
                    seconds_to_close=seconds_to_close,
                    volatility=volatility,
                    urgency=urgency,
                )
            return self.execution.plan_order(
                snapshot,
                "buy",
                notional_usdc=self.order_notional,
                seconds_to_close=seconds_to_close,
                volatility=volatility,
                urgency=urgency,
            )
        except (ExecutionError, SlippageToleranceExceeded) as exc:
            log.info("execution plan rejected: %s", exc)
            return None

    def _dispatch(self, plan: OrderPlan) -> None:
        """Hand the plan to the injected broker. No broker => nothing is sent."""
        if self.broker is None:
            log.info(
                "plan ready (%s children, %.4f shares) - no broker injected, not sending",
                len(plan.children),
                plan.filled_shares,
            )
            return
        if self.mode is not Mode.LIVE:
            log.info("mode=%s: broker not called for a %s plan", self.mode.value, plan.side)
            return
        try:
            response = self.broker(plan)
            self.orders_sent.append(
                {"plan": plan.as_dict(), "response": str(response)[:500]}
            )
        except Exception as exc:  # noqa: BLE001 - broker boundary
            log.error("broker rejected the order: %s", exc)

    # -- signal strength ---------------------------------------------------
    def _signal_strength(self, edge: float, move_pct: float, obi: float, side: str) -> float:
        """Blend the three things that make a snipe credible.

        * ``edge``  - how far the market is from our fair value
        * ``move``  - how decisive the underlying move is versus the threshold
        * ``obi``   - whether the book's own lean agrees with us
        """
        edge_term = max(0.0, min(1.0, edge / max(1e-9, self.gap_max)))
        move_term = max(0.0, min(1.0, abs(move_pct) / max(1e-9, self.threshold_pct * 3.0)))
        aligned_obi = obi if side == UP else -obi
        obi_term = max(0.0, min(1.0, (aligned_obi + 1.0) / 2.0))
        strength = 0.5 * edge_term + 0.3 * move_term + 0.2 * obi_term
        return float(max(0.0, min(1.0, strength)))

    # -- record keeping ----------------------------------------------------
    def summary(self) -> dict:
        return {
            "mode": self.mode.value,
            "stats": self.stats.as_dict(),
            "positions": self.risk.exposure_report(),
            "decisions_logged": len(self.decisions),
            "orders_sent": len(self.orders_sent),
            "params": {
                "scan_window_s": self.scan_window_s,
                "observation_time_s": self.observation_time_s,
                "threshold_pct": self.threshold_pct,
                "take_profit": self.take_profit,
                "stop_loss": self.stop_loss,
                "min_entry_gap_s": self.min_entry_gap_s,
                "hedge_ratio": self.risk.limits.hedge_ratio,
            },
        }

    def decisions_as_rows(self) -> list[dict]:
        return [d.as_dict() for d in self.decisions]


# --------------------------------------------------------------- offline replay
def replay_observations(
    strategy: SniperStrategy,
    observations: Sequence[MarketObservation],
    *,
    hub_factory: Callable[[MarketObservation], MarketDataHub] | None = None,
) -> dict:
    """Drive the strategy over recorded observations without touching a network.

    Each observation is turned into a hub with one book and a synthetic spot
    tick, then fed through :meth:`SniperStrategy.evaluate`. Used by the paper
    mode and by the tests to prove the decision path works end to end.
    """
    from core.market_data import OrderBook

    entries = 0
    hedges = 0
    holds = 0
    for observation in observations:
        hub = (
            hub_factory(observation)
            if hub_factory
            else _hub_from_observation(observation)
        )
        decision = strategy.evaluate(
            hub,
            now=observation.observation_epoch,
            window_open_price=observation.btc_open,
        )
        if decision.is_entry:
            entries += 1
        elif decision.is_hedge:
            hedges += 1
        else:
            holds += 1
    return {
        "observations": len(observations),
        "entries": entries,
        "hedges": hedges,
        "holds": holds,
        "summary": strategy.summary(),
    }


def _hub_from_observation(observation: MarketObservation) -> MarketDataHub:
    """Build an offline hub from one observation (book + spot + window clock)."""
    from core.backtest_engine import SyntheticBookModel, coerce_midpoint
    from core.market_data import SpotTick

    price = observation.up_price
    if price is None:
        price = 1.0 - (observation.down_price or 0.5)
    model = SyntheticBookModel()
    up_book = model.build(
        coerce_midpoint(price, model.tick_size),
        t_minus_s=observation.t_minus_s,
        seed_key=observation.market_id,
    )
    down_price = observation.down_price
    if down_price is None:
        down_price = 1.0 - price
    down_book = model.build(
        coerce_midpoint(down_price, model.tick_size),
        t_minus_s=observation.t_minus_s,
        seed_key=f"{observation.market_id}:down",
    )

    hub = MarketDataHub(
        token_ids={UP: up_book.token_id, DOWN: down_book.token_id},
        book_depth=5,
    )
    for side, book in ((UP, up_book), (DOWN, down_book)):
        hub.books[side].apply_snapshot(
            {
                "bids": [list(level) for level in book.bids],
                "asks": [list(level) for level in book.asks],
                "timestamp": observation.observation_epoch,
                "window_close_epoch": observation.window_close_epoch,
            },
            source="replay",
        )
    spot = observation.btc_spot or observation.btc_close
    if spot is not None:
        hub.update_spot(
            SpotTick(
                symbol="BTCUSDT",
                price=float(spot),
                timestamp=observation.observation_epoch,
                source="replay",
            )
        )
    return hub

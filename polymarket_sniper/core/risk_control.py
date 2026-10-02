"""Module 3 - dynamic risk control and reverse hedging.

The core idea: a fixed stop distance is wrong for a market whose volatility
changes every second. Stops here widen with realised volatility, and the
inventory itself can be defended by opening the *opposite* outcome instead of
dumping the position - which is cheaper in a book that has already moved away
from you.

Decision outputs
----------------
:meth:`RiskController.evaluate` returns a :class:`RiskDecision` whose
``action`` is one of:

``hold``
    nothing to do;
``stop_loss``
    hard stop - probability moved against the position beyond the stop level;
``trailing_stop``
    give-back stop - price retraced from its favourable peak;
``hedge``
    open the opposite side with ``hedge_ratio`` of the position size;
``hedge_topup``
    widen an existing hedge because the signal got stronger;
``take_profit``
    target reached;
``flatten``
    force flat (window ending / data untrustworthy).

Hedging is deliberately bounded: ratio <= ``max_hedge_ratio``, a minimum signal
strength, and one top-up at most.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.market_data import BookSnapshot, UP, DOWN
from utils.logger import get_logger, utc_now

log = get_logger("risk_control")


class Action(str, Enum):
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    HEDGE = "hedge"
    HEDGE_TOPUP = "hedge_topup"
    FLATTEN = "flatten"


class RiskError(RuntimeError):
    """Raised for invalid position/limit combinations."""


@dataclass
class Position:
    """One open outcome position."""

    side: str
    shares: float
    entry_price: float
    entry_ts: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    peak_price: float | None = None
    hedge_shares: float = 0.0
    hedge_entry_price: float | None = None
    hedge_topups: int = 0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    @property
    def notional(self) -> float:
        return self.shares * self.entry_price

    @property
    def is_hedged(self) -> bool:
        return self.hedge_shares > 0

    @property
    def hedge_ratio(self) -> float:
        if self.shares <= 0:
            return 0.0
        return self.hedge_shares / self.shares

    def update_peak(self, price: float) -> None:
        if self.peak_price is None or price > self.peak_price:
            self.peak_price = price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares

    def hedge_pnl(self, opposite_price: float) -> float:
        if not self.is_hedged or self.hedge_entry_price is None:
            return 0.0
        return (opposite_price - self.hedge_entry_price) * self.hedge_shares

    def net_pnl(self, price: float, opposite_price: float | None = None) -> float:
        pnl = self.unrealized_pnl(price) - self.fees_paid
        if opposite_price is not None:
            pnl += self.hedge_pnl(opposite_price)
        return pnl

    def as_dict(self, price: float | None = None, opposite_price: float | None = None) -> dict:
        data = {
            "side": self.side,
            "shares": round(self.shares, 8),
            "entry_price": round(self.entry_price, 6),
            "notional": round(self.notional, 6),
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "peak_price": self.peak_price,
            "hedge_shares": round(self.hedge_shares, 8),
            "hedge_entry_price": self.hedge_entry_price,
            "hedge_ratio": round(self.hedge_ratio, 4),
            "fees_paid": round(self.fees_paid, 8),
        }
        if price is not None:
            data["unrealized_pnl"] = round(self.unrealized_pnl(price), 6)
            data["net_pnl"] = round(self.net_pnl(price, opposite_price), 6)
        return data


@dataclass(frozen=True)
class RiskLimits:
    """Static ceilings; dynamic levels are computed per evaluation."""

    max_position_notional: float = 500.0
    max_window_notional: float = 1500.0
    stop_loss_prob: float = 0.05
    trailing_stop_prob: float = 0.04
    obi_extreme_abs: float = 0.85
    obi_flip_abs: float = 0.60
    volatility_stop_mult: float = 2.5
    hedge_enabled: bool = True
    hedge_ratio: float = 0.60
    max_hedge_ratio: float = 1.0
    hedge_min_signal_strength: float = 0.55
    min_stop_prob: float = 0.01
    max_stop_prob: float = 0.35


@dataclass(frozen=True)
class RiskDecision:
    """What the risk layer wants done, with the reasoning attached."""

    action: Action
    reason: str
    side: str | None = None
    shares: float = 0.0
    reference_price: float | None = None
    stop_price: float | None = None
    dynamic_stop_prob: float = 0.0
    obi: float = 0.0
    volatility: float | None = None
    signal_strength: float = 0.0
    urgency: float = 0.0

    @property
    def is_exit(self) -> bool:
        return self.action in {Action.STOP_LOSS, Action.TRAILING_STOP, Action.FLATTEN}

    @property
    def is_hedge(self) -> bool:
        return self.action in {Action.HEDGE, Action.HEDGE_TOPUP}

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "side": self.side,
            "shares": round(self.shares, 8),
            "reference_price": self.reference_price,
            "stop_price": self.stop_price,
            "dynamic_stop_prob": round(self.dynamic_stop_prob, 6),
            "obi": round(self.obi, 4),
            "volatility": self.volatility,
            "signal_strength": round(self.signal_strength, 4),
            "urgency": round(self.urgency, 4),
        }


def opposite_side(side: str) -> str:
    return DOWN if side == UP else UP


def dynamic_stop_probability(
    *,
    base_stop: float,
    volatility: float | None,
    obi: float,
    mult: float = 2.5,
    min_stop: float = 0.01,
    max_stop: float = 0.35,
) -> float:
    """Volatility- and imbalance-aware stop distance in probability units.

    ``stop = clamp(base + mult * vol + |obi| * base, min, max)``

    Rationale: in a fast, lopsided book a tight stop is just a donation to the
    other side, so the stop widens with both realised volatility and the size of
    the imbalance working against a calm exit.
    """
    if base_stop < 0:
        raise RiskError("base_stop must be >= 0")
    vol_term = 0.0 if volatility is None else max(0.0, float(volatility)) * mult
    obi_term = abs(float(obi)) * base_stop
    stop = base_stop + vol_term + obi_term
    return float(max(min_stop, min(max_stop, stop)))


class RiskController:
    """Stateful risk layer: tracks positions and emits defended decisions."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        *,
        take_profit_prob: float = 0.10,
        window_notional_used: float = 0.0,
    ) -> None:
        self.limits = limits or RiskLimits()
        if not 0.0 <= self.limits.hedge_ratio <= self.limits.max_hedge_ratio <= 1.0:
            raise RiskError("require 0 <= hedge_ratio <= max_hedge_ratio <= 1")
        self.take_profit_prob = float(take_profit_prob)
        self.positions: dict[str, Position] = {}
        self.window_notional_used = float(window_notional_used)
        self.events: list[dict] = []

    # -- position bookkeeping --------------------------------------------
    def open_position(
        self,
        side: str,
        shares: float,
        price: float,
        *,
        timestamp: float = 0.0,
        fee_usdc: float = 0.0,
    ) -> Position:
        if shares <= 0:
            raise RiskError("shares must be > 0")
        if not 0.0 < price < 1.0:
            raise RiskError("price must be in (0, 1)")
        notional = shares * price
        if side in self.positions:
            existing = self.positions[side]
            total_shares = existing.shares + shares
            existing.entry_price = (
                existing.entry_price * existing.shares + price * shares
            ) / total_shares
            existing.shares = total_shares
            existing.fees_paid += fee_usdc
            existing.update_peak(price)
            self.window_notional_used += notional
            return existing

        position = Position(
            side=side,
            shares=float(shares),
            entry_price=float(price),
            entry_ts=timestamp,
            peak_price=price,
            fees_paid=float(fee_usdc),
        )
        self.positions[side] = position
        self.window_notional_used += notional
        self._log("open", position, note=f"{shares:.4f} @ {price:.4f}")
        return position

    def close_position(self, side: str, price: float, *, reason: str = "") -> float:
        position = self.positions.pop(side, None)
        if position is None:
            return 0.0
        realized = position.unrealized_pnl(price) - position.fees_paid
        if position.is_hedged and position.hedge_entry_price is not None:
            realized += position.hedge_pnl(price)
        self._log("close", position, note=f"{reason} realized={realized:+.4f}")
        return realized

    def add_hedge(self, side: str, shares: float, price: float, *, topup: bool = False) -> Position:
        position = self.positions.get(side)
        if position is None:
            raise RiskError(f"no position on {side} to hedge")
        if shares <= 0:
            raise RiskError("hedge shares must be > 0")
        projected = position.hedge_shares + shares
        if projected > position.shares * self.limits.max_hedge_ratio + 1e-12:
            allowed = position.shares * self.limits.max_hedge_ratio - position.hedge_shares
            if allowed <= 0:
                raise RiskError("hedge already at max_hedge_ratio")
            shares = allowed
        if position.hedge_entry_price is None:
            position.hedge_entry_price = price
        else:
            total = position.hedge_shares + shares
            position.hedge_entry_price = (
                position.hedge_entry_price * position.hedge_shares + price * shares
            ) / total
        position.hedge_shares += shares
        if topup:
            position.hedge_topups += 1
        self._log(
            "hedge",
            position,
            note=f"+{shares:.4f} @ {price:.4f} ratio={position.hedge_ratio:.2f}",
        )
        return position

    def _log(self, event: str, position: Position, *, note: str = "") -> None:
        self.events.append(
            {
                "event": event,
                "side": position.side,
                "shares": round(position.shares, 8),
                "entry_price": round(position.entry_price, 6),
                "note": note,
                "ts": utc_now().isoformat(),
            }
        )

    # -- sizing -----------------------------------------------------------
    def can_open(self, notional: float) -> tuple[bool, str]:
        if notional <= 0:
            return False, "notional must be positive"
        if notional > self.limits.max_position_notional:
            return (
                False,
                f"order notional {notional:.2f} exceeds max_position_notional "
                f"{self.limits.max_position_notional:.2f}",
            )
        if self.window_notional_used + notional > self.limits.max_window_notional:
            return (
                False,
                f"window notional {self.window_notional_used + notional:.2f} exceeds "
                f"max_window_notional {self.limits.max_window_notional:.2f}",
            )
        return True, ""

    def size_position(self, equity: float, price: float, *, risk_fraction: float = 0.02) -> float:
        """Shares for a fixed fraction of equity, capped by position limits."""
        if equity <= 0 or not 0.0 < price < 1.0:
            return 0.0
        notional = min(equity * risk_fraction, self.limits.max_position_notional)
        remaining = self.limits.max_window_notional - self.window_notional_used
        notional = max(0.0, min(notional, remaining))
        return notional / price

    # -- the decision -----------------------------------------------------
    def evaluate(
        self,
        side: str,
        *,
        snapshot: BookSnapshot,
        opposite_snapshot: BookSnapshot | None = None,
        volatility: float | None = None,
        seconds_to_close: float | None = None,
        signal_strength: float = 0.0,
        signal_side: str | None = None,
        force_flat: bool = False,
    ) -> RiskDecision:
        """Evaluate an open position and return the risk action.

        ``signal_side`` is where the *strategy* now thinks the market is going;
        when it opposes the position with enough strength, hedging is preferred
        over a stop, because selling into a book that has moved away is the most
        expensive way out.
        """
        position = self.positions.get(side)
        price = snapshot.midpoint() or snapshot.microprice()
        obi = snapshot.obi()
        if position is None or price is None:
            return RiskDecision(
                action=Action.HOLD,
                reason="no position or no price",
                side=side,
                obi=obi,
                volatility=volatility,
            )

        position.update_peak(price)
        stop_prob = dynamic_stop_probability(
            base_stop=self.limits.stop_loss_prob,
            volatility=volatility,
            obi=obi,
            mult=self.limits.volatility_stop_mult,
            min_stop=self.limits.min_stop_prob,
            max_stop=self.limits.max_stop_prob,
        )
        stop_price = max(0.0, position.entry_price - stop_prob)
        position.stop_price = stop_price
        position.take_profit_price = min(1.0, position.entry_price + self.take_profit_prob)

        pnl = position.unrealized_pnl(price)
        pnl_per_share = price - position.entry_price
        opposing_signal = signal_side is not None and signal_side != side
        extreme_obi = abs(obi) >= self.limits.obi_extreme_abs
        obi_against = (side == UP and obi <= -self.limits.obi_flip_abs) or (
            side == DOWN and obi >= self.limits.obi_flip_abs
        )

        # 1) hard stop: probability moved against us
        if pnl_per_share <= -stop_prob:
            return RiskDecision(
                action=Action.STOP_LOSS,
                reason=(
                    f"stop hit: move {pnl_per_share:+.4f} <= -{stop_prob:.4f} "
                    f"(dynamic stop from vol={volatility}, OBI={obi:+.2f})"
                ),
                side=side,
                shares=position.shares,
                reference_price=price,
                stop_price=stop_price,
                dynamic_stop_prob=stop_prob,
                obi=obi,
                volatility=volatility,
                signal_strength=signal_strength,
                urgency=1.0,
            )

        # 2) take profit
        if pnl_per_share >= self.take_profit_prob:
            return RiskDecision(
                action=Action.TAKE_PROFIT,
                reason=f"take profit: move {pnl_per_share:+.4f} >= {self.take_profit_prob:.4f}",
                side=side,
                shares=position.shares,
                reference_price=price,
                stop_price=stop_price,
                dynamic_stop_prob=stop_prob,
                obi=obi,
                volatility=volatility,
                signal_strength=signal_strength,
            )

        # 3) reverse hedge - the signature behaviour of this module
        if self.limits.hedge_enabled and opposite_snapshot is not None:
            hedge_decision = self._hedge_decision(
                position=position,
                side=side,
                price=price,
                opposite_snapshot=opposite_snapshot,
                volatility=volatility,
                obi=obi,
                signal_side=signal_side,
                signal_strength=signal_strength,
                opposing_signal=opposing_signal or obi_against or extreme_obi,
            )
            if hedge_decision is not None:
                return hedge_decision

        # 4) trailing give-back
        if position.peak_price is not None:
            give_back = position.peak_price - price
            if give_back >= self.limits.trailing_stop_prob and pnl > 0:
                return RiskDecision(
                    action=Action.TRAILING_STOP,
                    reason=(
                        f"trailing stop: retraced {give_back:.4f} from peak "
                        f"{position.peak_price:.4f} while still up {pnl:+.4f}"
                    ),
                    side=side,
                    shares=position.shares,
                    reference_price=price,
                    stop_price=stop_price,
                    dynamic_stop_prob=stop_prob,
                    obi=obi,
                    volatility=volatility,
                    signal_strength=signal_strength,
                    urgency=0.8,
                )

        # 5) endgame / forced flat
        if force_flat or (seconds_to_close is not None and seconds_to_close <= 1.0):
            return RiskDecision(
                action=Action.FLATTEN,
                reason="window closing; flatten before settlement",
                side=side,
                shares=position.shares,
                reference_price=price,
                stop_price=stop_price,
                dynamic_stop_prob=stop_prob,
                obi=obi,
                volatility=volatility,
                signal_strength=signal_strength,
                urgency=1.0,
            )

        return RiskDecision(
            action=Action.HOLD,
            reason=(
                f"within limits: move {pnl_per_share:+.4f}, stop -{stop_prob:.4f}, "
                f"OBI {obi:+.2f}"
            ),
            side=side,
            reference_price=price,
            stop_price=stop_price,
            dynamic_stop_prob=stop_prob,
            obi=obi,
            volatility=volatility,
            signal_strength=signal_strength,
        )

    def _hedge_decision(
        self,
        *,
        position: Position,
        side: str,
        price: float,
        opposite_snapshot: BookSnapshot,
        volatility: float | None,
        obi: float,
        signal_side: str | None,
        signal_strength: float,
        opposing_signal: bool,
    ) -> RiskDecision | None:
        """Decide whether to open (or top up) the reverse-side hedge."""
        if not opposing_signal:
            return None
        if signal_strength < self.limits.hedge_min_signal_strength:
            return None
        opposite_price = opposite_snapshot.midpoint() or opposite_snapshot.microprice()
        if opposite_price is None:
            return None

        target_ratio = self._target_hedge_ratio(obi=obi, signal_strength=signal_strength)
        current_ratio = position.hedge_ratio
        target_shares = position.shares * target_ratio
        delta = target_shares - position.hedge_shares

        if delta <= 1e-9:
            return None

        action = Action.HEDGE_TOPUP if position.is_hedged else Action.HEDGE
        if position.is_hedged and position.hedge_topups >= 1:
            return None  # one top-up only; stop layering risk

        return RiskDecision(
            action=action,
            reason=(
                f"signal reversed to {signal_side or 'n/a'} (strength "
                f"{signal_strength:.2f}) with OBI {obi:+.2f}; hedge {target_ratio:.0%} "
                f"of {position.shares:.4f} shares on {opposite_side(side)} at "
                f"{opposite_price:.4f}"
            ),
            side=opposite_side(side),
            shares=delta,
            reference_price=opposite_price,
            stop_price=position.stop_price,
            dynamic_stop_prob=0.0,
            obi=obi,
            volatility=volatility,
            signal_strength=signal_strength,
            urgency=0.9,
        )

    def _target_hedge_ratio(self, *, obi: float, signal_strength: float) -> float:
        """Scale the hedge with conviction, bounded by ``max_hedge_ratio``."""
        base = self.limits.hedge_ratio
        conviction = max(0.0, min(1.0, signal_strength))
        imbalance = min(1.0, abs(obi))
        scaled = base * (0.5 + 0.5 * conviction) + 0.15 * imbalance
        return float(max(0.0, min(self.limits.max_hedge_ratio, scaled)))

    # -- portfolio level ---------------------------------------------------
    def portfolio_hedge_ratio(self) -> float:
        if not self.positions:
            return 0.0
        ratios = [p.hedge_ratio for p in self.positions.values()]
        return sum(ratios) / len(ratios)

    def total_net_delta(self, prices: dict[str, float]) -> float:
        """Signed share exposure: +ve net long UP, -ve net long DOWN."""
        delta = 0.0
        for side, position in self.positions.items():
            sign = 1.0 if side == UP else -1.0
            delta += sign * position.shares
            if position.is_hedged:
                delta -= sign * position.hedge_shares
        return delta

    def exposure_report(self, snapshots: dict[str, BookSnapshot] | None = None) -> dict:
        snapshots = snapshots or {}
        rows = []
        total = 0.0
        for side, position in self.positions.items():
            snapshot = snapshots.get(side)
            price = snapshot.midpoint() if snapshot else None
            opposite = snapshots.get(opposite_side(side))
            opposite_price = opposite.midpoint() if opposite else None
            notional = position.notional
            total += notional
            rows.append(position.as_dict(price, opposite_price))
        return {
            "positions": rows,
            "position_count": len(rows),
            "gross_notional": round(total, 6),
            "window_notional_used": round(self.window_notional_used, 6),
            "max_window_notional": self.limits.max_window_notional,
            "avg_hedge_ratio": round(self.portfolio_hedge_ratio(), 4),
            "events": len(self.events),
        }

"""Module 4 - precise slippage control and adaptive order routing.

Three responsibilities:

1. **VWAP pre-trade estimate** - price a would-be order against the live book
   before sending it, so the strategy knows its true cost, not the top-of-book
   fantasy (see :func:`estimate_execution`).
2. **Adaptive routing** - choose Passive Maker or Aggressive Taker per child
   order from spread, depth, imbalance, volatility and urgency
   (:meth:`ExecutionEngine.choose_route`).
3. **Split orders** - break large tickets into depth-aware child clips, each
   one gated by ``slippageTolerance`` so a thin book can never be walked into a
   catastrophic fill (:meth:`ExecutionEngine.plan_order`).

Every method is pure: it takes a book snapshot and returns a plan. Nothing here
sends an order or opens a connection - the caller's broker does that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from core.market_data import BookSnapshot
from utils.fee_calculator import FeeBreakdown, fee_breakdown
from utils.logger import get_logger
from utils.vwap_calculator import (
    Level,
    VwapResult,
    participation_size,
    shares_for_notional_on_ladder,
    split_notional,
    vwap_buy,
    vwap_sell,
)

log = get_logger("execution")


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


class LiquidityRole(str, Enum):
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


class ExecutionError(RuntimeError):
    """Raised when an order cannot be routed safely."""


class SlippageToleranceExceeded(ExecutionError):
    """Raised when market impact would breach the configured tolerance."""

    def __init__(self, message: str, *, estimated_bps: float, tolerance_bps: float) -> None:
        super().__init__(message)
        self.estimated_bps = estimated_bps
        self.tolerance_bps = tolerance_bps


@dataclass(frozen=True)
class ExecutionEstimate:
    """Pre-trade cost estimate for one order."""

    side: str
    requested_shares: float
    filled_shares: float
    vwap: float
    notional_usdc: float
    fee_usdc: float
    total_cost_usdc: float
    slippage_bps: float
    slippage_tolerance_bps: float
    within_tolerance: bool
    fully_filled: bool
    levels_consumed: int
    reference_price: float | None
    available_shares: float

    @property
    def effective_price(self) -> float:
        """All-in price per share including fees."""
        if self.filled_shares <= 0:
            return float("nan")
        return (self.notional_usdc + self.fee_usdc) / self.filled_shares

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "requested_shares": round(self.requested_shares, 8),
            "filled_shares": round(self.filled_shares, 8),
            "vwap": round(self.vwap, 6),
            "notional_usdc": round(self.notional_usdc, 6),
            "fee_usdc": round(self.fee_usdc, 8),
            "total_cost_usdc": round(self.total_cost_usdc, 6),
            "effective_price": (
                None if math.isnan(self.effective_price) else round(self.effective_price, 6)
            ),
            "slippage_bps": round(self.slippage_bps, 4),
            "slippage_tolerance_bps": round(self.slippage_tolerance_bps, 4),
            "within_tolerance": self.within_tolerance,
            "fully_filled": self.fully_filled,
        }


@dataclass
class ChildOrder:
    """One clip of a parent order."""

    index: int
    shares: float
    notional_usdc: float
    order_type: OrderType
    limit_price: float | None
    estimate: ExecutionEstimate
    role: LiquidityRole

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "shares": round(self.shares, 8),
            "notional_usdc": round(self.notional_usdc, 6),
            "order_type": self.order_type.value,
            "role": self.role.value,
            "limit_price": self.limit_price,
            "estimate": self.estimate.as_dict(),
        }


@dataclass
class OrderPlan:
    """The full routing decision for one parent order."""

    side: str
    requested_shares: float
    requested_notional: float
    children: list[ChildOrder] = field(default_factory=list)
    blocked_reason: str | None = None
    route_reason: str = ""
    unfilled_shares: float = 0.0
    total_fee_usdc: float = 0.0
    total_notional_usdc: float = 0.0
    blended_vwap: float = 0.0

    @property
    def executable(self) -> bool:
        return self.blocked_reason is None and bool(self.children)

    @property
    def filled_shares(self) -> float:
        return sum(child.shares for child in self.children)

    @property
    def maker_children(self) -> int:
        return sum(1 for c in self.children if c.order_type is OrderType.MAKER)

    @property
    def taker_children(self) -> int:
        return sum(1 for c in self.children if c.order_type is OrderType.TAKER)

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "requested_shares": round(self.requested_shares, 8),
            "requested_notional": round(self.requested_notional, 6),
            "executable": self.executable,
            "blocked_reason": self.blocked_reason,
            "route_reason": self.route_reason,
            "children": [c.as_dict() for c in self.children],
            "filled_shares": round(self.filled_shares, 8),
            "unfilled_shares": round(self.unfilled_shares, 8),
            "maker_children": self.maker_children,
            "taker_children": self.taker_children,
            "blended_vwap": round(self.blended_vwap, 6),
            "total_notional_usdc": round(self.total_notional_usdc, 6),
            "total_fee_usdc": round(self.total_fee_usdc, 8),
        }


@dataclass(frozen=True)
class Fill:
    """A realised fill (used by the backtester and by live reports)."""

    side: str
    shares: float
    price: float
    fee_usdc: float
    role: str
    timestamp: float = 0.0
    order_type: str = "taker"

    @property
    def notional_usdc(self) -> float:
        return self.shares * self.price

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "shares": round(self.shares, 8),
            "price": round(self.price, 6),
            "notional_usdc": round(self.notional_usdc, 6),
            "fee_usdc": round(self.fee_usdc, 8),
            "role": self.role,
            "order_type": self.order_type,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------- pure helpers
def book_side_levels(snapshot: BookSnapshot, side: Side | str) -> tuple[Level, ...]:
    """Levels an order will actually consume."""
    side_value = side.value if isinstance(side, Side) else str(side).lower()
    return snapshot.asks if side_value == "buy" else snapshot.bids


def estimate_execution(
    snapshot: BookSnapshot,
    side: Side | str,
    shares: float,
    *,
    is_taker: bool = True,
    slippage_tolerance: float = 0.01,
    fee_coefficient: float = 0.07,
    fee_curve: str = "documented",
    maker_rebate_share: float = 0.25,
    reference_price: float | None = None,
    reference_side: str = "touch",
) -> ExecutionEstimate:
    """VWAP pre-trade estimate including fees and slippage for one order.

    ``slippage_tolerance`` is a **fraction of price** (0.01 == 1%), matching the
    ``slippageTolerance`` field carried on live market orders.

    ``reference_side`` decides what slippage is measured against:

    ``touch`` (default)
        The best price on the side being consumed (best ask for a buy). Slippage
        is then the *ladder impact* - how much worse the average fill is than the
        top of book. Crossing the spread is not counted as slippage, because on a
        binary book where 0.50/0.54 is a normal quote, half the spread would
        otherwise consume the whole tolerance on every single trade.
    ``mid``
        The midpoint. Slippage then includes half the spread; useful when you
        want to measure total vs-fair-value cost instead of market impact.
    """
    if shares <= 0:
        raise ExecutionError("shares must be > 0")
    side_value = side.value if isinstance(side, Side) else str(side).lower()
    if side_value not in {"buy", "sell"}:
        raise ExecutionError("side must be 'buy' or 'sell'")

    levels = book_side_levels(snapshot, side_value)
    if not levels:
        raise ExecutionError(f"no {side_value} liquidity in book for {snapshot.side}")

    reference = reference_price
    if reference is None:
        if reference_side == "mid":
            reference = snapshot.midpoint()
        else:
            reference = snapshot.best_ask if side_value == "buy" else snapshot.best_bid
        if reference is None:
            reference = snapshot.midpoint()

    tolerance_abs = max(0.0, float(slippage_tolerance))
    if side_value == "buy":
        result: VwapResult = vwap_buy(
            levels, shares, reference_price=reference, slippage_tolerance=tolerance_abs
        )
    else:
        result = vwap_sell(
            levels, shares, reference_price=reference, slippage_tolerance=tolerance_abs
        )

    breakdown: FeeBreakdown = fee_breakdown(
        result.filled_shares,
        result.vwap if result.vwap > 0 else (reference or 0.5),
        is_taker=is_taker,
        coefficient=fee_coefficient,
        curve=fee_curve,  # type: ignore[arg-type]
        rebate_share=maker_rebate_share,
    )

    return ExecutionEstimate(
        side=side_value,
        requested_shares=float(shares),
        filled_shares=result.filled_shares,
        vwap=result.vwap,
        notional_usdc=result.notional_usdc,
        fee_usdc=breakdown.charged_usdc,
        total_cost_usdc=result.notional_usdc + breakdown.charged_usdc,
        slippage_bps=result.slippage_bps,
        slippage_tolerance_bps=tolerance_abs * 10_000.0,
        within_tolerance=abs(result.slippage_abs) <= tolerance_abs + 1e-12,
        fully_filled=result.fully_filled,
        levels_consumed=result.levels_consumed,
        reference_price=reference,
        available_shares=result.available_shares,
    )


def _impact_bps(snapshot: BookSnapshot, side: str, vwap: float) -> float:
    """Ladder impact: how much worse the VWAP is than the touch price."""
    touch = snapshot.best_ask if side == "buy" else snapshot.best_bid
    if not touch or vwap <= 0:
        return 0.0
    if side == "buy":
        return max(0.0, (vwap - touch) / touch * 10_000.0)
    return max(0.0, (touch - vwap) / touch * 10_000.0)


# -------------------------------------------------------------------- engine
class ExecutionEngine:
    """Routing brain: plans orders, never sends them."""

    def __init__(
        self,
        *,
        slippage_tolerance: float = 0.01,
        max_slippage_bps: float = 150.0,
        maker_first: bool = True,
        maker_timeout_s: float = 2.0,
        split_threshold_notional: float = 50.0,
        participation_rate: float = 0.25,
        min_children: int = 2,
        max_children: int = 6,
        tick_size: float = 0.001,
        price_improve_ticks: int = 1,
        fee_coefficient: float = 0.07,
        fee_curve: str = "documented",
        maker_rebate_share: float = 0.25,
        depth_levels: int = 5,
        reference_side: str = "touch",
    ) -> None:
        self.slippage_tolerance = float(slippage_tolerance)
        self.max_slippage_bps = float(max_slippage_bps)
        self.maker_first = bool(maker_first)
        self.maker_timeout_s = float(maker_timeout_s)
        self.split_threshold_notional = float(split_threshold_notional)
        self.participation_rate = float(participation_rate)
        self.min_children = int(min_children)
        self.max_children = int(max_children)
        self.tick_size = float(tick_size)
        self.price_improve_ticks = int(price_improve_ticks)
        self.fee_coefficient = float(fee_coefficient)
        self.fee_curve = fee_curve
        self.maker_rebate_share = float(maker_rebate_share)
        self.depth_levels = int(depth_levels)
        self.reference_side = str(reference_side)

    # -- routing decision -------------------------------------------------
    def choose_route(
        self,
        snapshot: BookSnapshot,
        side: Side | str,
        *,
        seconds_to_close: float | None = None,
        volatility: float | None = None,
        urgency: float = 0.0,
    ) -> tuple[OrderType, LiquidityRole, str]:
        """Pick maker vs taker from spread, depth, OBI, volatility and urgency.

        The heuristic is intentionally explainable - every branch returns the
        reason so a backtest can tell you *why* it routed that way:

        * endgame (``seconds_to_close`` <= 2s) -> always taker: a passive order
          will not be filled before the window closes;
        * very wide spread -> maker: crossing it is pure cost;
        * strong adverse imbalance (OBI against us) -> taker: the book is
          already leaning away;
        * high volatility -> taker: the quote you saw will not be there next tick;
        * otherwise -> maker first.
        """
        side_value = side.value if isinstance(side, Side) else str(side).lower()
        spread = snapshot.spread()
        mid = snapshot.midpoint()
        obi = snapshot.obi(self.depth_levels)

        if mid is None or spread is None:
            return OrderType.TAKER, LiquidityRole.AGGRESSIVE, "no two-sided quote; take what exists"

        spread_bps = snapshot.spread_bps() or 0.0

        if seconds_to_close is not None and seconds_to_close <= 2.0:
            return (
                OrderType.TAKER,
                LiquidityRole.AGGRESSIVE,
                f"endgame: {seconds_to_close:.1f}s to close leaves no time for a resting order",
            )

        if urgency >= 0.8:
            return OrderType.TAKER, LiquidityRole.AGGRESSIVE, f"high urgency ({urgency:.2f})"

        # OBI sign: positive = bid-heavy = supportive for a buy
        adverse = (side_value == "buy" and obi <= -0.6) or (side_value == "sell" and obi >= 0.6)
        if adverse:
            return (
                OrderType.TAKER,
                LiquidityRole.AGGRESSIVE,
                f"adverse imbalance OBI={obi:+.2f}; book is leaning against the order",
            )

        if spread_bps > max(80.0, self.max_slippage_bps * 0.5):
            return (
                OrderType.MAKER,
                LiquidityRole.PASSIVE,
                f"wide spread {spread_bps:.0f}bps; crossing is too expensive",
            )

        if volatility is not None and volatility > 0.02:
            return (
                OrderType.TAKER,
                LiquidityRole.AGGRESSIVE,
                f"volatility {volatility:.4f} high; quotes are unstable",
            )

        if self.maker_first:
            return (
                OrderType.MAKER,
                LiquidityRole.PASSIVE,
                f"balanced book (spread {spread_bps:.0f}bps, OBI={obi:+.2f}); post passively",
            )
        return OrderType.TAKER, LiquidityRole.AGGRESSIVE, "maker_first disabled"

    def passive_price(self, snapshot: BookSnapshot, side: Side | str) -> float | None:
        """Improve the inside quote by ``price_improve_ticks``."""
        side_value = side.value if isinstance(side, Side) else str(side).lower()
        improve = self.price_improve_ticks * self.tick_size
        if side_value == "buy":
            best = snapshot.best_bid
            if best is None:
                return None
            price = min(best + improve, (snapshot.best_ask or 1.0) - self.tick_size)
        else:
            best = snapshot.best_ask
            if best is None:
                return None
            price = max(best - improve, (snapshot.best_bid or 0.0) + self.tick_size)
        return round(max(self.tick_size, min(1.0 - self.tick_size, price)), 6)

    # -- planning ---------------------------------------------------------
    def plan_order(
        self,
        snapshot: BookSnapshot,
        side: Side | str,
        *,
        shares: float | None = None,
        notional_usdc: float | None = None,
        seconds_to_close: float | None = None,
        volatility: float | None = None,
        urgency: float = 0.0,
        allow_partial: bool = False,
    ) -> OrderPlan:
        """Full routing plan with split children and tolerance gating."""
        side_value = side.value if isinstance(side, Side) else str(side).lower()
        levels = book_side_levels(snapshot, side_value)
        if not levels:
            raise ExecutionError(f"no liquidity on the {side_value} side")

        if shares is None:
            if notional_usdc is None:
                raise ExecutionError("provide shares= or notional_usdc=")
            budget = float(notional_usdc)
            # What the caller asked for, expressed in shares at the reference price.
            # The book may only be able to absorb less than this - that shortfall is
            # reported (and can block the order) rather than silently shrinking the
            # request to whatever the book happens to hold.
            reference_for_size = (
                (snapshot.best_ask if side_value == "buy" else snapshot.best_bid)
                or snapshot.midpoint()
                or 0.5
            )
            fillable = shares_for_notional_on_ladder(levels, budget)
            shares = budget / reference_for_size
            shares_shortfall = shares - fillable
        else:
            shares_shortfall = shares - shares_for_notional_on_ladder(
                levels, shares * (snapshot.best_ask or snapshot.midpoint() or 0.5)
            )
        shares = float(shares)
        if shares <= 0:
            raise ExecutionError("computed order size is zero")

        reference = snapshot.midpoint() or snapshot.best_bid or snapshot.best_ask or 0.5
        plan = OrderPlan(
            side=side_value,
            requested_shares=shares,
            requested_notional=shares * reference,
        )

        order_type, role, reason = self.choose_route(
            snapshot,
            side_value,
            seconds_to_close=seconds_to_close,
            volatility=volatility,
            urgency=urgency,
        )

        # --- split decision -------------------------------------------------
        # The configured threshold is a ceiling, not a floor: on a book whose
        # visible depth is smaller than the threshold, a single child would try to
        # swallow the whole order and fail the availability gate. Splitting is
        # therefore triggered whenever the order exceeds a fraction of visible
        # depth, whichever bites first.
        notional_ref = shares * reference
        visible_notional = sum(price * size for price, size in levels[: self.depth_levels])
        effective_threshold = min(
            self.split_threshold_notional,
            max(reference, 0.5 * visible_notional) if visible_notional > 0 else self.split_threshold_notional,
        )
        if notional_ref >= effective_threshold:
            clips = split_notional(
                notional_ref,
                levels[: self.depth_levels],
                participation_rate=self.participation_rate,
                min_children=self.min_children,
                max_children=self.max_children,
            )
            children_shares = [clip / reference for clip in clips]
            plan.route_reason = (
                f"{reason} | split {len(children_shares)} children of "
                f"{notional_ref / len(children_shares):.2f} USDC "
                f"(participation {self.participation_rate:.0%} of {visible_notional:.2f} USDC visible)"
            )
        else:
            children_shares = [shares]
            plan.route_reason = reason

        # --- price each child ---------------------------------------------
        tolerance_rejections: list[str] = []
        for index, child_shares in enumerate(children_shares):
            if child_shares <= 0:
                continue
            is_taker = order_type is OrderType.TAKER
            estimate = estimate_execution(
                snapshot,
                side_value,
                child_shares,
                is_taker=is_taker,
                slippage_tolerance=self.slippage_tolerance,
                fee_coefficient=self.fee_coefficient,
                fee_curve=self.fee_curve,
                maker_rebate_share=self.maker_rebate_share,
                reference_price=None,
                reference_side=self.reference_side,
            )
            # the gate measures LADDER IMPACT (VWAP vs top of book). The passive
            # limit price is a better reference when posting rather than taking.
            impact_bps = _impact_bps(snapshot, side_value, estimate.vwap)
            over_bps = impact_bps > self.max_slippage_bps
            if not estimate.within_tolerance or over_bps:
                tolerance_rejections.append(
                    f"child {index}: slippage {estimate.slippage_bps:.1f}bps / "
                    f"impact {impact_bps:.1f}bps "
                    f"(tolerance {estimate.slippage_tolerance_bps:.1f}bps, "
                    f"cap {self.max_slippage_bps:.1f}bps)"
                )
                continue
            if not estimate.fully_filled and not allow_partial:
                tolerance_rejections.append(
                    f"child {index}: only {estimate.filled_shares:.4f} of "
                    f"{child_shares:.4f} shares available"
                )
                continue
            plan.children.append(
                ChildOrder(
                    index=index,
                    shares=estimate.filled_shares,
                    notional_usdc=estimate.notional_usdc,
                    order_type=order_type,
                    limit_price=(
                        self.passive_price(snapshot, side_value)
                        if order_type is OrderType.MAKER
                        else None
                    ),
                    estimate=estimate,
                    role=role,
                )
            )

        if not plan.children:
            plan.blocked_reason = (
                "no child passed the slippage/availability gate: "
                + "; ".join(tolerance_rejections[:3] or ["unknown"])
            )
            plan.unfilled_shares = shares
            return plan

        # The book could not absorb the full request: say so instead of quietly
        # trading a smaller order than the caller asked for.
        if shares_shortfall > max(1e-9, shares * 1e-6) and not allow_partial:
            plan.blocked_reason = (
                f"visible depth only supports {shares - shares_shortfall:.4f} of "
                f"{shares:.4f} shares requested "
                f"(shortfall {shares_shortfall:.4f}); raise depth, split further, "
                "or pass allow_partial=True"
            )
            plan.unfilled_shares = max(0.0, shares_shortfall)
            return plan

        filled = plan.filled_shares
        plan.unfilled_shares = max(0.0, shares - filled)
        plan.total_notional_usdc = sum(c.notional_usdc for c in plan.children)
        plan.total_fee_usdc = sum(c.estimate.fee_usdc for c in plan.children)
        plan.blended_vwap = (
            plan.total_notional_usdc / filled if filled > 0 else 0.0
        )
        if plan.unfilled_shares > 0 and not allow_partial:
            # unfilled remainder means the book could not absorb the parent size
            plan.blocked_reason = (
                f"only {filled:.4f} of {shares:.4f} shares fillable within tolerance"
            )
        return plan

    # -- post-trade defensive checks --------------------------------------
    def check_fill(
        self,
        estimate: ExecutionEstimate,
        actual_price: float,
        actual_shares: float,
    ) -> tuple[bool, float]:
        """Verify a real fill against the pre-trade estimate.

        Returns ``(ok, deviation_bps)``. A broker print materially worse than the
        estimate is the signature of a malicious/latency-arbitraged fill.
        """
        if actual_shares <= 0:
            return False, 0.0
        expected = estimate.vwap
        if not expected:
            return True, 0.0
        if estimate.side == "buy":
            deviation = (actual_price - expected) / expected * 10_000.0
        else:
            deviation = (expected - actual_price) / expected * 10_000.0
        ok = deviation <= self.max_slippage_bps
        if not ok:
            log.warning(
                "fill outside slippage cap: actual %.6f vs expected %.6f (%.1f bps)",
                actual_price,
                expected,
                deviation,
            )
        return ok, float(deviation)

    def apply_slippage_tolerance(
        self, price: float, side: Side | str, tolerance: float | None = None
    ) -> float:
        """Worst price a market order may accept, for the venue's slippageTolerance."""
        tol = self.slippage_tolerance if tolerance is None else float(tolerance)
        side_value = side.value if isinstance(side, Side) else str(side).lower()
        if side_value == "buy":
            worst = price * (1.0 + tol)
        else:
            worst = price * (1.0 - tol)
        return round(max(self.tick_size, min(1.0 - self.tick_size, worst)), 6)


def execution_cost_curve(
    snapshot: BookSnapshot,
    side: Side | str,
    notionals: Sequence[float],
    *,
    fee_coefficient: float = 0.07,
    slippage_tolerance: float = 0.01,
) -> list[dict]:
    """Cost vs size - shows where the book starts biting."""
    rows = []
    for notional in notionals:
        levels = book_side_levels(snapshot, side)
        from utils.vwap_calculator import shares_for_notional_on_ladder

        shares = shares_for_notional_on_ladder(levels, notional)
        if shares <= 0:
            continue
        estimate = estimate_execution(
            snapshot,
            side,
            shares,
            fee_coefficient=fee_coefficient,
            slippage_tolerance=slippage_tolerance,
        )
        row = estimate.as_dict()
        row["target_notional"] = notional
        rows.append(row)
    return rows

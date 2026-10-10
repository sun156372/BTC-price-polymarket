"""Execution guards - the checks that keep a backtest edge from dying live.

Each guard is independent and returns a structured refusal instead of raising, so
a run can report *why* it declined to trade. That breakdown is the most useful
diagnostic the engine produces: it tells you whether a bad result came from the
signal or from the market being untradeable.

History behind the defaults
---------------------------
Measured on the trade history, splitting entries by how old the last print was:

======================  ========  ===============  ===========
quote staleness         trades    direction acc.   net P&L
======================  ========  ===============  ===========
<= 10s                  774       76.0%            +4,723
> 10s                   65        35.4%            -1,915
======================  ========  ===============  ===========

The stale bucket is **worse than a coin flip** and accounts for the entire net
loss. Mechanically the reason is simple: when the newest print is 30s or 200s
old, the "confirmed direction" was read off a price that no longer exists. That
is why :class:`StalenessGuard` is the first guard and why its default is 10s.
"""
from __future__ import annotations

from dataclasses import dataclass

from .signals import Signal


class GuardName:
    STALENESS = "quote_staleness"
    SLIPPAGE = "slippage"
    DEPTH = "depth"
    PRICE_BAND = "price_band"


@dataclass(frozen=True)
class GuardDecision:
    """Allow/deny with the arithmetic that produced it."""

    allowed: bool
    guard: str
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "guard": self.guard,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Quote:
    """What we would actually trade against, right now.

    ``price`` is the price we would pay (the ask for a buy). ``size_at_price`` is
    the visible depth there when the feed provides it; ``None`` means unknown and
    the depth guard abstains rather than inventing a number.
    """

    ts: float
    price: float
    size_at_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None

    @property
    def staleness_s(self) -> float:
        return 0.0

    def age(self, now: float) -> float:
        return now - self.ts


class StalenessGuard:
    """Refuse to trade on a price that has stopped updating."""

    name = GuardName.STALENESS

    def __init__(self, max_staleness_s: float = 10.0) -> None:
        if max_staleness_s <= 0:
            raise ValueError("max_staleness_s must be > 0")
        self.max_staleness_s = float(max_staleness_s)

    def evaluate(self, *, now: float, quote: Quote, shares: float,
                 signal_price: float) -> GuardDecision:
        return self.check(quote, now=now)

    def check(self, quote: Quote, *, now: float) -> GuardDecision:
        age = quote.age(now)
        if age <= self.max_staleness_s:
            return GuardDecision(True, self.name, "fresh", f"{age:.1f}s old")
        return GuardDecision(
            False, self.name, "stale_quote",
            f"last print {age:.1f}s old (limit {self.max_staleness_s:.0f}s); "
            "direction read from a dead price",
        )


class SlippageGuard:
    """Refuse a fill materially worse than the price the signal assumed."""

    name = GuardName.SLIPPAGE

    def __init__(self, max_slippage_bps: float = 200.0) -> None:
        if max_slippage_bps <= 0:
            raise ValueError("max_slippage_bps must be > 0")
        self.max_slippage_bps = float(max_slippage_bps)

    def evaluate(self, *, now: float, quote: Quote, shares: float,
                 signal_price: float) -> GuardDecision:
        return self.check(signal_price=signal_price, quote=quote)

    def check(self, *, signal_price: float, quote: Quote) -> GuardDecision:
        if signal_price <= 0:
            return GuardDecision(False, self.name, "bad_reference",
                                 f"signal price {signal_price!r}")
        slippage_bps = (quote.price - signal_price) / signal_price * 10_000.0
        if slippage_bps <= self.max_slippage_bps:
            return GuardDecision(True, self.name, "within_tolerance",
                                 f"{slippage_bps:+.1f} bps")
        return GuardDecision(
            False, self.name, "slippage_exceeded",
            f"{slippage_bps:+.1f} bps worse than the signal price "
            f"(limit {self.max_slippage_bps:.0f} bps)",
        )


class DepthGuard:
    """Refuse to take more than a fraction of what is visibly offered."""

    name = GuardName.DEPTH

    def __init__(self, max_participation: float = 0.50) -> None:
        if not 0.0 < max_participation <= 1.0:
            raise ValueError("max_participation must be in (0, 1]")
        self.max_participation = float(max_participation)

    def evaluate(self, *, now: float, quote: Quote, shares: float,
                 signal_price: float) -> GuardDecision:
        return self.check(shares=shares, quote=quote)

    def check(self, *, shares: float, quote: Quote) -> GuardDecision:
        if quote.size_at_price is None:
            return GuardDecision(True, self.name, "depth_unknown",
                                 "feed provides no size; guard abstains")
        allowed = quote.size_at_price * self.max_participation
        if shares <= allowed:
            return GuardDecision(True, self.name, "within_depth",
                                 f"{shares:.2f} of {allowed:.2f} allowed shares")
        return GuardDecision(
            False, self.name, "insufficient_depth",
            f"want {shares:.2f} shares, only {allowed:.2f} allowed "
            f"({self.max_participation:.0%} of {quote.size_at_price:.2f} visible)",
        )


class PriceBandGuard:
    """Optional sanity band on the entry price.

    Note it is **disabled by default**, and that is deliberate: measured on this
    dataset, entries above 0.65 hit the direction 98.6% of the time and were
    *profitable* (the market had already priced the outcome). An "avoid expensive
    entries" rule sounds prudent and costs money here.
    """

    name = GuardName.PRICE_BAND

    def __init__(self, min_price: float = 0.0, max_price: float = 1.0) -> None:
        self.min_price = float(min_price)
        self.max_price = float(max_price)

    def evaluate(self, *, now: float, quote: Quote, shares: float,
                 signal_price: float) -> GuardDecision:
        return self.check(quote)

    def check(self, quote: Quote) -> GuardDecision:
        if self.min_price <= quote.price <= self.max_price:
            return GuardDecision(True, self.name, "in_band", f"{quote.price:.4f}")
        return GuardDecision(
            False, self.name, "out_of_band",
            f"price {quote.price:.4f} outside [{self.min_price}, {self.max_price}]",
        )


class GuardChain:
    """Runs every guard in cost order and reports all refusals.

    Each guard implements ``evaluate(context)``; the chain does not care which
    guard it is holding, so adding a new one is a one-line change.
    """

    def __init__(self, guards: list) -> None:
        self.guards = list(guards)

    @classmethod
    def from_config(cls, guards_section) -> "GuardChain":
        chain = [
            StalenessGuard(guards_section.max_quote_staleness_s),
            SlippageGuard(guards_section.max_slippage_bps),
            DepthGuard(guards_section.max_depth_participation),
        ]
        min_price = float(guards_section.get("min_price", 0.0) or 0.0)
        max_price = float(guards_section.get("max_price", 1.0) or 1.0)
        if min_price > 0.0 or max_price < 1.0:
            chain.append(PriceBandGuard(min_price, max_price))
        return cls(chain)

    def check(
        self,
        signal: Signal,
        quote: Quote,
        *,
        now: float,
        shares: float,
        signal_price: float | None = None,
    ) -> tuple[bool, list[GuardDecision]]:
        """Return ``(allowed, decisions)``, stopping at the first refusal.

        ``signal_price`` is the price the strategy believed it would pay. When it
        is omitted the quote is treated as its own reference, which means the
        slippage guard can only fire on an explicitly supplied reference - so
        callers that care about slippage should pass it.
        """
        decisions: list[GuardDecision] = []
        for guard in self.guards:
            decision = guard.evaluate(
                now=now, quote=quote, shares=shares,
                signal_price=signal_price if signal_price is not None else quote.price,
            )
            decisions.append(decision)
            if not decision.allowed:
                return False, decisions
        return True, decisions

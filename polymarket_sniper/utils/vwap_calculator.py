"""VWAP and order-book walking.

Everything here is pure arithmetic over ``[(price, size), ...]`` ladders, so it
can be unit-tested without a venue connection.

Conventions
-----------
* ``bids`` are sorted best-first (highest price first), ``asks`` best-first
  (lowest price first). :func:`sort_levels` enforces this.
* Buying consumes asks from the lowest price upward; selling consumes bids from
  the highest price downward.
* ``size`` is in shares; ``price`` is in probability space (0..1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

Level = tuple[float, float]


class VwapError(ValueError):
    """Raised for malformed ladders or impossible requests."""


def sort_levels(levels: Iterable[Sequence[float]], side: str) -> list[Level]:
    """Normalise a ladder to ``[(price, size), ...]``, best price first."""
    cleaned: list[Level] = []
    for level in levels:
        if len(level) < 2:
            raise VwapError(f"level needs (price, size), got {level!r}")
        price, size = float(level[0]), float(level[1])
        if price <= 0 or price >= 1:
            raise VwapError(f"price out of (0, 1): {price!r}")
        if size < 0:
            raise VwapError(f"size must be >= 0: {size!r}")
        cleaned.append((price, size))
    reverse = side.lower() == "bid"
    cleaned.sort(key=lambda item: item[0], reverse=reverse)
    return cleaned


def levels_for_side(
    order_book: dict, side: str, depth: int | None = None
) -> list[Level]:
    """Pull a ladder out of a raw book dict and normalise it."""
    key = "bids" if side.lower() in {"bid", "bids", "buy"} else "asks"
    levels = order_book.get(key) or []
    if not isinstance(levels, (list, tuple)):
        raise VwapError(f"{key} must be a list of levels")
    normalised_side = "bid" if key == "bids" else "ask"
    result = sort_levels(levels, normalised_side)
    if depth is not None:
        result = result[: max(0, int(depth))]
    return result


def book_depth(levels: Sequence[Level]) -> float:
    return float(sum(size for _, size in levels))


def book_notional(levels: Sequence[Level]) -> float:
    return float(sum(price * size for price, size in levels))


def best_price(levels: Sequence[Level]) -> float | None:
    return float(levels[0][0]) if levels else None


def walk_ladder(
    levels: Sequence[Level], shares: float
) -> tuple[float, float, float, bool]:
    """Consume ``shares`` through a ladder.

    Returns ``(filled_shares, notional_usdc, vwap, fully_filled)``.
    ``vwap`` is 0.0 when nothing fills.
    """
    if shares < 0:
        raise VwapError("shares must be >= 0")
    remaining = float(shares)
    filled = 0.0
    notional = 0.0
    for price, size in levels:
        if remaining <= 0:
            break
        take = min(remaining, size)
        if take <= 0:
            continue
        filled += take
        notional += take * price
        remaining -= take
    vwap = (notional / filled) if filled > 0 else 0.0
    return filled, notional, vwap, remaining <= 1e-12


@dataclass(frozen=True)
class VwapResult:
    """Outcome of pricing an order against a ladder."""

    side: str
    requested_shares: float
    filled_shares: float
    notional_usdc: float
    vwap: float
    best_price: float | None
    reference_price: float | None
    slippage_abs: float
    slippage_bps: float
    fully_filled: bool
    levels_consumed: int
    available_shares: float
    slippage_tolerance: float | None = None

    @property
    def fill_ratio(self) -> float:
        if self.requested_shares <= 0:
            return 0.0
        return self.filled_shares / self.requested_shares

    @property
    def unfilled_shares(self) -> float:
        return max(0.0, self.requested_shares - self.filled_shares)

    @property
    def within_tolerance(self) -> bool:
        if self.slippage_tolerance is None:
            return True
        return abs(self.slippage_abs) <= self.slippage_tolerance + 1e-12

    @property
    def average_price(self) -> float:
        return self.vwap

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "requested_shares": round(self.requested_shares, 8),
            "filled_shares": round(self.filled_shares, 8),
            "fill_ratio": round(self.fill_ratio, 6),
            "notional_usdc": round(self.notional_usdc, 6),
            "vwap": round(self.vwap, 6),
            "best_price": self.best_price,
            "reference_price": self.reference_price,
            "slippage_abs": round(self.slippage_abs, 8),
            "slippage_bps": round(self.slippage_bps, 4),
            "fully_filled": self.fully_filled,
            "levels_consumed": self.levels_consumed,
            "available_shares": round(self.available_shares, 6),
        }


def vwap_buy(
    asks: Sequence[Level],
    shares: float,
    *,
    reference_price: float | None = None,
    slippage_tolerance: float | None = None,
) -> VwapResult:
    """VWAP for a buy against an ask ladder."""
    return _vwap(asks, shares, "buy", reference_price, slippage_tolerance)


def vwap_sell(
    bids: Sequence[Level],
    shares: float,
    *,
    reference_price: float | None = None,
    slippage_tolerance: float | None = None,
) -> VwapResult:
    """VWAP for a sell against a bid ladder."""
    return _vwap(bids, shares, "sell", reference_price, slippage_tolerance)


def _vwap(
    levels: Sequence[Level],
    shares: float,
    side: str,
    reference_price: float | None,
    slippage_tolerance: float | None,
) -> VwapResult:
    if shares <= 0:
        raise VwapError("shares must be > 0")
    ladder = list(levels)
    best = best_price(ladder)
    filled, notional, vwap, complete = walk_ladder(ladder, shares)

    if reference_price is None:
        reference_price = best
    if vwap > 0 and reference_price:
        if side == "buy":
            slippage_abs = vwap - reference_price  # paying more is worse
        else:
            slippage_abs = reference_price - vwap  # receiving less is worse
        slippage_bps = slippage_abs / reference_price * 10_000.0
    else:
        slippage_abs = 0.0
        slippage_bps = 0.0

    consumed = 0
    running = 0.0
    for _, size in ladder:
        if running >= shares - 1e-12:
            break
        running += size
        consumed += 1

    return VwapResult(
        side=side,
        requested_shares=float(shares),
        filled_shares=filled,
        notional_usdc=notional,
        vwap=vwap,
        best_price=best,
        reference_price=reference_price,
        slippage_abs=slippage_abs,
        slippage_bps=slippage_bps,
        fully_filled=complete,
        levels_consumed=consumed,
        available_shares=book_depth(ladder),
        slippage_tolerance=slippage_tolerance,
    )


def shares_for_notional_on_ladder(
    levels: Sequence[Level], notional_usdc: float
) -> float:
    """How many shares a fixed USDC budget buys walking the ladder."""
    if notional_usdc < 0:
        raise VwapError("notional must be >= 0")
    remaining = float(notional_usdc)
    shares = 0.0
    for price, size in levels:
        if remaining <= 0:
            break
        level_notional = price * size
        if level_notional <= remaining:
            shares += size
            remaining -= level_notional
        else:
            shares += remaining / price
            remaining = 0.0
    return shares


def notional_for_shares_on_ladder(levels: Sequence[Level], shares: float) -> float:
    """USDC required to buy ``shares`` walking the ladder."""
    _, notional, _, _ = walk_ladder(levels, shares)
    return notional


def participation_size(
    levels: Sequence[Level], participation_rate: float, *, cap: float | None = None
) -> float:
    """Largest order that stays within ``participation_rate`` of visible depth."""
    if not 0.0 < participation_rate <= 1.0:
        raise VwapError("participation_rate must be in (0, 1]")
    size = book_depth(levels) * participation_rate
    if cap is not None:
        size = min(size, float(cap))
    return size


def split_notional(
    total_notional: float,
    levels: Sequence[Level],
    *,
    participation_rate: float = 0.25,
    min_children: int = 2,
    max_children: int = 6,
) -> list[float]:
    """Break a notional into child clips sized to visible depth.

    Returns a list of child notional amounts summing to ``total_notional``
    (within float tolerance). When visible depth is thin, the child count rises
    toward ``max_children``; when depth is deep, it falls to ``min_children``.
    """
    if total_notional <= 0:
        raise VwapError("total_notional must be > 0")
    if min_children < 1:
        raise VwapError("min_children must be >= 1")
    if max_children < min_children:
        raise VwapError("max_children must be >= min_children")

    depth = book_depth(levels)
    best = best_price(levels)
    if not depth or not best:
        return [total_notional]

    clip_shares = participation_size(levels, participation_rate)
    clip_notional = clip_shares * best
    if clip_notional <= 0:
        return [total_notional]

    children = int(total_notional // clip_notional) + (1 if total_notional % clip_notional else 0)
    children = max(min_children, min(max_children, children))
    if children <= 1:
        return [total_notional]

    base = total_notional / children
    clips = [base] * children
    # push rounding drift into the last clip so the sum is exact
    clips[-1] = total_notional - base * (children - 1)
    return clips


def depth_weighted_midpoint(
    bids: Sequence[Level], asks: Sequence[Level], depth: int = 5
) -> float | None:
    """Micro-price: size-weighted mid over the top ``depth`` levels.

    Better entry reference than the plain midpoint when the book is lopsided -
    which is exactly the case this system hunts for.
    """
    top_bids = list(bids)[:depth]
    top_asks = list(asks)[:depth]
    bid_size = book_depth(top_bids)
    ask_size = book_depth(top_asks)
    bid_px = best_price(top_bids)
    ask_px = best_price(top_asks)
    if bid_px is None and ask_px is None:
        return None
    if bid_px is None:
        return ask_px
    if ask_px is None:
        return bid_px
    total = bid_size + ask_size
    if total <= 0:
        return (bid_px + ask_px) / 2.0
    return (bid_px * ask_size + ask_px * bid_size) / total

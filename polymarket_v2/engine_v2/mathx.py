"""Fees and ladder maths - self-contained, no dependencies on the research code.

The taker fee is Polymarket's published curve::

    fee_usdc = shares * coefficient * p * (1 - p)      coefficient = 0.07

It peaks at ``p = 0.5`` and vanishes at both extremes, which is why a 50c
contract is the most expensive thing on the book and why the break-even hit rate
for a directional trade is (slightly above) the entry price.

The ladder helpers walk real depth to produce a VWAP, so a fill can be priced
against the book rather than against the touch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_COEFFICIENT = 0.07

Level = tuple[float, float]


class MathError(ValueError):
    """Raised for impossible inputs (bad price, negative size)."""


def _price(value: float) -> float:
    p = float(value)
    if not 0.0 < p < 1.0:
        raise MathError(f"price must be strictly inside (0, 1), got {value!r}")
    return p


def taker_fee(shares: float, price: float, *, coefficient: float = DEFAULT_COEFFICIENT) -> float:
    """Fee in USDC for an aggressive fill."""
    if shares < 0:
        raise MathError(f"shares must be >= 0, got {shares!r}")
    if coefficient < 0:
        raise MathError("coefficient must be >= 0")
    p = _price(price)
    return float(shares) * coefficient * p * (1.0 - p)


def fee_per_share(price: float, *, coefficient: float = DEFAULT_COEFFICIENT) -> float:
    """Fee expressed per share - the cost floor a move must clear."""
    return coefficient * _price(price) * (1.0 - _price(price))


def break_even_move(price: float, *, coefficient: float = DEFAULT_COEFFICIENT) -> float:
    """Probability move needed to cover the taker fee alone."""
    return fee_per_share(price, coefficient=coefficient)


def shares_for_notional(notional: float, price: float) -> float:
    if notional < 0:
        raise MathError("notional must be >= 0")
    return float(notional) / _price(price)


def sort_ladder(levels: Iterable[Sequence[float]], *, descending: bool) -> list[Level]:
    """Normalise to [(price, size), ...] with the best price first."""
    out: list[Level] = []
    for level in levels:
        if len(level) < 2:
            raise MathError(f"level needs (price, size): {level!r}")
        price, size = float(level[0]), float(level[1])
        if not 0.0 < price < 1.0:
            raise MathError(f"level price out of range: {price!r}")
        if size < 0:
            raise MathError(f"level size must be >= 0: {size!r}")
        out.append((price, size))
    out.sort(key=lambda item: item[0], reverse=descending)
    return out


def walk_ladder(levels: Sequence[Level], shares: float) -> tuple[float, float, bool]:
    """Consume ``shares`` through ``levels``.

    Returns ``(filled_shares, notional_usdc, fully_filled)``.
    """
    if shares < 0:
        raise MathError("shares must be >= 0")
    remaining = float(shares)
    filled = 0.0
    notional = 0.0
    for price, size in levels:
        if remaining <= 0:
            break
        take = min(remaining, size)
        filled += take
        notional += take * price
        remaining -= take
    return filled, notional, remaining <= 1e-12


def vwap_for_shares(levels: Sequence[Level], shares: float) -> float | None:
    filled, notional, _ = walk_ladder(levels, shares)
    return (notional / filled) if filled > 0 else None


def quoted_spread(best_bid: float | None, best_ask: float | None) -> float | None:
    if best_bid is None or best_ask is None:
        return None
    return float(best_ask) - float(best_bid)

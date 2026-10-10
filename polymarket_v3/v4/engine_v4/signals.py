"""Fee and break-even arithmetic for the shipped V4 strategy.

This module is deliberately tiny and has **no dependencies**, not even on the rest of
this package. It replaces the older ``signals`` module from the research tree, which
carried the V3 signal machinery (``Standardiser``, ``score``, ``modelled_entry_price``)
and pulled in a CSV data reader for classes the shipped strategy never touches. Those
parts were removed rather than copied, because shipping dead code into a live trading
package is how a live package acquires bugs it never exercises.

Fee model, confirmed against the venue documentation
----------------------------------------------------
::

    fee = C x feeRate x p x (1 - p)

with ``feeRate = 0.07`` for the Crypto category, charged to takers only. The fee is
symmetric in price (a 0.30 trade and a 0.70 trade cost the same dollars) and peaks at
p = 0.50. Settlement is not a trade, so no exit fee is charged.
"""
from __future__ import annotations

#: Crypto-category taker rate, as documented by the venue
DEFAULT_FEE_COEFFICIENT = 0.07


def taker_fee_per_share(price: float,
                        coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Taker fee per share: ``coefficient * p * (1 - p)``."""
    return coefficient * price * (1.0 - price)


def break_even_accuracy(entry_price: float,
                        coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Accuracy needed to break even at a given entry price.

    For a hold-to-settlement directional binary, ``accuracy * 1 = entry + fee``.
    Settlement is not a trade, so no exit fee is charged.
    """
    return min(1.0, entry_price + taker_fee_per_share(entry_price, coefficient))


def max_viable_entry(accuracy: float,
                     coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Highest entry price that still breaks even at this accuracy.

    Solves ``accuracy = p + c*p*(1-p)`` for p by bisection.
    """
    if accuracy >= 1.0:
        return 1.0
    low, high = 0.0, 0.9999
    for _ in range(80):
        mid = (low + high) / 2.0
        if break_even_accuracy(mid, coefficient) < accuracy:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def edge_points(accuracy: float, entry_price: float,
                coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Per-share edge of a directional bet, after fees, in probability points."""
    return accuracy - break_even_accuracy(entry_price, coefficient)

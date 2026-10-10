"""Token book imbalance - the feature carrying the V4 edge.

    from engine_v4.imbalance import token_imbalance, book_imbalance, orient

WHAT IT IS
----------
``full_depth_bid_notional`` and ``full_depth_ask_notional`` are ready-made scalar
columns in the dataset: the total notional resting on each side of the token book.
Their normalised difference is the feature::

    imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)

``+1`` means every dollar of resting liquidity is a bid; ``-1`` means every dollar is
an ask. Measured on 44 days / 12,670 markets, its relationship to winning is U-shaped::

    imbalance decile   1     2     3     4     5     6     7     8
    win rate (%)     93.6  93.1  79.5  67.9  70.3  75.1  93.3  94.4

Extremes are strong, the middle is weak - the mid buckets LOSE money badly, because
the entry is around 0.79 while the win rate is only about 70%. Hence the gate
``|imbalance| >= 0.30``, which removes the balanced-book trades.

ORIENTATION, WHICH IS EASY TO GET WRONG
---------------------------------------
The raw feature is in UP-token space. When the strategy buys DOWN the sign must be
FLIPPED, because a book that is bid-heavy on the UP token is ask-heavy on the DOWN
token. ``orient`` does that and is the only sanctioned way to do it. A gate on
``|imbalance|`` looks symmetric and would silently accept a sign error, so the
convention is asserted in ``tools_v4/review.py``.
"""
from __future__ import annotations

import pyarrow.parquet as pq

#: columns needed from the feature files
_NEEDED = ("market_id", "time_to_deadline_ms",
           "full_depth_bid_notional", "full_depth_ask_notional")


def book_imbalance(bid_notional: float | None,
                   ask_notional: float | None) -> float | None:
    """(bid - ask) / (bid + ask); None when the book is empty or unusable."""
    if bid_notional is None or ask_notional is None:
        return None
    total = bid_notional + ask_notional
    if total <= 0:
        return None
    return (bid_notional - ask_notional) / total


def orient(imbalance: float | None, side: str) -> float | None:
    """Flip the sign so it is always measured in favour of the side being bought."""
    if imbalance is None:
        return None
    return imbalance if side == "UP" else -imbalance


def token_imbalance(root: str, checkpoint_s: float = 120.0,
                    tolerance_ms: float = 1500.0) -> dict[str, float]:
    """Map ``market_id`` -> imbalance at the decision tick.

    Indexed by ``market_id`` because ``window_key`` is corrupted in this dataset (it
    holds the literal string ``"et"``), so joining on the obvious-looking key yields
    zero matches. The decision tick is the one nearest ``checkpoint_s`` seconds before
    the close, matching ``Market.tick_at_t_minus`` in the loader.

    Reads only four scalar columns, so it is far cheaper than a full market load.
    """
    from engine_v4 import hf

    out: dict[str, float] = {}
    target = checkpoint_s * 1000.0
    for path in hf.day_files(root, hf.FEATURES):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in _NEEDED):
            continue
        data = pq.read_table(path, columns=list(_NEEDED)).to_pydict()
        ids = data.get("market_id") or []
        if not ids:
            continue
        market_id = str(ids[0])
        deadline = data["time_to_deadline_ms"]
        bids = data["full_depth_bid_notional"]
        asks = data["full_depth_ask_notional"]
        best_i = None
        best_gap = None
        for i, value in enumerate(deadline):
            if value is None:
                continue
            gap = abs(value - target)
            if gap <= tolerance_ms and (best_gap is None or gap < best_gap):
                best_i, best_gap = i, gap
        if best_i is None:
            continue
        value = book_imbalance(bids[best_i], asks[best_i])
        if value is not None:
            out[market_id] = value
    return out

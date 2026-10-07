"""V3 signals, and the break-even maths that decides whether a signal is tradable.

Two signal families, both strictly backward-looking:

**A. Intra-window momentum (`intra`)**
The v1/v2 signal. At checkpoint ``T-k`` the spot has already moved some distance
from the window open. Reconstruct the price at entry from the realised
open->close path (exact at the endpoints, linearly interpolated inside), then
measure the move over ``lookback`` seconds and trade its direction.

**B. Pre-window momentum (`pre`)**
A composite of the pre-entry features the dataset provides, standardised per
walk-forward fold (never with statistics from the future).

**C. Blend (`blend`)** - weighted sum of the two standardised scores.

The break-even identity
-----------------------
For a directional binary held to settlement::

    E[P&L per share] = accuracy * 1.0 - entry_price - fee_per_share
    break_even_accuracy = entry_price + fee_per_share

So accuracy alone says nothing about profitability. Every signal is therefore
reported as an **accuracy frontier**: the highest entry price at which it still
breaks even. That is the model-free part of the analysis - it needs no pricing
assumption at all.

Entry price when it is not measured
-----------------------------------
This dataset has no token prices, so entry price is modelled with a digital
option priced off trailing volatility::

    entry ≈ Phi( moneyness / (vol_per_sqrt_s * sqrt(t_remaining)) )

It is an *assumption*, flagged as such everywhere it is used, and it is
deliberately conservative: the model ignores the spread, so realised entries are
worse than modelled ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .data import Window

# --------------------------------------------------------------- normal CDF
_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ------------------------------------------------------------------ signals
@dataclass(frozen=True)
class SignalDef:
    """A named, fully specified signal."""

    name: str
    family: str                     # "intra" | "pre" | "blend"
    lookback_s: float = 180.0       # intra: seconds of move measured
    vol_scale: bool = True          # intra: divide the move by volatility
    weights: tuple[float, float] = (1.0, 0.0)   # blend: (intra, pre)


#: the pre-window composite: momentum, trend and stretch, equal-weighted after
#: standardisation. Kept deliberately small - a 23-feature fitted model on 71k
#: windows would find plenty of noise to memorise.
PRE_FEATURES_MOMENTUM = ("btc_ret_15m", "btc_ret_1h", "btc_ret_4h",
                         "ind_roc_5", "ind_roc_15", "ind_roc_60",
                         "ind_macd_hist", "ind_ema_9_dist", "ind_ema_26_dist")


class Standardiser:
    """Per-fold standardisation, and the train/live contract.

    Fit on the training slice only, then applied to both. Fitting on the whole
    dataset and evaluating on part of it leaks future distribution information
    into the features - a subtle and very common mistake.

    The same fitted object must be used live. If the live path standardises with
    different statistics (or with none), the threshold means something different
    and the strategy being traded is not the strategy that was measured. That is
    why :meth:`save`/:meth:`load` exist.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.mean = {n: 0.0 for n in self.names}
        self.std = {n: 1.0 for n in self.names}

    def fit(self, windows: Sequence[Window]) -> "Standardiser":
        for name in self.names:
            values = [w.features.get(name, 0.0) for w in windows]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            self.mean[name] = mean
            self.std[name] = math.sqrt(var) or 1.0
        return self

    def as_dict(self) -> dict:
        return {"names": list(self.names), "mean": dict(self.mean), "std": dict(self.std)}

    def save(self, path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "Standardiser":
        import json
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        obj = cls(raw["names"])
        obj.mean = {k: float(v) for k, v in raw["mean"].items()}
        obj.std = {k: float(v) for k, v in raw["std"].items()}
        return obj

    def z(self, window: Window, name: str) -> float:
        return (window.features.get(name, 0.0) - self.mean[name]) / self.std[name]

    def z_value(self, name: str, value: float) -> float:
        """Standardise a raw value, for callers that hold a feature dict."""
        return (value - self.mean.get(name, 0.0)) / self.std.get(name, 1.0)

    def pre_score(self, window: Window,
                  names: Sequence[str] = PRE_FEATURES_MOMENTUM) -> float:
        return sum(self.z(window, n) for n in names) / len(names)


# ------------------------------------------------------------ price at entry
def price_at(window: Window, ts: float) -> float:
    """BTC price at ``ts``, reconstructed from the realised open->close path.

    Exact at the endpoints; linear in between. This is the only intra-window
    price information this dataset carries, so the intra signal is only as good
    as that interpolation - stated here rather than buried.
    """
    if ts <= window.start_ts:
        return window.open_px
    if ts >= window.end_ts:
        return window.close_px
    frac = (ts - window.start_ts) / (window.end_ts - window.start_ts)
    return window.open_px + frac * (window.close_px - window.open_px)


def intra_move(window: Window, entry_ts: float, lookback_s: float) -> float:
    """Fractional BTC move over ``lookback_s`` ending at ``entry_ts``."""
    past = price_at(window, entry_ts - lookback_s)
    now = price_at(window, entry_ts)
    if not past:
        return 0.0
    return now / past - 1.0


def score(window: Window, entry_ts: float, sig: SignalDef,
          standardiser: Standardiser) -> float:
    """The raw signal value (sign = direction, magnitude = conviction)."""
    intra = 0.0
    if sig.family in ("intra", "blend"):
        intra = intra_move(window, entry_ts, sig.lookback_s)
        if sig.vol_scale and window.features.get("btc_vol_1h"):
            intra /= window.features["btc_vol_1h"]
    pre = 0.0
    if sig.family in ("pre", "blend"):
        pre = standardiser.pre_score(window)
    w_intra, w_pre = sig.weights
    return w_intra * intra + w_pre * pre


# ------------------------------------------------------- modelled entry price
def modelled_entry_price(window: Window, entry_ts: float, *,
                         threshold: float, side: str,
                         vol_floor: float = 1e-4) -> float:
    """Digital-option entry price for the chosen side. **Modelled, not measured.**

    A near-the-money binary is worth ~0.5; as the window's remaining time shrinks
    and the spot drifts, it converges to 0 or 1. Moneyness is the BTC move since
    the window open; ``btc_vol_1h`` (a 1-hour realised volatility) is scaled to
    the remaining time.
    """
    remaining = max(1.0, window.end_ts - entry_ts)
    vol_per_sqrt_s = max(vol_floor, window.features.get("btc_vol_1h", 0.0)) / math.sqrt(3600.0)
    sigma = vol_per_sqrt_s * math.sqrt(remaining)
    if sigma <= 0:
        return 0.5
    moneyness = (price_at(window, entry_ts) / window.open_px - 1.0)
    p_up = norm_cdf(moneyness / sigma)
    price = p_up if side == "UP" else 1.0 - p_up
    # never quote a binary outside a sane band
    return min(0.995, max(0.005, price))


# ------------------------------------------------------------- break-even
def taker_fee_per_share(price: float, coefficient: float = 0.07) -> float:
    """Polymarket taker fee per share: coefficient * p * (1 - p)."""
    return coefficient * price * (1.0 - price)


def break_even_accuracy(entry_price: float, coefficient: float = 0.07) -> float:
    """Accuracy needed to break even at a given entry price.

    ``accuracy * 1 = entry + fee`` for a hold-to-settlement directional binary.
    Settlement is not a trade, so no exit fee is charged.
    """
    return min(1.0, entry_price + taker_fee_per_share(entry_price, coefficient))


def max_viable_entry(accuracy: float, coefficient: float = 0.07) -> float:
    """Highest entry price that still breaks even at this accuracy.

    Solves ``accuracy = p + c*p*(1-p)`` for p by bisection.
    """
    if accuracy >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if break_even_accuracy(mid, coefficient) < accuracy:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def edge_points(accuracy: float, entry_price: float, coefficient: float = 0.07) -> float:
    """Accuracy surplus in percentage points over the break-even requirement."""
    return (accuracy - break_even_accuracy(entry_price, coefficient)) * 100.0

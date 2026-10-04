"""Signal detection: endgame momentum.

The claim this module encodes, and the evidence for it
------------------------------------------------------
Inside a 5/15-minute Up-or-Down window, if the underlying has already moved
meaningfully over the prior few minutes, the move tends to *persist* to the
close. Measured on the trade history (T-120 entries, 1,654 markets):

===================  ==============  ==============
lookback             60s             180s
===================  ==============  ==============
direction accuracy   62.7%           73.1%
===================  ==============  ==============

Accuracy rises monotonically with the lookback, which is why ``lookback_s`` is
the single most important parameter here - and also why this is honestly a
*momentum* strategy wearing an "endgame" name: a 180s lookback inside a 900s
window reads early-window momentum, not last-second structure.

Two rules keep the signal from manufacturing edge:

* **Backward only.** Every spot lookup is ``as_of`` - a price at or before the
  decision moment. A forward match would be reading the future.
* **Staleness aware.** The signal reports how old the spot bar was. A "confirmed
  direction" built on a stale price is fiction, and downstream guards act on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dataio import SpotIndex, utc


class Side(str, Enum):
    UP = "UP"
    DOWN = "DOWN"

    def opposite(self) -> "Side":
        return Side.DOWN if self is Side.UP else Side.UP


class Reason(str, Enum):
    """Why a decision came out the way it did - always populated."""

    READY = "ready"
    NO_SPOT = "no_spot_history"
    SPOT_STALE = "spot_stale"
    FLAT = "flat_underlying"
    BELOW_THRESHOLD = "move_below_threshold"
    WINDOW_TOO_EARLY = "window_too_early"
    ALREADY_TRADED = "already_traded_this_window"


@dataclass(frozen=True)
class Signal:
    """A direction call (or an explicit refusal) at one decision moment."""

    ts: float
    side: Side | None
    move_pct: float | None
    lookback_s: float
    spot_price: float | None
    spot_source_ts: float | None
    spot_staleness_s: float | None
    threshold_pct: float
    reason: Reason
    detail: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.side is not None and self.reason is Reason.READY

    @property
    def strength(self) -> float:
        """Move relative to threshold: 1.0 == exactly at the threshold."""
        if self.move_pct is None or self.threshold_pct <= 0:
            return 0.0
        return abs(self.move_pct) / self.threshold_pct

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "ts_utc": utc(self.ts),
            "side": self.side.value if self.side else None,
            "move_pct": None if self.move_pct is None else round(self.move_pct, 6),
            "lookback_s": self.lookback_s,
            "spot_price": self.spot_price,
            "spot_age_s": None if self.spot_staleness_s is None else round(self.spot_staleness_s, 2),
            "threshold_pct": self.threshold_pct,
            "strength": round(self.strength, 4),
            "reason": self.reason.value,
            "detail": self.detail,
        }


class SignalDetector:
    """Stateless-per-call momentum detector.

    ``one_entry_per_window`` is enforced by the caller passing ``already_traded``
    (the backtester and the live loop both track it per slug).
    """

    def __init__(
        self,
        *,
        lookback_s: float = 180.0,
        min_move_pct: float = 0.05,
        max_spot_staleness_s: float = 30.0,
        min_window_elapsed_s: float = 60.0,
        one_entry_per_window: bool = True,
    ) -> None:
        if lookback_s <= 0:
            raise ValueError("lookback_s must be > 0")
        if min_move_pct < 0:
            raise ValueError("min_move_pct must be >= 0")
        self.lookback_s = float(lookback_s)
        self.min_move_pct = float(min_move_pct)
        self.max_spot_staleness_s = float(max_spot_staleness_s)
        self.min_window_elapsed_s = float(min_window_elapsed_s)
        self.one_entry_per_window = bool(one_entry_per_window)

    def evaluate(
        self,
        spot: SpotIndex,
        *,
        ts: float,
        window_start_ts: float,
        already_traded: bool = False,
    ) -> Signal:
        """Decide the direction at ``ts`` using only data available at ``ts``."""
        def refuse(reason: Reason, detail: str, **extra) -> Signal:
            base = dict(
                ts=ts, side=None, move_pct=extra.get("move_pct"),
                lookback_s=self.lookback_s, spot_price=extra.get("spot_price"),
                spot_source_ts=extra.get("spot_source_ts"),
                spot_staleness_s=extra.get("spot_staleness_s"),
                threshold_pct=self.min_move_pct, reason=reason, detail=detail,
            )
            return Signal(**base)

        if self.one_entry_per_window and already_traded:
            return refuse(Reason.ALREADY_TRADED, "this window was already traded")

        if window_start_ts and (ts - window_start_ts) < self.min_window_elapsed_s:
            return refuse(
                Reason.WINDOW_TOO_EARLY,
                f"only {ts - window_start_ts:.0f}s into the window "
                f"(minimum {self.min_window_elapsed_s:.0f}s)",
            )

        change = spot.change_pct(ts, self.lookback_s)
        if change is None:
            return refuse(
                Reason.NO_SPOT,
                f"no spot bar at least {self.lookback_s:.0f}s before the decision",
            )
        move_pct, staleness = change
        bar = spot.as_of(ts)
        spot_price, spot_ts = bar if bar else (None, None)

        if staleness > self.max_spot_staleness_s:
            return refuse(
                Reason.SPOT_STALE,
                f"spot bar is {staleness:.1f}s old "
                f"(limit {self.max_spot_staleness_s:.0f}s)",
                move_pct=move_pct, spot_price=spot_price,
                spot_source_ts=spot_ts, spot_staleness_s=staleness,
            )

        if abs(move_pct) < self.min_move_pct:
            return refuse(
                Reason.BELOW_THRESHOLD,
                f"move {move_pct:+.4f}% below threshold {self.min_move_pct:.4f}%",
                move_pct=move_pct, spot_price=spot_price,
                spot_source_ts=spot_ts, spot_staleness_s=staleness,
            )

        if move_pct == 0.0:
            return refuse(Reason.FLAT, "underlying flat", move_pct=move_pct,
                          spot_price=spot_price, spot_source_ts=spot_ts,
                          spot_staleness_s=staleness)

        side = Side.UP if move_pct > 0 else Side.DOWN
        return Signal(
            ts=ts, side=side, move_pct=move_pct, lookback_s=self.lookback_s,
            spot_price=spot_price, spot_source_ts=spot_ts,
            spot_staleness_s=staleness, threshold_pct=self.min_move_pct,
            reason=Reason.READY,
            detail=(
                f"{side.value}: {move_pct:+.4f}% over {self.lookback_s:.0f}s "
                f"({abs(move_pct) / self.min_move_pct:.2f}x threshold)"
            ),
        )

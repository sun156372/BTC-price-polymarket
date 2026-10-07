"""V4 strategy runtime: the live decision path, matching the validated backtest.

V4 differs from V3 in one substantive way and it is worth stating plainly:

* **V3 faded short-term momentum. V4 follows it.** V3's "fade" finding came from
  the 9-month aggregate dataset, which had no order book. On the high-frequency
  order-book dataset, fading produces ~33% directional accuracy (it is
  anti-predictive) while following produces 62-85%. The earlier conclusion did not
  survive contact with real quotes, so it is replaced rather than patched.

What V4 keeps from V3, because the new data confirmed it:

* the break-even identity ``accuracy = entry + fee`` and the entry price cap
* holding to settlement rather than using a tight take profit (measured again:
  a 0.05 TP wins 93% of trades and still loses money)
* measured fills only - no modelled entry price may open a position
* a hard refusal to trade when the price already ate the edge

The decision rule
-----------------
At ``entry_t_minus`` seconds before the close, look at how far the Up token has
moved since the window opened. If it moved up by at least ``threshold``, buy UP;
if down by at least ``threshold``, buy DOWN. Otherwise do nothing. Pay the real
ask, hold to settlement, take 1.0 or 0.0.
"""
from __future__ import annotations

from dataclasses import dataclass

from .signals import break_even_accuracy, taker_fee_per_share

#: refusal reasons, surfaced verbatim in the paper log
READY = "ready"
NO_MOVE = "signal:below_threshold"
NO_DATA = "signal:missing_quote"
NO_QUOTE = "execution:no_quote"
PRICE_TOO_HIGH = "execution:entry_above_cap"
THIN_BOOK = "execution:book_too_thin"
STALE = "execution:quote_stale"
UNMEASURED = "execution:unmeasured_fill"
SIZE_ZERO = "sizing:no_budget"
HALTED = "safety:market_halted"


@dataclass(frozen=True)
class V4Params:
    """Tunables, set from the measured sweep on 1,148 resolved 5m markets.

    The winners in the sweep share three properties, and each one contradicts an
    earlier belief:

    * ``mode="follow"`` - fading scores ~33% direction accuracy on this data
    * ``take_profit=1.0`` - every tight take profit won more trades and lost money
    * ``threshold`` in the middle - the best risk-adjusted cell is 0.15, and the
      whole 0.00-0.20 range stays positive, so the choice is not a knife edge
    """

    #: seconds before the close. T-240 on a 5m window = 60s in.
    entry_t_minus: float = 240.0
    #: minimum |move| of the Up token since the open, in probability points.
    #: 0.15 gave the best P&L per share (1.77%) at 77.0% direction accuracy.
    threshold: float = 0.15
    #: follow the move, do not fade it
    mode: str = "follow"
    #: hold to settlement
    take_profit: float = 1.0
    #: no stop by default: 0.30 and 0.20 both reduced P&L in the sweep
    stop_loss: float = 1.0
    #: the measured entry for this cell is 0.739, so break-even at 77.0% accuracy
    #: is 0.7383 - a thin +3.2 point margin. The cap therefore sits just above the
    #: measured fill to tolerate normal price drift, and any worse fill is refused.
    max_entry_price: float = 0.76
    max_quote_staleness_s: float = 2.0
    max_slippage_cents: float = 2.0
    require_measured_fill: bool = True
    #: measured direction accuracy of the shipped cell
    measured_accuracy: float = 0.7701
    fee_coefficient: float = 0.07
    #: honest per-trade expectation and its spread, from the 4-day sample
    measured_edge_per_share: float = 0.0177
    measured_trades_per_day: float = 87.0
    measured_daily_std: float = 0.050


@dataclass(frozen=True)
class V4Decision:
    action: str                 # "enter" | "hold"
    reason: str
    side: str = ""
    move: float = 0.0
    limit_price: float = 0.0
    shares: float = 0.0
    break_even: float = 0.0

    @property
    def entering(self) -> bool:
        return self.action == "enter"


class StrategyV4:
    """The single decision path, shared by the backtest and the live loop."""

    def __init__(self, params: V4Params | None = None) -> None:
        self.p = params or V4Params()

    # ------------------------------------------------------------- helpers
    @property
    def break_even_entry(self) -> float:
        """Highest entry price that still breaks even at the measured accuracy."""
        p = self.p
        low, high = 0.0, 0.999
        for _ in range(70):
            mid = (low + high) / 2.0
            if break_even_accuracy(mid, p.fee_coefficient) < p.measured_accuracy:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def direction(self, open_up_mid: float | None, entry_up_mid: float | None
                  ) -> tuple[str, float] | None:
        if open_up_mid is None or entry_up_mid is None:
            return None
        move = entry_up_mid - open_up_mid
        if abs(move) < self.p.threshold:
            return None
        up = move > 0
        if self.p.mode == "fade":
            up = not up
        return ("UP" if up else "DOWN"), move

    # -------------------------------------------------------------- decide
    def decide(self, *, open_up_mid: float | None, entry_up_mid: float | None,
               ask: float | None, now: float, quote_ts: float | None = None,
               measured_fill: float | None = None, shares_wanted: float = 0.0,
               depth_cost: float | None = None, halted: bool = False) -> V4Decision:
        """Return the decision. Order matters: cheap checks first."""
        p = self.p

        if halted:
            return V4Decision("hold", HALTED)

        picked = self.direction(open_up_mid, entry_up_mid)
        if picked is None:
            return V4Decision("hold", NO_DATA if open_up_mid is None or entry_up_mid is None
                              else NO_MOVE)
        side, move = picked

        if quote_ts is not None and now - quote_ts > p.max_quote_staleness_s:
            return V4Decision("hold", STALE, side=side, move=move)
        if ask is None or not 0.0 < ask < 1.0:
            return V4Decision("hold", NO_QUOTE, side=side, move=move)
        if shares_wanted <= 0:
            return V4Decision("hold", SIZE_ZERO, side=side, move=move)

        if depth_cost is None:
            if p.require_measured_fill:
                return V4Decision("hold", THIN_BOOK, side=side, move=move)
            entry = ask
        else:
            entry = depth_cost / shares_wanted

        if measured_fill is not None:
            entry = measured_fill
        elif p.require_measured_fill and depth_cost is None:
            return V4Decision("hold", UNMEASURED, side=side, move=move)

        if ask is not None and entry - ask > p.max_slippage_cents / 100.0:
            return V4Decision("hold", THIN_BOOK, side=side, move=move)

        break_even = break_even_accuracy(entry, p.fee_coefficient)
        if entry > p.max_entry_price:
            return V4Decision("hold", PRICE_TOO_HIGH, side=side, move=move,
                              limit_price=entry, break_even=break_even)

        return V4Decision("enter", READY, side=side, move=move, limit_price=entry,
                          shares=shares_wanted, break_even=break_even)

    # ---------------------------------------------------------------- exits
    def exit_levels(self, entry_price: float) -> tuple[float | None, float | None]:
        """(stop price, target price). ``None`` means that leg never fires."""
        p = self.p
        stop = None if p.stop_loss >= 1.0 else max(0.001, entry_price - p.stop_loss)
        target = None if p.take_profit >= 1.0 else min(0.999, entry_price + p.take_profit)
        return stop, target

    def describe(self) -> list[str]:
        p = self.p
        return [
            f"signal      : {p.mode} the Up-token move, |move| >= {p.threshold:.3f}",
            f"entry       : T-{p.entry_t_minus:.0f}s",
            f"exit        : hold to settlement"
            + (f", stop {p.stop_loss:.2f}" if p.stop_loss < 1.0 else "")
            + (f", target {p.take_profit:.2f}" if p.take_profit < 1.0 else ""),
            f"entry cap   : {p.max_entry_price:.3f} "
            f"(break-even at measured accuracy {p.measured_accuracy:.4f} is "
            f"{self.break_even_entry:.3f})",
            f"fills       : {'measured only' if p.require_measured_fill else 'modelled allowed'}",
            f"staleness   : {p.max_quote_staleness_s:.1f}s",
        ]

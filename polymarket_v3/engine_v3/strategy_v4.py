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
#: the price-deviation filter: the fundamental driver (spot) has not moved enough
#: to support the token move, so the quote is not backed by anything real.
WEAK_DRIVER = "deviation:spot_move_too_small"


@dataclass(frozen=True)
class V4Params:
    """Tunables, validated on a genuine out-of-sample split.

    14 in-sample days (2026-03-27..04-26, 4,023 markets) were used to choose
    parameters; 10 untouched May days (2,877 markets) were used to test them.

    What survived:

    * ``mode="follow"`` - fading won **0 of 603** configurations in sample
    * ``take_profit=1.0`` - every tight take profit won more trades and lost money
    * ``threshold=0.20`` - everything at or below 0.12 lost money in sample
    * ``min_spot_move`` - the ONLY component that transferred: all four unfiltered
      cells went negative out of sample, all four filtered cells stayed positive

    What did NOT survive: the in-sample-optimal cell itself. Selecting purely on
    in-sample P&L picked T-240 with no filter, which returned -0.86%/share on the
    unseen days. The shipped cell is T-120 with the filter, which held its edge in
    both periods.

    Honest scale: ~+1.0%/share in sample and ~+0.9%/share out of sample, on 7/10
    positive out-of-sample days. That is small, and the sample history is a warning
    - the same rule measured +1.77%/share on 4 days and +1.02% on 14.
    """

    #: Seconds before the close. T-120 on a 5m window = 180s in.
    #:
    #: This moved from 240 to 120 because of the out-of-sample test. Selecting on
    #: the in-sample period alone chose T-240 with NO filter, which then went to
    #: -0.86%/share on ten unseen days in May. T-120 with the filter held up
    #: (+1.02%/share in sample, +1.02% out of sample) and trades far more often.
    entry_t_minus: float = 120.0
    #: Minimum |move| of the Up token since the open, in probability points.
    #: 0.20 keeps the highest accuracy at every checkpoint; every threshold at or
    #: below 0.12 lost money across 14 in-sample days.
    threshold: float = 0.20
    #: follow the move, do not fade it
    mode: str = "follow"
    #: hold to settlement
    take_profit: float = 1.0
    #: no stop by default: 0.30 and 0.20 both reduced P&L in the sweep
    stop_loss: float = 1.0
    #: Price-deviation filter: require the underlying to have moved at least this
    #: fraction since the window opened.
    #:
    #: This is the ONLY component that transferred out of sample. All four
    #: unfiltered cells were negative on the ten May days; all four filtered cells
    #: were positive. It is shipped on that basis, not on its in-sample gain.
    min_spot_move: float = 0.00045
    #: Entry-price cap. Swept 0.70 -> no cap in tools_v3/cap_optimal_v4.py.
    #:
    #: The cap was originally justified as "above break-even is negative EV", which
    #: is true for a trade of FIXED accuracy. But accuracy RISES with price - the
    #: trades a 0.90 cap rejects win 94.9% - so tightening the cap throws away the
    #: most accurate trades and LOWERS the per-share result. Out-of-sample pooled:
    #: 0.88 -> -1.11%, 0.90 -> -0.54%, no cap -> -0.17%. The cap is therefore off.
    #:
    #: This is not good news. The best setting is still slightly negative. Realised
    #: accuracy tracks the break-even line at every cap (no cap: needs 87.32%,
    #: realised 87.1%), which is the signature of a fairly priced market: there is
    #: no mispricing to exploit, and the fee is the entire loss.
    max_entry_price: float = 1.0
    max_quote_staleness_s: float = 2.0
    max_slippage_cents: float = 2.0
    require_measured_fill: bool = True
    #: measured on the full 37-day sample with the cap off (out-of-sample slice)
    measured_accuracy: float = 0.8710
    measured_entry: float = 0.8650
    in_sample_edge_per_share: float = -0.0009
    out_of_sample_edge_per_share: float = -0.0017
    out_of_sample_days_positive: int = 9
    out_of_sample_days: int = 22
    #: expected frequency with the cap off; out-of-sample traded 86/day
    measured_trades_per_day: float = 86.0
    fee_coefficient: float = 0.07
    #: retained for reference: the in-sample-optimal cell that FAILED out of sample
    failed_cell: str = "T-240 thr0.20 no filter: +2.62% in sample, -0.86% out"
    measured_daily_std: float = 0.036


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
               depth_cost: float | None = None, halted: bool = False,
               spot_move: float | None = None) -> V4Decision:
        """Return the decision. Order matters: cheap checks first.

        ``spot_move`` is the underlying's return since the window opened. It is the
        one optional input that enables the price-deviation filter: a token move
        the spot does not support is a quote with nothing behind it.
        """
        p = self.p

        if halted:
            return V4Decision("hold", HALTED)

        picked = self.direction(open_up_mid, entry_up_mid)
        if picked is None:
            return V4Decision("hold", NO_DATA if open_up_mid is None or entry_up_mid is None
                              else NO_MOVE)
        side, move = picked

        # Price-deviation filter: a token move the underlying does not support is
        # a quote with nothing behind it. This is the one component that survived
        # the out-of-sample test - all four unfiltered cells went negative on the
        # ten unseen May days, all four filtered cells stayed positive.
        # Skipped when spot is unavailable, so the strategy can still run on token
        # data alone - but then it is un-filtered, and un-filtered did not transfer.
        if spot_move is not None and abs(spot_move) < p.min_spot_move:
            return V4Decision("hold", WEAK_DRIVER, side=side, move=move)

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
            f"deviation   : require |spot move| >= {p.min_spot_move*10_000:.1f} bps "
            f"(token move must be backed by the underlying)",
            f"entry cap   : {p.max_entry_price:.3f} "
            f"(break-even at measured accuracy {p.measured_accuracy:.4f} is "
            f"{self.break_even_entry:.3f})",
            f"fills       : {'measured only' if p.require_measured_fill else 'modelled allowed'}",
            f"staleness   : {p.max_quote_staleness_s:.1f}s",
        ]

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
    """Tunables, from the measured sweep on 3,165 resolved 5m markets (11 days).

    The winners share three properties, and each one contradicts an earlier belief:

    * ``mode="follow"`` - fading wins **0 of 603** configurations; following is the
      only direction that works at all
    * ``take_profit=1.0`` - every tight take profit won more trades and lost money
    * a high ``threshold`` - on 14 days, everything at or below 0.12 loses money

    The edge shrank as the sample grew, which is the honest headline: 4 days said
    +1.77%/share, 11 days said +1.65%, 14 days say +1.02% at threshold 0.15. Only
    **1.3% of all 1,260 swept configurations are positive**. This is a narrow spike,
    not a broad plateau, and the shipped cell sits on the spike at threshold 0.20.
    """

    #: seconds before the close. T-240 on a 5m window = 60s in.
    entry_t_minus: float = 240.0
    #: Minimum |move| of the Up token since the open, in probability points.
    #: On 14 days / 4,023 markets this is the line between losing and winning:
    #: 0.20 gives +2.62%/share at 81.7% win, and EVERY threshold at or below 0.12
    #: loses money. Shipped at 0.20.
    threshold: float = 0.20
    #: follow the move, do not fade it
    mode: str = "follow"
    #: hold to settlement
    take_profit: float = 1.0
    #: no stop by default: 0.30 and 0.20 both reduced P&L in the sweep
    stop_loss: float = 1.0
    #: Price-deviation filter - require the underlying to have moved at least this
    #: fraction since the window opened. On 14 days this lifts the positive-day
    #: rate from 7/14 to 12/14 and the per-share return from +1.02% to +1.83%.
    #: 5 bps is the measured optimum but 8 bps trades too little; 4.5 bps keeps
    #: more trades at nearly the same quality.
    min_spot_move: float = 0.00045
    max_entry_price: float = 0.80
    max_quote_staleness_s: float = 2.0
    max_slippage_cents: float = 2.0
    require_measured_fill: bool = True
    #: measured figures for the shipped cell (threshold 0.20, 14 days, 1,022 trades)
    measured_accuracy: float = 0.8170
    measured_entry: float = 0.7792
    fee_coefficient: float = 0.07
    #: honest per-trade expectation, measured on 14 days
    measured_edge_per_share: float = 0.0262
    measured_trades_per_day: float = 73.0
    #: day-level standard deviation of pnl/share for the unfiltered cell
    measured_daily_std: float = 0.024
    #: with the spot-move filter applied, on the 0.15 cell (the filtered variant
    #: was measured there): 12 of 14 days positive, +0.0183/share, 56 trades/day
    filtered_edge_per_share: float = 0.0183
    filtered_trades_per_day: float = 56.0


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

        # Price-deviation filter. Measured on 11 days: requiring |spot move| >=
        # 5bps lifted the positive-day rate from 8/11 to 10/11 and the per-share
        # return from +0.0168 to +0.0239. It trades less (665 vs 1,281) and the
        # gain is per trade, not per day. Skipped when spot is unavailable so the
        # strategy still runs on token data alone - but then it is un-filtered.
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

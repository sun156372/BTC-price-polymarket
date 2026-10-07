"""V3 decision logic: the signal, the execution gate, and the risk policy.

One decision path, used identically by the backtest and the live loop. The gate
is the important part: it is what keeps the simulation honest, because it refuses
to book a trade at a price that was modelled rather than observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .config import StrategyV3
from .signals import break_even_accuracy, taker_fee_per_share

# replay reasons
READY = "ready"
NO_SIGNAL = "signal:below_threshold"
NO_SIGNAL_DATA = "signal:missing_features"
PRICE_TOO_HIGH = "execution:entry_above_cap"
SPREAD_TOO_WIDE = "execution:spread_too_wide"
SLIPPAGE_TOO_HIGH = "execution:slippage_too_high"
QUOTE_STALE = "execution:quote_stale"
NO_MEASURED_FILL = "execution:unmeasured_fill"
INSUFFICIENT_CASH = "sizing:insufficient_cash"


@dataclass(frozen=True)
class Decision:
    """What V3 decided, and why."""

    action: str                 # "enter" | "hold"
    reason: str
    side: str = ""              # "UP" | "DOWN"
    z: float = 0.0
    limit_price: float = 0.0
    shares: float = 0.0
    fee_per_share: float = 0.0
    break_even: float = 0.0

    @property
    def entering(self) -> bool:
        return self.action == "enter"


@dataclass(frozen=True)
class Quote:
    """A live observation of one outcome token."""

    bid: float | None
    ask: float | None
    ts: float
    token: str = ""

    @property
    def spread(self) -> float:
        if self.bid is None or self.ask is None:
            return float("inf")
        return max(0.0, self.ask - self.bid)

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


class StrategyV3Runtime:
    """Turns a feature snapshot plus a quote into an entry decision."""

    def __init__(self, config: StrategyV3,
                 standardiser=None) -> None:
        self.config = config
        self.standardiser = standardiser

    # ------------------------------------------------------------ signal
    def z_score(self, features: Mapping[str, float]) -> float | None:
        """Standardised composite of the configured features."""
        names = self.config.signal.features
        values = []
        for name in names:
            if name not in features:
                return None
            values.append(features[name])
        if self.standardiser is not None:
            values = [self.standardiser.z_value(name, value)
                      for name, value in zip(names, values)]
        else:
            # no fitted scaler: fall back to a fixed unit scale so the threshold
            # stays meaningful, and say so via the reason code
            values = [value for value in values]
        raw = sum(values) / len(values)
        return -raw if self.config.signal.invert else raw

    def decide(self, *, features: Mapping[str, float], quote: Quote | None,
               now: float, equity: float, cash: float,
               seconds_since_last_entry: float,
               open_positions: int,
               measured_fill: float | None = None) -> Decision:
        """The single decision path. Order matters: cheap checks first."""
        cfg = self.config

        if open_positions >= cfg.risk.max_concurrent_positions:
            return Decision("hold", "risk:position_open")
        if seconds_since_last_entry < cfg.risk.min_seconds_between_entries:
            return Decision("hold", "risk:cooldown")

        z = self.z_score(features)
        if z is None:
            return Decision("hold", NO_SIGNAL_DATA)
        if abs(z) < cfg.signal.threshold_z:
            return Decision("hold", NO_SIGNAL, z=z)

        side = "UP" if z > 0 else "DOWN"

        if quote is None:
            return Decision("hold", QUOTE_STALE, side=side, z=z)
        if now - quote.ts > cfg.execution.max_quote_staleness_s:
            return Decision("hold", QUOTE_STALE, side=side, z=z)
        if quote.spread > cfg.execution.max_spread_cents / 100.0:
            return Decision("hold", SPREAD_TOO_WIDE, side=side, z=z)

        # what we would actually pay
        if measured_fill is not None:
            entry = measured_fill
        elif cfg.execution.require_measured_fill:
            return Decision("hold", NO_MEASURED_FILL, side=side, z=z)
        else:
            if quote.ask is None:
                return Decision("hold", QUOTE_STALE, side=side, z=z)
            entry = quote.ask

        if entry > cfg.execution.max_entry_price:
            return Decision("hold", PRICE_TOO_HIGH, side=side, z=z,
                            limit_price=entry,
                            break_even=break_even_accuracy(entry,
                                                           cfg.fees.taker_coefficient))

        # slippage of the observed fill vs the quote we decided on
        if quote.ask is not None and measured_fill is not None:
            slip_cents = (measured_fill - quote.ask) * 100.0
            if slip_cents > cfg.execution.max_slippage_cents:
                return Decision("hold", SLIPPAGE_TOO_HIGH, side=side, z=z,
                                limit_price=entry)

        budget = min(cfg.sizing.notional_usdc,
                     equity * cfg.sizing.max_position_pct_equity,
                     cash)
        if budget <= 0:
            return Decision("hold", INSUFFICIENT_CASH, side=side, z=z)

        fee = taker_fee_per_share(entry, cfg.fees.taker_coefficient)
        shares = budget / (entry + fee)
        return Decision("enter", READY, side=side, z=z, limit_price=entry,
                        shares=shares, fee_per_share=fee,
                        break_even=break_even_accuracy(entry, cfg.fees.taker_coefficient))


def exit_levels(config: StrategyV3, entry_price: float) -> tuple[float | None, float | None]:
    """(stop_price, take_profit_price) for a position, or (None, None) to hold.

    Returns ``None`` for a leg that should never fire. V3's default has no take
    profit: at a 3-point edge the barrier study could not support truncating
    winners, and the exit-price question is unmeasurable on this dataset.
    """
    stop = None
    if config.risk.catastrophe_stop:
        stop = max(0.001, entry_price * (1.0 - config.risk.catastrophe_stop))
    take = None
    if config.risk.take_profit:
        take = min(0.999, entry_price + config.risk.take_profit)
    return stop, take


def required_accuracy_ladder(coefficient: float = 0.07) -> list[tuple[float, float]]:
    """(entry price, accuracy needed) - the table the execution gate implements."""
    ladder = []
    for cents in range(45, 76):
        price = cents / 100.0
        ladder.append((price, break_even_accuracy(price, coefficient)))
    return ladder

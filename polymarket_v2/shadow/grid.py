"""Parameter grid and the per-parameter virtual account.

Memory layout note
------------------
The 1200 accounts are held as one list of :class:`VirtualAccount` objects, and the
grid is a parallel list of :class:`ShadowParams`. The account loop walks a single
``for index, account in enumerate(accounts)`` - no threads, no coroutines, no
per-account queues. Accounts share the market data by reference and own only
their own cash/equity/positions/trades.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable

FIELDS = [
    "combo_id", "time", "threshold", "tp", "sl", "min_gap",
    "dynamic_tp_sl", "hedge_enabled", "hedge_ratio",
]


class GridError(ValueError):
    """Raised when the parameter file is missing columns or malformed."""


def _as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on", "t"}


@dataclass(frozen=True)
class ShadowParams:
    """One row of ``param_combinations.csv``, typed."""

    combo_id: str
    time_s: float            # entry checkpoint, seconds before close
    threshold_pct: float     # minimum spot move over the lookback, in percent
    tp: float                # take profit, probability points
    sl: float                # stop loss, probability points
    min_gap_s: float         # minimum seconds between entries
    dynamic_tp_sl: bool      # scale tp/sl by realised ATR
    hedge_enabled: bool      # answer a stop by hedging instead of closing
    hedge_ratio: float       # fraction of the position to hedge

    def as_dict(self) -> dict:
        return {
            "combo_id": self.combo_id,
            "time": self.time_s,
            "threshold": self.threshold_pct,
            "tp": self.tp,
            "sl": self.sl,
            "min_gap": self.min_gap_s,
            "dynamic_tp_sl": self.dynamic_tp_sl,
            "hedge_enabled": self.hedge_enabled,
            "hedge_ratio": self.hedge_ratio,
        }


def load_grid(path: str, *, limit: int | None = None) -> list[ShadowParams]:
    """Read the grid, tolerating the older schema (no hedge/dynamic columns)."""
    if not os.path.exists(path):
        raise GridError(
            f"parameter file not found: {path}\n"
            "generate one with:  python scripts/make_param_grid.py"
        )
    rows: list[ShadowParams] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = {h.strip().lower() for h in (reader.fieldnames or [])}
        required = {"time", "threshold", "tp", "sl"}
        missing = required - headers
        if missing:
            raise GridError(f"{path} is missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            try:
                rows.append(ShadowParams(
                    combo_id=str(row.get("combo_id") or f"p{index + 1:05d}"),
                    time_s=float(row["time"]),
                    threshold_pct=float(row["threshold"]),
                    tp=float(row["tp"]),
                    sl=float(row["sl"]),
                    min_gap_s=float(row.get("min_gap") or 0.0),
                    dynamic_tp_sl=_as_bool(row.get("dynamic_tp_sl", False)),
                    hedge_enabled=_as_bool(row.get("hedge_enabled", False)),
                    hedge_ratio=float(row.get("hedge_ratio") or 0.0),
                ))
            except (TypeError, ValueError) as exc:
                raise GridError(f"{path} row {index + 1} is malformed: {exc}") from exc
    if not rows:
        raise GridError(f"{path} contains no parameter rows")
    return rows


def entry_times(grid: Iterable[ShadowParams]) -> list[float]:
    """Distinct entry checkpoints - used to pre-compute spot moves once."""
    return sorted({row.time_s for row in grid})


# ------------------------------------------------------------------- accounts
@dataclass
class Leg:
    """One filled leg (the primary entry, or its hedge)."""

    token: str
    side: str                 # "UP" / "DOWN"
    shares: float
    price: float
    fee: float
    ts: float

    @property
    def cost(self) -> float:
        return self.shares * self.price + self.fee


@dataclass
class OpenPosition:
    """A position in one window, possibly hedged."""

    slug: str
    asset: str
    window_end_ts: float
    side: str
    token: str
    shares: float
    entry_price: float
    entry_fee: float
    entry_ts: float
    tp_price: float
    sl_price: float
    atr: float | None = None
    hedged: bool = False
    hedge_leg: Leg | None = None
    closed: bool = False

    @property
    def primary_leg(self) -> Leg:
        return Leg(self.token, self.side, self.shares, self.entry_price,
                   self.entry_fee, self.entry_ts)

    @property
    def cost(self) -> float:
        total = self.primary_leg.cost
        if self.hedge_leg is not None:
            total += self.hedge_leg.cost
        return total

    def payout(self, winner: str) -> float:
        """Settlement value of every leg, given the winning side."""
        value = self.shares if winner == self.side else 0.0
        if self.hedge_leg is not None:
            value += self.hedge_leg.shares if winner == self.hedge_leg.side else 0.0
        return value


@dataclass
class VirtualAccount:
    """One parameter set's independent book."""

    params: ShadowParams
    starting_cash: float
    cash: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    hedges: int = 0
    stops: int = 0
    take_profits: int = 0
    settlements: int = 0
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    last_entry_ts: float = float("-inf")
    realised_pnl: float = 0.0
    last_reason: str = ""
    #: windows this account has already decided on, so it can never re-decide
    #: the same window if a replay catches up on checkpoints
    decided: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.cash:
            self.cash = float(self.starting_cash)
            self.equity = float(self.starting_cash)
            self.peak_equity = float(self.starting_cash)

    # -- accounting ------------------------------------------------------
    def apply_pnl(self, pnl: float, gross: float, fees: float,
                  reason: str = "") -> None:
        self.cash += pnl
        self.realised_pnl += pnl
        self.gross_pnl += gross
        self.fees_paid += fees
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.last_reason = reason
        self.mark()

    def mark(self) -> None:
        self.equity = self.cash
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades else 0.0

    @property
    def return_pct(self) -> float:
        return ((self.equity / self.starting_cash) - 1.0) * 100.0 if self.starting_cash else 0.0

    def can_enter(self, notional: float) -> bool:
        return self.cash >= notional and not self.positions

    def as_row(self, *, now: float) -> dict:
        return {
            "param_id": self.params.combo_id,
            "equity": round(self.equity, 4),
            "return_pct": round(self.return_pct, 4),
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "hedges": self.hedges,
            "stops": self.stops,
            "take_profits": self.take_profits,
            "settlements": self.settlements,
            "fees_paid": round(self.fees_paid, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "open_positions": len(self.positions),
            "time": self.params.time_s,
            "threshold": self.params.threshold_pct,
            "tp": self.params.tp,
            "sl": self.params.sl,
            "min_gap": self.params.min_gap_s,
            "dynamic_tp_sl": self.params.dynamic_tp_sl,
            "hedge_enabled": self.params.hedge_enabled,
            "hedge_ratio": self.params.hedge_ratio,
            "updated_ts": round(now, 3),
        }


def build_accounts(grid: Iterable[ShadowParams], *, starting_cash: float = 1_000.0) -> list[VirtualAccount]:
    """One account per parameter set, all sharing the same starting cash."""
    return [VirtualAccount(params=row, starting_cash=starting_cash) for row in grid]

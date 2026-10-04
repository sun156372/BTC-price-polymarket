"""Broker boundary: what the strategy calls to actually trade.

The engine never touches a venue directly. It talks to a :class:`Broker`, and the
only broker that ships enabled is :class:`PaperBroker`. :class:`LiveBroker` exists
to make the boundary explicit and to fail loudly rather than pretend - you
subclass it and implement ``submit`` with your own signed-order client.

Why the boundary is this strict
-------------------------------
A strategy that can place an order by accident will eventually do it. Making the
live path an abstract method with no default means the *only* way to trade real
money is to write code that says so.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from .mathx import taker_fee


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OrderRequest:
    slug: str
    token_id: str
    side: OrderSide
    shares: float
    limit_price: float
    reason: str = ""
    reduce_only: bool = False


@dataclass(frozen=True)
class Fill:
    slug: str
    token_id: str
    side: OrderSide
    shares: float
    price: float
    fee_usdc: float
    status: OrderStatus
    order_id: str = ""
    note: str = ""

    @property
    def notional_usdc(self) -> float:
        return self.shares * self.price

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "slug": self.slug,
            "token_id": self.token_id,
            "side": self.side.value,
            "shares": round(self.shares, 6),
            "price": round(self.price, 6),
            "notional_usdc": round(self.notional_usdc, 4),
            "fee_usdc": round(self.fee_usdc, 8),
            "status": self.status.value,
            "note": self.note,
        }


@runtime_checkable
class Broker(Protocol):
    """Minimal surface the engine depends on."""

    def submit(self, request: OrderRequest, *, at_price: float) -> Fill:  # pragma: no cover
        ...


class BrokerError(RuntimeError):
    """Raised when a broker cannot service a request."""


class PaperBroker:
    """Simulated fills with explicit, conservative assumptions.

    * The fill price is supplied by the caller (``at_price``) - normally the last
      real print, so the backtest cannot silently invent liquidity.
    * ``adverse_bps`` worsens every fill, which is how the live-realistic run is
      produced. A backtest that fills at the exact touch is the standard way to
      overstate a strategy.
    * Fees use the venue curve, charged on entry and on any exit that is a real
      trade. Settlement is not a trade and therefore pays no fee.
    """

    def __init__(self, *, coefficient: float = 0.07, adverse_bps: float = 0.0) -> None:
        self.coefficient = float(coefficient)
        self.adverse_bps = float(adverse_bps)
        self._ids = itertools.count(1)
        self.orders: list[OrderRequest] = []
        self.fills: list[Fill] = []

    def submit(self, request: OrderRequest, *, at_price: float) -> Fill:
        if request.shares <= 0:
            raise BrokerError(f"shares must be > 0, got {request.shares!r}")
        if not 0.0 < at_price < 1.0:
            raise BrokerError(f"fill price must be in (0, 1), got {at_price!r}")

        slip = self.adverse_bps / 10_000.0
        price = at_price * (1.0 + slip) if request.side is OrderSide.BUY else at_price * (1.0 - slip)
        price = min(0.999, max(0.001, price))
        fee = taker_fee(request.shares, price, coefficient=self.coefficient)

        self.orders.append(request)
        fill = Fill(
            slug=request.slug, token_id=request.token_id, side=request.side,
            shares=request.shares, price=price, fee_usdc=fee,
            status=OrderStatus.FILLED, order_id=f"paper-{next(self._ids)}",
            note=request.reason,
        )
        self.fills.append(fill)
        return fill

    def reset(self) -> None:
        self.orders.clear()
        self.fills.clear()
        self._ids = itertools.count(1)


class LiveBroker:
    """Base class for a real venue client. Intentionally unusable as-is.

    Implement :meth:`submit` with your authenticated client, then construct your
    engine with it. Until you do, every call raises - there is no silent fallback
    to simulation, because a silent fallback in live mode is how a strategy
    "works" in testing and does nothing (or something wrong) in production.
    """

    def __init__(self, *, confirm_live: bool = False) -> None:
        if not confirm_live:
            raise BrokerError(
                "LiveBroker requires confirm_live=True - refusing to construct a "
                "live trading path implicitly"
            )

    def submit(self, request: OrderRequest, *, at_price: float) -> Fill:  # pragma: no cover
        raise NotImplementedError(
            "LiveBroker.submit is not implemented. Subclass LiveBroker and submit "
            "signed orders with your own client (see README 'Going live')."
        )


@dataclass
class Position:
    """Open exposure in one outcome token."""

    slug: str
    token_id: str
    side: str                 # "UP" / "DOWN"
    shares: float
    entry_price: float
    entry_fee: float
    entry_ts: float
    take_profit_price: float
    stop_loss_price: float
    fills: list[Fill] = field(default_factory=list)

    @property
    def cost_usdc(self) -> float:
        return self.shares * self.entry_price + self.entry_fee

    def unrealised(self, price: float) -> float:
        return self.shares * (price - self.entry_price) - self.entry_fee

    def as_dict(self) -> dict:
        return {
            "slug": self.slug,
            "token_id": self.token_id,
            "side": self.side,
            "shares": round(self.shares, 6),
            "entry_price": round(self.entry_price, 6),
            "entry_fee": round(self.entry_fee, 8),
            "take_profit_price": round(self.take_profit_price, 6),
            "stop_loss_price": round(self.stop_loss_price, 6),
            "cost_usdc": round(self.cost_usdc, 4),
        }

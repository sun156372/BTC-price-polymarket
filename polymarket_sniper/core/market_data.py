"""Module 1 - multi-source market data and depth recognition.

Design rule: **never trust a single price source.**

* :class:`OrderBook` keeps local state for one outcome token, with L2 delta
  application, midpoint/micro-price, liquidity scoring, order-book imbalance
  (OBI) and liquidity-wall detection.
* :class:`MarketDataHub` merges a streaming source (WebSocket deltas) with a
  periodic REST snapshot, cross-validates them, and can veto trading when the
  two disagree beyond tolerance or when data goes stale.

Offline by default
------------------
This module performs **no I/O of its own**. Live access is injected:

* ``BookStreamTransport`` - implement ``connect``/``subscribe``/``poll``;
  nothing here opens a socket.
* ``RestSnapshotTransport`` - implement ``fetch_json``; see
  :class:`RecordingRestTransport` for an offline implementation used in tests
  and backtests.

That boundary is deliberate: it keeps the analytics testable and guarantees the
backtest engine can never accidentally reach the network.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, Iterator, Protocol, Sequence

from utils.logger import get_logger
from utils.vwap_calculator import (
    Level,
    best_price,
    book_depth,
    depth_weighted_midpoint,
    levels_for_side,
)

log = get_logger("market_data")

UP = "up"
DOWN = "down"


class DataQualityError(RuntimeError):
    """Raised when the book is unusable (crossed, empty, stale, divergent)."""


# --------------------------------------------------------------------- levels
@dataclass(frozen=True)
class DepthMetrics:
    """Depth analytics for one side of a book."""

    side: str
    depth: int
    shares: float
    notional: float
    liquidity_score: float
    weighted_avg_price: float | None
    best_price: float | None
    levels: tuple[Level, ...] = ()

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "depth": self.depth,
            "shares": round(self.shares, 6),
            "notional": round(self.notional, 6),
            "liquidity_score": round(self.liquidity_score, 6),
            "weighted_avg_price": (
                None if self.weighted_avg_price is None else round(self.weighted_avg_price, 6)
            ),
            "best_price": self.best_price,
        }


@dataclass(frozen=True)
class Wall:
    """A liquidity wall: one level holding an outsized share of top-N depth."""

    side: str
    price: float
    size: float
    share_of_depth: float
    rank: int
    distance_from_best: float

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "price": round(self.price, 6),
            "size": round(self.size, 6),
            "share_of_depth": round(self.share_of_depth, 4),
            "rank": self.rank,
            "distance_from_best": round(self.distance_from_best, 6),
        }


# ------------------------------------------------------------------ analytics
def liquidity_score(
    levels: Sequence[Level],
    *,
    depth: int = 5,
    decay: float = 0.5,
    notional_scale: float = 100.0,
) -> float:
    """Top-N liquidity score in roughly [0, 1].

    Two intuitive properties, both deliberate:

    ``shape``
        Depth quality - how evenly the size is spread across the top ``depth``
        levels. A book with 5 decent levels scores higher than one where all
        size sits on a single level, even at equal total size.
    ``scale``
        Size quality - total notional saturating at ``notional_scale`` USDC, so
        the score stays bounded.

    ``decay`` weights level importance geometrically (0.5 => each level counts
    half as much as the one above it).
    """
    if depth <= 0:
        raise ValueError("depth must be > 0")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    if notional_scale <= 0:
        raise ValueError("notional_scale must be > 0")

    top = list(levels)[:depth]
    if not top:
        return 0.0

    weights = [decay**i for i in range(len(top))]
    sizes = [size for _, size in top]
    weight_sum = sum(weights)
    weighted_size = sum(w * s for w, s in zip(weights, sizes))

    # concentration on the very best level (HHI-style on weighted share)
    if weighted_size <= 0:
        return 0.0
    top_share = (weights[0] * sizes[0]) / weighted_size
    shape = max(0.0, 1.0 - top_share)

    notional = sum(price * size for price, size in top)
    scale = notional / (notional + notional_scale)

    score = 0.5 * shape + 0.5 * scale
    return float(min(1.0, max(0.0, score)))


def order_book_imbalance(
    bids: Sequence[Level], asks: Sequence[Level], *, depth: int = 5
) -> float:
    """OBI = (bid_size - ask_size) / (bid_size + ask_size) over top ``depth``.

    ``+1`` = bids only (buy pressure), ``-1`` = asks only (sell pressure),
    ``0`` = balanced.
    """
    bid_size = sum(size for _, size in list(bids)[:depth])
    ask_size = sum(size for _, size in list(asks)[:depth])
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return float((bid_size - ask_size) / total)


def detect_walls(
    levels: Sequence[Level],
    *,
    side: str,
    depth: int = 5,
    min_share: float = 0.30,
) -> list[Wall]:
    """Find levels holding at least ``min_share`` of top-``depth`` size."""
    top = list(levels)[:depth]
    if not top:
        return []
    total = book_depth(top)
    if total <= 0:
        return []
    best = best_price(top)
    walls: list[Wall] = []
    for rank, (price, size) in enumerate(top):
        share = size / total
        if share >= min_share:
            walls.append(
                Wall(
                    side=side,
                    price=float(price),
                    size=float(size),
                    share_of_depth=float(share),
                    rank=rank,
                    distance_from_best=(
                        0.0 if best is None else abs(float(price) - best)
                    ),
                )
            )
    walls.sort(key=lambda w: w.size, reverse=True)
    return walls


# ------------------------------------------------------------------- snapshot
@dataclass
class BookSnapshot:
    """One consistent view of a single outcome token's book."""

    token_id: str
    side: str
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    sequence: int | None = None
    window_close_epoch: float | None = None

    # -- basic derived values --------------------------------------------
    @property
    def best_bid(self) -> float | None:
        return best_price(self.bids)

    @property
    def best_ask(self) -> float | None:
        return best_price(self.asks)

    def midpoint(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def microprice(self, depth: int = 5) -> float | None:
        return depth_weighted_midpoint(self.bids, self.asks, depth=depth)

    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def spread_bps(self) -> float | None:
        spread = self.spread()
        mid = self.midpoint()
        if spread is None or not mid:
            return None
        return spread / mid * 10_000.0

    def is_crossed(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask

    def depth(self, side: str, top: int = 5) -> DepthMetrics:
        levels = self.bids if side == "bid" else self.asks
        top_levels = tuple(list(levels)[:top])
        return DepthMetrics(
            side=side,
            depth=top,
            shares=book_depth(top_levels),
            notional=sum(p * s for p, s in top_levels),
            liquidity_score=liquidity_score(top_levels, depth=top),
            weighted_avg_price=(
                (sum(p * s for p, s in top_levels) / book_depth(top_levels))
                if book_depth(top_levels) > 0
                else None
            ),
            best_price=best_price(top_levels),
            levels=top_levels,
        )

    def obi(self, top: int = 5) -> float:
        return order_book_imbalance(self.bids, self.asks, depth=top)

    def walls(self, depth: int = 5, min_share: float = 0.30) -> list[Wall]:
        """Outsized levels on both sides of this book."""
        return detect_walls(self.bids, side="bid", depth=depth, min_share=min_share) + detect_walls(
            self.asks, side="ask", depth=depth, min_share=min_share
        )

    def distance_to_close(self, now: float | None = None) -> float | None:
        if self.window_close_epoch is None:
            return None
        return self.window_close_epoch - (now if now is not None else self.timestamp)

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.timestamp

    def as_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "side": self.side,
            "timestamp": round(self.timestamp, 3),
            "source": self.source,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midpoint": (
                None if self.midpoint() is None else round(self.midpoint() or 0.0, 6)
            ),
            "microprice": (
                None if self.microprice() is None else round(self.microprice() or 0.0, 6)
            ),
            "spread": self.spread(),
            "spread_bps": None if self.spread_bps() is None else round(self.spread_bps() or 0.0, 2),
            "obi": round(self.obi(), 4),
            "bid_depth": self.depth("bid").as_dict(),
            "ask_depth": self.depth("ask").as_dict(),
            "walls": [w.as_dict() for w in self.walls()],
            "crossed": self.is_crossed(),
        }


# ----------------------------------------------------------------- order book
class OrderBook:
    """Mutable L2 book for one token: applies snapshots and deltas."""

    def __init__(self, token_id: str, side: str, *, max_levels: int = 50) -> None:
        self.token_id = token_id
        self.side = side
        self.max_levels = max_levels
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self.sequence: int | None = None
        self.timestamp: float = 0.0
        self.source: str = "unknown"
        self.window_close_epoch: float | None = None
        self.applied_updates = 0
        self.rejected_updates = 0

    # -- mutation ---------------------------------------------------------
    def apply_snapshot(self, payload: dict, *, source: str = "snapshot") -> None:
        """Replace the whole book. ``payload``: {"bids": [[p,s],...], "asks": ...}"""
        try:
            self._bids = {float(p): float(s) for p, s in (payload.get("bids") or [])}
            self._asks = {float(p): float(s) for p, s in (payload.get("asks") or [])}
        except (TypeError, ValueError) as exc:
            raise DataQualityError(f"malformed snapshot for {self.token_id}: {exc}") from exc
        self._prune()
        self.timestamp = float(payload.get("timestamp") or time.time())
        self.sequence = payload.get("sequence", self.sequence)
        self.window_close_epoch = payload.get("window_close_epoch", self.window_close_epoch)
        self.source = source
        self.applied_updates += 1

    def apply_delta(self, payload: dict, *, source: str = "websocket") -> bool:
        """Apply a change message. Returns False when it arrived out of order."""
        sequence = payload.get("sequence")
        if sequence is not None and self.sequence is not None and sequence <= self.sequence:
            self.rejected_updates += 1
            return False

        for change in payload.get("changes") or []:
            try:
                side = str(change.get("side", "")).lower()
                price = float(change["price"])
                size = float(change.get("size", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise DataQualityError(f"malformed delta for {self.token_id}: {change!r}") from exc
            book = self._bids if side.startswith("b") else self._asks
            if size <= 0:
                book.pop(price, None)
            else:
                book[price] = size

        self._prune()
        self.timestamp = float(payload.get("timestamp") or time.time())
        if sequence is not None:
            self.sequence = sequence
        self.source = source
        self.applied_updates += 1
        return True

    def _prune(self) -> None:
        if len(self._bids) > self.max_levels:
            keep = sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True)[: self.max_levels]
            self._bids = dict(keep)
        if len(self._asks) > self.max_levels:
            keep = sorted(self._asks.items(), key=lambda kv: kv[0])[: self.max_levels]
            self._asks = dict(keep)

    # -- read -------------------------------------------------------------
    @property
    def bids(self) -> tuple[Level, ...]:
        return tuple(sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True))

    @property
    def asks(self) -> tuple[Level, ...]:
        return tuple(sorted(self._asks.items(), key=lambda kv: kv[0]))

    def snapshot(self, *, source: str | None = None) -> BookSnapshot:
        return BookSnapshot(
            token_id=self.token_id,
            side=self.side,
            bids=self.bids,
            asks=self.asks,
            timestamp=self.timestamp or time.time(),
            source=source or self.source,
            sequence=self.sequence,
            window_close_epoch=self.window_close_epoch,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OrderBook({self.side} {self.token_id[:10]}… "
            f"{len(self._bids)}b/{len(self._asks)}a seq={self.sequence})"
        )


# ------------------------------------------------------------- raw feed types
@dataclass(frozen=True)
class TradePrint:
    price: float
    size: float
    timestamp: float
    side: str = ""
    source: str = ""


@dataclass(frozen=True)
class SpotTick:
    """Reference price from the underlying venue (e.g. Binance BTCUSDT)."""

    symbol: str
    price: float
    timestamp: float
    source: str = "binance"

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.timestamp


# -------------------------------------------------------------- transport API
class RestSnapshotTransport(Protocol):
    """Anything that can fetch a JSON snapshot. Implemented by the caller."""

    def fetch_json(self, url: str, params: dict[str, Any] | None = ...) -> Any:  # pragma: no cover
        ...


class BookStreamTransport(Protocol):
    """Streaming transport. ``poll`` must be non-blocking and return [] when idle."""

    def connect(self) -> None:  # pragma: no cover
        ...

    def subscribe(self, token_ids: Sequence[str]) -> None:  # pragma: no cover
        ...

    def poll(self) -> list[dict]:  # pragma: no cover
        ...

    def close(self) -> None:  # pragma: no cover
        ...


class OfflineRestTransport:
    """Offline transport used in backtests/tests; never opens a socket.

    Construct with canned payloads keyed by URL substring, or register them with
    :meth:`register`. Any unregistered URL raises, so a missing mock is loud
    rather than silently returning None.
    """

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self._responses: dict[str, Any] = dict(responses or {})
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def register(self, url_key: str, payload: Any) -> None:
        self._responses[url_key] = payload

    def fetch_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self.requests.append((url, params))
        for key, payload in self._responses.items():
            if key in url:
                return payload
        raise DataQualityError(
            f"OfflineRestTransport has no canned response for {url!r}; "
            "this system never performs live requests unless you inject a real transport"
        )


class RecordingStreamTransport:
    """Replays a fixed list of messages - the offline stand-in for a WebSocket."""

    def __init__(self, messages: Iterable[dict] | None = None) -> None:
        self._messages: Deque[dict] = deque(messages or [])
        self.connected = False
        self.subscribed: tuple[str, ...] = ()
        self.closed = False

    def push(self, message: dict) -> None:
        self._messages.append(message)

    def connect(self) -> None:
        self.connected = True
        self.closed = False

    def subscribe(self, token_ids: Sequence[str]) -> None:
        self.subscribed = tuple(token_ids)

    def poll(self) -> list[dict]:
        out: list[dict] = []
        while self._messages:
            out.append(self._messages.popleft())
        return out

    def close(self) -> None:
        self.closed = True
        self.connected = False


class LiveWebSocketTransport:
    """Real WebSocket transport - **opt-in only**.

    Requires the optional ``websockets`` package and an explicit URL. It is
    deliberately not instantiated anywhere in this codebase; wire it yourself
    when you want live data::

        transport = LiveWebSocketTransport("wss://.../ws/market", token_ids=[...])
        hub = MarketDataHub(token_ids=..., stream=transport, rest=my_rest)

    Parsing is delegated to :meth:`parse_message`, which understands the CLOB
    ``book``/``price_change`` shapes and falls back to a generic
    ``{"bids": [...], "asks": [...]}`` payload.
    """

    def __init__(
        self,
        url: str,
        *,
        token_ids: Sequence[str] = (),
        subscribe_message: Callable[[Sequence[str]], dict] | None = None,
    ) -> None:
        self.url = url
        self.token_ids = tuple(token_ids)
        self.subscribe_message = subscribe_message or (
            lambda ids: {"type": "market", "assets_ids": list(ids)}
        )
        self._ws = None
        self._inbox: Deque[dict] = deque()

    def connect(self) -> None:  # pragma: no cover - requires network
        try:
            import websockets.sync.client as ws_client  # type: ignore
        except ImportError as exc:
            raise DataQualityError(
                "LiveWebSocketTransport needs the optional 'websockets' package; "
                "this system ships offline by default"
            ) from exc
        self._ws = ws_client.connect(self.url)

    def subscribe(self, token_ids: Sequence[str]) -> None:  # pragma: no cover
        if self._ws is None:
            raise DataQualityError("connect() first")
        self._ws.send(json.dumps(self.subscribe_message(token_ids)))

    def poll(self) -> list[dict]:  # pragma: no cover - requires network
        if self._ws is None:
            return []
        while True:
            try:
                raw = self._ws.recv(timeout=0.05)
            except TimeoutError:
                break
            except Exception:  # noqa: BLE001 - transport-level failure
                break
            if raw:
                self._inbox.append(json.loads(raw))
        out: list[dict] = []
        while self._inbox:
            out.append(self._inbox.popleft())
        return out

    def close(self) -> None:  # pragma: no cover
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    @staticmethod
    def parse_message(message: dict) -> dict | None:
        """Normalise a venue message into a snapshot/delta payload."""
        if not isinstance(message, dict):
            return None
        event_type = str(message.get("event_type") or message.get("type") or "").lower()
        asset = message.get("asset_id") or message.get("token_id") or message.get("market")

        if "bids" in message and "asks" in message:
            return {
                "kind": "snapshot",
                "token_id": asset,
                "bids": message.get("bids") or [],
                "asks": message.get("asks") or [],
                "timestamp": message.get("timestamp"),
                "sequence": message.get("sequence") or message.get("hash"),
            }
        if event_type in {"book", "snapshot"}:
            return {
                "kind": "snapshot",
                "token_id": asset,
                "bids": message.get("bids") or message.get("buys") or [],
                "asks": message.get("asks") or message.get("sells") or [],
                "timestamp": message.get("timestamp"),
                "sequence": message.get("sequence") or message.get("hash"),
            }
        if event_type in {"price_change", "delta", "l2update"}:
            changes = message.get("changes")
            if changes is None:
                changes = [
                    {
                        "side": change.get("side"),
                        "price": change.get("price"),
                        "size": change.get("size"),
                    }
                    for change in (message.get("price_changes") or [])
                ]
            return {
                "kind": "delta",
                "token_id": asset,
                "changes": changes,
                "timestamp": message.get("timestamp"),
                "sequence": message.get("sequence") or message.get("hash"),
            }
        return None


# ------------------------------------------------------------------ market hub
@dataclass
class DataQualityReport:
    """Cross-validation verdict for one polling cycle."""

    healthy: bool
    reasons: tuple[str, ...] = ()
    ws_mid: float | None = None
    rest_mid: float | None = None
    divergence: float | None = None
    ws_age: float | None = None
    rest_age: float | None = None
    spot_age: float | None = None
    crossed: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "reasons": list(self.reasons),
            "ws_mid": self.ws_mid,
            "rest_mid": self.rest_mid,
            "divergence": self.divergence,
            "ws_age": self.ws_age,
            "rest_age": self.rest_age,
            "spot_age": self.spot_age,
            "crossed": list(self.crossed),
        }


class MarketDataHub:
    """Merges streamed books with periodic REST snapshots and validates them.

    ``max_divergence``
        Absolute midpoint difference (in probability) above which the two
        sources are considered to disagree and trading should pause.
    ``max_ws_age`` / ``max_rest_age`` / ``max_spot_age``
        Staleness limits in seconds.
    """

    def __init__(
        self,
        token_ids: dict[str, str] | None = None,
        *,
        stream: BookStreamTransport | None = None,
        rest: RestSnapshotTransport | None = None,
        clob_rest_url: str = "",
        gamma_rest_url: str = "",
        book_depth: int = 5,
        max_divergence: float = 0.03,
        max_ws_age: float = 5.0,
        max_rest_age: float = 30.0,
        max_spot_age: float = 5.0,
        wall_min_share: float = 0.30,
    ) -> None:
        self.token_ids = dict(token_ids or {})
        self.stream = stream
        self.rest = rest
        self.clob_rest_url = clob_rest_url
        self.gamma_rest_url = gamma_rest_url
        self.depth = book_depth
        self.max_divergence = max_divergence
        self.max_ws_age = max_ws_age
        self.max_rest_age = max_rest_age
        self.max_spot_age = max_spot_age
        self.wall_min_share = wall_min_share

        self.books: dict[str, OrderBook] = {
            side: OrderBook(token_id=token_id, side=side)
            for side, token_id in self.token_ids.items()
        }
        self._rest_snapshots: dict[str, BookSnapshot] = {}
        self.spot: SpotTick | None = None
        self.trades: Deque[TradePrint] = deque(maxlen=500)
        self.last_report: DataQualityReport | None = None
        self.price_history: dict[str, Deque[float]] = {
            side: deque(maxlen=600) for side in self.books
        }

    # -- wiring -----------------------------------------------------------
    def start(self) -> None:
        if self.stream is not None:
            self.stream.connect()
            self.stream.subscribe(tuple(self.token_ids.values()))

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.close()

    # -- ingestion --------------------------------------------------------
    def ingest_stream(self) -> int:
        """Drain the stream, applying snapshots/deltas. Returns messages applied."""
        if self.stream is None:
            return 0
        applied = 0
        for message in self.stream.poll():
            payload = (
                LiveWebSocketTransport.parse_message(message)
                if message.get("kind") not in {"snapshot", "delta"}
                else message
            )
            if payload is None:
                continue
            token_id = payload.get("token_id")
            book = self._book_for_token(token_id) if token_id else self._sole_book()
            if book is None:
                continue
            if payload["kind"] == "snapshot":
                book.apply_snapshot(payload, source="websocket")
            else:
                book.apply_delta(payload, source="websocket")
            applied += 1
        for side, book in self.books.items():
            mid = book.snapshot().midpoint()
            if mid is not None:
                self.price_history[side].append(mid)
        return applied

    def _book_for_token(self, token_id: str) -> OrderBook | None:
        for book in self.books.values():
            if book.token_id == token_id:
                return book
        return None

    def _sole_book(self) -> OrderBook | None:
        if len(self.books) == 1:
            return next(iter(self.books.values()))
        return None

    def ingest_rest_snapshot(self, payload: dict, *, side: str | None = None) -> BookSnapshot:
        """Accept a REST book payload and record it for cross-validation.

        ``side`` selects which token this payload belongs to; when omitted the
        token id in the payload is matched, else the single book is used.
        """
        token_id = payload.get("token_id") or payload.get("asset_id")
        book = self.books.get(side) if side else None
        if book is None:
            book = self._book_for_token(str(token_id)) if token_id else self._sole_book()
        if book is None:
            raise DataQualityError(
                "cannot route REST snapshot: provide side= or a known token_id"
            )
        snapshot = BookSnapshot(
            token_id=book.token_id,
            side=book.side,
            bids=tuple(levels_for_side(payload, "bid")),
            asks=tuple(levels_for_side(payload, "ask")),
            timestamp=float(payload.get("timestamp") or time.time()),
            source="rest",
            window_close_epoch=payload.get("window_close_epoch"),
        )
        self._rest_snapshots[book.side] = snapshot
        return snapshot

    def fetch_rest_snapshot(self, side: str) -> BookSnapshot | None:
        """Pull a REST snapshot through the injected transport (never implicit)."""
        if self.rest is None or not self.clob_rest_url:
            return None
        book = self.books.get(side)
        if book is None:
            return None
        payload = self.rest.fetch_json(
            f"{self.clob_rest_url}/book", {"token_id": book.token_id}
        )
        if not isinstance(payload, dict):
            raise DataQualityError("REST snapshot transport returned a non-object payload")
        return self.ingest_rest_snapshot(payload, side=side)

    def update_spot(self, tick: SpotTick) -> None:
        self.spot = tick

    def record_trade(self, trade: TradePrint) -> None:
        self.trades.append(trade)

    # -- analytics --------------------------------------------------------
    def snapshot(self, side: str) -> BookSnapshot:
        return self.books[side].snapshot()

    def midpoint(self, side: str) -> float | None:
        return self.books[side].snapshot().midpoint()

    def microprice(self, side: str) -> float | None:
        return self.books[side].snapshot().microprice(self.depth)

    def obi(self, side: str, top: int | None = None) -> float:
        return self.books[side].snapshot().obi(top or self.depth)

    def liquidity(self, side: str, book_side: str = "bid", top: int | None = None) -> DepthMetrics:
        """Depth metrics for one side of one outcome token's book."""
        if book_side not in {"bid", "ask"}:
            raise ValueError("book_side must be 'bid' or 'ask'")
        return self.books[side].snapshot().depth(book_side, top or self.depth)

    def book_liquidity(self, side: str, top: int | None = None) -> tuple[DepthMetrics, DepthMetrics]:
        """(bid_depth, ask_depth) for one outcome token."""
        snapshot = self.books[side].snapshot()
        return snapshot.depth("bid", top or self.depth), snapshot.depth("ask", top or self.depth)

    def walls(self, side: str) -> list[Wall]:
        return self.books[side].snapshot().walls(self.depth, self.wall_min_share)

    def implied_probability(self, side: str) -> float | None:
        """Probability the market assigns to this outcome (microprice preferred)."""
        micro = self.microprice(side)
        if micro is not None:
            return micro
        return self.midpoint(side)

    def complement_implied_probability(self) -> float | None:
        """Implied prob of ``down`` derived from ``up`` (binary constraint)."""
        up = self.implied_probability(UP)
        return None if up is None else 1.0 - up

    # -- validation -------------------------------------------------------
    def validate(self, now: float | None = None, *, verbose: bool = False) -> DataQualityReport:
        """Cross-check the two sources and flag staleness/crossed books."""
        moment = now if now is not None else time.time()
        reasons: list[str] = []
        crossed: list[str] = []

        ws_mid = self.midpoint(UP)
        rest = self._rest_snapshots.get(UP)
        rest_mid = rest.midpoint() if rest else None

        for side, book in self.books.items():
            snap = book.snapshot()
            if snap.is_crossed():
                crossed.append(side)
            if book.timestamp and (moment - book.timestamp) > self.max_ws_age:
                reasons.append(
                    f"{side}: stream stale by {moment - book.timestamp:.1f}s "
                    f"(limit {self.max_ws_age:.1f}s)"
                )
            if not snap.bids or not snap.asks:
                reasons.append(f"{side}: book has no two-sided quotes")

        divergence: float | None = None
        if ws_mid is not None and rest_mid is not None:
            divergence = abs(ws_mid - rest_mid)
            if divergence > self.max_divergence:
                reasons.append(
                    f"stream/REST midpoint divergence {divergence:.4f} exceeds "
                    f"{self.max_divergence:.4f}"
                )
        elif rest is not None and (moment - rest.timestamp) > self.max_rest_age:
            reasons.append(
                f"REST snapshot stale by {moment - rest.timestamp:.1f}s "
                f"(limit {self.max_rest_age:.1f}s)"
            )

        if crossed:
            reasons.append(f"crossed book on: {', '.join(crossed)}")

        spot_age = self.spot.age(moment) if self.spot else None
        if self.spot is None:
            reasons.append("no reference spot price received")
        elif spot_age is not None and spot_age > self.max_spot_age:
            reasons.append(f"spot stale by {spot_age:.1f}s (limit {self.max_spot_age:.1f}s)")

        report = DataQualityReport(
            healthy=not reasons,
            reasons=tuple(reasons),
            ws_mid=ws_mid,
            rest_mid=rest_mid,
            divergence=divergence,
            ws_age=(moment - self.books[UP].timestamp) if UP in self.books and self.books[UP].timestamp else None,
            rest_age=(moment - rest.timestamp) if rest else None,
            spot_age=spot_age,
            crossed=tuple(crossed),
        )
        self.last_report = report
        if verbose and reasons:
            for reason in reasons:
                log.warning("data quality: %s", reason)
        return report

    def is_tradeable(self, now: float | None = None) -> bool:
        return self.validate(now).healthy

    # -- rolling stats ----------------------------------------------------
    def realized_volatility(self, side: str = UP, window: int = 30) -> float | None:
        """Std-dev of recent midpoint changes - the risk module's volatility input."""
        history = list(self.price_history.get(side, ()))[-window:]
        if len(history) < 3:
            return None
        diffs = [b - a for a, b in zip(history, history[1:])]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        return math.sqrt(var)

    def iter_sides(self) -> Iterator[tuple[str, OrderBook]]:
        return iter(self.books.items())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MarketDataHub(tokens={list(self.token_ids)}, depth={self.depth})"

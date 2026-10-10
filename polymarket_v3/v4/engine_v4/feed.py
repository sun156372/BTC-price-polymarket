"""Public, read-only market data feed for live paper trading.

    from engine_v4.feed import PolymarketFeed, SpotFeed

NO API KEYS, NO SIGNING, NO ORDER PLACEMENT. Every call here is a GET against a
public endpoint. There is no code path in this package that can submit an order, so
"paper only" is structural rather than a configuration flag.

.. warning::

   **This module has never been executed against the live endpoints.** It was
   written in an environment with no network egress, so every URL, field name and
   response shape below is from the documented API rather than from an observed
   response. Treat the first run as a debugging session: call
   ``python run_live_v4.py probe`` before starting the loop, which fetches one
   market and one book and prints exactly what came back. If a field has moved, the
   probe will show it immediately instead of failing silently mid-session.

Endpoints used (all public GET):

* ``https://gamma-api.polymarket.com/markets?slug=...``  market metadata + token ids
* ``https://clob.polymarket.com/book?token_id=...``      full order book
* ``https://api.exchange.coinbase.com/products/BTC-USD/ticker``  BTC/USD spot
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com"
BINANCE = "https://api.binance.com"

#: slug format used by the 5-minute BTC up/down markets
SLUG_TEMPLATE = "btc-updown-5m-{epoch}"
WINDOW_SECONDS = 300

DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 2


class FeedError(RuntimeError):
    """Any failure to obtain usable market data. Never raised as a silent default."""


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT,
              retries: int = DEFAULT_RETRIES, params: dict | None = None):
    """GET a JSON document, with a bounded retry. Raises FeedError on failure."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "dsh-v4-paper/1.0",
                              "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
    raise FeedError(f"GET {url} failed after {retries + 1} attempts: {last}")


# --------------------------------------------------------------------- market


@dataclass
class MarketInfo:
    """Everything needed to price one window."""

    slug: str
    condition_id: str
    up_token: str
    down_token: str
    start_ts: float
    end_ts: float
    raw: dict = field(default_factory=dict)
    #: outcome labels aligned to the token ids, as returned by the venue
    outcomes: list = field(default_factory=list)


@dataclass
class Book:
    """One side's full order book at a moment in time."""

    token_id: str
    bids: list[tuple[float, float]]   # (price, size), best first
    asks: list[tuple[float, float]]
    fetched_at: float

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def bid_notional(self) -> float:
        return sum(p * s for p, s in self.bids)

    def ask_notional(self) -> float:
        return sum(p * s for p, s in self.asks)

    def imbalance(self) -> float | None:
        """(bid - ask) notional over the total.

        This reproduces the ``full_depth_bid_notional`` / ``full_depth_ask_notional``
        ratio the backtest relies on. If the venue's book ends up being only a
        shallow top-N view rather than the full book, this number will NOT equal the
        backtested feature and the gate threshold would need recalibrating - the
        live log therefore stores both notionals so the difference is visible.
        """
        b, a = self.bid_notional(), self.ask_notional()
        if b + a <= 0:
            return None
        return (b - a) / (b + a)

    def cost_to_buy(self, shares: float) -> float | None:
        """Walk the ask side for ``shares``; None if the book cannot fill it."""
        remaining = shares
        cost = 0.0
        for price, size in self.asks:
            if remaining <= 0:
                break
            take = min(remaining, size)
            cost += take * price
            remaining -= take
        if remaining > 1e-9:
            return None
        return cost


class PolymarketFeed:
    """Read-only access to Polymarket's public market data."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def market_for_epoch(self, epoch: int) -> MarketInfo | None:
        """Look up the 5m market whose window opens at ``epoch``."""
        slug = SLUG_TEMPLATE.format(epoch=int(epoch))
        payload = _get_json(f"{GAMMA}/markets", timeout=self.timeout,
                            params={"slug": slug})
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        return self._to_market_info(row, slug)

    def _to_market_info(self, row: dict, slug: str) -> MarketInfo:
        def maybe_json(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return None
            return value

        tokens = maybe_json(row.get("clobTokenIds")) or []
        outcomes = maybe_json(row.get("outcomes")) or []
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise FeedError(f"market {slug} returned no usable clobTokenIds: "
                            f"{row.get('clobTokenIds')!r}")
        # outcomes are conventionally ["Up", "Down"]; do not assume the order
        up_token = down_token = None
        for token, label in zip(tokens, outcomes or ["Up", "Down"]):
            text = str(label).strip().lower()
            if text in ("up", "yes"):
                up_token = token
            elif text in ("down", "no"):
                down_token = token
        if up_token is None or down_token is None:
            up_token, down_token = tokens[0], tokens[1]

        start = row.get("startDate") or row.get("start_date")
        end = row.get("endDate") or row.get("end_date")
        start_ts = _parse_ts(start)
        end_ts = _parse_ts(end)
        if start_ts is None:
            # fall back to the epoch in the slug, which is what the slug encodes
            try:
                start_ts = float(slug.rsplit("-", 1)[-1])
            except (ValueError, IndexError):
                start_ts = None
        if start_ts is None:
            raise FeedError(f"cannot determine window start for {slug}")
        if end_ts is None:
            end_ts = start_ts + WINDOW_SECONDS
        return MarketInfo(slug=slug,
                          condition_id=str(row.get("conditionId")
                                           or row.get("condition_id") or ""),
                          up_token=str(up_token), down_token=str(down_token),
                          start_ts=start_ts, end_ts=end_ts, raw=row,
                          outcomes=list(outcomes) if outcomes else [])

    def book(self, token_id: str) -> Book:
        payload = _get_json(f"{CLOB}/book", timeout=self.timeout,
                            params={"token_id": token_id})
        bids = _levels(payload.get("bids"))
        asks = _levels(payload.get("asks"))
        # venue returns bids ascending by price and asks descending; normalise so
        # index 0 is always the best level
        bids.sort(key=lambda lv: lv[0], reverse=True)
        asks.sort(key=lambda lv: lv[0])
        return Book(token_id=token_id, bids=bids, asks=asks,
                    fetched_at=time.time())

    def resolved_outcome(self, market: MarketInfo) -> str | None:
        """'UP', 'DOWN', or None if not yet resolved."""
        try:
            payload = _get_json(f"{GAMMA}/markets", timeout=self.timeout,
                                params={"slug": market.slug})
        except FeedError:
            return None
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        if not row.get("closed"):
            return None
        prices = row.get("outcomePrices")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (ValueError, TypeError):
                return None
        if not isinstance(prices, list) or len(prices) < 2:
            return None
        try:
            values = [float(p) for p in prices]
        except (TypeError, ValueError):
            return None
        return "UP" if values[0] >= values[1] else "DOWN"


def _levels(raw) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for level in raw or []:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size > 0:
            out.append((price, size))
    return out


def _parse_ts(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        from datetime import datetime, timezone
        cleaned = text.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc).timestamp()
    except (ValueError, TypeError):
        pass
    try:
        return float(text)
    except ValueError:
        return None


# ----------------------------------------------------------------------- spot


@dataclass
class SpotQuote:
    ts: float
    price: float
    source: str


class SpotFeed:
    """BTC/USD spot, public endpoints, with a fallback chain.

    The backtest used Hyperliquid BTC/USDC at ~117ms. A polling fallback is coarser,
    which matters for the 4.5bps spot-move gate: a 5bp move is small, so a stale
    price can flip that gate. The live log stores the quote timestamp and source so
    this can be audited rather than assumed.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def quote(self) -> SpotQuote | None:
        for source, fetch in (("coinbase", self._coinbase),
                              ("binance", self._binance)):
            try:
                price = fetch()
            except FeedError:
                continue
            if price:
                return SpotQuote(ts=time.time(), price=price, source=source)
        return None

    def _coinbase(self) -> float | None:
        payload = _get_json(f"{COINBASE}/products/BTC-USD/ticker",
                            timeout=self.timeout)
        price = payload.get("price")
        return float(price) if price else None

    def _binance(self) -> float | None:
        payload = _get_json(f"{BINANCE}/api/v3/ticker/price",
                            timeout=self.timeout, params={"symbol": "BTCUSDT"})
        price = payload.get("price")
        return float(price) if price else None

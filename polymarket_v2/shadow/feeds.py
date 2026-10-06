"""Live feeds for the shadow engine, plus the offline replay used for testing.

Both feeds only ever *fill* the pre-parsed containers from :mod:`shadow.markets`;
no parsing happens downstream, so the 1200-account loop stays arithmetic-only.

* :class:`BinanceSpotFeed` - 1s klines into a :class:`SpotRing` (REST polling via
  ccxt when available, otherwise plain HTTP).
* :class:`PolymarketFeed` - CLOB market websocket into per-token
  :class:`PrintSeries`, plus token discovery through the Gamma REST API.
* :class:`ReplayFeed` - reads the parquet history and drives the identical engine
  offline, which is how the shadow book is verified before going live.
"""
from __future__ import annotations

import bisect
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from .markets import MarketBook, MarketFrame, PrintSeries, SpotRing, now_utc

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_REST = "https://api.binance.com/api/v3/klines"
SPOT_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
WINDOW_SECONDS = {"5m": 300, "15m": 900}

try:
    import requests
except ImportError:                                     # pragma: no cover
    requests = None
try:
    import websockets
except ImportError:                                     # pragma: no cover
    websockets = None
try:
    import ccxt
except ImportError:                                     # pragma: no cover
    ccxt = None


# ------------------------------------------------------------------ spot feed
class BinanceSpotFeed:
    """Polls Binance 1s klines into one SpotRing per asset."""

    def __init__(self, assets: Sequence[str], *,
                 log: Callable[[str], None] = print) -> None:
        self.log = log
        self.assets = [a.upper() for a in assets]
        self.rings: dict[str, SpotRing] = {a: SpotRing(a, capacity=3600) for a in self.assets}
        self.symbols = {a: SPOT_SYMBOLS.get(a, f"{a}USDT") for a in self.assets}
        self._client = ccxt.binance({"enableRateLimit": True}) if ccxt else None
        self._last_ms: dict[str, int] = {}
        self.errors = 0

    def _klines(self, symbol: str, since_ms: int) -> list:
        if self._client is not None:
            return self._client.public_get_klines(
                {"symbol": symbol, "interval": "1s", "startTime": since_ms, "limit": 300})
        if requests is None:
            raise RuntimeError("need requests or ccxt to poll Binance")
        response = requests.get(BINANCE_REST, timeout=10, params={
            "symbol": symbol, "interval": "1s", "startTime": since_ms, "limit": 300})
        response.raise_for_status()
        return response.json()

    def poll(self, now: float) -> int:
        added = 0
        for asset in self.assets:
            symbol = self.symbols[asset]
            since_ms = self._last_ms.get(asset, int((now - 900) * 1000)) + 1000
            try:
                rows = self._klines(symbol, since_ms)
            except Exception as exc:  # noqa: BLE001 - network boundary
                self.errors += 1
                self.log(f"  [spot] {symbol} poll failed: {type(exc).__name__}: {exc}")
                continue
            ring = self.rings[asset]
            for row in rows:
                try:
                    ts = float(row[0]) / 1000.0
                    close = float(row[4])
                except (IndexError, TypeError, ValueError):
                    continue
                if ts * 1000 > self._last_ms.get(asset, 0):
                    ring.append(ts, close)
                    self._last_ms[asset] = int(ts * 1000)
                    added += 1
        return added

    def last_price(self, asset: str) -> float | None:
        bar = self.rings.get(asset.upper())
        newest = bar.newest if bar else None
        return newest[1] if newest else None

    def age(self, asset: str, now: float) -> float:
        bar = self.rings.get(asset.upper())
        return float("inf") if bar is None else bar.age(now)


# ----------------------------------------------------------- polymarket feed
class PolymarketFeed:
    """Discovers active windows and streams their books into PrintSeries."""

    def __init__(self, assets: Sequence[str], *, window: str = "5m",
                 print_interval_s: float = 1.0,
                 log: Callable[[str], None] = print) -> None:
        self.log = log
        self.assets = [a.upper() for a in assets]
        self.window = window
        self.window_s = WINDOW_SECONDS.get(window, 300)
        self.print_interval_s = float(print_interval_s)
        self.tokens: dict[str, tuple[str, str]] = {}    # slug -> (UP, DOWN)
        self.quotes: dict[str, dict[str, tuple[float, float | None, float | None]]] = {}
        self.series: dict[str, dict[str, PrintSeries]] = {}
        self._pending: set[str] = set()
        self._last_print: dict[str, float] = {}
        self._ws = None
        self.messages = 0

    # -- discovery -------------------------------------------------------
    def candidate_slugs(self, now: float) -> list[str]:
        slot = int(now // self.window_s) * self.window_s
        out = []
        for asset in self.assets:
            for start in (slot, slot + self.window_s):
                out.append(f"{asset.lower()}-updown-{self.window}-{start}")
        return out

    def resolve(self, now: float) -> list[str]:
        """Fetch token ids for the active/next window of each asset."""
        if requests is None:
            return []
        found: list[str] = []
        for slug in self.candidate_slugs(now):
            if slug in self.tokens:
                found.append(slug)
                continue
            try:
                response = requests.get(f"{GAMMA_URL}/events", timeout=8,
                                        params={"slug": slug})
                response.raise_for_status()
                events = response.json()
            except Exception as exc:  # noqa: BLE001
                self.log(f"  [pm] resolve {slug}: {type(exc).__name__}")
                continue
            if not events:
                continue
            market = (events[0].get("markets") or [None])[0]
            if not market:
                continue
            raw = market.get("clobTokenIds")
            ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if len(ids) < 2:
                continue
            end_ts = self._epoch(market.get("endDate"))
            if end_ts is not None and end_ts < now - 30:
                continue
            self.tokens[slug] = (str(ids[0]), str(ids[1]))   # Polymarket: [Up, Down]
            self.series[slug] = {str(ids[0]): PrintSeries(str(ids[0])),
                                 str(ids[1]): PrintSeries(str(ids[1]))}
            self._pending.update(self.tokens[slug])
            self.log(f"  [pm] {slug} resolved"
                     + (f", ends in {end_ts - now:.0f}s" if end_ts else ""))
            found.append(slug)
        return found

    @staticmethod
    def _epoch(value: Any) -> float | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def frame_for(self, slug: str, now: float) -> MarketFrame | None:
        mapping = self.tokens.get(slug)
        if not mapping:
            return None
        try:
            start = float(slug.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return None
        return MarketFrame(
            slug=slug, asset=slug.split("-")[0].upper(), start_ts=start,
            end_ts=start + self.window_s, tokens=mapping,
            prints=dict(self.series.get(slug, {})),
        )

    # -- websocket -------------------------------------------------------
    async def stream(self) -> None:
        if websockets is None:
            self.log("  [pm] websockets not installed - no order book feed")
            return
        import asyncio

        while True:
            try:
                async with websockets.connect(CLOB_WS, ping_interval=20,
                                              close_timeout=5) as ws:
                    self._ws = ws
                    self.log("  [pm] websocket connected")
                    while True:
                        if self._pending:
                            ids = sorted(self._pending)
                            self._pending.clear()
                            await ws.send(json.dumps({"assets_ids": ids, "type": "market"}))
                            self.log(f"  [pm] subscribed {len(ids)} token(s)")
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        self.ingest(raw)
            except Exception as exc:  # noqa: BLE001 - connection boundary
                self.log(f"  [pm] ws error: {type(exc).__name__}: {exc}; reconnecting")
                await asyncio.sleep(3)

    def ingest(self, raw: str, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        for item in (payload if isinstance(payload, list) else [payload]):
            if not isinstance(item, dict):
                continue
            token = str(item.get("asset_id") or item.get("token_id") or "")
            if not token:
                continue
            slug = self._slug_for(token)
            if slug is None:
                continue
            event = str(item.get("event_type") or item.get("type") or "").lower()
            if "bids" in item or "asks" in item or event in {"book", "snapshot"}:
                bid = self._best(item.get("bids") or item.get("buys"), highest=True)
                ask = self._best(item.get("asks") or item.get("sells"), highest=False)
                if bid is None and ask is None:
                    continue
                mid = ((bid + ask) / 2.0) if (bid and ask) else (bid or ask)
                self.quotes.setdefault(slug, {})[token] = (mid, bid, ask)
                self._append(slug, token, mid, moment)
            elif event in {"price_change", "last_trade_price", "tick"}:
                try:
                    price = float(item.get("price"))
                except (TypeError, ValueError):
                    continue
                if 0.0 < price < 1.0:
                    self._append(slug, token, price, moment)
            self.messages += 1

    def _append(self, slug: str, token: str, price: float, now: float) -> None:
        key = f"{slug}:{token}"
        if now - self._last_print.get(key, 0.0) < self.print_interval_s:
            return
        self._last_print[key] = now
        series = self.series.setdefault(slug, {}).get(token)
        if series is None:
            return
        ts = list(series.ts)
        px = list(series.px)
        ts.append(now)
        px.append(price)
        if len(ts) > 2400:                       # keep ~40 min at 1s resolution
            ts, px = ts[-2400:], px[-2400:]
        self.series[slug][token] = PrintSeries(token, tuple(ts), tuple(px))

    @staticmethod
    def _best(levels: Any, *, highest: bool) -> float | None:
        best = None
        for level in levels or []:
            if isinstance(level, dict):
                price = level.get("price")
            elif isinstance(level, (list, tuple)) and level:
                price = level[0]
            else:
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            if not 0.0 < price < 1.0:
                continue
            if best is None or (price > best if highest else price < best):
                best = price
        return best

    def _slug_for(self, token: str) -> str | None:
        for slug, mapping in self.tokens.items():
            if token in mapping:
                return slug
        return None


# ------------------------------------------------------------------ replay
def to_epoch(value: Any) -> float:
    """Convert a parquet timestamp to epoch seconds, **assuming UTC**.

    DuckDB writes these columns as naive ``datetime`` objects. Calling
    ``.timestamp()`` on a naive datetime interprets it as *local* time, which on a
    UTC+8 machine shifts every print by 8 hours and silently breaks the join with
    window timestamps derived from the market slug. Pinning tzinfo=UTC is the fix.
    """
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return float(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return float(value)


@dataclass
class ReplayData:
    """Everything a replay needs, with spot bars sorted rather than pre-ringed."""

    book: MarketBook
    bars: dict[str, list[tuple[float, float]]]      # asset -> sorted (ts, close)
    mapping: dict[str, tuple[str, str]]             # slug -> (UP, DOWN)


@dataclass
class ReplayFeed:
    """Drives the shadow engine from the parquet history, offline.

    Reads one trades file plus the crypto_prices bars, then steps the engine
    window by window in the same order live data would arrive.
    """

    data_dir: str
    trades_file: str = "auto"
    assets: Sequence[str] = ("BTC", "ETH", "SOL", "XRP")

    def _resolve_trades(self) -> str:
        import glob
        import os
        if self.trades_file != "auto":
            path = self.trades_file if os.path.isabs(self.trades_file) else os.path.join(
                self.data_dir, self.trades_file)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path
        best = None
        for path in glob.glob(os.path.join(self.data_dir, "*.parquet")):
            try:
                import pyarrow.parquet as pq
                names = set(pq.ParquetFile(path).schema_arrow.names)
            except Exception:  # noqa: BLE001
                continue
            if {"timestamp", "market_id", "token_id", "price"} <= names:
                size = os.path.getsize(path)
                if best is None or size > best[0]:
                    best = (size, path)
        if best is None:
            raise FileNotFoundError(f"no trades parquet in {self.data_dir}")
        return best[1]

    def load(self) -> "ReplayData":
        """Return replay data: market book, sorted spot bars, token mapping.

        Spot bars are returned **sorted rather than pre-loaded into a ring**. The
        ring is a live construct holding the last N seconds; filling it with the
        whole file would leave its newest bar hours after the window being
        replayed, and every cutoff-guarded lookup would then correctly return
        nothing. The runner advances the ring with the timeline instead.
        """
        import pyarrow.parquet as pq
        import os
        import glob

        wanted = {a.upper() for a in self.assets}

        # -- spot bars, sorted per asset
        bars: dict[str, list[tuple[float, float]]] = {}
        for path in glob.glob(os.path.join(self.data_dir, "*.parquet")):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if not {"timestamp", "asset", "close"} <= names:
                continue
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=200_000, columns=["timestamp", "asset", "close"]):
                for row in batch.to_pylist():
                    asset = str(row.get("asset") or "").upper()
                    if asset not in wanted:
                        continue
                    bars.setdefault(asset, []).append(
                        (to_epoch(row["timestamp"]), float(row["close"])))
        for asset in bars:
            bars[asset].sort(key=lambda item: item[0])

        # -- prints + token mapping
        path = self._resolve_trades()
        prints: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=200_000,
                columns=["timestamp", "asset", "market_id", "token_id", "price"]):
            for row in batch.to_pylist():
                asset = str(row.get("asset") or "").upper()
                if asset not in wanted:
                    continue
                slug = str(row.get("market_id") or "")
                token = str(row.get("token_id") or "")
                price = row.get("price")
                if not slug or not token or price is None:
                    continue
                price = float(price)
                if not 0.0 < price < 1.0:
                    continue
                ts = to_epoch(row["timestamp"])
                prints.setdefault(slug, {}).setdefault(token, []).append((ts, price))

        # -- which token is UP: the one that ends nearer 1.0
        book = MarketBook()
        mapping: dict[str, tuple[str, str]] = {}
        for slug, tokens in prints.items():
            if len(tokens) < 2:
                continue
            try:
                start = float(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            end = start + (900 if "-15m" in slug else 300)
            late = {}
            for token, series in tokens.items():
                tail = [p for ts, p in series if end - ts <= 120]
                if tail:
                    late[token] = sum(tail) / len(tail)
            if len(late) < 2:
                continue
            ranked = sorted(late.items(), key=lambda kv: kv[1], reverse=True)
            if ranked[0][1] - ranked[1][1] < 0.05:
                continue
            up_token, down_token = ranked[0][0], ranked[1][0]
            top_mid = ranked[0][1]
            winner = "UP" if top_mid > 0.5 else "DOWN"
            series = {}
            for token, points in tokens.items():
                points.sort(key=lambda item: item[0])
                series[token] = PrintSeries(token,
                                            tuple(p[0] for p in points),
                                            tuple(p[1] for p in points))
            mapping[slug] = (up_token, down_token)
            book.upsert(MarketFrame(
                slug=slug, asset=slug.split("-")[0].upper(), start_ts=start,
                end_ts=end, tokens=(up_token, down_token), prints=series,
                winner=winner, mid={ranked[0][0]: ranked[0][1], ranked[1][0]: ranked[1][1]},
            ))
        return ReplayData(book=book, bars=bars, mapping=mapping)

"""Data access: parquet readers, the market/outcome join, and as-of lookups.

Dataset layout this expects (DuckDB exports)::

    all.parquet          market_id(=slug), asset, start_time, end_time, slug, ...
    all (1).parquet      market_id(=numeric), asset, start_time, end_time, outcome
    trades/*.parquet     timestamp, asset, market_id(=slug), token_id, price, [side, size]
    crypto_prices/*.pq   timestamp, asset, open, high, low, close, volume
    orderbooks/*.parquet timestamp, asset, market_id, token_id, best_bid, best_ask, ...

Two traps this module handles explicitly, because both silently corrupt results:

1. **The two resolution files disagree on identifiers.** One carries the slug but
   no outcome, the other the outcome but only a numeric id. They are merged on
   ``(asset, start_time, end_time)`` - the window they both describe.
2. **Time units vary.** Native TIMESTAMP columns, epoch seconds, milliseconds and
   ISO strings all appear in the wild; :func:`to_epoch` normalises them.

Spot prices are indexed for **backward-only (as-of) lookup** - a forward match
would let the backtest read a price that did not exist yet.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

MAGIC_HINT = "install pyarrow"


class DataError(RuntimeError):
    """Raised when the dataset is missing, malformed or unreadable."""


# --------------------------------------------------------------- dependencies
def parquet_backend() -> str:
    """Return 'pyarrow' or 'pandas'; raise with instructions if neither exists."""
    try:
        import pyarrow.parquet  # noqa: F401

        return "pyarrow"
    except ImportError:
        pass
    try:
        import pandas  # noqa: F401

        return "pandas"
    except ImportError as exc:
        raise DataError(
            "reading parquet needs pyarrow (recommended) or pandas:\n"
            "    pip install pyarrow\n"
            "CSV export alternative:\n"
            "    duckdb -c \"COPY (SELECT * FROM read_parquet('trades/*.parquet')) "
            "TO 'trades.csv' (HEADER)\""
        ) from exc


def read_parquet(
    path: str,
    columns: Sequence[str] | None = None,
    batch_size: int = 100_000,
) -> Iterator[list[dict]]:
    """Stream a parquet file in bounded batches - never loads the whole file."""
    backend = parquet_backend()
    if backend == "pyarrow":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        wanted = [c for c in (columns or names) if c in names] or None
        for batch in parquet.iter_batches(batch_size=batch_size, columns=wanted):
            yield batch.to_pylist()
        return

    import pandas as pd

    reader = pd.read_parquet(path, columns=list(columns) if columns else None,
                             chunksize=batch_size)
    for frame in reader:
        yield frame.to_dict("records")


def parquet_columns(path: str) -> list[str]:
    backend = parquet_backend()
    if backend == "pyarrow":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    import pandas as pd

    return list(pd.read_parquet(path, columns=[]).columns)


# ------------------------------------------------------------------ time utils
def to_epoch(value: Any) -> float | None:
    """Normalise datetime / epoch (s, ms, us, ns) / ISO string to epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            try:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.timestamp()
        value = number
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return None
    if magnitude >= 1e17:
        return float(value) / 1e9
    if magnitude >= 1e14:
        return float(value) / 1e6
    if magnitude >= 1e11:
        return float(value) / 1e3
    return float(value)


def utc(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# --------------------------------------------------------------------- records
@dataclass(frozen=True)
class Market:
    """One 5/15-minute Up-or-Down window with its settled outcome."""

    slug: str
    asset: str
    start_ts: float
    end_ts: float
    outcome: str            # "UP" or "DOWN"

    @property
    def window_s(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def up_won(self) -> bool:
        return self.outcome.startswith("UP")

    def t_minus_to_epoch(self, t_minus_s: float) -> float:
        return self.end_ts - t_minus_s


@dataclass
class MarketTrades:
    """Executed prints for one market, split by outcome token."""

    slug: str
    by_token: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def tokens(self) -> list[str]:
        return list(self.by_token)

    def series(self, token: str) -> list[tuple[float, float]]:
        return self.by_token.get(token) or []

    def last_at_or_before(self, token: str, ts: float) -> tuple[float, float] | None:
        """Most recent print at or before ``ts`` - the only fill we may assume."""
        series = self.by_token.get(token) or []
        index = bisect.bisect_right(series, (ts, float("inf"))) - 1
        if index < 0:
            return None
        return series[index]

    def prints_after(self, token: str, ts: float) -> list[tuple[float, float]]:
        series = self.by_token.get(token) or []
        index = bisect.bisect_right(series, (ts, float("inf")))
        return series[index:]

    def mean_price_between(self, token: str, start: float, end: float) -> float | None:
        prices = [p for ts, p in (self.by_token.get(token) or []) if start <= ts <= end]
        if not prices:
            return None
        return sum(prices) / len(prices)


class SpotIndex:
    """Backward-only price lookup for one asset."""

    def __init__(self, asset: str) -> None:
        self.asset = asset
        self.times: list[float] = []
        self.prices: list[float] = []

    def __len__(self) -> int:
        return len(self.times)

    def add(self, ts: float | None, price: float | None) -> None:
        if ts is None or price is None:
            return
        self.times.append(float(ts))
        self.prices.append(float(price))

    def finalise(self) -> None:
        if not self.times:
            return
        pairs = sorted(zip(self.times, self.prices), key=lambda item: item[0])
        times: list[float] = []
        prices: list[float] = []
        for ts, price in pairs:
            if times and ts == times[-1]:
                prices[-1] = price          # keep the last print per timestamp
            else:
                times.append(ts)
                prices.append(price)
        self.times, self.prices = times, prices

    def as_of(self, ts: float, *, max_staleness: float | None = None) -> tuple[float, float] | None:
        """(price, source_ts) from the most recent bar at or before ``ts``."""
        if not self.times:
            return None
        index = bisect.bisect_right(self.times, ts) - 1
        if index < 0:
            return None
        source_ts = self.times[index]
        if max_staleness is not None and (ts - source_ts) > max_staleness:
            return None
        return self.prices[index], source_ts

    def change_pct(self, ts: float, lookback_s: float, *, max_staleness: float | None = None) -> tuple[float, float] | None:
        """(move_pct, spot_staleness_s) using only prices at or before ``ts``.

        Returns ``None`` when there is no bar early enough to form a base, which
        is the honest answer for the start of a dataset.
        """
        now = self.as_of(ts, max_staleness=max_staleness)
        past = self.as_of(ts - lookback_s, max_staleness=max_staleness)
        if now is None or past is None:
            return None
        price_now, source_ts = now
        price_past, _ = past
        if not price_past:
            return None
        return (price_now / price_past - 1.0) * 100.0, ts - source_ts


# --------------------------------------------------------------------- loader
class Dataset:
    """Lazily loads and caches everything the engine needs from ``data_dir``."""

    def __init__(self, data_dir: str, *, assets: Sequence[str] | None = None,
                 batch_size: int = 100_000) -> None:
        self.data_dir = os.path.abspath(data_dir)
        if not os.path.isdir(self.data_dir):
            raise DataError(f"data directory not found: {self.data_dir}")
        self.assets = [a.strip().upper() for a in (assets or []) if a.strip()]
        self.batch_size = batch_size
        self._markets: dict[str, Market] | None = None
        self._spot: dict[str, SpotIndex] | None = None
        self._files: dict[str, list[str]] | None = None

    # -- file discovery --------------------------------------------------
    def files(self) -> dict[str, list[str]]:
        if self._files is None:
            found = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.parquet"),
                                     recursive=True))
            groups: dict[str, list[str]] = {
                "markets": [], "outcomes": [], "trades": [], "spot": [], "books": [],
            }
            for path in found:
                name = os.path.basename(path).lower()
                try:
                    columns = set(parquet_columns(path))
                except Exception:  # noqa: BLE001 - skip unreadable files
                    continue
                if {"best_bid", "best_ask"} <= columns:
                    groups["books"].append(path)
                elif {"open", "high", "low", "close"} <= columns:
                    groups["spot"].append(path)
                elif {"price", "token_id"} <= columns and "side" in columns:
                    groups["trades"].append(path)
                elif "outcome" in columns and "end_time" in columns:
                    groups["outcomes"].append(path)
                elif "end_time" in columns and "slug" in columns:
                    groups["markets"].append(path)
                elif {"price", "token_id"} <= columns:
                    groups["trades"].append(path)
            self._files = groups
        return self._files

    # -- markets ---------------------------------------------------------
    def markets(self) -> dict[str, Market]:
        """slug -> Market, merging the metadata file with the outcome file."""
        if self._markets is not None:
            return self._markets

        groups = self.files()
        by_window: dict[tuple, Market] = {}

        def key(asset: str, start: float | None, end: float | None) -> tuple:
            return (asset.strip().upper(), round(start or 0.0), round(end or 0.0))

        for path in groups["markets"] + groups["outcomes"]:
            columns = parquet_columns(path)
            wanted = [c for c in ("market_id", "asset", "start_time", "end_time",
                                  "outcome", "slug") if c in columns]
            for batch in read_parquet(path, wanted, self.batch_size):
                for row in batch:
                    asset = str(row.get("asset") or "").strip().upper()
                    if self.assets and asset not in self.assets:
                        continue
                    start = to_epoch(row.get("start_time"))
                    end = to_epoch(row.get("end_time"))
                    if end is None:
                        continue
                    raw_slug = str(row.get("slug") or "")
                    raw_id = str(row.get("market_id") or "")
                    slug = raw_slug or (raw_id if "-updown-" in raw_id else "")
                    outcome = str(row.get("outcome") or "").strip().upper()

                    slot = key(asset, start, end)
                    existing = by_window.get(slot)
                    if existing is None:
                        if not slug and not outcome:
                            continue
                        by_window[slot] = Market(slug=slug, asset=asset,
                                                 start_ts=start or 0.0, end_ts=end,
                                                 outcome=outcome)
                    else:
                        if not existing.outcome and outcome:
                            by_window[slot] = Market(existing.slug, existing.asset,
                                                     existing.start_ts, existing.end_ts,
                                                     outcome)
                        elif not existing.slug and slug:
                            by_window[slot] = Market(slug, existing.asset,
                                                     existing.start_ts, existing.end_ts,
                                                     existing.outcome)

        self._markets = {
            m.slug: m for m in by_window.values() if m.slug and m.outcome
        }
        return self._markets

    # -- spot ------------------------------------------------------------
    def spot(self) -> dict[str, SpotIndex]:
        if self._spot is not None:
            return self._spot
        groups = self.files()
        if not groups["spot"]:
            raise DataError(
                "no crypto_prices files found (need timestamp, asset, close). "
                "Direction cannot be determined without them."
            )
        indexes: dict[str, SpotIndex] = {}
        for path in groups["spot"]:
            columns = parquet_columns(path)
            wanted = [c for c in ("timestamp", "asset", "close") if c in columns]
            for batch in read_parquet(path, wanted, self.batch_size):
                for row in batch:
                    asset = str(row.get("asset") or "").strip().upper()
                    if self.assets and asset not in self.assets:
                        continue
                    index = indexes.setdefault(asset, SpotIndex(asset))
                    index.add(to_epoch(row.get("timestamp")), row.get("close"))
        for index in indexes.values():
            index.finalise()
        self._spot = indexes
        return indexes

    # -- trades ----------------------------------------------------------
    def pick_trades_file(self, requested: str = "auto") -> str:
        """Choose the trade export to use, preferring the richest and largest."""
        groups = self.files()
        if requested and requested != "auto":
            path = requested if os.path.isabs(requested) else os.path.join(
                self.data_dir, requested)
            if not os.path.exists(path):
                raise DataError(f"trades file not found: {path}")
            return path
        if not groups["trades"]:
            raise DataError(
                "no trades files found (need timestamp, market_id, token_id, price)"
            )
        ranked = []
        for path in groups["trades"]:
            columns = set(parquet_columns(path))
            ranked.append(("side" in columns, os.path.getsize(path), path))
        ranked.sort(reverse=True)
        return ranked[0][2]

    def load_trades(self, path: str, *, max_markets: int = 0) -> dict[str, MarketTrades]:
        """slug -> MarketTrades, streamed and grouped."""
        columns = parquet_columns(path)
        wanted = [c for c in ("timestamp", "asset", "market_id", "token_id", "price")
                  if c in columns]
        missing = {"timestamp", "market_id", "token_id", "price"} - set(wanted)
        if missing:
            raise DataError(f"{os.path.basename(path)} is missing columns: {sorted(missing)}")

        out: dict[str, MarketTrades] = {}
        for batch in read_parquet(path, wanted, self.batch_size):
            for row in batch:
                asset = str(row.get("asset") or "").strip().upper()
                if self.assets and asset not in self.assets:
                    continue
                slug = str(row.get("market_id") or "")
                token = str(row.get("token_id") or "")
                ts = to_epoch(row.get("timestamp"))
                price = row.get("price")
                if not slug or not token or ts is None or price is None:
                    continue
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue
                if not 0.0 < price < 1.0:
                    continue
                entry = out.get(slug)
                if entry is None:
                    if max_markets and len(out) >= max_markets:
                        continue
                    entry = MarketTrades(slug=slug)
                    out[slug] = entry
                entry.by_token.setdefault(token, []).append((ts, price))

        for entry in out.values():
            for token in entry.by_token:
                entry.by_token[token].sort(key=lambda item: item[0])
        return out

    # -- summary ---------------------------------------------------------
    def describe(self) -> dict:
        groups = self.files()
        return {
            "data_dir": self.data_dir,
            "files": {name: len(paths) for name, paths in groups.items()},
            "markets_with_outcome": len(self.markets()),
            "spot_assets": {asset: len(index) for asset, index in self.spot().items()},
        }


def classify_tokens(trades: MarketTrades, market: Market, *, late_s: float = 120.0,
                    min_gap: float = 0.05) -> tuple[str, str] | None:
    """Work out which token is UP and which is DOWN.

    The trade export carries no side label, so the assignment is inferred from
    late prices (the winning token converges to 1.0). Returns ``None`` when the
    two tokens are too close to call - a market that cannot be labelled must be
    skipped, never guessed at.
    """
    late: dict[str, float] = {}
    for token in trades.tokens():
        mean = trades.mean_price_between(token, market.end_ts - late_s, market.end_ts)
        if mean is not None:
            late[token] = mean
    if len(late) < 2:
        return None
    ranked = sorted(late.items(), key=lambda item: item[1], reverse=True)
    (top_token, top_price), (bottom_token, bottom_price) = ranked[0], ranked[1]
    if top_price - bottom_price < min_gap:
        return None
    return (top_token, bottom_token) if market.up_won else (bottom_token, top_token)

"""Schema-driven, memory-bounded loading and time alignment.

Everything here is written for a 1.88 GB folder on a laptop:

* **Chunked reads.** CSV/JSONL are streamed in bounded batches; parquet is read
  row-group by row-group (or with ``pandas.read_parquet`` when available). No
  function in this module ever loads a whole file into memory.
* **Column mapping by detection, not assumption.** Real exports disagree on names
  (``timestamp`` / ``t`` / ``ts`` / ``open_time``; ``best_bid`` / ``bid`` /
  ``bids``). :func:`detect_mapping` finds them and *reports what it picked*, so a
  wrong guess is visible instead of silently producing empty columns.
* **Timestamp units normalised.** Seconds / milliseconds / microseconds /
  nanoseconds and ISO-8601 strings all collapse to epoch seconds via
  :func:`normalize_timestamp_column`, using the same magnitude heuristic the
  audit tool reports.
* **As-of alignment.** Order-book rows are matched to the most recent spot price
  at or before their timestamp - never a forward-looking match, which would
  invent an edge that did not exist in real time.

Pure standard library for CSV/JSONL (works anywhere). ``pandas``/``pyarrow`` are
used only if you have them and want parquet support.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Sequence

# --------------------------------------------------------------------- helpers
SLUG_EPOCH_RE = re.compile(r"(\d{10})")


def iso(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class DataError(RuntimeError):
    """Raised when the data cannot be interpreted safely."""


# ------------------------------------------------------------- time handling
def detect_epoch_divisor(values: Iterable[Any]) -> tuple[str, float]:
    """Infer the timestamp unit from magnitudes. Returns (name, divisor)."""
    magnitudes = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            magnitudes.append(abs(float(value)))
        except (TypeError, ValueError):
            continue
    if not magnitudes:
        return "unknown", 1.0
    peak = max(magnitudes)
    if peak < 1e11:
        return "seconds", 1.0
    if peak < 1e14:
        return "milliseconds", 1e3
    if peak < 1e17:
        return "microseconds", 1e6
    return "nanoseconds", 1e9


def to_epoch_seconds(value: Any, divisor: float = 1.0) -> float | None:
    """One cell -> epoch seconds, accepting ISO-8601 strings too."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
            try:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.timestamp()
        value = float(text)
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return None


def normalize_timestamp_column(
    rows: Sequence[dict], column: str, divisor: float | None = None
) -> tuple[list[float | None], str, float]:
    """Normalise a timestamp column across ``rows``; auto-detects the unit."""
    raw = [row.get(column) for row in rows]
    if divisor is None:
        unit, divisor = detect_epoch_divisor(raw)
    else:
        unit = "explicit"
    return [to_epoch_seconds(value, divisor) for value in raw], unit, divisor


# ---------------------------------------------------------- column detection
@dataclass
class ColumnMapping:
    """Which source column plays which role, and how confident we are."""

    timestamp: str | None = None
    slug: str | None = None
    outcome: str | None = None
    best_bid: str | None = None
    best_ask: str | None = None
    bid_size: str | None = None
    ask_size: str | None = None
    bids: str | None = None
    asks: str | None = None
    price: str | None = None
    size: str | None = None
    side: str | None = None
    close: str | None = None
    open: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "slug": self.slug,
            "outcome": self.outcome,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "bids": self.bids,
            "asks": self.asks,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "close": self.close,
            "open": self.open,
            "notes": list(self.notes),
        }


#: ordered candidate names per role - first hit wins
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "timestamp_ms", "timestamp_us", "ts", "time", "open_time",
                  "t", "datetime", "date", "created_at", "snapshot_ts", "start_time",
                  "end_time", "resolved_at"),
    "slug": ("market_slug", "slug", "market_id", "condition_id", "token_id", "asset_id",
             "market", "event_slug"),
    "outcome": ("outcome", "side_outcome", "winner", "result", "label", "direction"),
    "best_bid": ("best_bid", "bid", "top_bid", "bestbid", "bid_price", "bid_px"),
    "best_ask": ("best_ask", "ask", "top_ask", "bestask", "ask_price", "ask_px"),
    "bid_size": ("bid_size", "best_bid_size", "bid_qty", "bid_amount", "bidsize"),
    "ask_size": ("ask_size", "best_ask_size", "ask_qty", "ask_amount", "asksize"),
    # 'bid_levels'/'ask_levels' is what the DuckDB exports in this dataset use
    "bids": ("bids", "bid_levels", "buy_levels", "bids_json", "bid_ladder"),
    "asks": ("asks", "ask_levels", "sell_levels", "asks_json", "ask_ladder"),
    "price": ("price", "p", "trade_price", "last_price", "matched_price", "close",
              "mid_price"),
    "size": ("size", "qty", "quantity", "amount", "shares", "volume"),
    "side": ("side", "taker_side", "aggressor", "direction"),
    "close": ("close", "close_price", "c", "last", "mark_price"),
    "open": ("open", "open_price", "o", "first"),
}


def detect_mapping(columns: Sequence[str], *, kind: str = "generic") -> ColumnMapping:
    """Pick source columns for each role. ``kind`` tweaks the preference order."""
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    mapping = ColumnMapping()

    def take(role: str) -> str | None:
        for candidate in _CANDIDATES[role]:
            if candidate in lowered:
                return lowered[candidate]
        return None

    for role in _CANDIDATES:
        setattr(mapping, role, take(role))

    # disambiguate: a single 'price' column should not double as 'close'
    if mapping.close and mapping.close == mapping.price:
        mapping.close = None
    # a 'side' column that only ever holds up/down is the outcome column
    if mapping.outcome is None and mapping.side:
        mapping.outcome = mapping.side
        mapping.side = None

    missing = [
        role
        for role in ("timestamp", "best_bid", "best_ask", "bids", "asks")
        if getattr(mapping, role) is None
    ]
    if mapping.best_bid is None and mapping.bids is None:
        mapping.notes.append("no bid column found (neither best_bid nor a bids ladder)")
    if mapping.best_ask is None and mapping.asks is None:
        mapping.notes.append("no ask column found (neither best_ask nor an asks ladder)")

    if kind == "spot":
        # first close-like column wins over generic price
        if mapping.close is None:
            for candidate in ("close", "c", "last", "price", "p"):
                if candidate in lowered:
                    mapping.close = lowered[candidate]
                    break
    return mapping


def parse_ladder(value: Any) -> list[tuple[float, float]]:
    """Parse a ladder cell into [(price, size), ...].

    Handles the shapes seen in the wild: a JSON string, bytes (BSON/UTF-8), an
    already-decoded list, a dict with ``price``/``size`` keys, or a
    ``{"0.52": 100}`` price->size map.
    """
    if value in (None, ""):
        return []
    data = value
    # DuckDB writes JSON-ish list columns as BSON, which arrives as bytes
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        for candidate in (raw, raw[4:] if len(raw) > 4 else raw):
            try:
                data = json.loads(candidate.decode("utf-8", "replace"))
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        else:
            data = raw.decode("utf-8", "replace")
    if isinstance(data, str):
        text = value.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # "0.52:100|0.53:50" style
            out: list[tuple[float, float]] = []
            for part in re.split(r"[|;]", text):
                bits = re.split(r"[:@,]", part)
                if len(bits) >= 2:
                    try:
                        out.append((float(bits[0]), float(bits[1])))
                    except ValueError:
                        continue
            return out
    if isinstance(data, dict):
        pairs: list[tuple[float, float]] = []
        for key, val in data.items():
            try:
                if isinstance(val, dict):
                    pairs.append((float(key), float(val.get("size", 0))))
                else:
                    pairs.append((float(key), float(val)))
            except (TypeError, ValueError):
                continue
        return pairs
    if isinstance(data, (list, tuple)):
        pairs = []
        for item in data:
            if isinstance(item, dict):
                price = item.get("price", item.get("p"))
                size = item.get("size", item.get("s", item.get("qty", 0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price, size = item[0], item[1]
            else:
                continue
            try:
                pairs.append((float(price), float(size)))
            except (TypeError, ValueError):
                continue
        return pairs
    return []


def top_of_book(
    row: dict, mapping: ColumnMapping
) -> tuple[float | None, float | None, float | None, float | None]:
    """(best_bid, bid_size, best_ask, ask_size) from either direct or ladder columns."""
    bid = ask = bid_size = ask_size = None

    if mapping.best_bid:
        bid = _float(row.get(mapping.best_bid))
    if mapping.best_ask:
        ask = _float(row.get(mapping.best_ask))
    if mapping.bid_size:
        bid_size = _float(row.get(mapping.bid_size))
    if mapping.ask_size:
        ask_size = _float(row.get(mapping.ask_size))

    if mapping.bids and (bid is None or bid_size is None):
        ladder = parse_ladder(row.get(mapping.bids))
        if ladder:
            best = max(ladder, key=lambda pair: pair[0])
            if bid is None:
                bid = best[0]
            if bid_size is None:
                bid_size = best[1]
    if mapping.asks and (ask is None or ask_size is None):
        ladder = parse_ladder(row.get(mapping.asks))
        if ladder:
            best = min(ladder, key=lambda pair: pair[0])
            if ask is None:
                ask = best[0]
            if ask_size is None:
                ask_size = best[1]
    return bid, bid_size, ask, ask_size


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- chunk reading
@dataclass
class ReadStats:
    rows_read: int = 0
    rows_emitted: int = 0
    chunks: int = 0
    bytes_read: int = 0

    def as_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_emitted": self.rows_emitted,
            "chunks": self.chunks,
            "bytes_read": self.bytes_read,
        }


def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    counts = {d: head.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def iter_rows(
    path: str,
    *,
    chunk_size: int = 200_000,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    file_format: str | None = None,
) -> Iterator[tuple[list[dict], ReadStats]]:
    """Yield bounded chunks of row dicts, whatever the format.

    Yields ``(rows, stats)`` where ``stats`` accumulates progress so callers can
    log/abort. Never materialises more than ``chunk_size`` rows.
    """
    fmt = (file_format or os.path.splitext(path)[1].lstrip(".")).lower()
    stats = ReadStats()
    if fmt in ("parquet", "pq"):
        yield from _iter_parquet(path, chunk_size, columns, max_rows, stats)
    elif fmt in ("jsonl", "ndjson", "json"):
        yield from _iter_jsonl(path, chunk_size, columns, max_rows, stats)
    else:
        yield from _iter_csv(path, chunk_size, columns, max_rows, stats)


def _iter_csv(path, chunk_size, columns, max_rows, stats) -> Iterator[tuple[list[dict], ReadStats]]:
    delimiter = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        buffer: list[dict] = []
        for row in reader:
            if columns:
                buffer.append({c: row.get(c) for c in columns})
            else:
                buffer.append(row)
            stats.rows_read += 1
            if len(buffer) >= chunk_size:
                buffer = _apply_max(buffer, stats, max_rows)
                stats.chunks += 1
                stats.rows_emitted += len(buffer)
                yield buffer, stats
                buffer = []
                if max_rows is not None and stats.rows_emitted >= max_rows:
                    return
        if buffer:
            buffer = _apply_max(buffer, stats, max_rows)
            stats.chunks += 1
            stats.rows_emitted += len(buffer)
            yield buffer, stats


def _apply_max(buffer: list[dict], stats: ReadStats, max_rows: int | None) -> list[dict]:
    if max_rows is None:
        return buffer
    remaining = max_rows - stats.rows_emitted
    return buffer[:remaining] if remaining < len(buffer) else buffer


def _iter_jsonl(path, chunk_size, columns, max_rows, stats) -> Iterator[tuple[list[dict], ReadStats]]:
    buffer: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if columns:
                row = {c: row.get(c) for c in columns}
            buffer.append(row)
            stats.rows_read += 1
            if len(buffer) >= chunk_size:
                buffer = _apply_max(buffer, stats, max_rows)
                stats.chunks += 1
                stats.rows_emitted += len(buffer)
                yield buffer, stats
                buffer = []
                if max_rows is not None and stats.rows_emitted >= max_rows:
                    return
    if buffer:
        buffer = _apply_max(buffer, stats, max_rows)
        stats.chunks += 1
        stats.rows_emitted += len(buffer)
        yield buffer, stats


def _iter_parquet(path, chunk_size, columns, max_rows, stats) -> Iterator[tuple[list[dict], ReadStats]]:
    """Row-group streaming. Prefers pyarrow, falls back to pandas chunks."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunk_size, columns=list(columns) if columns else None):
            rows = batch.to_pylist()
            stats.rows_read += len(rows)
            rows = _apply_max(rows, stats, max_rows)
            stats.chunks += 1
            stats.rows_emitted += len(rows)
            yield rows, stats
            if max_rows is not None and stats.rows_emitted >= max_rows:
                return
        return
    except ImportError:
        pass

    try:
        import pandas as pd  # type: ignore

        reader = pd.read_parquet(
            path, columns=list(columns) if columns else None, chunksize=chunk_size
        )
        for frame in reader:
            rows = frame.to_dict("records")
            stats.rows_read += len(rows)
            rows = _apply_max(rows, stats, max_rows)
            stats.chunks += 1
            stats.rows_emitted += len(rows)
            yield rows, stats
            if max_rows is not None and stats.rows_emitted >= max_rows:
                return
        return
    except ImportError as exc:
        raise DataError(
            "reading parquet needs pyarrow or pandas:\n"
            "  pip install pyarrow        # recommended, streams row groups\n"
            "  pip install pandas pyarrow # alternative\n"
            "CSV and JSONL need no extra packages."
        ) from exc


def peek(path: str, rows: int = 5, file_format: str | None = None) -> tuple[list[str], list[dict]]:
    """Cheap 'what is in this file' helper: (columns, first N rows)."""
    columns: list[str] = []
    sample: list[dict] = []
    for chunk, _stats in iter_rows(path, chunk_size=max(1, rows), max_rows=rows, file_format=file_format):
        for row in chunk:
            if not columns:
                columns = list(row.keys())
            if len(sample) < rows:
                sample.append(row)
        if len(sample) >= rows:
            break
    return columns, sample


# ------------------------------------------------------------------- features
def slug_window_start(slug: str) -> int | None:
    """Extract the window start epoch embedded in a Polymarket slug.

    Slugs look like ``btc-updown-5m-1780319100`` (5m/15m) - the trailing 10-digit
    number is the window start. Returns None when the pattern is absent (the 1h
    and 1d slugs are human-readable dates and need a different lookup).
    """
    if not slug:
        return None
    match = SLUG_EPOCH_RE.search(str(slug))
    return int(match.group(1)) if match else None


def infer_window_seconds(slug: str, default: int = 900) -> int:
    """5m / 15m / 1h windows are encoded in the slug text."""
    text = str(slug).lower()
    if "-5m" in text:
        return 300
    if "-15m" in text:
        return 900
    if "-1h" in text:
        return 3600
    if "-4h" in text:
        return 14400
    return default


@dataclass
class OrderBookRow:
    """One aligned observation: book level, spot context and clock position."""

    timestamp: float
    slug: str
    outcome: str
    best_bid: float | None
    bid_size: float | None
    best_ask: float | None
    ask_size: float | None
    midpoint: float | None
    spot_price: float | None
    spot_ts: float | None
    spot_change_pct: float | None
    window_start: float | None
    window_close: float | None
    seconds_to_close: float | None
    result: str | None = None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def as_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "timestamp_utc": iso(self.timestamp),
            "slug": self.slug,
            "outcome": self.outcome,
            "best_bid": self.best_bid,
            "bid_size": self.bid_size,
            "best_ask": self.best_ask,
            "ask_size": self.ask_size,
            "midpoint": None if self.midpoint is None else round(self.midpoint, 6),
            "spread": self.spread,
            "spot_price": self.spot_price,
            "spot_ts": self.spot_ts,
            "spot_change_pct": (
                None if self.spot_change_pct is None else round(self.spot_change_pct, 6)
            ),
            "window_start": self.window_start,
            "window_close": self.window_close,
            "seconds_to_close": self.seconds_to_close,
            "result": self.result,
        }


class SpotSeries:
    """In-memory spot price index supporting as-of (backward) lookups.

    Only the spot series is held in memory - it is one row per second at most,
    which is ~2.6M rows per symbol per month, fine for a laptop. If even that is
    too much, pass ``--spot-dir`` and read per window instead.
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self.timestamps: list[float] = []
        self.prices: list[float] = []

    def __len__(self) -> int:
        return len(self.timestamps)

    def add(self, timestamp: float | None, price: float | None) -> None:
        if timestamp is None or price is None:
            return
        self.timestamps.append(float(timestamp))
        self.prices.append(float(price))

    def finalize(self) -> None:
        """Sort and de-duplicate (keeps the last price per timestamp)."""
        if not self.timestamps:
            return
        pairs = sorted(zip(self.timestamps, self.prices), key=lambda pair: pair[0])
        deduped_ts: list[float] = []
        deduped_px: list[float] = []
        for timestamp, price in pairs:
            if deduped_ts and timestamp == deduped_ts[-1]:
                deduped_px[-1] = price
            else:
                deduped_ts.append(timestamp)
                deduped_px.append(price)
        self.timestamps, self.prices = deduped_ts, deduped_px

    def price_asof(self, timestamp: float, max_lag_s: float | None = None) -> tuple[float | None, float | None]:
        """Most recent price at or BEFORE ``timestamp`` (never forward-looking)."""
        if not self.timestamps:
            return None, None
        import bisect

        index = bisect.bisect_right(self.timestamps, timestamp) - 1
        if index < 0:
            return None, None
        spot_ts = self.timestamps[index]
        if max_lag_s is not None and (timestamp - spot_ts) > max_lag_s:
            return None, spot_ts
        return self.prices[index], spot_ts

    def change_pct(self, end_ts: float, lookback_s: float, max_lag_s: float | None = None) -> float | None:
        """Percent change over ``lookback_s`` ending at ``end_ts``."""
        now_price, now_ts = self.price_asof(end_ts, max_lag_s)
        past_price, past_ts = self.price_asof(end_ts - lookback_s, max_lag_s)
        if now_price is None or not past_price:
            return None
        return (now_price / past_price - 1.0) * 100.0


def build_spot_series(
    path: str,
    *,
    symbol_column: str | None = None,
    symbol_value: str | None = None,
    chunk_size: int = 200_000,
    max_rows: int | None = None,
    mapping: ColumnMapping | None = None,
) -> SpotSeries:
    """Load a crypto_prices file into a :class:`SpotSeries` (chunked, memory-bounded)."""
    columns, first = peek(path, rows=1)
    if not first:
        raise DataError(f"empty file: {path}")
    mapping = mapping or detect_mapping(columns, kind="spot")
    if not mapping.timestamp:
        raise DataError(f"no timestamp column recognised in {path}: {columns}")
    price_col = mapping.close or mapping.price
    if not price_col:
        raise DataError(f"no price column recognised in {path}: {columns}")

    series = SpotSeries(symbol=symbol_value or "")
    divisor: float | None = None
    for chunk, stats in iter_rows(path, chunk_size=chunk_size, max_rows=max_rows):
        if divisor is None:
            divisor = detect_epoch_divisor([row.get(mapping.timestamp) for row in chunk])[1]
        for row in chunk:
            if symbol_column and symbol_value:
                if str(row.get(symbol_column, "")).strip() != symbol_value:
                    continue
            ts = to_epoch_seconds(row.get(mapping.timestamp), divisor)
            price = _float(row.get(price_col))
            series.add(ts, price)
    series.finalize()
    return series


def align_book_with_spot(
    book_path: str,
    spot: SpotSeries,
    *,
    book_mapping: ColumnMapping | None = None,
    lookback_s: float = 3.0,
    default_window_s: int = 900,
    results: dict[str, str] | None = None,
    chunk_size: int = 200_000,
    max_rows: int | None = None,
    max_spot_lag_s: float | None = 5.0,
    outcomes: Sequence[str] | None = None,
    on_progress: Callable[[ReadStats], None] | None = None,
) -> Iterator[list[OrderBookRow]]:
    """Stream aligned observations from an orderbook file.

    Yields lists of :class:`OrderBookRow` (bounded by ``chunk_size``), so the
    caller can analyse or write them without holding the dataset in memory.
    """
    columns, first = peek(book_path, rows=1)
    if not first:
        raise DataError(f"empty file: {book_path}")
    mapping = book_mapping or detect_mapping(columns, kind="book")
    if not mapping.timestamp:
        raise DataError(f"no timestamp column recognised in {book_path}: {columns}")
    if not (mapping.best_bid or mapping.bids) or not (mapping.best_ask or mapping.asks):
        raise DataError(
            f"orderbook file {book_path} has neither top-of-book nor ladder columns; "
            f"detected mapping: {mapping.as_dict()}"
        )

    wanted_outcomes = {o.lower() for o in outcomes} if outcomes else None
    divisor: float | None = None

    for chunk, stats in iter_rows(book_path, chunk_size=chunk_size, max_rows=max_rows):
        if divisor is None:
            divisor = detect_epoch_divisor([row.get(mapping.timestamp) for row in chunk])[1]
        aligned: list[OrderBookRow] = []
        for row in chunk:
            ts = to_epoch_seconds(row.get(mapping.timestamp), divisor)
            if ts is None:
                continue
            slug = str(row.get(mapping.slug, "")) if mapping.slug else ""
            outcome = str(row.get(mapping.outcome, "")).strip() if mapping.outcome else ""
            if wanted_outcomes and outcome.lower() not in wanted_outcomes:
                continue

            bid, bid_size, ask, ask_size = top_of_book(row, mapping)
            if bid is None and ask is None:
                continue
            # a crossed or impossible book is data corruption, not a signal
            if bid is not None and (bid <= 0 or bid >= 1):
                bid = None
            if ask is not None and (ask <= 0 or ask >= 1):
                ask = None
            if bid is None and ask is None:
                continue

            midpoint = (
                (bid + ask) / 2.0 if bid is not None and ask is not None else (bid or ask)
            )
            spot_price, spot_ts = spot.price_asof(ts, max_spot_lag_s)
            spot_change = (
                spot.change_pct(ts, lookback_s, max_spot_lag_s) if len(spot) else None
            )

            window_start = slug_window_start(slug)
            window_close = None
            if window_start is not None:
                window_close = float(window_start + infer_window_seconds(slug, default_window_s))

            aligned.append(
                OrderBookRow(
                    timestamp=ts,
                    slug=slug,
                    outcome=outcome,
                    best_bid=bid,
                    bid_size=bid_size,
                    best_ask=ask,
                    ask_size=ask_size,
                    midpoint=midpoint,
                    spot_price=spot_price,
                    spot_ts=spot_ts,
                    spot_change_pct=spot_change,
                    window_start=float(window_start) if window_start else None,
                    window_close=window_close,
                    seconds_to_close=(window_close - ts) if window_close else None,
                    result=(results or {}).get(slug),
                )
            )
        if on_progress is not None:
            on_progress(stats)
        if aligned:
            yield aligned


def write_joint_csv(path: str, rows: Iterable[OrderBookRow]) -> int:
    """Write aligned rows to CSV; returns the number written."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fields = [
        "timestamp", "timestamp_utc", "slug", "outcome", "best_bid", "bid_size",
        "best_ask", "ask_size", "midpoint", "spread", "spot_price", "spot_ts",
        "spot_change_pct", "window_start", "window_close", "seconds_to_close", "result",
    ]
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
            count += 1
    return count


def load_results_from_markets_csv(path: str) -> dict[str, str]:
    """Optional: slug -> settlement outcome, to label historical windows."""
    if not path or not os.path.exists(path):
        return {}
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        lowered = {str(c).lower(): c for c in (reader.fieldnames or [])}
        slug_col = lowered.get("slug") or lowered.get("market_slug")
        result_col = lowered.get("result") or lowered.get("winning_outcome") or lowered.get("outcome")
        if not slug_col or not result_col:
            return {}
        for row in reader:
            slug = str(row.get(slug_col, "")).strip()
            if slug:
                mapping[slug] = str(row.get(result_col, "")).strip().upper()
    return mapping

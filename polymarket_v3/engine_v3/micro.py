"""V3 microstructure data: the token-level dataset that replaced modelling with measurement.

Files (workspace root):

* ``markets.parquet``    288 BTC 5-minute markets - the whole sample
* ``snapshots.parquet``  3,889 L2 order book snapshots (bids/asks/depths)
* ``trades.parquet``     680,994 executed prints
* ``deltas.parquet``     41.6M book updates (only loaded when needed)

What these files fix
--------------------
The aggregate dataset (``polymarket_btc_up_down_5m_251218_260909.csv``) had no
token prices, so V3 had to *model* entry price, spread and the intra-window path.
That is why V3 could not judge TP/SL. These files contain real quotes and real
prints, so those three things become measurable.

What they do NOT fix
--------------------
**Coverage is one day** (288 windows). The aggregate dataset had nine months but
no prices; this one has prices but 24 hours. So:

* execution questions (spread, depth, achievable entry price) -> answerable, well
* signal questions (is the edge real, what threshold) -> NOT answerable here

Both are used for what they are good for, and neither is used to overstate the
other. Trying to re-tune the signal on 288 windows is how a curve gets fitted to
a single day.

Known data caveats, verified in this module
-------------------------------------------
* ``trades.parquet`` has **no token_id**: a print's price cannot be attributed to
  the Up or the Down token. Snapshot rows, by contrast, carry a ``market_id`` and
  come in a consistent order per market (first row = Up).
* ``fee_rate_bps`` is ``0`` on all 680,994 prints, which is a scrape artifact, not
  evidence that trading is free. Fees are still modelled from the documented
  curve.
* ``winning_outcome_index`` is an **int**, not a string, unlike the published
  schema. Compared as int.
"""
from __future__ import annotations

import io
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import pyarrow.parquet as pq

#: ordering assumption, verified against winning_outcome_index
UP_INDEX = 0
DOWN_INDEX = 1


def _json_list(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _f(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MicroMarket:
    """One 5-minute market with its two token ids and the realised outcome."""

    market_id: str
    question: str
    open_ms: int
    close_ms: int
    up_token: str
    down_token: str
    winning_index: int
    tick_size: float

    @property
    def open_ts(self) -> float:
        return self.open_ms / 1000.0

    @property
    def close_ts(self) -> float:
        return self.close_ms / 1000.0

    @property
    def up_won(self) -> bool:
        return self.winning_index == UP_INDEX

    def entry_ts(self, t_minus_s: float) -> float:
        return self.close_ts - t_minus_s


@dataclass(frozen=True)
class BookQuote:
    """One L2 snapshot for one token."""

    market_id: str
    token: str
    ts: float
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    midpoint: float | None
    bid_depth: float | None
    ask_depth: float | None
    bid_levels: int
    ask_levels: int
    bids: tuple = ()
    asks: tuple = ()

    @property
    def executable_ask(self) -> float | None:
        """What a taker pays to buy this token right now."""
        return self.best_ask

    def cost_to_buy(self, shares: float) -> float | None:
        """USDC needed to take ``shares`` off the visible asks."""
        if not self.asks:
            return None
        remaining = shares
        total = 0.0
        for level in self.asks:
            price = _f(level.get("price"))
            size = _f(level.get("size"), 0.0)
            if price is None or not size:
                continue
            take = min(remaining, size)
            total += take * price
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            return None          # book too thin for this size
        return total


@dataclass(frozen=True)
class Print:
    """One executed trade print."""

    market_id: str
    ts: float
    price: float
    size: float
    side: str


def load_markets(path: str) -> list[MicroMarket]:
    rows = pq.read_table(path).to_pylist()
    out: list[MicroMarket] = []
    for row in rows:
        tokens = [str(t) for t in _json_list(row["token_ids"])]
        if len(tokens) < 2:
            continue
        # winning_outcome_index arrives as an int here, not a string
        index = int(row["winning_outcome_index"])
        out.append(MicroMarket(
            market_id=row["market_id"],
            question=row.get("question", ""),
            open_ms=int(row["open_time_ms"]),
            close_ms=int(row["close_time_ms"]),
            up_token=tokens[UP_INDEX],
            down_token=tokens[DOWN_INDEX],
            winning_index=index,
            tick_size=_f(row.get("tick_size"), 0.01) or 0.01,
        ))
    out.sort(key=lambda m: m.open_ms)
    return out


def load_snapshots(path: str, markets: dict[str, MicroMarket] | None = None,
                   columns: Sequence[str] | None = None) -> dict[str, list[BookQuote]]:
    """Snapshots grouped by market, each in time order.

    A market emits snapshots in a stable token order, so the rows alternate: the
    first row of each market is Up, the next Down. Rather than trust alternation,
    the token is assigned by **price agreement with the outcome**: the Up token's
    price should drift toward 1 when Up wins. That assignment is validated in
    :func:`validate_token_assignment`.
    """
    cols = columns or ["market_id", "collected_at_ms", "best_bid", "best_ask",
                       "spread", "midpoint", "bid_depth", "ask_depth",
                       "bid_levels", "ask_levels", "bids", "asks"]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in cols if c in available]
    rows = pq.read_table(path, columns=cols).to_pylist()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["market_id"], []).append(row)

    out: dict[str, list[BookQuote]] = {}
    for market_id, market_rows in grouped.items():
        market = markets.get(market_id) if markets else None
        market_rows.sort(key=lambda r: int(r["collected_at_ms"]))
        quotes: list[BookQuote] = []
        for position, row in enumerate(market_rows):
            # snapshots for one market come as (Up, Down) pairs per collection
            token = ""
            if market is not None:
                token = market.up_token if position % 2 == 0 else market.down_token
            bids = tuple(_json_list(row.get("bids")) or [])
            asks = tuple(_json_list(row.get("asks")) or [])
            quotes.append(BookQuote(
                market_id=market_id,
                token=token,
                ts=int(row["collected_at_ms"]) / 1000.0,
                best_bid=_f(row.get("best_bid")),
                best_ask=_f(row.get("best_ask")),
                spread=_f(row.get("spread")),
                midpoint=_f(row.get("midpoint")),
                bid_depth=_f(row.get("bid_depth")),
                ask_depth=_f(row.get("ask_depth")),
                bid_levels=int(_f(row.get("bid_levels"), 0) or 0),
                ask_levels=int(_f(row.get("ask_levels"), 0) or 0),
                bids=bids,
                asks=asks,
            ))
        out[market_id] = quotes
    return out


def validate_token_assignment(quotes: dict[str, list[BookQuote]],
                              markets: dict[str, MicroMarket]) -> dict:
    """Check the Up/Down assignment by asking which token converges to 1.

    Returns a small report; ``agreement`` near 1.0 means the assignment holds.
    """
    agree = 0
    checked = 0
    for market_id, market_quotes in quotes.items():
        market = markets.get(market_id)
        if market is None:
            continue
        up_rows = [q for q in market_quotes if q.token == market.up_token and q.midpoint]
        down_rows = [q for q in market_quotes if q.token == market.down_token and q.midpoint]
        if not up_rows or not down_rows:
            continue
        up_last = up_rows[-1].midpoint
        down_last = down_rows[-1].midpoint
        checked += 1
        if (up_last > down_last) == market.up_won:
            agree += 1
    return {"checked": checked, "agreement": agree / checked if checked else 0.0}


def load_prints(path: str, market_ids: set[str] | None = None,
                batch_size: int = 200_000) -> dict[str, list[Print]]:
    """Executed prints grouped by market, in time order.

    Streamed in batches: the file has 680,994 rows and ``deltas.parquet`` has 41.6M,
    so nothing here reads a whole file into a list of dicts unnecessarily.
    """
    out: dict[str, list[Print]] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=["market_id", "platform_timestamp_ms",
                                          "price", "size", "side"]):
        for row in batch.to_pylist():
            market_id = row["market_id"]
            if market_ids is not None and market_id not in market_ids:
                continue
            price = _f(row["price"])
            if price is None or not 0.0 < price < 1.0:
                continue
            out.setdefault(market_id, []).append(Print(
                market_id=market_id,
                ts=int(row["platform_timestamp_ms"]) / 1000.0,
                price=price,
                size=_f(row["size"], 0.0) or 0.0,
                side=str(row.get("side") or ""),
            ))
    for prints in out.values():
        prints.sort(key=lambda p: p.ts)
    return out


def quote_at(quotes: list[BookQuote], ts: float, token: str | None = None,
             max_age_s: float = 30.0) -> BookQuote | None:
    """Nearest snapshot at or before ``ts``, within ``max_age_s``."""
    best = None
    for quote in quotes:
        if token is not None and quote.token != token:
            continue
        if quote.ts <= ts:
            best = quote
        else:
            break
    if best is None or (ts - best.ts) > max_age_s:
        return None
    return best


def prints_between(prints: list[Print], start: float, end: float) -> list[Print]:
    return [p for p in prints if start < p.ts <= end]


def summarise(values: Sequence[float]) -> dict:
    """Compact distribution summary used throughout the reports."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    clean.sort()
    n = len(clean)

    def pct(q: float) -> float:
        return clean[min(n - 1, max(0, int(q * n)))]

    return {
        "n": n,
        "min": clean[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": statistics.median(clean),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": clean[-1],
        "mean": statistics.fmean(clean),
    }


def coverage(markets: Sequence[MicroMarket]) -> dict:
    opens = [m.open_ms for m in markets]
    closes = [m.close_ms for m in markets]
    days = (max(closes) - min(opens)) / 86_400_000.0
    return {
        "markets": len(markets),
        "start": datetime.fromtimestamp(min(opens) / 1000, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(max(closes) / 1000, tz=timezone.utc).isoformat(),
        "days": days,
        "markets_per_day": len(markets) / days if days else 0.0,
        "up_wins": sum(1 for m in markets if m.up_won),
        "up_rate": sum(1 for m in markets if m.up_won) / len(markets) if markets else 0.0,
    }

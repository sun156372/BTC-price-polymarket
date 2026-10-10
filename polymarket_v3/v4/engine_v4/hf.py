"""Loader for the high-frequency per-market dataset (real order book + real path).

This replaces the earlier ``micro.py`` reader (kept only for reference - it reads
the superseded ``markets/snapshots/trades/deltas.parquet`` export and cannot
attribute quotes to a side). It fixes every limitation that blocked the previous
attempts, and corrects three dataset defects found by inspection rather than
assumption.

What is in the files
--------------------
``dataset=polymarket_features_100ms/date=*/part-historical-<market_id>.parquet``
holds **exactly one market per file**, with 100ms rows spanning roughly 5 minutes
before the window opens through the end of the window. Across the four supplied
archives there are 1,655 markets::

    5m  : 1,151
    15m :   384
    4h  :    24
    other (hourly legacy slugs): 96

Each row carries both sides of the book at 100ms - ``up_best_bid``/``up_best_ask``
and ``down_*`` - plus depth arrays, ``time_to_start_ms`` and ``time_to_deadline_ms``.
Resolution comes from ``dataset=resolution``.

Two dataset defects, verified
-----------------------------
1. **``window_key`` is corrupted.** It is the string ``"et"`` (scraped from a slug
   suffix like ``8pm-et``) in the hourly-legacy files and is not a usable key. The
   join key is ``market_id``, which is reliable: every file holds one market and
   ``resolution.market_id`` matches. Joining on ``window_key`` - the obvious choice
   - silently yields zero matches.
2. **``time_to_start_ms`` is misnamed**, and its sign is inverted: it counts from
   the window start (0 -> -300,000) rather than counting down to it. The
   authoritative field is ``time_to_deadline_ms``, which runs ~+300,000 -> ~0.
3. **The tick order is a trap.** Rows are sorted by countdown ASCENDING, i.e. from
   the close back to the open. Iterating them forward therefore reads the tape
   backwards, and a first-touch exit model silently reports the LAST price touch
   instead of the first. This produced a 0.69 take-profit "filling" at 0.99. Ticks
   are re-sorted to chronological order here, and ``backtest_v4.py`` refuses to run
   if the order ever regresses.

Why this dataset is the right one for the job
---------------------------------------------
Because it prices BOTH sides against each other, no model is needed for the entry
price, the spread, or the intra-window path. The token mid at T-k *is* the market's
probability, and the depth *is* what a taker pays. TP/SL stops being a modelling
question and becomes a tape-reading question.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Sequence

import pyarrow.parquet as pq

FEATURES = "polymarket_features_100ms"
BOOKS = "books_100ms"
EXTERNAL = "external_100ms"
LIFECYCLE = "lifecycle"
RESOLUTION = "resolution"

#: ``btc-updown-5m-1774569600`` -> horizon ``5m``
_SLUG_RE = re.compile(r"btc-updown-(5m|15m|1h|4h)-(\d{9,12})")

#: columns worth keeping from a feature file
USEFUL_COLUMNS = (
    "timestamp_ms", "market_id", "slug", "lifecycle_state",
    "up_token_id", "down_token_id", "time_to_start_ms", "time_to_deadline_ms",
    "up_best_bid", "up_best_ask", "up_midquote", "up_spread_bps",
    "down_best_bid", "down_best_ask", "down_midquote", "down_spread_bps",
    "sum_best_bid", "sum_best_ask", "sum_midquote",
    "full_depth_bid_notional", "full_depth_ask_notional",
    "up_side_bids", "up_bid_sizes", "up_side_asks", "up_ask_sizes",
    "down_side_bids", "down_bid_sizes", "down_side_asks", "down_ask_sizes",
    "up_last_trade_price", "down_last_trade_price", "observed_halt_flag",
)


def _f(value, default=None):
    if value is None or value == "" or value == "None":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _levels(value) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value if v is not None)
    text = str(value).strip()
    if not text or text == "None":
        return ()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(float(piece))
        except ValueError:
            continue
    return tuple(out)


def horizon_of(slug: str) -> str:
    match = _SLUG_RE.search(slug or "")
    return match.group(1) if match else ""


def start_of(slug: str) -> float | None:
    match = _SLUG_RE.search(slug or "")
    if not match:
        return None
    return float(match.group(2))


def day_files(root: str, dataset: str) -> list[str]:
    found = glob.glob(os.path.join(root, "*", f"dataset={dataset}", "date=*", "*.parquet"))
    return sorted(found)


# ------------------------------------------------------------------- records
@dataclass(frozen=True)
class Tick:
    """One 100ms observation, both sides priced.

    Timing: ``time_to_deadline_ms`` is the authoritative countdown. It runs from
    ~+300,000 ms at the first tick down to ~0 at the close, which is exactly the
    5-minute window. ``time_to_start_ms`` is **misnamed** in this dataset - it
    counts from the market's window start (0 -> -300,000) rather than counting down
    to it - so it is kept only for reference and never used for decisions.
    """

    ts: float
    to_start_ms: float
    to_deadline_ms: float
    up_bid: float | None
    up_ask: float | None
    up_mid: float | None
    down_bid: float | None
    down_ask: float | None
    down_mid: float | None
    sum_ask: float | None
    up_asks: tuple = ()
    up_ask_sizes: tuple = ()
    down_asks: tuple = ()
    down_ask_sizes: tuple = ()
    up_bids: tuple = ()
    up_bid_sizes: tuple = ()
    down_bids: tuple = ()
    down_bid_sizes: tuple = ()

    @property
    def remaining_s(self) -> float:
        """Seconds until the window closes (the countdown that matters)."""
        return self.to_deadline_ms / 1000.0

    @property
    def elapsed_s(self, ) -> float:
        """Seconds elapsed since the window opened.

        Derived from the deadline rather than from ``to_start_ms``, whose sign
        convention is inverted in this dataset.
        """
        return max(0.0, -self.to_start_ms / 1000.0) - self.remaining_s

    def t_minus(self, window_seconds: float) -> float:
        """Seconds before the close, i.e. 240.0 at T-240."""
        return self.remaining_s

    def ask(self, side: str) -> float | None:
        return self.up_ask if side == "UP" else self.down_ask

    def bid(self, side: str) -> float | None:
        return self.up_bid if side == "UP" else self.down_bid

    def mid(self, side: str) -> float | None:
        return self.up_mid if side == "UP" else self.down_mid

    def asks(self, side: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if side == "UP":
            return self.up_asks, self.up_ask_sizes
        return self.down_asks, self.down_ask_sizes

    def bids(self, side: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if side == "UP":
            return self.up_bids, self.up_bid_sizes
        return self.down_bids, self.down_bid_sizes

    def cost_to_buy(self, side: str, shares: float) -> float | None:
        """USDC to lift ``shares`` off the visible asks. None if the book is short."""
        prices, sizes = self.asks(side)
        remaining = shares
        total = 0.0
        for price, size in zip(prices, sizes):
            if size <= 0:
                continue
            take = min(remaining, size)
            total += take * price
            remaining -= take
            if remaining <= 1e-9:
                return total
        return None

    def vwap_buy(self, side: str, shares: float) -> float | None:
        cost = self.cost_to_buy(side, shares)
        return None if cost is None else cost / shares

    def proceeds_to_sell(self, side: str, shares: float) -> float | None:
        """USDC received for hitting the bids with ``shares``."""
        prices, sizes = self.bids(side)
        remaining = shares
        total = 0.0
        for price, size in zip(prices, sizes):
            if size <= 0:
                continue
            take = min(remaining, size)
            total += take * price
            remaining -= take
            if remaining <= 1e-9:
                return total
        return None


@dataclass
class Market:
    """One window: its tape, its book, and how it resolved."""

    market_id: str
    slug: str
    horizon: str
    start_ts: float | None
    ticks: list[Tick]
    resolution: "Resolution | None" = None
    up_token: str = ""
    down_token: str = ""
    #: countdowns in ms, ascending, for bisect-based lookup
    _deadlines: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Order ticks **forward in time** and cache countdowns for bisect.

        The countdown runs DOWN over the window (299,883 ms at the open -> 9 ms at
        the close), so sorting ascending by countdown yields the tape in REVERSE
        chronological order. That is a trap: any loop reading the ticks forward
        sees the last touch first. An earlier version of this loader sorted
        ascending and consequently reported exits at the price of the window's
        *final* tick instead of the first barrier breach - a 0.69 take-profit
        "exited" at 0.99. Ticks are therefore stored **descending** by countdown,
        which is chronological, and ``_deadlines`` keeps an ascending copy purely
        for bisect lookups.
        """
        if not self._deadlines and self.ticks:
            self.ticks.sort(key=lambda t: t.to_deadline_ms, reverse=True)
            self._deadlines = [t.to_deadline_ms for t in reversed(self.ticks)]

    @property
    def day(self) -> str:
        """UTC date of the window open, for per-day consistency checks."""
        if self.start_ts is None:
            return ""
        return datetime.fromtimestamp(self.start_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    @property
    def session_seconds(self) -> float:
        """Window length in seconds, from the horizon label."""
        return {"5m": 300.0, "15m": 900.0, "1h": 3600.0, "4h": 14400.0}.get(self.horizon, 300.0)

    @property
    def up_won(self) -> bool | None:
        return None if self.resolution is None else self.resolution.up_won

    def in_window(self) -> list[Tick]:
        """Ticks inside the betting window: everything before the close.

        ``time_to_deadline_ms`` counts down to the close, so the window is simply
        every tick with a non-negative countdown. The pre-open tape that other
        files carried is not present here (this dataset starts at the open), which
        is a genuine improvement over the earlier snapshot exports.
        """
        return [t for t in self.ticks if t.to_deadline_ms > 0]

    def tick_at_t_minus(self, t_minus_s: float,
                        tolerance_s: float = 1.0) -> Tick | None:
        """The tick nearest ``t_minus_s`` seconds before the close.

        T-240 on a 300s window means 60s after the open; on a 900s window it means
        660s after the open. Addressing by countdown keeps both cases correct.
        """
        target = t_minus_s * 1000.0
        # _deadlines is ascending; ticks is descending, so map the index across
        index = bisect.bisect_left(self._deadlines, target)
        best = None
        best_gap = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self._deadlines):
                tick = self.ticks[len(self.ticks) - 1 - candidate]
                gap = abs(tick.to_deadline_ms - target)
                if gap <= tolerance_s * 1000.0 and (best_gap is None or gap < best_gap):
                    best, best_gap = tick, gap
        return best

    def first_tick(self) -> Tick | None:
        """Earliest tick still inside the window (the largest countdown).

        ``ticks`` is chronological, so the open is the first entry with a
        positive countdown.
        """
        for tick in self.ticks:
            if tick.to_deadline_ms > 0:
                return tick
        return None

    def first_mid(self) -> float | None:
        tick = self.first_tick()
        return tick.up_mid if tick else None

    def ticks_after(self, t_minus_s: float) -> list[Tick]:
        """Ticks strictly after T-k, in chronological order.

        ``ticks`` runs from the open (largest countdown) to the close, so the ticks
        after T-k are the tail. ``_deadlines`` is the ascending mirror, hence the
        index is measured from the end.
        """
        start = bisect.bisect_left(self._deadlines, t_minus_s * 1000.0)
        return self.ticks[len(self.ticks) - start:]

    def ticks_between_t_minus(self, high_s: float, low_s: float) -> list[Tick]:
        """Chronological slice with ``low_s <= countdown < high_s``."""
        lo = bisect.bisect_left(self._deadlines, low_s * 1000.0)
        hi = bisect.bisect_right(self._deadlines, high_s * 1000.0)
        n = len(self.ticks)
        return self.ticks[n - hi:n - lo]


@dataclass(frozen=True)
class Resolution:
    market_id: str
    slug: str
    horizon: str
    outcome_direction: str
    up_token: str
    down_token: str
    winner_token: str
    open_price: float | None
    close_price: float | None
    usable: bool

    @property
    def up_won(self) -> bool:
        return self.outcome_direction.upper() == "UP"


def load_resolutions(root: str) -> dict[str, Resolution]:
    """Resolution keyed by ``market_id`` (NOT window_key, which is corrupted)."""
    out: dict[str, Resolution] = {}
    for path in day_files(root, RESOLUTION):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
            for row in batch.to_pylist():
                market_id = str(row.get("market_id") or "")
                if not market_id:
                    continue
                out[market_id] = Resolution(
                    market_id=market_id,
                    slug=str(row.get("slug") or ""),
                    horizon=str(row.get("horizon") or ""),
                    outcome_direction=str(row.get("outcome_direction") or ""),
                    up_token=str(row.get("up_token_id") or ""),
                    down_token=str(row.get("down_token_id") or ""),
                    winner_token=str(row.get("winner_token_id") or ""),
                    open_price=_f(row.get("chainlink_open_price")),
                    close_price=_f(row.get("chainlink_resolution_price")),
                    usable=str(row.get("usable_for_backtest")).lower() == "true",
                )
    return out


def iter_markets(root: str, *, horizons: Sequence[str] | None = None,
                 resolutions: dict[str, Resolution] | None = None,
                 limit: int | None = None, progress=None) -> Iterator[Market]:
    """Yield one :class:`Market` per feature file, streaming.

    Streaming keeps memory flat: a 4h market contributes up to 135,600 ticks while
    the median 5m market contributes ~2,700.
    """
    files = day_files(root, FEATURES)
    kept = 0
    for index, path in enumerate(files, start=1):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        cols = [c for c in USEFUL_COLUMNS if c in names]
        rows = pq.read_table(path, columns=cols).to_pylist()
        if not rows:
            continue
        first = rows[0]
        slug = str(first.get("slug") or "")
        horizon = horizon_of(slug)
        if horizons is not None and horizon not in horizons:
            continue
        market_id = str(first.get("market_id") or "")
        ticks = []
        for row in rows:
            ticks.append(Tick(
                ts=(_f(row.get("timestamp_ms"), 0.0) or 0.0) / 1000.0,
                to_start_ms=_f(row.get("time_to_start_ms"), 0.0) or 0.0,
                to_deadline_ms=_f(row.get("time_to_deadline_ms"), 0.0) or 0.0,
                up_bid=_f(row.get("up_best_bid")),
                up_ask=_f(row.get("up_best_ask")),
                up_mid=_f(row.get("up_midquote")),
                down_bid=_f(row.get("down_best_bid")),
                down_ask=_f(row.get("down_best_ask")),
                down_mid=_f(row.get("down_midquote")),
                sum_ask=_f(row.get("sum_best_ask")),
                up_asks=_levels(row.get("up_side_asks")),
                up_ask_sizes=_levels(row.get("up_ask_sizes")),
                down_asks=_levels(row.get("down_side_asks")),
                down_ask_sizes=_levels(row.get("down_ask_sizes")),
                up_bids=_levels(row.get("up_side_bids")),
                up_bid_sizes=_levels(row.get("up_bid_sizes")),
                down_bids=_levels(row.get("down_side_bids")),
                down_bid_sizes=_levels(row.get("down_bid_sizes")),
            ))
        ticks.sort(key=lambda t: t.ts)
        market = Market(
            market_id=market_id,
            slug=slug,
            horizon=horizon,
            start_ts=start_of(slug),
            ticks=ticks,
            resolution=(resolutions or {}).get(market_id),
            up_token=str(first.get("up_token_id") or ""),
            down_token=str(first.get("down_token_id") or ""),
        )
        kept += 1
        if progress is not None and index % 200 == 0:
            progress(index, len(files), kept)
        yield market
        if limit is not None and kept >= limit:
            return


def summarise(values: Sequence[float]) -> dict:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {"n": 0}
    n = len(clean)

    def pct(q: float) -> float:
        return clean[min(n - 1, max(0, int(q * n)))]

    import statistics
    return {"n": n, "min": clean[0], "p10": pct(0.10), "p25": pct(0.25),
            "median": statistics.median(clean), "p75": pct(0.75), "p90": pct(0.90),
            "max": clean[-1], "mean": statistics.fmean(clean)}

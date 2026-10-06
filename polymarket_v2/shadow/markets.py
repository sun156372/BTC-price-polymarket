"""Pre-parsed market data for the shadow engine.

Everything expensive happens **before** the 1200-account loop:

* timestamps are already floats (epoch seconds)
* print series are plain lists sorted by time, with a binary-search cursor
* token ids are resolved once per window, not per account
* ATR for dynamic exits is computed once per window and shared by all accounts

The account loop therefore only does float comparisons, dict lookups and
arithmetic. That is the whole point of this module existing separately.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Sequence

K = 2  # number of outcome tokens per market (UP, DOWN)


def now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


class SpotRing:
    """Fixed-capacity ring of 1-second spot bars with O(1) append.

    ``cutoff`` (optional) hides every bar after a given timestamp. The live feed
    never sets it; the replay harness does, so a walk through history cannot read
    a price that had not happened yet - the classic way a backtest invents edge.
    """

    __slots__ = ("capacity", "_ts", "_px", "_start", "_count", "asset", "cutoff",
                 "_view", "_view_key")

    def __init__(self, asset: str, capacity: int = 3600) -> None:
        self.asset = asset
        self.capacity = int(capacity)
        self._ts = [0.0] * self.capacity
        self._px = [0.0] * self.capacity
        self._start = 0
        self._count = 0
        self.cutoff: float | None = None
        self._view: list[float] = []
        self._view_key: float = -1.0

    def __len__(self) -> int:
        return self._count

    def append(self, ts: float, price: float) -> None:
        if self._count and ts <= self._ts[(self._start + self._count - 1) % self.capacity]:
            return                                    # out of order or duplicate
        index = (self._start + self._count) % self.capacity
        self._ts[index] = ts
        self._px[index] = price
        if self._count < self.capacity:
            self._count += 1
        else:
            self._start = (self._start + 1) % self.capacity

    def _at(self, offset: int) -> tuple[float, float]:
        index = (self._start + offset) % self.capacity
        return self._ts[index], self._px[index]

    @property
    def newest(self) -> tuple[float, float] | None:
        if not self._count:
            return None
        return self._at(self._count - 1)

    def age(self, ts: float) -> float:
        bar = self.newest
        return float("inf") if bar is None else ts - bar[0]

    def price_as_of(self, ts: float) -> tuple[float, float] | None:
        """Backward-only lookup: last bar at or before ``ts``.

        Bisects over ring **positions** (0..count-1), so the per-cycle cost is
        logarithmic rather than O(n).

        ``cutoff`` restricts the *search range* (hi), it never extends it: a
        lookup for an early timestamp must not return a later bar merely because
        that bar happens to be before the cutoff.
        """
        if not self._count:
            return None
        # shrink the range to bars with t <= min(ts, cutoff)
        hi = self._count
        if self.cutoff is not None:
            low, high = 0, self._count
            while low < high:
                mid = (low + high) // 2
                if self._at(mid)[0] <= self.cutoff:
                    low = mid + 1
                else:
                    high = mid
            hi = low
        index = bisect.bisect_right(self._ts_view(), ts, 0, hi) - 1
        if index < 0:
            return None
        return self._at(index)

    def _ts_view(self) -> list[float]:
        """Ring timestamps in logical order, rebuilt only when the ring changes.

        Exists so :func:`bisect` can run over logical positions. The cache key is
        the newest timestamp rather than head/length, because at capacity an append
        changes ``_start`` *and* keeps ``_count`` - keying on those would serve a
        stale view after every single append.
        """
        newest = self._at(self._count - 1)[0] if self._count else 0.0
        if self._view_key != newest:
            self._view = [self._at(i)[0] for i in range(self._count)]
            self._view_key = newest
        return self._view

    def _visible(self, ts: float) -> float:
        """The furthest moment this ring will answer questions about."""
        return ts if self.cutoff is None else min(ts, self.cutoff)

    def move_pct(self, ts: float, lookback_s: float) -> float | None:
        """Percent change over ``lookback_s`` ending at ``ts`` (backward only).

        Both ends of the comparison are clamped to ``cutoff``, so a query about a
        future moment still cannot see a price from the future.
        """
        moment = self._visible(ts)
        now = self.price_as_of(moment)
        past = self.price_as_of(moment - lookback_s)
        if now is None or past is None or not past[1]:
            return None
        return (now[1] / past[1] - 1.0) * 100.0

    def atr(self, ts: float, window_s: float = 60.0) -> float | None:
        """Mean absolute 1s bar change over the window, as a fraction of price.

        This is the volatility estimate dynamic TP/SL scales against: quiet market
        -> small ATR -> tighter exits; violent market -> large ATR -> wider exits.
        """
        moment = self._visible(ts)
        if self._count < 3:
            return None
        prices: list[float] = []
        for offset in range(self._count - 1, -1, -1):
            bar_ts, price = self._at(offset)
            if bar_ts > moment:
                continue                      # hidden by the cutoff
            if moment - bar_ts > window_s:
                break
            prices.append(price)
        if len(prices) < 3:
            return None
        prices.reverse()
        total = 0.0
        moves = 0
        for previous, current in zip(prices, prices[1:]):
            if previous:
                total += abs(current / previous - 1.0)
                moves += 1
        if not moves:
            return None
        return total / moves


@dataclass(frozen=True)
class PrintSeries:
    """One token's print history as parallel arrays (no objects in the hot path)."""

    token: str
    ts: tuple[float, ...] = ()
    px: tuple[float, ...] = ()
    #: when set, every read stops at this timestamp (replay lookahead guard)
    cutoff: float | None = None

    def __len__(self) -> int:
        return len(self.ts)

    def _end(self) -> int:
        """Index one past the last visible print."""
        if self.cutoff is None:
            return len(self.ts)
        return bisect.bisect_right(self.ts, self.cutoff)

    def last_at_or_before(self, ts: float) -> tuple[float, float] | None:
        limit = ts if self.cutoff is None else min(ts, self.cutoff)
        index = bisect.bisect_right(self.ts, limit, 0, self._end()) - 1
        if index < 0:
            return None
        return self.ts[index], self.px[index]

    def first_strictly_after(self, ts: float) -> int:
        start = bisect.bisect_right(self.ts, ts)
        return min(start, self._end())

    def walk_from(self, start_index: int) -> Iterator[tuple[float, float]]:
        for index in range(start_index, self._end()):
            yield self.ts[index], self.px[index]

    def with_cutoff(self, cutoff: float | None) -> "PrintSeries":
        """Cheap copy sharing the same tuples, with a visibility bound."""
        return PrintSeries(self.token, self.ts, self.px, cutoff)


@dataclass
class MarketFrame:
    """One 5/15m window, fully prepared for the account loop."""

    slug: str
    asset: str
    start_ts: float
    end_ts: float
    tokens: tuple[str, str]                 # (UP, DOWN)
    prints: dict[str, PrintSeries] = field(default_factory=dict)
    atr: float | None = None                # computed once, shared by all accounts
    spot_move: dict[float, float | None] = field(default_factory=dict)  # t_minus -> move
    mid: dict[str, float] = field(default_factory=dict)                 # token -> last mid
    settled: bool = False
    winner: str = ""                        # "UP" / "DOWN" once known
    #: when set, all reads inside this frame stop at this timestamp (replay guard)
    cutoff: float | None = None

    @property
    def window_s(self) -> float:
        return self.end_ts - self.start_ts

    def t_minus(self, ts: float) -> float:
        return self.end_ts - ts

    def entry_ts_for(self, t_minus_s: float) -> float:
        return self.end_ts - t_minus_s

    def series(self, token: str) -> PrintSeries:
        return self.prints.get(token) or PrintSeries(token)

    def prepare_signal(self, spot: SpotRing, times: Sequence[float], lookback_s: float) -> None:
        """Pre-compute the spot move for every entry checkpoint (once per window)."""
        for t_minus in times:
            entry_ts = self.entry_ts_for(t_minus)
            self.spot_move[t_minus] = spot.move_pct(entry_ts, lookback_s)

    def refresh_mid(self) -> None:
        for token, series in self.prints.items():
            if len(series):
                self.mid[token] = series.px[-1]


class MarketBook:
    """All live frames keyed by slug; the account loop iterates this dict."""

    def __init__(self) -> None:
        self.frames: dict[str, MarketFrame] = {}
        self.completed: list[str] = []

    def upsert(self, frame: MarketFrame) -> None:
        existing = self.frames.get(frame.slug)
        if existing is None:
            self.frames[frame.slug] = frame
            return
        existing.atr = frame.atr
        existing.tokens = frame.tokens
        existing.end_ts = frame.end_ts
        existing.start_ts = frame.start_ts

    def retire(self, slug: str) -> MarketFrame | None:
        frame = self.frames.pop(slug, None)
        if frame is not None:
            self.completed.append(slug)
        return frame

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[MarketFrame]:
        return iter(self.frames.values())

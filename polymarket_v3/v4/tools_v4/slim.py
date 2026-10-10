"""Fast reader for the slim tick cache.

    from slim import load_ticks, load_index, has_cache
    for market in load_ticks(root, horizons=("5m",)):
        print(market.market_id, market.entry_at(120.0))

Why this exists
---------------
Streaming the raw ``polymarket_features_100ms`` files costs ~40 minutes per full
pass because each row carries book-level JSON that no experiment reads. The slim
cache (``tools_v4/build_cache.py``) holds only the six price columns and
loads a whole 15,316-market sample in seconds.

The cache is optional. If it is missing, ``has_cache`` returns False and callers
should fall back to :func:`engine_v4.hf.iter_markets`.
"""
from __future__ import annotations

import bisect
import dataclasses
import os

import pyarrow.parquet as pq

CACHE_DIR = "_cache"
TICKS = "slim_ticks.parquet"
INDEX = "slim_index.parquet"
RESOLUTIONS = "slim_resolutions.parquet"


def cache_dir(root: str) -> str:
    return os.path.join(root, CACHE_DIR)


def has_cache(root: str) -> bool:
    d = cache_dir(root)
    return all(os.path.exists(os.path.join(d, n)) for n in (TICKS, INDEX))


@dataclasses.dataclass(slots=True)
class SlimTick:
    """One 100ms observation. ``to_deadline_ms`` is the authoritative countdown."""

    ts: float
    to_deadline_ms: float
    up_bid: float | None
    up_ask: float | None
    up_mid: float | None
    down_bid: float | None
    down_ask: float | None
    down_mid: float | None

    def ask(self, side: str) -> float | None:
        return self.up_ask if side == "UP" else self.down_ask

    def bid(self, side: str) -> float | None:
        return self.up_bid if side == "UP" else self.down_bid

    def mid(self, side: str) -> float | None:
        return self.up_mid if side == "UP" else self.down_mid

    def cost_to_buy(self, side: str, shares: float, levels: int = 8) -> float | None:
        """Not available from the slim cache - depth was deliberately not cached.

        Callers that need real book depth must use the full loader. Returning None
        makes that explicit instead of silently substituting the touch price.
        """
        return None


@dataclasses.dataclass(slots=True)
class SlimMarket:
    market_id: str
    slug: str
    day: str
    horizon: str
    start_ts: float
    up_won: bool | None
    ticks: list[SlimTick]
    _deadlines: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        # ``ticks`` is chronological, so its countdowns DESCEND. Bisecting needs an
        # ASCENDING list, so keep one - but then the index must be translated:
        #     _deadlines[i] corresponds to ticks[len(ticks) - 1 - i]
        # Indexing both lists with the same number is a silent ~57-second error that
        # still returns a plausible price. tools_v4/parity_check.py guards against it.
        self._deadlines = sorted(t.to_deadline_ms for t in self.ticks)

    def _tick_for_deadline_index(self, i: int) -> SlimTick:
        return self.ticks[len(self.ticks) - 1 - i]

    def first_tick(self) -> SlimTick | None:
        return self.ticks[0] if self.ticks else None

    def first_mid(self) -> float | None:
        t = self.first_tick()
        return None if t is None else t.up_mid

    def tick_at_t_minus(self, t_minus_s: float,
                        tolerance_s: float = 1.0) -> SlimTick | None:
        """The tick nearest ``t_minus_s`` seconds before the close.

        Byte-for-byte the same semantics as ``engine_v4.hf.Market.tick_at_t_minus``:
        bisect to the target, then consider the three candidates around the
        insertion point and keep the CLOSEST one within tolerance. The only
        difference is the index translation, because ``ticks`` is descending while
        ``_deadlines`` ascends.

        Getting this wrong is not loud. An earlier version scanned outward from a
        mis-mapped index and returned a real price from ~57 seconds earlier, which
        every downstream number accepted without complaint. tools_v4/parity_check.py
        now asserts equality against the authoritative loader.
        """
        if not self.ticks:
            return None
        target = t_minus_s * 1000.0
        index = bisect.bisect_left(self._deadlines, target)
        best = None
        best_gap = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self._deadlines):
                tick = self._tick_for_deadline_index(candidate)
                gap = abs(tick.to_deadline_ms - target)
                if gap <= tolerance_s * 1000.0 and (best_gap is None or gap < best_gap):
                    best, best_gap = tick, gap
        return best


def load_index(root: str) -> list[dict]:
    path = os.path.join(cache_dir(root), INDEX)
    tbl = pq.read_table(path)
    return tbl.to_pylist()


def load_resolutions(root: str) -> dict[str, dict]:
    path = os.path.join(cache_dir(root), RESOLUTIONS)
    tbl = pq.read_table(path)
    return {str(r["market_id"]): r for r in tbl.to_pylist()}


def load_ticks(root: str, horizons: tuple[str, ...] | None = ("5m",),
               days: tuple[str, ...] | None = None):
    """Stream :class:`SlimMarket` objects from the cache.

    The cache is sorted by market, so a single pass groups rows by ``market_id``.
    """
    index = load_index(root)
    if horizons is not None:
        index = [r for r in index if r["horizon"] in horizons]
    if days is not None:
        index = [r for r in index if r["day"] in days]
    meta = {str(r["market_id"]): r for r in index}
    if not meta:
        return

    path = os.path.join(cache_dir(root), TICKS)
    pf = pq.ParquetFile(path)
    current_id: str | None = None
    bucket: list[SlimTick] = []

    def emit(market_id: str, ticks: list[SlimTick]):
        info = meta[market_id]
        return SlimMarket(
            market_id=market_id, slug=str(info["slug"]), day=str(info["day"]),
            horizon=str(info["horizon"]), start_ts=float(info["start_ts"] or 0.0),
            up_won=(None if info["up_won"] is None else bool(info["up_won"])),
            ticks=ticks,
        )

    for batch in pf.iter_batches(batch_size=200_000):
        cols = batch.to_pydict()
        ids = cols["market_id"]
        for i, mid in enumerate(ids):
            if mid != current_id:
                if current_id is not None and current_id in meta and bucket:
                    yield emit(current_id, bucket)
                current_id = mid
                bucket = []
                if mid not in meta:
                    continue
            if mid not in meta:
                continue
            bucket.append(SlimTick(
                ts=cols["ts"][i], to_deadline_ms=cols["to_deadline_ms"][i],
                up_bid=cols["up_bid"][i], up_ask=cols["up_ask"][i],
                up_mid=cols["up_mid"][i], down_bid=cols["down_bid"][i],
                down_ask=cols["down_ask"][i], down_mid=cols["down_mid"][i],
            ))
    if current_id is not None and current_id in meta and bucket:
        yield emit(current_id, bucket)

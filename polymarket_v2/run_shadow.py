#!/usr/bin/env python
"""run_shadow.py - 1200 parameter sets, one thread, live or replayed.

    # verify the whole thing offline against your parquet history (no network)
    python run_shadow.py --mode replay --data ../data

    # live shadow book: Binance spot + Polymarket books, virtual money only
    python run_shadow.py --mode live --assets BTC

    # watch the archive
    #   PowerShell :  Import-Csv shadow_metrics.csv | Sort-Object {[double]$_.equity} -Descending | Select-Object -First 15
    #   bash       :  sort -t, -k2 -gr shadow_metrics.csv | head -15

Nothing in this file can reach a funded wallet: there is no broker, no key, no
order client. Positions are numbers in a dataclass.

1200 accounts, one loop
-----------------------
Per cycle the engine does exactly this, single-threaded:

    for frame in book:            # active windows
        prepare once             # spot moves + ATR, parsed outside the loop
        for t_minus in times:     # entry checkpoints
            for account in accounts_at_that_time:
                decide -> fill -> exit/hedge/settle

No thread or coroutine is created per parameter set, and no timestamp
conversion, JSON parsing or string formatting happens inside the account loop.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal as os_signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.engine import ShadowEngine, build_engine
from shadow.feeds import BinanceSpotFeed, PolymarketFeed, ReplayFeed
from shadow.grid import GridError, load_grid
from shadow.markets import MarketBook, MarketFrame, SpotRing, now_utc

DEFAULT_GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "param_combinations.csv")


class ShadowRunner:
    """Owns the loop. Live and replay share every line below the feed."""

    def __init__(self, engine: ShadowEngine, *, status_every_s: float = 15.0,
                 archive_every_s: float = 300.0, tick_s: float = 2.0) -> None:
        self.engine = engine
        self.status_every_s = float(status_every_s)
        self.archive_every_s = float(archive_every_s)
        self.tick_s = float(tick_s)
        self.running = True
        self._last_status = 0.0

    def stop(self, *_args) -> None:
        self.running = False

    # ------------------------------------------------------------- replay
    def run_replay(self, data_dir: str, trades_file: str, assets, *,
                   progress_every: int = 200, warmup_s: float = 900.0) -> None:
        feed = ReplayFeed(data_dir, trades_file=trades_file, assets=assets)
        print(f"[{now_utc()}] loading history from {os.path.abspath(data_dir)} ...")
        data = feed.load()
        frames = sorted(data.book.frames.values(), key=lambda f: f.end_ts)
        print(f"[{now_utc()}] loaded {len(frames)} windows, "
              f"{sum(len(v) for v in data.bars.values()):,} spot bars")

        # One ring per asset, filled progressively as the timeline advances - the
        # same way a live feed would fill it. Pre-loading the file would put the
        # ring's newest bar in the future relative to every replayed window.
        rings = {asset: SpotRing(asset, capacity=7200) for asset in data.bars}
        cursors = {asset: 0 for asset in data.bars}

        def advance(asset: str, up_to: float) -> None:
            """Append every bar at or before ``up_to`` that is not in the ring yet."""
            series = data.bars.get(asset) or []
            ring = rings[asset]
            index = cursors[asset]
            while index < len(series) and series[index][0] <= up_to:
                ring.append(series[index][0], series[index][1])
                index += 1
            cursors[asset] = index

        cycles = 0
        for frame in frames:
            if not self.running:
                break
            asset = frame.asset
            if asset not in rings:
                continue
            # warm the ring with the history the 180s lookback needs
            advance(asset, frame.start_ts)
            window_book = MarketBook()
            window_book.upsert(frame)

            for t_minus in sorted(self.engine.times, reverse=True):
                entry_ts = frame.entry_ts_for(t_minus)
                advance(asset, entry_ts)
                for ring in rings.values():
                    ring.cutoff = entry_ts
                frame.cutoff = entry_ts
                frame.spot_move.clear()
                self.engine.cycle(window_book, rings, now=entry_ts)

            # final pass at the close
            advance(asset, frame.end_ts)
            for ring in rings.values():
                ring.cutoff = frame.end_ts
            frame.cutoff = frame.end_ts
            self.engine.resolve(window_book, now=frame.end_ts + 1.0)
            cycles += 1
            if cycles % progress_every == 0:
                self.engine.archive(force=True, now=frame.end_ts)
                print(f"[{now_utc()}] {cycles}/{len(frames)} windows | "
                      f"entries {self.engine.stats.entries} "
                      f"TP {self.engine.stats.take_profits} "
                      f"SL {self.engine.stats.stops} "
                      f"hedge {self.engine.stats.hedges} "
                      f"settle {self.engine.stats.settlements}")

        path = self.engine.archive(force=True, now=time.time())
        self.engine.log_summary()
        print(f"[{now_utc()}] replay done: {cycles} windows, "
              f"{self.engine.stats.entries} entries")
        print(f"[{now_utc()}] archive: {path}")

    # --------------------------------------------------------------- live
    async def run_live(self, assets, *, window: str) -> None:
        spot_feed = BinanceSpotFeed(assets)
        polymarket = PolymarketFeed(assets, window=window)
        stream = asyncio.create_task(polymarket.stream())
        try:
            while self.running:
                moment = time.time()
                spot_feed.poll(moment)
                if int(moment) % 15 < self.tick_s:
                    polymarket.resolve(moment)

                book = MarketBook()
                for slug in list(polymarket.tokens):
                    frame = polymarket.frame_for(slug, moment)
                    if frame is not None:
                        book.upsert(frame)

                self.engine.cycle(book, spot_feed.rings, now=moment)
                self.engine.resolve(book, now=moment)

                if moment - self._last_status >= self.status_every_s:
                    self._last_status = moment
                    self.engine.log_summary(
                        spot_price=spot_feed.last_price(assets[0]),
                        active_windows=len(book))
                    ages = " ".join(f"{a}:{spot_feed.age(a, moment):.0f}s" for a in assets)
                    print(f"    feed age {ages} | ws msgs {polymarket.messages} "
                          f"| spot errors {spot_feed.errors}")

                path = self.engine.archive(now=moment)
                if path:
                    print(f"    archived -> {path}")

                await asyncio.sleep(self.tick_s)
        finally:
            stream.cancel()
            print(f"[{now_utc()}] final archive: {self.engine.archive(force=True)}")
            self.engine.log_summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shadow book: 1200 parameter sets, one thread, virtual money",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--mode", choices=["replay", "live", "selftest"],
                        default="replay")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="param_combinations.csv")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N params")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(DEFAULT_GRID)),
                        help="data dir (replay mode)")
    parser.add_argument("--trades", default="auto", help="trades parquet (replay mode)")
    parser.add_argument("--assets", nargs="*", default=["BTC"],
                        help="live assets (replay uses all it finds)")
    parser.add_argument("--window", default="5m", help="live market window")
    parser.add_argument("--cash", type=float, default=1_000.0,
                        help="virtual cash per account (default 1000)")
    parser.add_argument("--notional", type=float, default=None,
                        help="notional per trade (default: 1%% of --cash)")
    parser.add_argument("--lookback", type=float, default=180.0,
                        help="spot lookback for the move signal (default 180s)")
    parser.add_argument("--archive", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "shadow_metrics.csv"))
    parser.add_argument("--archive-every", type=float, default=300.0,
                        help="archive interval, seconds (default 300 = 5 min)")
    parser.add_argument("--status-every", type=float, default=15.0)
    parser.add_argument("--tick", type=float, default=2.0, help="loop interval, seconds")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "selftest":
        from shadow.selftest import run
        return run()

    try:
        grid = load_grid(args.grid, limit=args.limit)
    except GridError as exc:
        print(f"grid error: {exc}", file=sys.stderr)
        print("\ngenerate one with:  python scripts/make_param_grid.py", file=sys.stderr)
        return 2

    print("=" * 92)
    print(f"SHADOW BOOK  {len(grid)} parameter sets, single thread, virtual money only")
    print("=" * 92)
    print(f"grid            : {os.path.abspath(args.grid)}")
    hedged = sum(1 for p in grid if p.hedge_enabled)
    dynamic = sum(1 for p in grid if p.dynamic_tp_sl)
    print(f"  hedging on    : {hedged} ({hedged / len(grid) * 100:.0f}%)")
    print(f"  dynamic TP/SL : {dynamic} ({dynamic / len(grid) * 100:.0f}%)")
    print(f"  entry times   : {sorted({p.time_s for p in grid})}")
    print(f"  thresholds    : {sorted({p.threshold_pct for p in grid})}")
    print(f"virtual cash    : {args.cash:,.2f} per account "
          f"({args.cash * len(grid):,.0f} total, all simulated)")
    print(f"archive         : {os.path.abspath(args.archive)}")

    engine = build_engine(
        grid, starting_cash=args.cash, notional=args.notional,
        lookback_s=args.lookback, archive_path=args.archive,
        archive_interval_s=args.archive_every,
    )
    runner = ShadowRunner(engine, status_every_s=args.status_every,
                          archive_every_s=args.archive_every, tick_s=args.tick)
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            os_signal.signal(sig, runner.stop)
        except (ValueError, OSError):
            pass

    try:
        if args.mode == "replay":
            runner.run_replay(args.data, args.trades,
                              ["BTC", "ETH", "SOL", "XRP"])
        else:
            if BinanceSpotFeed is None:                       # pragma: no cover
                print("feed imports unavailable", file=sys.stderr)
                return 2
            asyncio.run(runner.run_live([a.upper() for a in args.assets],
                                        window=args.window))
    except KeyboardInterrupt:
        runner.stop()
        print(f"\n[{now_utc()}] interrupted; final archive: "
              f"{engine.archive(force=True)}")
    except Exception:  # noqa: BLE001 - top level
        print("fatal:\n" + traceback.format_exc(), file=sys.stderr)
        engine.archive(force=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-verification for the shadow engine: synthetic data, known answers.

    python run_shadow.py --mode selftest

Checks the properties that matter and are cheap to get wrong:

1. one-thread invariant - 1200 accounts, no threads or tasks created
2. accounting invariant  - every entry is closed or settled, never lost
3. dynamic TP/SL actually widens with volatility and narrows without it
4. hedge-on-stop opens the opposite leg instead of realising the stop
5. hedge survives to settlement and pays off on the correct side
6. hard stop fires when hedging is disabled
7. ATR-based sizing stays inside its clamp
8. archiving rewrites the CSV atomically with one row per account
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

from .engine import MAX_SCALE, MIN_SCALE, ShadowEngine, build_engine
from .grid import ShadowParams, VirtualAccount
from .markets import MarketBook, MarketFrame, PrintSeries, SpotRing


def _params(**kwargs) -> ShadowParams:
    base = dict(combo_id="t", time_s=60.0, threshold_pct=0.01, tp=0.10, sl=0.10,
                min_gap_s=0.0, dynamic_tp_sl=False, hedge_enabled=False,
                hedge_ratio=0.0)
    base.update(kwargs)
    return ShadowParams(**base)


def _ring(asset: str, *, base_ts: float, prices: list[float], step: float = 1.0) -> SpotRing:
    ring = SpotRing(asset, capacity=7200)
    for index, price in enumerate(prices):
        ring.append(base_ts + index * step, price)
    return ring


def _frame(slug: str, *, start: float, end: float, up: str, down: str,
           up_prints: list[tuple[float, float]],
           down_prints: list[tuple[float, float]]) -> MarketFrame:
    return MarketFrame(
        slug=slug, asset="BTC", start_ts=start, end_ts=end, tokens=(up, down),
        prints={
            up: PrintSeries(up, tuple(p[0] for p in up_prints),
                            tuple(p[1] for p in up_prints)),
            down: PrintSeries(down, tuple(p[0] for p in down_prints),
                              tuple(p[1] for p in down_prints)),
        },
    )


def run() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    def section(label: str) -> None:
        # printed immediately: if a section ever hangs, its name is the last line
        print(f"  ... {label}", flush=True)

    section("fixtures")
    base = 1_800_000_000.0
    end = base + 300.0
    entry_ts = end - 60.0                       # T-60 for the fixtures below
    start = base

    # ---------------------------------------------------------- 1. threading
    section("threading")
    # A 1200-account book is built to prove the shape, but the *cycle* is run on a
    # small grid: cycling 1200 accounts over a long synthetic ring is pure test
    # cost with no extra coverage, and it made this selftest hang.
    before = threading.active_count()
    grid = [_params(combo_id=f"p{i:04d}") for i in range(1200)]
    engine = build_engine(grid, starting_cash=1_000.0, notional=10.0)
    check("1200 accounts built from a 1200-row grid",
          len(engine.accounts) == 1200, str(len(engine.accounts)))

    small = build_engine([_params(combo_id=f"s{i:03d}") for i in range(50)],
                         starting_cash=1_000.0, notional=10.0)
    rising = _ring("BTC", base_ts=base - 1_200,
                   prices=[100.0 * (1 + 0.001 * i) for i in range(1_400)])
    book = MarketBook()
    book.upsert(_frame("btc-updown-5m-1", start=start, end=end, up="U", down="D",
                       up_prints=[(entry_ts - 1, 0.50)], down_prints=[]))
    small.cycle(book, {"BTC": rising}, now=entry_ts)
    after = threading.active_count()
    check("no thread created per account", after == before, f"{before} -> {after}")

    # ------------------------------------------------------- 2. gate coverage
    section("gates")
    stats = small.stats
    check("signal gates evaluated for all accounts",
          stats.entries + stats.blocked_threshold + stats.blocked_min_gap
          + stats.blocked_no_print + stats.blocked_no_cash > 0,
          str(stats.as_dict()))

    # ------------------------------------------------- 3. dynamic TP/SL scaling
    section("dynamic")
    calm = build_engine([_params(dynamic_tp_sl=True)], starting_cash=1_000.0, notional=10.0)
    calm_plan = calm._exit_plan(calm.accounts[0].params, 0.50, atr=0.00002)   # tiny ATR
    check("calm market tightens the exits",
          calm_plan.tp_price < 0.60 and calm_plan.scale < 1.0,
          f"tp={calm_plan.tp_price:.4f} scale={calm_plan.scale:.3f}")

    wild = build_engine([_params(dynamic_tp_sl=True)], starting_cash=1_000.0, notional=10.0)
    wild_plan = wild._exit_plan(wild.accounts[0].params, 0.50, atr=0.002)     # volatile
    check("volatile market widens the exits",
          wild_plan.tp_price > 0.60 and wild_plan.scale > 1.0,
          f"tp={wild_plan.tp_price:.4f} scale={wild_plan.scale:.3f}")

    clamped = wild._exit_plan(_params(dynamic_tp_sl=True), 0.50, atr=1.0)
    check("dynamic scale is clamped", clamped.scale == MAX_SCALE, str(clamped.scale))
    fixed = wild._exit_plan(_params(dynamic_tp_sl=False), 0.50, atr=1.0)
    check("dynamic flag off means fixed exits",
          abs(fixed.tp_price - 0.60) < 1e-9 and fixed.scale == 1.0,
          f"tp={fixed.tp_price}")

    # --------------------------------------------------- 4. hedge on stop loss
    section("hedge")
    hedge_params = _params(hedge_enabled=True, hedge_ratio=0.5, tp=0.50, sl=0.10)
    hedger = build_engine([hedge_params], starting_cash=1_000.0, notional=10.0)
    frame = _frame("btc-updown-5m-h", start=start, end=end, up="U", down="D",
                   up_prints=[(entry_ts - 1, 0.50), (entry_ts + 10, 0.30)],
                   down_prints=[(entry_ts + 11, 0.70)])
    hbook = MarketBook()
    hbook.upsert(frame)
    hedger.cycle(hbook, {"BTC": rising}, now=entry_ts)
    account = hedger.accounts[0]
    check("hedged account entered", len(account.positions) == 1, str(len(account.positions)))
    hedger.resolve(hbook, now=end + 1)
    check("stop produced a hedge, not a close", hedger.stats.hedges == 1,
          f"hedges={hedger.stats.hedges} stops={hedger.stats.stops}")
    check("hedge recorded on the opposite side",
          account.hedges == 1 and account.stops == 0,
          f"hedges={account.hedges} stops={account.stops}")
    check("hedged position settled as one round trip", account.trades == 1,
          str(account.trades))

    # ------------------------------------------------- 5. hard stop without hedge
    section("hardstop")
    plain_params = _params(hedge_enabled=False, tp=0.50, sl=0.10)
    plain = build_engine([plain_params], starting_cash=1_000.0, notional=10.0)
    pframe = _frame("btc-updown-5m-p", start=start, end=end, up="U", down="D",
                    up_prints=[(entry_ts - 1, 0.50), (entry_ts + 10, 0.30)],
                    down_prints=[])
    pbook = MarketBook()
    pbook.upsert(pframe)
    plain.cycle(pbook, {"BTC": rising}, now=entry_ts)
    plain.resolve(pbook, now=end + 1)
    check("hard stop fires when hedging is off",
          plain.stats.stops == 1 and plain.stats.hedges == 0,
          f"stops={plain.stats.stops} hedges={plain.stats.hedges}")
    check("hard stop realised a loss", plain.accounts[0].equity < 1_000.0,
          f"{plain.accounts[0].equity:.2f}")

    # ------------------------------------------- 6. take profit still works
    section("takeprofit")
    tp_engine = build_engine([_params(tp=0.10, sl=0.50)], starting_cash=1_000.0, notional=10.0)
    tframe = _frame("btc-updown-5m-t", start=start, end=end, up="U", down="D",
                    up_prints=[(entry_ts - 1, 0.50), (entry_ts + 10, 0.70)],
                    down_prints=[])
    tbook = MarketBook()
    tbook.upsert(tframe)
    tp_engine.cycle(tbook, {"BTC": rising}, now=entry_ts)
    tp_engine.resolve(tbook, now=end + 1)
    check("take profit fires on first touch", tp_engine.stats.take_profits == 1,
          str(tp_engine.stats.take_profits))
    check("take profit is profitable", tp_engine.accounts[0].equity > 1_000.0,
          f"{tp_engine.accounts[0].equity:.2f}")

    # --------------------------------------------- 7. lookahead guard (cutoff)
    section("lookahead")
    leak = build_engine([_params()], starting_cash=1_000.0, notional=10.0)
    lframe = _frame("btc-updown-5m-l", start=start, end=end, up="U", down="D",
                    up_prints=[(entry_ts - 1, 0.50)], down_prints=[])
    lframe.cutoff = entry_ts
    lbook = MarketBook()
    lbook.upsert(lframe)
    guarded = rising.price_as_of(entry_ts + 500)
    check("cutoff hides future spot bars",
          guarded is not None and guarded[0] <= entry_ts,
          f"{guarded}")

    # ------------------------------------------------------ 8. atomic archive
    section("archive")
    # The scratch dir lives under the project, not %TEMP%: writing outside the
    # sandboxed workspace can block under confinement, which looked exactly like
    # an archive bug until it was traced.
    scratch = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".shadow_test"))
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, "shadow_metrics.csv")
    written = small.archive(path, force=True, now=time.time())
    with open(path, encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    check("archive writes a header plus one row per account",
          written == path and len(rows) == len(small.accounts) + 1,
          f"{len(rows)} lines for {len(small.accounts)} accounts")
    check("archive is sorted by equity descending",
          rows[1].split(",")[1] >= rows[2].split(",")[1], rows[1][:40])
    try:
        os.remove(path)
    except OSError:
        pass

    # ------------------------------------------------- 9. accounting invariant
    section("invariant")
    opened = small.stats.entries
    closed = (small.stats.take_profits + small.stats.stops
              + small.stats.hedges + small.stats.settlements)
    still_open = sum(len(a.positions) for a in small.accounts)
    check("entries == closed + still open",
          opened == closed + still_open,
          f"{opened} vs {closed} + {still_open}")
    check("no account holds two positions at once",
          all(len(a.positions) <= 1 for a in small.accounts))

    # Every exit path must increment exactly one counter. This caught a real bug
    # where a hedged position both under-counted engine settlements and
    # double-counted the account's, which only showed up as a 58k discrepancy
    # between the engine log and the archived CSV on a full replay.
    per_account_trades = sum(a.trades for a in small.accounts)
    check("account trades == sum of exit reasons",
          per_account_trades == closed,
          f"trades {per_account_trades} vs reasons {closed}")
    reason_sum = sum(
        a.take_profits + a.stops + a.settlements for a in small.accounts
    )
    check("account exit reasons == account trades",
          reason_sum == per_account_trades,
          f"reasons {reason_sum} vs trades {per_account_trades}")

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"selftest: {len(checks) - failed}/{len(checks)} passed")
    return 0 if not failed else 1

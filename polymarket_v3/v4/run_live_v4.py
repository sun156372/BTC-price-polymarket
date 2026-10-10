"""Live paper-trading loop for the V4 imbalance strategy.

    python run_live_v4.py probe                  # verify the feed first
    python run_live_v4.py run --stake 10         # trade paper, log everything
    python run_live_v4.py report                 # summarise a session

PAPER ONLY, STRUCTURALLY. This process never signs or sends an order. It reads
public market data, evaluates the strategy's gates, records what it WOULD have done,
and then scores the result against the venue's own resolution. Cash is a float in a
log file.

.. warning::

   The network layer (``engine_v4/feed.py``) has never been executed against the
   live endpoints, because it was written where there was no network egress. Run
   ``probe`` first. If a field name has moved, the probe says so immediately rather
   than the loop failing silently at 3am.

What this measures that the backtest could not
----------------------------------------------
The backtest filled at the quoted ask. The whole live question is how far the real
fill drifts from that. So every decision records:

* the ask at the decision instant
* the ask again one and two seconds later (the latency drift)
* how deep the book was at that price

and every settled trade is scored twice: once at the decision ask, once at the
2-second-later ask. The gap between those two columns IS the latency cost, measured
on live data instead of assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine_v4.strategy import (  # noqa: E402
    BALANCED_BOOK, READY, V4Params, StrategyV4,
)
from engine_v4.feed import (  # noqa: E402
    FeedError, PolymarketFeed, SpotFeed, SLUG_TEMPLATE, WINDOW_SECONDS,
)

DEFAULT_LOG_DIR = os.path.join(HERE, "paper_v4")
DECISION_FIELDS = [
    "decided_at_utc", "slug", "window_start", "secs_to_close", "action", "reason",
    "side", "up_mid_open", "up_mid_now", "move", "spot_open", "spot_now",
    "spot_move_bps", "ask", "ask_1s", "ask_2s", "bid", "spread",
    "book_imbalance", "bid_notional", "ask_notional", "book_levels",
    "breaks_even_at", "would_win", "pnl_at_ask", "pnl_at_ask_2s",
    "spot_source", "spot_age_s",
]
SETTLE_FIELDS = ["slug", "resolved", "side", "would_win", "ask",
                 "pnl_at_ask", "pnl_at_ask_2s", "settled_at_utc"]


# ------------------------------------------------------------------- helpers


def utc(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def current_window_start(now: float | None = None) -> int:
    now = now or time.time()
    return int(now // WINDOW_SECONDS) * WINDOW_SECONDS


def log_path(log_dir: str, name: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, name)


def append_row(path: str, fields: list[str], row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------- probe


def cmd_probe(params: V4Params, log_dir: str) -> int:
    print("=" * 92)
    print("FEED PROBE - run this before any session")
    print("=" * 92)
    print("  This module has never been run against the live endpoints. Expect the")
    print("  possibility of a field or URL having moved; that is what this checks.")
    print()

    feed = PolymarketFeed()
    spot_feed = SpotFeed()
    problems = 0

    print("1. SPOT")
    try:
        quote = spot_feed.quote()
    except FeedError as exc:
        quote = None
        print(f"   [FAIL] {exc}")
    if quote is None:
        print("   [FAIL] no spot quote from coinbase or binance")
        problems += 1
    else:
        print(f"   [ok] BTC/USD {quote.price:,.2f} from {quote.source}")
    print()

    epoch = current_window_start()
    print(f"2. MARKET  (window opening {utc(epoch)}, slug "
          f"{SLUG_TEMPLATE.format(epoch=epoch)})")
    market = None
    try:
        market = feed.market_for_epoch(epoch)
    except FeedError as exc:
        print(f"   [FAIL] {exc}")
        problems += 1
    if market is None:
        print("   [FAIL] market not found. If the 5-minute series has ended or the")
        print("          slug format changed, the loop cannot run.")
        problems += 1
    else:
        print(f"   [ok] slug       {market.slug}")
        print(f"        up_token   {market.up_token[:24]}...")
        print(f"        down_token {market.down_token[:24]}...")
        print(f"        outcomes   {market.outcomes}")
        print(f"        window     {utc(market.start_ts)} -> {utc(market.end_ts)}")
    print()

    print("3. ORDER BOOK")
    if market is not None:
        for label, token in (("UP", market.up_token), ("DOWN", market.down_token)):
            try:
                book = feed.book(token)
            except FeedError as exc:
                print(f"   [FAIL] {label}: {exc}")
                problems += 1
                continue
            print(f"   [ok] {label}: bid {book.best_bid} ask {book.best_ask} "
                  f"spread {book.spread}")
            print(f"        levels {len(book.bids)} bid / {len(book.asks)} ask")
            print(f"        bid_notional {book.bid_notional():,.2f}  "
                  f"ask_notional {book.ask_notional():,.2f}  "
                  f"imbalance {book.imbalance():+.4f}" if book.imbalance() is not None
                  else "        imbalance unavailable")
            # the live imbalance must be comparable to the backtest feature
            print("        NOTE: the gate was calibrated on the dataset's "
                  "full_depth_*_notional.")
            print("              If this book is a top-N view, the scale differs and "
                  "the 0.30")
            print("              threshold would need recalibrating. Compare the "
                  "printed numbers")
            print("              against a market you know before trusting the gate.")
    print()
    print("=" * 92)
    if problems:
        print(f"PROBE FAILED with {problems} problem(s). Do not start a session.")
        return 1
    print("PROBE PASSED. The feed returns usable data.")
    return 0


# ----------------------------------------------------------------------- run


class WindowState:
    """What the loop observed about one 5-minute window.

    The strategy needs two things that only exist at a specific moment:

    * the Up-token mid AT THE WINDOW OPEN, to measure how far it has moved
    * the spot price AT THE WINDOW OPEN, to measure the underlying's move

    Neither can be fetched after the fact from the public endpoints used here, so
    the loop has to be present when the window opens and record them. This class
    holds that record, and makes the difference between "recorded" and "missed"
    explicit: a missing open is a REFUSAL (``signal:missing_quote``), never a
    silent substitution of the current price, because substituting would make the
    measured move identically zero and quietly disable the signal.
    """

    __slots__ = ("slug", "epoch", "up_mid_open", "spot_open", "opened_at")

    def __init__(self, slug: str, epoch: int):
        self.slug = slug
        self.epoch = epoch
        self.up_mid_open: float | None = None
        self.spot_open: float | None = None
        self.opened_at: float | None = None

    @property
    def complete(self) -> bool:
        return self.up_mid_open is not None and self.spot_open is not None


class Session:
    """Tracks windows and their observed opens across one live session."""

    def __init__(self):
        self.windows: dict[str, WindowState] = {}

    def observe_open(self, epoch: int, up_mid: float | None,
                     spot: float | None, now: float) -> None:
        """Record the open for ``epoch``. First observation wins."""
        slug = SLUG_TEMPLATE.format(epoch=epoch)
        state = self.windows.setdefault(slug, WindowState(slug, epoch))
        if state.up_mid_open is None and up_mid is not None:
            state.up_mid_open = up_mid
            state.opened_at = now
        if state.spot_open is None and spot is not None:
            state.spot_open = spot

    def get(self, slug: str) -> WindowState | None:
        return self.windows.get(slug)

    def prune(self, before_epoch: int) -> None:
        for slug in [s for s, w in self.windows.items() if w.epoch < before_epoch]:
            self.windows.pop(slug, None)


def cmd_run(params: V4Params, stake: float, log_dir: str, seconds: float) -> int:
    decisions_path = log_path(log_dir, "decisions.csv")
    settle_path = log_path(log_dir, "settled.csv")
    print("=" * 92)
    print("V4 LIVE PAPER SESSION")
    print("=" * 92)
    print(f"  stake {stake:.2f} USDC per trade, NO orders are ever sent")
    print(f"  logs: {decisions_path}")
    print(f"        {settle_path}")
    print(f"  gates: T-{params.entry_t_minus:.0f} | thr {params.threshold:.2f} | "
          f"spot >= {params.min_spot_move*1e4:.1f}bps | "
          f"|imb| >= {params.min_book_imbalance:.2f}")
    print()
    print("  The loop polls the open mid and the spot price through each window so it")
    print("  can measure the move. Ctrl-C to stop; every window is decided once.")
    print()

    strategy = StrategyV4(params)
    feed = PolymarketFeed()
    spot_feed = SpotFeed()
    session = Session()
    deadline = time.time() + seconds if seconds > 0 else None
    decided: set[str] = set()
    warned_open = False

    while True:
        now = time.time()
        if deadline is not None and now >= deadline:
            print("  session time reached")
            break
        epoch = current_window_start(now)
        slug = SLUG_TEMPLATE.format(epoch=epoch)
        secs_to_close = epoch + WINDOW_SECONDS - now

        try:
            market = feed.market_for_epoch(epoch)
        except FeedError as exc:
            print(f"  [{utc(now)}] feed error: {exc}")
            time.sleep(2.0)
            continue
        if market is None:
            if not warned_open:
                print(f"  [{utc(now)}] no market for {slug} - is the series live?")
                warned_open = True
            time.sleep(2.0)
            continue

        try:
            quote = spot_feed.quote()
            book = feed.book(market.up_token)
        except FeedError as exc:
            print(f"  [{utc(now)}] feed error: {exc}")
            time.sleep(2.0)
            continue

        # record the open while the window is young
        if secs_to_close > WINDOW_SECONDS - 25:
            session.observe_open(epoch, book.mid, quote.price if quote else None, now)

        if slug not in decided and secs_to_close <= params.entry_t_minus:
            decided.add(slug)
            state = session.get(slug)
            try:
                _evaluate_window(strategy, feed, spot_feed, market, state, params,
                                 stake, decisions_path, slug)
            except FeedError as exc:
                print(f"  [{utc()}] {slug}: feed error - {exc}")
            except Exception as exc:                      # noqa: BLE001
                print(f"  [{utc()}] {slug}: unexpected {type(exc).__name__}: {exc}")

        session.prune(epoch - WINDOW_SECONDS)
        time.sleep(1.0)

    _settle_open_positions(feed, decisions_path, settle_path)
    return 0


def _evaluate_window(strategy: StrategyV4, feed: PolymarketFeed,
                     spot_feed: SpotFeed, market, state: WindowState | None,
                     params: V4Params, stake: float, decisions_path: str,
                     slug: str) -> None:
    """Score one window and append the decision to the log.

    Every name used below is either a parameter or assigned here. This function once
    lost two of them during a refactor (``up_mid_now`` and ``slug``) and the only
    symptom was a NameError on the first live window, so the assignments are kept
    together and explicit rather than spread across branches.
    """
    decided_at = time.time()
    spot_now_q = spot_feed.quote()
    up_book = feed.book(market.up_token)

    # the mid now, and the mid at the window open (recorded when the window opened)
    up_mid_now = up_book.mid
    up_mid_open = state.up_mid_open if state else None
    spot_open = state.spot_open if state else None
    spot_now = spot_now_q.price if spot_now_q else None
    spot_move = (spot_now / spot_open - 1.0 if (spot_open and spot_now) else None)

    picked = strategy.direction(up_mid_open, up_mid_now)
    side = picked[0] if picked else ""
    move = picked[1] if picked else 0.0
    # the book on the side actually being bought; fetched only once the side is known
    book = up_book if side != "DOWN" else feed.book(market.down_token)
    ask = book.best_ask
    ask_1s = ask_2s = None
    if ask is not None:
        time.sleep(1.0)
        ask_1s = feed.book(book.token_id).best_ask
        time.sleep(1.0)
        ask_2s = feed.book(book.token_id).best_ask

    imbalance_signed = None
    if side:
        raw_imb = book.imbalance()
        if raw_imb is not None:
            imbalance_signed = raw_imb if side == "UP" else -raw_imb

    shares = stake / ask if ask else 0.0
    decision = strategy.decide(
        open_up_mid=up_mid_open, entry_up_mid=up_mid_now, ask=ask,
        now=decided_at, quote_ts=book.fetched_at, measured_fill=ask,
        shares_wanted=shares, depth_cost=(ask * shares if ask else None),
        spot_move=spot_move, book_imbalance=imbalance_signed,
    )
    breaks_even = None
    if ask:
        breaks_even = ask + params.fee_coefficient * ask * (1 - ask)

    row = {
        "decided_at_utc": utc(decided_at), "slug": slug,
        "window_start": utc(market.start_ts),
        "secs_to_close": round(market.end_ts - decided_at, 1),
        "action": decision.action, "reason": decision.reason,
        "side": decision.side or side, "up_mid_open": up_mid_open,
        "up_mid_now": up_mid_now, "move": round(move, 5),
        "spot_open": spot_open, "spot_now": spot_now,
        "spot_move_bps": (round(spot_move * 1e4, 3)
                          if spot_move is not None else None),
        "ask": ask, "ask_1s": ask_1s, "ask_2s": ask_2s,
        "bid": book.best_bid, "spread": book.spread,
        "book_imbalance": (round(imbalance_signed, 5)
                           if imbalance_signed is not None else None),
        "bid_notional": round(book.bid_notional(), 4),
        "ask_notional": round(book.ask_notional(), 4),
        "book_levels": f"{len(book.bids)}/{len(book.asks)}",
        "breaks_even_at": (round(breaks_even, 5) if breaks_even else None),
        "spot_source": spot_now_q.source if spot_now_q else "",
        "spot_age_s": (round(decided_at - spot_now_q.ts, 2) if spot_now_q else None),
    }
    append_row(decisions_path, DECISION_FIELDS, row)
    marker = "ENTER" if decision.entering else "hold "
    print(f"  [{utc(decided_at)}] {marker} {slug}  {decision.side or side or '-':<4} "
          f"ask={ask if ask is None else round(ask,3)} "
          f"imb={None if imbalance_signed is None else round(imbalance_signed,3)} "
          f"spot={None if spot_move is None else round(spot_move*1e4,2)}bps  "
          f"-> {decision.reason}")


def _settle_open_positions(feed: PolymarketFeed, decisions_path: str,
                           settle_path: str) -> None:
    if not os.path.exists(decisions_path):
        return
    with open(decisions_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    settled = set()
    if os.path.exists(settle_path):
        with open(settle_path, encoding="utf-8") as handle:
            settled = {r["slug"] for r in csv.DictReader(handle)}
    for row in rows:
        if row["action"] != "enter" or row["slug"] in settled:
            continue
        try:
            epoch = int(row["slug"].rsplit("-", 1)[-1])
            market = feed.market_for_epoch(epoch)
            if market is None or time.time() < market.end_ts + 20:
                continue
            outcome = feed.resolved_outcome(market)
        except FeedError:
            continue
        if outcome is None:
            continue
        side = row["side"]
        won = (side == outcome)
        ask = float(row["ask"])
        ask2 = float(row["ask_2s"]) if row.get("ask_2s") else ask
        fee_rate = 0.07
        def pnl(entry: float) -> float:
            gross = (1.0 - entry) if won else (-entry)
            return gross - fee_rate * entry * (1.0 - entry)
        append_row(settle_path, SETTLE_FIELDS, {
            "slug": row["slug"], "resolved": outcome, "side": side,
            "would_win": won, "ask": ask,
            "pnl_at_ask": round(pnl(ask), 5),
            "pnl_at_ask_2s": round(pnl(ask2), 5),
            "settled_at_utc": utc(),
        })
        print(f"  [{utc()}] SETTLED {row['slug']} {side} -> {outcome} "
              f"{'WIN' if won else 'loss'} pnl/share {pnl(ask):+.4f}")


# -------------------------------------------------------------------- report


def cmd_report(log_dir: str, stake: float) -> int:
    path = log_path(log_dir, "settled.csv")
    dpath = log_path(log_dir, "decisions.csv")
    if not os.path.exists(dpath):
        print(f"no decisions log at {dpath}")
        return 1
    with open(dpath, encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    print("=" * 92)
    print("SESSION REPORT")
    print("=" * 92)
    print(f"  windows evaluated : {len(decisions)}")
    reasons: dict[str, int] = defaultdict(int)
    for r in decisions:
        reasons[r["reason"]] += 1
    print("  refusals by reason:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>6}  {reason}")
    entries = [r for r in decisions if r["action"] == "enter"]
    print(f"  would have entered: {len(entries)}")
    if entries:
        shares = [stake / float(r["ask"]) for r in entries if r.get("ask")]
        if shares:
            print(f"  mean shares/trade : {statistics.fmean(shares):.2f}")
        drifts = [float(r["ask_2s"]) - float(r["ask"])
                  for r in entries if r.get("ask") and r.get("ask_2s")]
        if drifts:
            drifts.sort()
            n = len(drifts)
            print(f"  latency drift over 2s (cents): median {drifts[n//2]*100:+.2f} "
                  f"p90 {drifts[9*n//10]*100:+.2f} mean "
                  f"{statistics.fmean(drifts)*100:+.2f}")
            print("    This is the number that decides live profitability. The")
            print("    backtest assumed 0.00; the edge tolerates up to ~3 cents.")

    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            settled = list(csv.DictReader(handle))
        if settled:
            wins = sum(1 for r in settled if r["would_win"] == "True")
            per = [float(r["pnl_at_ask"]) for r in settled]
            per2 = [float(r["pnl_at_ask_2s"]) for r in settled]
            print()
            print(f"  settled trades    : {len(settled)}")
            print(f"  win rate          : {wins/len(settled)*100:.1f}%")
            print(f"  pnl/share at ask  : {statistics.fmean(per)*100:+.2f}%")
            print(f"  pnl/share at +2s  : {statistics.fmean(per2)*100:+.2f}%")
            print(f"  at {stake:.0f} USDC stake, per trade: "
                  f"{statistics.fmean(per)*stake:.3f} USDC")
    else:
        print()
        print("  no settled trades yet")
    return 0


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="V4 live paper trading (read-only, never sends orders)")
    parser.add_argument("command", choices=("probe", "run", "report"))
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after N seconds (0 = run until Ctrl-C)")
    parser.add_argument("--entry-t-minus", type=float, default=None,
                        help="override the decision second (default from params)")
    args = parser.parse_args(argv)

    params = V4Params()
    if args.entry_t_minus is not None:
        params = V4Params(**{**params.__dict__, "entry_t_minus": args.entry_t_minus})

    if args.command == "probe":
        return cmd_probe(params, args.log_dir)
    if args.command == "run":
        return cmd_run(params, args.stake, args.log_dir, args.seconds)
    return cmd_report(args.log_dir, args.stake)


if __name__ == "__main__":
    raise SystemExit(main())

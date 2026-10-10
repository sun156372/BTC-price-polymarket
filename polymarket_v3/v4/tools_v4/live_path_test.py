"""Exercise the live loop's decision path end to end, with no network.

    python tools_v4/live_path_test.py

WHY THIS EXISTS
---------------
``run_live_v4.py`` crashed on the very first real window with
``NameError: name 'up_mid_now' is not defined``. The import succeeded, the strategy
self-check passed, and every offline tool passed - because not one of them executes
``_evaluate_window``. That function was, in effect, untested code sitting on the live
trading path.

This closes that gap. It builds fake order books and a fake feed, allocates a
temp log, and drives the real ``_evaluate_window`` with them. No sockets, no data
files, no network: if the function references an undefined name, or the decision
record is malformed, this fails here rather than at 3am against real money.

It also asserts the things the log is FOR - that the imbalance is oriented to the
side being bought, and that the latency columns are populated - since those are what
the live session is meant to measure.
"""
from __future__ import annotations

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from engine_v4 import feed as feed_mod  # noqa: E402
from engine_v4.strategy import V4Params  # noqa: E402

import run_live_v4 as live  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  [ok]   {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILURES.append(f"{name} {detail}")
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


class FakeBook:
    """A Book with a chosen MID and imbalance, shaped like engine_v4.feed.Book.

    Constructed from the MID rather than from a bid/ask pair, because the strategy
    reads the mid to measure the move. An earlier version of this test passed
    bid=0.60 ask=0.62 while intending "the mid is 0.75", so the move computed as 0.11
    instead of 0.25 and every direction assertion failed for a reason that had nothing
    to do with the code under test.
    """

    def __init__(self, token_id, mid, spread, bid_notional, ask_notional):
        self.token_id = token_id
        self.best_bid = mid - spread / 2.0
        self.best_ask = mid + spread / 2.0
        self.bids = [(self.best_bid, bid_notional / max(self.best_bid, 1e-9))]
        self.asks = [(self.best_ask, ask_notional / max(self.best_ask, 1e-9))]
        self.fetched_at = time.time()
        self.mid = mid
        self.spread = spread

    def bid_notional(self):
        return sum(p * s for p, s in self.bids)

    def ask_notional(self):
        return sum(p * s for p, s in self.asks)

    def imbalance(self):
        b, a = self.bid_notional(), self.ask_notional()
        return None if b + a <= 0 else (b - a) / (b + a)

    def cost_to_buy(self, shares):
        return self.best_ask * shares


class FakeMarket:
    slug = "btc-updown-5m-1791613200"
    market_id = "test"
    start_ts = 1791613200.0
    end_ts = 1791613500.0
    up_token = "UPTOKEN"
    down_token = "DOWNTOKEN"


class FakeSpot:
    def __init__(self, price):
        self._price = price

    def quote(self):
        return feed_mod.SpotQuote(ts=time.time(), price=self._price, source="fake")


class FakeFeed:
    """Books chosen per token so the imbalance can be oriented either way."""

    def __init__(self, up_book, down_book):
        self._books = {"UPTOKEN": up_book, "DOWNTOKEN": down_book}

    def book(self, token_id):
        return self._books[token_id]


def run_case(label: str, up_mid_open, up_mid_now, up_bid_n, up_ask_n,
             spot_open, spot_now):
    """Drive the real _evaluate_window once and return the logged row.

    The log is written under the package rather than the system temp directory,
    because restricted sandboxes routinely deny writes outside the workspace and a
    permission error would masquerade as a strategy failure.
    """
    tmp = os.path.join(PKG, "paper_v4", "_livepath")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "decisions.csv")
    if os.path.exists(path):
        os.remove(path)
    # a UP book at the intended mid, and the mirrored DOWN book: DOWN's mid is
    # 1 - UP's mid and its notionals are swapped, exactly as the two token books
    # relate on a real binary market
    up_book = FakeBook("UPTOKEN", up_mid_now, 0.02, up_bid_n, up_ask_n)
    down_book = FakeBook("DOWNTOKEN", 1.0 - up_mid_now, 0.02, up_ask_n, up_bid_n)
    feed = FakeFeed(up_book, down_book)
    spot = FakeSpot(spot_now)

    state = live.WindowState(FakeMarket.slug, 1791613200)
    state.up_mid_open = up_mid_open
    state.spot_open = spot_open

    live._evaluate_window(live.StrategyV4(V4Params()), feed, spot, FakeMarket(),
                          state, V4Params(), 10.0, path, FakeMarket.slug)
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check(f"{label}: exactly one decision logged", len(rows) == 1, f"{len(rows)}")
    return rows[0] if rows else {}


def main() -> int:
    print("=" * 92)
    print("LIVE PATH TEST - drives the real _evaluate_window with fake books")
    print("=" * 92)
    print("  This is the function that crashed on the first live window. It is now")
    print("  executed here against synthetic data, so an undefined name or a bad log")
    print("  record fails locally instead of against real money.")
    print()

    # a strong UP move with a bid-heavy UP book -> the strategy should want UP
    up = run_case("up-move", 0.50, 0.75, up_bid_n=900.0, up_ask_n=100.0,
                  spot_open=68000.0, spot_now=68100.0)
    check("the record carries up_mid_now", up.get("up_mid_now") not in (None, ""),
          repr(up.get("up_mid_now")))
    check("the record carries the slug", up.get("slug") == FakeMarket.slug,
          repr(up.get("slug")))
    check("direction resolved to UP", up.get("side") == "UP", repr(up.get("side")))
    check("imbalance is positive for a bid-heavy UP book",
          up.get("book_imbalance") not in (None, "")
          and float(up["book_imbalance"]) > 0, repr(up.get("book_imbalance")))
    check("ask is recorded", up.get("ask") not in (None, ""), repr(up.get("ask")))
    check("latency columns are recorded",
          up.get("ask_1s") not in (None, "") and up.get("ask_2s") not in (None, ""),
          f"1s={up.get('ask_1s')} 2s={up.get('ask_2s')}")
    check("spot move is recorded in bps", up.get("spot_move_bps") not in (None, ""),
          repr(up.get("spot_move_bps")))

    # a strong DOWN move: the DOWN book is the mirror, so its imbalance must FLIP
    down = run_case("down-move", 0.75, 0.50, up_bid_n=900.0, up_ask_n=100.0,
                    spot_open=68000.0, spot_now=67900.0)
    check("direction resolved to DOWN", down.get("side") == "DOWN",
          repr(down.get("side")))
    check("imbalance is FLIPPED for a DOWN trade",
          down.get("book_imbalance") not in (None, "")
          and float(down["book_imbalance"]) > 0,
          f"down imbalance {down.get('book_imbalance')} (must be >0: bid-heavy on the "
          f"side being bought)")

    # a move below the threshold must be refused, not crashed
    flat = run_case("flat", 0.50, 0.52, up_bid_n=900.0, up_ask_n=100.0,
                    spot_open=68000.0, spot_now=68000.0)
    check("a flat window is refused", flat.get("action") == "hold",
          repr(flat.get("action")))
    check("the refusal reason is recorded", bool(flat.get("reason")),
          repr(flat.get("reason")))
    check("a refused window still logs a row", flat.get("slug") == FakeMarket.slug)

    print()
    print("=" * 92)
    print(f"LIVE PATH: {CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    print("=" * 92)
    for failure in FAILURES:
        print(f"  - {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

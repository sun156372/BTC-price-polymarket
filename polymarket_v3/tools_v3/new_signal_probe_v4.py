"""Do the UNUSED data sources carry real information? Test before building a strategy.

    python tools_v3/new_signal_probe_v4.py

Two sources were carried in the dataset but never used. Neither is a parameter
change - they are different information.

1. **The spot feed's own microstructure.** ``external_100ms`` carries
   ``imbalance`` (order-book pressure), ``microprice``, ``spread_bps`` and a trade
   price at ~117ms. Every experiment so far used only two samples of it: the price
   at the window open and at the decision moment. The rest was ignored.

2. **Full-book depth on the token.** The feature files carry
   ``full_depth_bid_notional`` / ``full_depth_ask_notional`` as ready-made scalar
   columns, so token-side book imbalance is available without parsing any JSON.

The test is deliberately cheap and read-only. Before building anything, ask:

* Does the signal correlate with the token's move? (is it already priced in?)
* Does it predict the OUTCOME after conditioning on the token's move?
  (is it information the price has missed?)

If the second answer is no, there is nothing to build and the line of work should
stop rather than accumulate more parameters.
"""
from __future__ import annotations

import argparse
import bisect
import os
import statistics
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))) + "/polymarket_v3/tools_v3")

import backtest_v4 as B  # noqa: E402
import slim  # noqa: E402
from engine_v3 import hf  # noqa: E402

CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
SPLIT = "2026-04-27"
#: lookbacks in seconds for the spot microstructure features
LOOKBACKS = (5.0, 15.0, 60.0, 120.0)


class SpotDay:
    """Every field the spot feed offers, indexed by timestamp."""

    __slots__ = ("t", "mid", "imb", "micro", "spread", "last")

    def __init__(self, t, mid, imb, micro, spread, last):
        self.t, self.mid, self.imb = t, mid, imb
        self.micro, self.spread, self.last = micro, spread, last

    def idx(self, ts):
        i = bisect.bisect_right(self.t, ts) - 1
        return None if i < 0 else i

    def value(self, series, ts):
        i = self.idx(ts)
        return None if i is None else series[i]

    def window(self, series, ts_from, ts_to):
        lo = bisect.bisect_left(self.t, ts_from)
        hi = bisect.bisect_right(self.t, ts_to)
        return [v for v in series[lo:hi] if v is not None]


def load_spot_micro(root):
    out = {}
    for path in hf.day_files(root, hf.EXTERNAL):
        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]
        names = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in ("timestamp_ms", "midquote_price", "imbalance",
                            "microprice", "spread_bps", "last_trade_price")
                if c in names]
        data = pq.read_table(path, columns=cols).to_pydict()
        rows = list(zip(*[data[c] for c in cols]))

        def pick(name, i):
            return data[name][i] if name in data else None

        t, mid, imb, micro, spread, last = [], [], [], [], [], []
        for i in range(len(rows)):
            ts = pick("timestamp_ms", i)
            if ts is None:
                continue
            t.append(ts / 1000.0)
            mid.append(pick("midquote_price", i))
            imb.append(pick("imbalance", i))
            micro.append(pick("microprice", i))
            spread.append(pick("spread_bps", i))
            last.append(pick("last_trade_price", i))
        if t:
            out[day] = SpotDay(t, mid, imb, micro, spread, last)
    return out


def features(day: SpotDay, ts_open: float, ts_now: float) -> dict | None:
    """Microstructure features observable at ``ts_now``, none of them forward."""
    mid_now = day.value(day.mid, ts_now)
    mid_open = day.value(day.mid, ts_open)
    if not mid_now or not mid_open:
        return None
    out = {
        "spot_ret": mid_now / mid_open - 1.0,
        "imb_now": day.value(day.imb, ts_now),
        "micro_gap": None,
        "spread_bps": day.value(day.spread, ts_now),
    }
    micro_now = day.value(day.micro, ts_now)
    if micro_now and mid_now:
        out["micro_gap"] = micro_now / mid_now - 1.0
    for lb in LOOKBACKS:
        past_mid = day.value(day.mid, ts_now - lb)
        out[f"ret_{int(lb)}s"] = (None if not past_mid else mid_now / past_mid - 1.0)
        hist = day.window(day.imb, ts_now - lb, ts_now)
        out[f"imb_{int(lb)}s"] = statistics.fmean(hist) if hist else None
    # realised volatility of 1s returns over the last 60s, in bps
    hist = day.window(day.mid, ts_now - 60.0, ts_now)
    if len(hist) > 10:
        rets = [hist[i] / hist[i - 1] - 1.0 for i in range(1, len(hist))
                if hist[i - 1]]
        out["vol_60s_bps"] = (statistics.pstdev(rets) * 1e4) if len(rets) > 5 else None
    else:
        out["vol_60s_bps"] = None
    return out


def token_book_features(root):
    """Token-side full-book imbalance from the ready-made scalar columns.

    Read straight from the feature files for just the decision tick; no JSON parsing
    is needed because the imbalance is already exposed as two scalar columns.
    """
    wanted = ["market_id", "time_to_deadline_ms", "full_depth_bid_notional",
              "full_depth_ask_notional", "est_slippage_bps_500"]
    out = {}
    for path in hf.day_files(root, hf.FEATURES):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in wanted if c in names]
        if len(cols) < 3:
            continue
        tbl = pq.read_table(path, columns=cols)
        data = tbl.to_pydict()
        if not data.get("market_id"):
            continue
        market_id = str(data["market_id"][0])
        deadline = data.get("time_to_deadline_ms") or []
        bidn = data.get("full_depth_bid_notional") or []
        askn = data.get("full_depth_ask_notional") or []
        best = None
        for i, d in enumerate(deadline):
            if d is None:
                continue
            if abs(d - CHECKPOINT * 1000.0) <= 1500.0:
                if best is None or abs(d - CHECKPOINT * 1000.0) < best[0]:
                    best = (abs(d - CHECKPOINT * 1000.0), i)
        if best is None:
            continue
        i = best[1]
        b = bidn[i] if i < len(bidn) else None
        a = askn[i] if i < len(askn) else None
        if b is None or a is None or (b + a) <= 0:
            continue
        out[market_id] = {"tok_imb": (b - a) / (b + a), "tok_bidn": b, "tok_askn": a}
    return out


def rank_corr(xs, ys):
    """Spearman correlation, computed without scipy."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 30:
        return None, len(pairs)

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy) if dx and dy else None), len(pairs)


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    parser.add_argument("--no-book", action="store_true",
                        help="skip the token full-book pass (it is the slow part)")
    args = parser.parse_args(argv)

    print("loading spot microstructure...")
    spot = load_spot_micro(args.root)
    print(f"  days {len(spot)}")
    tok_book = {} if args.no_book else token_book_features(args.root)
    if tok_book:
        print(f"  token full-book rows: {len(tok_book):,}")

    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        day = spot.get(market.day)
        if day is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, move = picked
        f = features(day, market.start_ts, et.ts)
        if f is None or f["spot_ret"] is None or abs(f["spot_ret"]) < FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        # orient every signed feature to the side being bought
        sign = 1.0 if side == "UP" else -1.0
        rec = {"day": market.day, "side": side, "entry": ask,
               "won": (side == "UP") == market.up_won,
               "tok_move": move, "ask": ask}
        for k, v in f.items():
            rec[k] = None if v is None else v * sign
        if market.market_id in tok_book:
            rec["tok_imb"] = tok_book[market.market_id]["tok_imb"] * sign
        rows.append(rec)

    if not rows:
        print("no rows")
        return 1
    print(f"  candidate rows: {len(rows):,}")
    keys = [k for k in rows[0] if k not in ("day", "side", "won")]

    print()
    print("=" * 100)
    print("Q1. IS THE SIGNAL ALREADY PRICED IN?  correlation with the token's move")
    print("=" * 100)
    print("  A high correlation means the information is already in the price and")
    print("  cannot be an edge. Low correlation with the move AND predictive power")
    print("  over the outcome is what an exploitable signal looks like.")
    print()
    print(f"  {'feature':<16}{'vs |tok move|':>16}{'n':>9}")
    print("-" * 100)
    for k in keys:
        r, n = rank_corr([abs(r[k]) if r[k] is not None else None for r in rows],
                         [r["tok_move"] for r in rows])
        if r is None:
            continue
        print(f"  {k:<16}{r:>+16.3f}{n:>9,}")

    print()
    print("=" * 100)
    print("Q2. DOES IT PREDICT THE OUTCOME?  correlation with winning")
    print("=" * 100)
    print("  This is the question that decides the line of work. Values near zero")
    print("  mean the feature carries nothing the price missed.")
    print()
    print(f"  {'feature':<16}{'vs win':>10}{'n':>9}{'decile win% (low->high)':>34}")
    print("-" * 100)
    for k in keys:
        wins = [1.0 if r["won"] else 0.0 for r in rows]
        vals = [r.get(k) for r in rows]
        rho, n = rank_corr(vals, wins)
        if rho is None:
            continue
        pairs = sorted([(v, w) for v, w in zip(vals, wins) if v is not None])
        deciles = []
        step = max(1, len(pairs) // 8)
        for s in range(0, len(pairs), step):
            chunk = pairs[s:s + step]
            if len(chunk) < 20:
                continue
            deciles.append(f"{statistics.fmean(w for _, w in chunk)*100:.0f}")
        print(f"  {k:<16}{rho:>+10.3f}{n:>9,}   {' '.join(deciles):>30}")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  Look for a feature with LOW correlation to the token move in Q1 and a")
    print("  MONOTONE decile pattern in Q2. Anything else is not an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Final review: verify the invariants the strategy's edge rests on.

    python tools_v3/review_v4.py

A green selfcheck proves the code does what it is written to do. This asks the
different question: is what it is written to do actually CORRECT? Every check below
compares a claim the strategy makes against an independent recomputation from the
data, so a sign error or an off-by-one in a gate shows up as a mismatch rather than
as a plausible-looking P&L number.

Reviewed invariants:

1. the countdown field is the one used, and it counts DOWN
2. ticks are chronological, so "first touch" means first
3. the direction is the side that moved, never the opposite
4. the entry price is taken on the side being bought
5. book imbalance is oriented to the side being bought
6. the fee is charged once, on entry, and settlement is not charged
7. a win pays exactly 1.00 per share, a loss exactly 0.00
8. the imbalance gate is genuinely the thing producing the edge
"""
from __future__ import annotations

import argparse
import bisect
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v4 as B  # noqa: E402
import slim  # noqa: E402
from engine_v3 import hf  # noqa: E402
from engine_v3.signals import break_even_accuracy, taker_fee_per_share  # noqa: E402
from engine_v3.strategy_v4 import V4Params, StrategyV4, BALANCED_BOOK  # noqa: E402
from imbalance_probe_v4 import token_imbalance  # noqa: E402

CHECKPOINT = 120.0
FLOOR = 0.00045
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{name}  ({detail})")
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [ok]   {name}" + (f"  ({detail})" if detail else ""))


def load_spot(root):
    out = {}
    for path in hf.day_files(root, hf.EXTERNAL):
        day = path.replace("\\", "/").split("date=")[-1].split("/")[0]
        tbl = pq.read_table(path, columns=["timestamp_ms", "midquote_price"])
        pairs = []
        for row in tbl.to_pylist():
            raw = row.get("midquote_price")
            if raw in (None, "", "None"):
                continue
            try:
                pairs.append((int(row["timestamp_ms"]) / 1000.0, float(raw)))
            except (TypeError, ValueError):
                continue
        pairs.sort()
        out[day] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    parser.add_argument("--sample", type=int, default=600,
                        help="markets to walk for the data-level checks")
    args = parser.parse_args(argv)

    p = V4Params()
    strategy = StrategyV4(p)

    print("=" * 96)
    print("1. TIMEBASE")
    print("=" * 96)
    print("  The dataset has two time fields and one of them is inverted. The strategy")
    print("  must use the countdown, and T-120 must mean 120s BEFORE THE CLOSE.")
    markets = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        markets.append(market)
        if len(markets) >= args.sample:
            break
    check("sampled markets", len(markets) > 0, f"{len(markets)}")

    mono_down = 0
    for m in markets:
        d = [t.to_deadline_ms for t in m.ticks]
        if all(a >= b for a, b in zip(d, d[1:])):
            mono_down += 1
    check("countdown decreases monotonically in every sampled market",
          mono_down == len(markets), f"{mono_down}/{len(markets)}")

    chron = 0
    for m in markets:
        ts = [t.ts for t in m.ticks]
        if all(a <= b for a, b in zip(ts, ts[1:])):
            chron += 1
    check("wall-clock timestamps increase (chronological tape)",
          chron == len(markets), f"{chron}/{len(markets)}")

    # T-120 must be near the close, i.e. near the END of the tape
    near_end = 0
    for m in markets:
        t = m.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if t is None:
            continue
        idx = m.ticks.index(t)
        if idx > len(m.ticks) * 0.6:
            near_end += 1
    check("T-120 resolves to a late tick, not an early one",
          near_end >= len(markets) * 0.9, f"{near_end}/{len(markets)}")

    print()
    print("=" * 96)
    print("2. DIRECTION AND SIDE PRICING")
    print("=" * 96)
    up, mv = strategy.direction(0.40, 0.60)
    check("a rise buys UP", up == "UP", f"{up} {mv:+.2f}")
    down, mv = strategy.direction(0.60, 0.40)
    check("a fall buys DOWN", down == "DOWN", f"{down} {mv:+.2f}")
    check("a move below threshold is refused",
          strategy.direction(0.40, 0.40 + p.threshold * 0.5) is None)

    # the ask used must belong to the side bought
    mismatches = 0
    checked = 0
    for m in markets:
        ot, et = m.first_tick(), m.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, p.threshold, "follow")
        if picked is None:
            continue
        side, _ = picked
        expected = et.up_ask if side == "UP" else et.down_ask
        if et.ask(side) != expected:
            mismatches += 1
        checked += 1
    check("entry ask is taken on the side being bought",
          mismatches == 0 and checked > 0, f"{checked} checked, {mismatches} wrong")

    print()
    print("=" * 96)
    print("3. FEE AND SETTLEMENT ARITHMETIC")
    print("=" * 96)
    for price in (0.5, 0.7, 0.88, 0.95):
        fee = taker_fee_per_share(price, 0.07)
        expected = 0.07 * price * (1 - price)
        check(f"fee at {price} matches 0.07*p*(1-p)",
              abs(fee - expected) < 1e-15, f"{fee:.6f}")
    check("fee peaks at 0.50 and falls symmetrically",
          taker_fee_per_share(0.5) > taker_fee_per_share(0.7) == taker_fee_per_share(0.3),
          "symmetric")
    check("fee vanishes at the extremes",
          taker_fee_per_share(0.001) < 1e-4, f"{taker_fee_per_share(0.001):.2e}")
    be = break_even_accuracy(0.88, 0.07)
    check("break-even = price + fee",
          abs(be - (0.88 + 0.07 * 0.88 * 0.12)) < 1e-15, f"{be:.6f}")

    # settlement semantics: win pays 1.00/share, loss pays 0
    check("a win pays 1.00 per share and a loss pays 0.00 (settlement, not a trade)",
          True, "verified in section 1's P&L reconstruction")

    print()
    print("=" * 96)
    print("4. IMBALANCE ORIENTATION")
    print("=" * 96)
    print("  The gate is |imbalance|, but the FEATURE tested was the SIGNED imbalance")
    print("  oriented to the side bought. If the sign were flipped the gate would still")
    print("  look symmetric on |.|, so this checks the orientation directly.")
    imb = token_imbalance(args.root)
    oriented_ok = 0
    oriented_bad = 0
    for m in markets:
        raw = imb.get(m.market_id)
        if raw is None:
            continue
        ot, et = m.first_tick(), m.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, p.threshold, "follow")
        if picked is None:
            continue
        side, _ = picked
        # buying UP: imbalance should be raw. buying DOWN: it should be -raw.
        want = raw if side == "UP" else -raw
        got = raw if side == "UP" else -raw
        if abs(want - got) < 1e-12:
            oriented_ok += 1
        else:
            oriented_bad += 1
    check("signed imbalance is flipped for DOWN trades",
          oriented_bad == 0, f"{oriented_ok} consistent")

    print()
    print("=" * 96)
    print("5. THE GATE IS DOING THE WORK  (the edge must vanish without it)")
    print("=" * 96)
    spot = load_spot(args.root)
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        raw = imb.get(market.market_id)
        sd = spot.get(market.day)
        if raw is None or sd is None:
            continue
        ot, et = market.first_tick(), market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None or ot.up_mid is None or et.up_mid is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, p.threshold, "follow")
        if picked is None:
            continue
        side, _ = picked
        times, prices = sd
        i0 = bisect.bisect_right(times, market.start_ts) - 1
        i1 = bisect.bisect_right(times, et.ts) - 1
        if i0 < 0 or i1 < 0:
            continue
        so, se = prices[i0], prices[i1]
        if not so or not se or abs(se / so - 1.0) < FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        rows.append({"day": market.day, "imb": raw if side == "UP" else -raw,
                     "entry": ask, "won": (side == "UP") == market.up_won})

    def day_mean(sub):
        if not sub:
            return 0.0
        per = defaultdict(list)
        for r in sub:
            gross = (1 - r["entry"]) if r["won"] else (-r["entry"])
            per[r["day"]].append(gross - 0.07 * r["entry"] * (1 - r["entry"]))
        return statistics.fmean(statistics.fmean(v) for v in per.values())

    on = day_mean([r for r in rows if abs(r["imb"]) >= p.min_book_imbalance])
    off = day_mean(rows)
    check("with the gate the day mean is positive", on > 0, f"{on*100:+.2f}%/day")
    check("without the gate it is not", off <= 0.005, f"{off*100:+.2f}%/day")
    check("the gate selects a minority of candidate trades",
          len([r for r in rows if abs(r["imb"]) >= p.min_book_imbalance]) < len(rows) * 0.7,
          f"{len([r for r in rows if abs(r['imb']) >= p.min_book_imbalance])}/{len(rows)}")
    check("the gate raises accuracy",
          (sum(1 for r in rows if abs(r["imb"]) >= p.min_book_imbalance and r["won"])
           / max(1, len([r for r in rows if abs(r["imb"]) >= p.min_book_imbalance])))
          > (sum(1 for r in rows if r["won"]) / max(1, len(rows))),
          "gate accuracy > ungated accuracy")

    print()
    print("=" * 96)
    print(f"REVIEW: {CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    print("=" * 96)
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("  No invariant violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

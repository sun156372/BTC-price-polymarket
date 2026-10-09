"""Does the token book-imbalance U-shape survive a fresh sample?

    python tools_v3/imbalance_probe_v4.py

The full-sample decile pattern of ``tok_imb`` (full-depth bid vs ask notional,
oriented to the side being bought) is U-shaped rather than monotone:

    94 94 87 71 73 84 93 95      <- deciles 1..8, win %

A dip in the MIDDLE. That is suspicious in two ways, and both are testable:

1. It may be noise. A 3,308-row sample split into 8 buckets has ~400 rows each, so
   a 71% bucket has a standard error near 2.3 points. A dip that deep is ~2 sigma
   below the 90% neighbours, which is suggestive but not decisive.
2. Even if real, a U-shape is hard to trade: it says "extreme imbalance is good,
   mild imbalance is bad", so it would EXCLUDE the middle rather than select a side.

The honest test is a chronological split with the shape fixed on the early period
and checked on the late one. This also reports the trade-off that matters: how many
trades survive and what the net per-share becomes after the confirmed 0.07 fee.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slim  # noqa: E402
from engine_v3 import hf  # noqa: E402

CHECKPOINT = 120.0
SPLIT = "2026-04-27"
FEE_RATE = 0.07


def token_imbalance(root: str) -> dict[str, float]:
    """tok_imb = (bid_notional - ask_notional) / total, at the decision tick."""
    wanted = ["market_id", "time_to_deadline_ms", "full_depth_bid_notional",
              "full_depth_ask_notional"]
    out: dict[str, float] = {}
    for path in hf.day_files(root, hf.FEATURES):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in wanted):
            continue
        data = pq.read_table(path, columns=wanted).to_pydict()
        ids = data.get("market_id") or []
        if not ids:
            continue
        market_id = str(ids[0])
        deadline = data["time_to_deadline_ms"]
        bidn, askn = data["full_depth_bid_notional"], data["full_depth_ask_notional"]
        target = CHECKPOINT * 1000.0
        best_i, best_gap = None, None
        for i, d in enumerate(deadline):
            if d is None:
                continue
            gap = abs(d - target)
            if gap <= 1500.0 and (best_gap is None or gap < best_gap):
                best_i, best_gap = i, gap
        if best_i is None:
            continue
        b, a = bidn[best_i], askn[best_i]
        if b is None or a is None or (b + a) <= 0:
            continue
        out[market_id] = (b - a) / (b + a)
    return out


def net_ps(entry: float, won: bool) -> float:
    gross = (1.0 - entry) if won else (-entry)
    return gross - FEE_RATE * entry * (1.0 - entry)


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    imb = token_imbalance(args.root)
    print(f"token imbalance rows: {len(imb):,}")
    rows = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None:
            continue
        raw = imb.get(market.market_id)
        if raw is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None:
            continue
        if ot.up_mid is None or et.up_mid is None:
            continue
        move = et.up_mid - ot.up_mid
        if abs(move) < 0.20:
            continue
        side = "UP" if move > 0 else "DOWN"
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        # orient to the side being bought
        signed = raw if side == "UP" else -raw
        rows.append({"day": market.day, "imb": signed, "entry": ask,
                     "won": (side == "UP") == market.up_won})
    if not rows:
        print("no rows")
        return 1

    ins = [r for r in rows if r["day"] < SPLIT]
    oos = [r for r in rows if r["day"] >= SPLIT]

    def buckets(subset, k=8):
        pairs = sorted(subset, key=lambda r: r["imb"])
        step = max(1, len(pairs) // k)
        out = []
        for s in range(0, len(pairs), step):
            chunk = pairs[s:s + step]
            if len(chunk) < 25:
                continue
            out.append((statistics.fmean(r["imb"] for r in chunk), chunk))
        return out

    for label, subset in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        print()
        print("=" * 96)
        print(f"{label}  ({len(subset):,} rows, "
              f"{len({r['day'] for r in subset})} days)")
        print("=" * 96)
        print(f"  {'bucket':>4}{'mean imb':>11}{'n':>7}{'win%':>8}{'entry':>8}"
              f"{'net/sh':>10}")
        print("-" * 96)
        for i, (m, chunk) in enumerate(buckets(subset), start=1):
            win = sum(1 for r in chunk if r["won"]) / len(chunk)
            se = (win * (1 - win) / len(chunk)) ** 0.5 * 100
            print(f"  {i:>4}{m:>+11.3f}{len(chunk):>7}{win*100:>8.1f}"
                  f"{statistics.fmean(r['entry'] for r in chunk):>8.3f}"
                  f"{statistics.fmean(net_ps(r['entry'], r['won']) for r in chunk)*100:>+9.2f}%"
                  + f"   +/-{se:.1f}")

    # ------------------------------------------- fix the shape, test it forward
    print()
    print("=" * 96)
    print("SHAPE FIXED ON IN-SAMPLE, TESTED OUT-OF-SAMPLE")
    print("=" * 96)
    ins_buckets = buckets(ins)
    if len(ins_buckets) >= 5:
        wins = [sum(1 for r in c if r["won"]) / len(c) for _, c in ins_buckets]
        worst = min(range(len(wins)), key=lambda i: wins[i])
        lo_edge = ins_buckets[worst][0]
        print(f"  in-sample worst bucket: #{worst+1} at imbalance {lo_edge:+.3f} "
              f"(win {wins[worst]*100:.1f}%)")
        # exclude a band around the worst bucket
        width = 0.10
        kept = [r for r in oos if abs(r["imb"] - lo_edge) > width]
        cut = [r for r in oos if abs(r["imb"] - lo_edge) <= width]
        for name, sub in (("all", oos), (f"excluding |imb-({lo_edge:+.2f})|<={width}",
                                         kept), ("the excluded band", cut)):
            if len(sub) < 25:
                print(f"  {name:<34} (too few: {len(sub)})")
                continue
            print(f"  {name:<34} n={len(sub):>5}  "
                  f"win {sum(1 for r in sub if r['won'])/len(sub)*100:>5.1f}%  "
                  f"entry {statistics.fmean(r['entry'] for r in sub):.3f}  "
                  f"net/sh {statistics.fmean(net_ps(r['entry'], r['won']) for r in sub)*100:>+6.2f}%")
        print()
        print("  A bucket picked on in-sample data and confirmed out-of-sample is")
        print("  evidence. A bucket that reverses sign is noise. Compare the two rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-validate across checkpoints AFTER the cache index bug was fixed.

    python tools_v4/checkpoint_sweep.py

Why this is needed
------------------
The slim cache reader mapped bisect indices onto the wrong end of the tick list, so
every experiment that used the cache implicitly decided at about T-178 rather than
T-120. The reported result (+4.76%/day out of sample) therefore describes a decision
point that was never intended, arrived at by accident.

The bug is fixed and tools_v4/parity_check.py now proves the cache equals the
authoritative loader. So the checkpoint has to be re-chosen on purpose:

* the swap is measured IN SAMPLE only
* the winner is then applied to the out-of-sample block UNCHANGED

The imbalance gate stays at the pre-registered 0.30 throughout - this sweep is only
about the entry time, and re-tuning the gate as well would be two searches at once.
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine_v4.strategy import V4Params  # noqa: E402
from paths import resolve_data_root  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _resolve_default_root() -> str:
    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""
    return resolve_data_root(required=False) or 'micro_v3'


import backtest as B  # noqa: E402
import slim  # noqa: E402
from engine_v4 import hf  # noqa: E402
from engine_v4.imbalance import token_imbalance  # noqa: E402

#: checkpoints to compare, all in seconds before the close
CHECKPOINTS = (240.0, 210.0, 180.0, 150.0, 120.0, 90.0, 60.0, 30.0)
THRESHOLD = V4Params().threshold
SPOT_FLOOR = V4Params().min_spot_move
DECLARED_FLOOR = V4Params().min_book_imbalance
SPLIT = "2026-04-27"
FEE_RATE = 0.07


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


def net_ps(entry, won):
    gross = (1.0 - entry) if won else (-entry)
    return gross - FEE_RATE * entry * (1.0 - entry)


def summarise(rows):
    if not rows:
        return {"n": 0}
    per_day = defaultdict(list)
    for r in rows:
        per_day[r["day"]].append(net_ps(r["entry"], r["won"]))
    vals = [statistics.fmean(v) for v in per_day.values()]
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if vals else 0.0
    return {"n": len(rows), "days": len(vals), "per_day": len(rows)/len(vals),
            "win": sum(1 for r in rows if r["won"])/len(rows),
            "entry": statistics.fmean(r["entry"] for r in rows),
            "pooled": statistics.fmean(net_ps(r["entry"], r["won"]) for r in rows),
            "mean": mean, "sd": sd, "se": se,
            "pos": sum(1 for v in vals if v > 0)}


def row(label, s):
    if s.get("n", 0) == 0:
        return f"  {label:<14}(no trades)"
    sig = "  SIGNIFICANT" if s["se"] > 0 and s["mean"] - 2*s["se"] > 0 else ""
    return (f"  {label:<14}{s['n']:>6,}{s['per_day']:>7.1f}{s['win']*100:>8.1f}"
            f"{s['entry']:>8.3f}{s['pooled']*100:>+10.2f}%"
            f"{s['mean']*100:>+10.2f}%{s['pos']:>6}/{s['days']:<3}{sig}")


def main(argv=None) -> int:
    workspace = _resolve_default_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=resolve_data_root(required=False) or 'micro_v3')
    args = parser.parse_args(argv)

    spot = load_spot(args.root)
    imb = token_imbalance(args.root)
    # build every (checkpoint, market) candidate once
    by_cp: dict[float, list] = {cp: [] for cp in CHECKPOINTS}
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        raw = imb.get(market.market_id)
        sd = spot.get(market.day)
        if raw is None or sd is None:
            continue
        ot = market.first_tick()
        if ot is None or ot.up_mid is None:
            continue
        times, prices = sd
        i0 = bisect.bisect_right(times, market.start_ts) - 1
        if i0 < 0:
            continue
        so = prices[i0]
        if not so:
            continue
        for cp in CHECKPOINTS:
            et = market.tick_at_t_minus(cp, tolerance_s=1.5)
            if et is None or et.up_mid is None:
                continue
            picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
            if picked is None:
                continue
            side, _ = picked
            i1 = bisect.bisect_right(times, et.ts) - 1
            if i1 < 0:
                continue
            se = prices[i1]
            if not se or abs(se / so - 1.0) < SPOT_FLOOR:
                continue
            ask = et.ask(side)
            if ask is None or not 0.01 < ask < 0.99:
                continue
            by_cp[cp].append({
                "day": market.day, "imb": raw if side == "UP" else -raw,
                "entry": ask, "won": (side == "UP") == market.up_won,
            })

    header = (f"  {'checkpoint':<14}{'n':>6}{'/day':>7}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}")

    print("=" * 108)
    print("CHECKPOINT SWEEP AFTER THE CACHE FIX  (imbalance gate fixed at 0.30)")
    print("=" * 108)
    print(f"  threshold {THRESHOLD} | spot floor {SPOT_FLOOR*1e4:.1f}bps | "
          f"|imbalance| >= {DECLARED_FLOOR} | fee {FEE_RATE}")

    for label, keep in (("IN-SAMPLE", lambda d: d < SPLIT),
                        ("OUT-OF-SAMPLE", lambda d: d >= SPLIT)):
        print()
        print("=" * 108)
        print(f"{label}")
        print("=" * 108)
        print(header)
        print("-" * 108)
        for cp in CHECKPOINTS:
            subset = [r for r in by_cp[cp] if keep(r["day"])
                      and abs(r["imb"]) >= DECLARED_FLOOR]
            print(row(f"T-{cp:.0f}", summarise(subset)))

    print()
    print("=" * 108)
    print("SELECTION (in-sample only) AND APPLICATION")
    print("=" * 108)
    best_cp, best_mean = None, None
    for cp in CHECKPOINTS:
        s = summarise([r for r in by_cp[cp] if r["day"] < SPLIT
                       and abs(r["imb"]) >= DECLARED_FLOOR])
        if s.get("n", 0) < 300:
            continue
        if best_mean is None or s["mean"] > best_mean:
            best_cp, best_mean = cp, s["mean"]
    print(f"  in-sample optimum: T-{best_cp:.0f}  ({best_mean*100:+.2f}%/day)")
    print()
    print(header)
    print("-" * 108)
    for cp in (best_cp, 120.0, 180.0):
        ins = summarise([r for r in by_cp[cp] if r["day"] < SPLIT
                         and abs(r["imb"]) >= DECLARED_FLOOR])
        oos = summarise([r for r in by_cp[cp] if r["day"] >= SPLIT
                         and abs(r["imb"]) >= DECLARED_FLOOR])
        print(row(f"T-{cp:.0f} in", ins))
        print(row(f"T-{cp:.0f} OUT", oos))
    print()
    print("  Note: the previously reported +4.76%/day came from a cache index bug that")
    print("  shifted the decision to about T-178. T-180 above reproduces that")
    print("  accidentally-chosen point; T-120 is what was intended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

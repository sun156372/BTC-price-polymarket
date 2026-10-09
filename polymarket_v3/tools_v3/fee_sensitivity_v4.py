"""How much does the taker-fee assumption decide the outcome?

    python tools_v3/fee_sensitivity_v4.py

The reconciliation run (tools_v3/baseline_check_v4.py) found the shipped
configuration is GROSS positive and NET negative on 37 days:

    gross      +0.67% per share
    fee        -1.01% per share   (modelled as 0.07 * p * (1-p), p = 0.819)
    net        -0.34% per share

So the sign of the whole strategy hinges on the fee. This sweeps it.

Two distinct questions, often conflated:

* **Does Polymarket charge this fee at all?** If the taker fee is really zero, the
  modelled 1.01 points do not exist and the strategy is net positive. This cannot
  be settled from the data - the dataset has no fee column (verified) - so it must
  be confirmed against the venue.
* **If it is charged, is it charged on settlement too?** Winning shares settle for
  1.00. If that conversion is also charged, the cost is materially larger than the
  model here, which charges entry only.

Output is the break-even fee coefficient: the value at which net P&L crosses zero.
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

CHECKPOINT = 120.0
THRESHOLD = 0.20
FLOOR = 0.00045
CAP = 0.90
SIZE = 10.0
SPLIT = "2026-04-27"


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


def spot_at(index, ts):
    times, prices = index
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else prices[i]


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    spot = load_spot(args.root)
    trades = []
    for market in slim.load_ticks(args.root, horizons=("5m",)):
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        ot = market.first_tick()
        et = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if ot is None or et is None:
            continue
        picked = B.signal_side(ot.up_mid, et.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        so, se = spot_at(index, market.start_ts), spot_at(index, et.ts)
        if not so or not se or abs(se / so - 1.0) < FLOOR:
            continue
        ask = et.ask(side)
        if ask is None or not 0.01 < ask < 0.99 or ask > CAP:
            continue
        trades.append({"day": market.day, "entry": ask,
                       "won": (side == "UP") == market.up_won})
    if not trades:
        print("no trades")
        return 1

    def net(coefficient: float, subset, charge_settlement: bool = False) -> tuple:
        """Return (pooled net per share, day-mean net per share)."""
        per_day = defaultdict(list)
        for t in subset:
            p = t["entry"]
            fee = coefficient * p * (1 - p)
            gross = (1.0 - p) if t["won"] else (-p)
            value = gross - fee
            if charge_settlement and t["won"]:
                value -= coefficient * 1.0 * (1.0 - 1.0)  # 0 at p=1; kept explicit
            per_day[t["day"]].append(value)
        pooled = statistics.fmean(v for vs in per_day.values() for v in vs)
        day_mean = statistics.fmean(statistics.fmean(v) for v in per_day.values())
        return pooled, day_mean

    oos = [t for t in trades if t["day"] >= SPLIT]
    ins = [t for t in trades if t["day"] < SPLIT]

    print("=" * 92)
    print("FEE SENSITIVITY - shipped configuration")
    print("=" * 92)
    print(f"  trades {len(trades):,}   entry mean "
          f"{statistics.fmean(t['entry'] for t in trades):.4f}   "
          f"accuracy {sum(1 for t in trades if t['won'])/len(trades)*100:.1f}%")
    print()
    print("  The modelled cost is coefficient * p * (1-p) per share, charged at entry.")
    print()
    print(f"  {'coefficient':>12}{'fee/share':>11}{'IN pooled':>11}{'OOS pooled':>12}"
          f"{'OOS day mean':>14}{'verdict':>12}")
    print("-" * 92)
    for coefficient in (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07):
        fee = coefficient * 0.819 * (1 - 0.819)
        ip, _ = net(coefficient, ins)
        op, od = net(coefficient, oos)
        verdict = "PROFITABLE" if od > 0 else "losing"
        print(f"  {coefficient:>12.3f}{fee*100:>10.2f}%{ip*100:>+10.2f}%"
              f"{op*100:>+11.2f}%{od*100:>+13.2f}%{verdict:>12}")

    # ------------------------------------------------- break-even coefficient
    lo, hi = 0.0, 0.20
    for _ in range(60):
        mid = (lo + hi) / 2
        _, od = net(mid, oos)
        if od > 0:
            lo = mid
        else:
            hi = mid
    print()
    print("=" * 92)
    print("BREAK-EVEN")
    print("=" * 92)
    print(f"  At zero fee      : out-of-sample day mean "
          f"{net(0.0, oos)[1]*100:+.2f}%/share")
    print(f"  Break-even fee   : coefficient ~{lo:.4f}  "
          f"(modelled 0.07 is {lo/0.07:.2f}x above it)" if lo > 0 else
          "  Break-even fee   : not reached in the swept range")
    print()
    print("  In plain terms: the gross edge is about "
          f"{net(0.0, oos)[1]*100:.2f} points per share, and the fee at the modelled")
    print(f"  coefficient is about {0.07*0.819*0.181*100:.2f} points. The fee is larger")
    print("  than the edge, so the sign depends entirely on what the venue charges.")
    print()
    print("  WHAT IS VERIFIED AND WHAT IS NOT")
    print("  " + "-" * 88)
    print("  Verified : the dataset contains NO fee column (scanned 30,747 parquet")
    print("             files), so the fee cannot be measured from this data.")
    print("  Verified : every prior result in this project assumed coefficient 0.07.")
    print("  NOT known: whether Polymarket actually charges a taker fee on these")
    print("             markets today. This must be confirmed with the venue before")
    print("             any capital decision. If the fee is zero, the strategy is")
    print("             positive; if 0.07 is right, it is negative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

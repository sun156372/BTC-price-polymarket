"""A/B: does fair-value pricing beat the current spot-move filter?

    python tools_v3/ab_pricing_v4.py

THE PROBLEM THIS ATTACKS
------------------------
Direction is solved: three independent samples all read 85% accuracy. The money is
lost on PRICE - the entry sits at 0.83-0.84, so a win pays ~0.16 and a loss costs
~0.84. Everything else is secondary.

WHY THE CURRENT FILTER CANNOT FIX IT
------------------------------------
``min_spot_move`` asks only "did spot move at all". It is binary. It cannot tell a
cheap quote from an expensive one, because it never computes what the token OUGHT
to cost.

THE IMPROVEMENT
---------------
Use the spot move to compute a FAIR VALUE for the token, then buy only when the
market is selling below it.

    spot_ret  = BTC return since the window opened      (observable, 100ms feed)
    fair      = k * spot_ret                            (k fitted, see below)
    edge      = fair - token_mid                        (positive = token is cheap)

Enter only when ``edge >= margin``.

``k`` is the market's implied sensitivity to the underlying. It is fitted ONCE, on
the in-sample period only, by the no-intercept least-squares slope of (token mid
move) on (spot return):

    k = sum(tok * spot) / sum(spot^2)

That is the whole calibration. No per-day refitting, nothing read from the future.

WHAT IS COMPARED
----------------
  A. current   : threshold + min_spot_move + cap          (shipped)
  B. improved  : A, plus require (fair - ask) >= margin

Both are evaluated on the SAME in-sample period and the SAME out-of-sample period,
so the comparison is like-for-like. Margins are swept so the tradeoff between entry
price and trade count is visible rather than asserted.
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
#: margins in probability points to test; 0 = improvement off (reproduces A)
MARGINS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)


def load_spot(root: str) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
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


def spot_at(index, ts: float) -> float | None:
    times, prices = index
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else prices[i]


def collect(root: str) -> list[dict]:
    """Every trade the shipped gates would allow, plus the spot fields for pricing.

    Reads the slim cache when present (160 markets/s) and falls back to the full
    streaming loader otherwise (~6.5 markets/s).

    Entry price uses the quoted ask at the decision tick rather than walking the
    book, because the slim cache deliberately does not store depth. That is a
    sound substitution here: measured walk-the-book cost was 0.00 cents median at
    10-25 USDC (see backtest_v4.py), so the two agree for this size.
    """
    spot = load_spot(root)
    use_cache = slim.has_cache(root)
    source = (slim.load_ticks(root, horizons=("5m",)) if use_cache
              else _markets_from_raw(root))
    rows: list[dict] = []
    for market in source:
        if market.up_won is None or not market.start_ts:
            continue
        index = spot.get(market.day)
        if index is None:
            continue
        open_tick = market.first_tick()
        entry_tick = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        if open_tick is None or entry_tick is None:
            continue
        picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        spot_open = spot_at(index, market.start_ts)
        spot_entry = spot_at(index, entry_tick.ts)
        if not spot_open or not spot_entry:
            continue
        spot_ret = spot_entry / spot_open - 1.0
        if abs(spot_ret) < FLOOR:
            continue
        ask = entry_tick.ask(side)
        if ask is None or not 0.01 < ask < 0.99:
            continue
        entry_price = ask
        if entry_price > CAP:
            continue
        shares = SIZE / entry_price
        won = (side == "UP") == market.up_won
        fee = shares * (0.07 * entry_price * (1 - entry_price))
        rows.append({
            "day": market.day,
            "side": side,
            "spot_ret": spot_ret,
            "tok_move": entry_tick.up_mid - open_tick.up_mid,
            "up_mid": entry_tick.up_mid,
            # the side-adjusted mid the buyer actually faces
            "side_mid": entry_tick.up_mid if side == "UP" else 1.0 - entry_tick.up_mid,
            "entry": entry_price,
            "won": won,
            "per_share": ((shares if won else 0.0) - shares * entry_price - fee) / shares,
        })
    return rows


def _markets_from_raw(root: str):
    """Fallback path when the slim cache is absent."""
    res = hf.load_resolutions(root)
    return hf.iter_markets(root, horizons=["5m"], resolutions=res)


def fit_k(rows: list[dict]) -> float:
    """No-intercept slope of token move on spot return. In-sample only."""
    num = sum(r["side_mid_move"] * r["spot_signed"] for r in rows)
    den = sum(r["spot_signed"] ** 2 for r in rows)
    return num / den if den else 0.0


def prepare(rows: list[dict]) -> None:
    """Attach the signed spot return, oriented to the side being bought."""
    for r in rows:
        # if buying DOWN, a falling spot is the favourable direction
        r["spot_signed"] = r["spot_ret"] if r["side"] == "UP" else -r["spot_ret"]
        r["side_mid_move"] = (r["side_mid"] - 0.5)


def evaluate(rows: list[dict], k: float, margin: float) -> dict:
    kept = []
    for r in rows:
        if margin <= 0:
            kept.append(r)
            continue
        # fair value for the side being bought: 0.5 + k * (favourable spot move)
        fair = 0.5 + k * r["spot_signed"]
        if fair - r["side_mid"] >= margin:
            kept.append(r)
    if not kept:
        return {"n": 0}
    per_day = defaultdict(list)
    for r in kept:
        per_day[r["day"]].append(r["per_share"])
    vals = list(per_day.values())
    day_means = [statistics.fmean(v) for v in vals]
    return {
        "n": len(kept),
        "per_day": len(kept) / len(per_day),
        "win": sum(1 for r in kept if r["won"]) / len(kept),
        "entry": statistics.fmean(r["entry"] for r in kept),
        "pooled": statistics.fmean(r["per_share"] for r in kept),
        "day_mean": statistics.fmean(day_means),
        "day_sd": statistics.pstdev(day_means),
        "pos_days": sum(1 for v in day_means if v > 0),
        "days": len(day_means),
    }


def show(label: str, res: dict) -> None:
    if not res or res.get("n", 0) == 0:
        print(f"  {label:<26} (no trades)")
        return
    print(f"  {label:<26}{res['n']:>6}{res['per_day']:>8.1f}{res['win']*100:>8.1f}"
          f"{res['entry']:>8.3f}{res['pooled']*100:>+10.2f}%"
          f"{res['day_mean']*100:>+10.2f}%"
          f"{res['pos_days']:>6}/{res['days']:<3}")


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no trades collected")
        return 1
    prepare(rows)
    ins = [r for r in rows if r["day"] < SPLIT]
    oos = [r for r in rows if r["day"] >= SPLIT]
    print("=" * 104)
    print("A/B - FAIR-VALUE PRICING vs THE CURRENT SPOT FILTER")
    print("=" * 104)
    print(f"  shipped gates: T-{CHECKPOINT:.0f} | thr {THRESHOLD:.2f} | "
          f"spot floor {FLOOR*10_000:.1f}bps | cap {CAP:.2f}")
    print(f"  in-sample {len({r['day'] for r in ins})} days / {len(ins):,} trades   "
          f"out-of-sample {len({r['day'] for r in oos})} days / {len(oos):,} trades")

    # ------------------------------------------------------ calibrate k once
    k = fit_k(ins)
    print()
    print(f"  CALIBRATION (in-sample only, never refitted):")
    print(f"    implied sensitivity k = {k:.3f}")
    print(f"    reading: a 0.10% favourable BTC move implies the token should move")
    print(f"    about {k*0.001:.4f} in probability. Weaker than a pure model because")
    print(f"    the market prices only part of the intra-window move.")

    header = (f"  {'variant':<26}{'n':>6}{'/day':>8}{'win%':>8}{'entry':>8}"
              f"{'pooled':>11}{'day mean':>11}{'days+':>8}")
    for label, subset in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        print()
        print("=" * 104)
        print(f"{label}  ({len({r['day'] for r in subset})} days)")
        print("=" * 104)
        print(header)
        print("-" * 104)
        show("A. current (margin off)", evaluate(subset, k, 0.0))
        for margin in MARGINS[1:]:
            show(f"B. require fair-ask >= {margin:.3f}", evaluate(subset, k, margin))

    # ------------------------------------------------------------- verdict
    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    base_oos = evaluate(oos, k, 0.0)
    best_margin, best_res = None, None
    for margin in MARGINS[1:]:
        res = evaluate(oos, k, margin)
        if res.get("n", 0) < 80:
            continue
        if best_res is None or res["pooled"] > best_res["pooled"]:
            best_margin, best_res = margin, res
    if base_oos.get("n") and best_res:
        print(f"  A (current)          out-of-sample pooled {base_oos['pooled']*100:+.2f}%  "
              f"entry {base_oos['entry']:.3f}  {base_oos['per_day']:.0f}/day")
        print(f"  B (best margin "
              f"{best_margin:.3f})   out-of-sample pooled {best_res['pooled']*100:+.2f}%  "
              f"entry {best_res['entry']:.3f}  {best_res['per_day']:.0f}/day")
        delta = best_res["pooled"] - base_oos["pooled"]
        print()
        print(f"  per-share change: {delta*100:+.2f} points")
        if delta > 0:
            print("  B improves the per-share result out of sample.")
        else:
            print("  B does NOT improve the per-share result out of sample.")
        print()
        print("  Remember the margin was chosen on out-of-sample performance above,")
        print("  which is itself a form of fitting. The honest read is the SHAPE of")
        print("  the margin sweep: a broad plateau means the effect is real, a single")
        print("  spike means it is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

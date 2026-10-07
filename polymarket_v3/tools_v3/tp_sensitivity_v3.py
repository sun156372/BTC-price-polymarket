"""V3 TP/SL sensitivity: how much does the "optimal" exit depend on the path model?

    python tools_v3/tp_sensitivity_v3.py

The barrier study needs an intra-window price path, and this dataset does not
contain one, so the path is modelled. That means any "best TP/SL" it reports is a
property of the model as much as of the market.

This tool makes that explicit: it re-runs the same comparison across a range of
defensible bridge assumptions and prints whether the winning exit is stable. If
the "best TP" column jumps around, then no TP recommendation from this dataset is
defensible - which is exactly the conclusion V3's default risk policy rests on.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_v3 import (  # noqa: E402
    SIGNAL_FEATURES_USED,
    SIGNAL_INVERT,
    SIGNAL_THRESHOLD,
    pnl_for,
    token_price,
    unit_path,
    walk_path,
)
from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, load  # noqa: E402
from engine_v3.signals import (  # noqa: E402
    Standardiser,
    break_even_accuracy,
    taker_fee_per_share,
)

#: bridge inflation factors. 0.0 = fully determined bridge (no uncertainty left
#: beyond the endpoints), 0.85 = a fractional Brownian bridge, 1.4 = extra noise.
FRACTIONS = (0.0, 0.35, 0.6, 0.85, 1.1, 1.4)

#: (take profit, stop loss); 1.00 on a leg means that leg is disabled
EXITS = (
    (1.00, 1.00),   # hold to settlement
    (0.30, 1.00),
    (0.20, 1.00),
    (0.10, 1.00),
    (0.20, 0.30),
    (0.20, 0.20),
    (0.10, 0.30),
)


def build_trades(windows, folds=4, entry_t_minus=240.0):
    """Out-of-sample trades under walk-forward standardisation."""
    size = len(windows) // folds
    trades = []
    for fold in range(1, folds):
        train = windows[: fold * size]
        test = windows[fold * size: (fold + 1) * size] if fold < folds - 1 \
            else windows[fold * size:]
        standardiser = Standardiser(SIGNAL_FEATURES).fit(train)
        for window in test:
            raw = sum(standardiser.z(window, f) for f in SIGNAL_FEATURES_USED) \
                / len(SIGNAL_FEATURES_USED)
            if SIGNAL_INVERT:
                raw = -raw
            if abs(raw) < SIGNAL_THRESHOLD:
                continue
            side = "UP" if raw > 0 else "DOWN"
            trades.append((window, token_price(window, window.end_ts - entry_t_minus,
                                                side=side), side))
    return trades


def run(dataset: str, *, sims: int, seed: int, slip: float) -> int:
    windows = load(dataset, features=SIGNAL_FEATURES)
    trades = build_trades(windows)
    if not trades:
        print("no trades produced")
        return 1
    win_rate = sum(1 for x, _, s in trades
                   if (s == "UP") == (x.label_up == 1)) / len(trades)
    entry = statistics.fmean(p for _, p, _ in trades) + slip

    print("=" * 100)
    print("V3 TP/SL SENSITIVITY TO THE PATH MODEL")
    print("=" * 100)
    print(f"out-of-sample trades : {len(trades):,}")
    print(f"true win rate        : {win_rate:.4f}")
    print(f"mean entry (with {slip*100:.0f}c slippage): {entry:.4f}")
    print(f"break-even accuracy  : {break_even_accuracy(entry):.4f}")
    print(f"paths per trade      : {sims}")
    print()
    print("A take profit with NO stop loss can only exit at the target or settle,")
    print("so it converts winners into a smaller certain gain while leaving losses")
    print("open. Whether that helps depends entirely on how often the target is")
    print("reached - which is the thing this dataset cannot tell us.")
    print()

    header = f"{'leftover':>9}" + "".join(
        f"{('tp' + format(tp, '.2f') + '/sl' + format(sl, '.2f')):>16}" for tp, sl in EXITS)
    print(header)
    print("-" * 100)

    best_by_fraction = []
    for fraction in FRACTIONS:
        row = []
        for tp, sl in EXITS:
            rng = random.Random(seed)
            net = 0.0
            count = 0
            for window, model_price, side in trades:
                paid = min(0.999, model_price + slip)
                for _ in range(sims):
                    outcome = walk_path(window, window.end_ts - 240.0, side=side,
                                        entry_price=paid, tp=tp, sl=sl,
                                        noise=unit_path(rng),
                                        leftover_fraction=fraction)
                    net += pnl_for(outcome, paid, side, window)
                    count += 1
            row.append(net / count)
        best_index = max(range(len(EXITS)), key=lambda i: row[i])
        best_by_fraction.append(EXITS[best_index])
        print(f"{fraction:>9.2f}" + "".join(f"{v:>+16.4f}" for v in row))

    print("-" * 100)
    print()
    distinct = sorted({f"tp{tp:.2f}/sl{sl:.2f}" for tp, sl in best_by_fraction})
    print(f"best exit per bridge assumption: {distinct}")
    print()
    if len(distinct) == 1:
        print(f"STABLE: every assumption picks {distinct[0]} - a recommendation is defensible.")
    else:
        print(f"UNSTABLE: {len(distinct)} different 'optimal' exits across "
              f"{len(FRACTIONS)} defensible assumptions.")
        print("No TP/SL recommendation from this dataset is defensible. V3 therefore")
        print("defaults to hold-to-settlement plus a catastrophe stop, and says so.")
    print()
    print("The one conclusion that does NOT depend on the path model: the settlement")
    print(f"expectancy, {win_rate:.4f} - {entry:.3f} - fee = "
          f"{win_rate - entry - taker_fee_per_share(entry):+.4f} per share.")
    print("That is exact given the win rate, and it is what the execution gate acts on.")
    return 0


def main(argv=None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(root), DEFAULT_CSV))
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--slip", type=float, default=0.01)
    args = parser.parse_args(argv)
    return run(args.dataset, sims=args.sims, seed=args.seed, slip=args.slip)


if __name__ == "__main__":
    raise SystemExit(main())

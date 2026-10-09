"""Why an edge shrinks as more days are added - the mechanical part.

    python tools_v3/selection_bias_demo.py

The V4 sample history looked alarming:

    4 days   -> +1.77%/share
    11 days  -> +1.65%
    14 days  -> +1.02%
    10 unseen days -> +0.39%

The tempting reading is "the market adapted / the edge died". This simulation
shows that no market change is needed to produce exactly that pattern.

Mechanism: the strategy's parameters were chosen as the best of many cells on a
short sample. On a short sample the "best" cell is mostly the luckiest cell, so
its measured edge sits far above the truth. As the sample grows, the same
selection is averaged over more days and the luck washes out, so the reported
number falls toward the true edge.

This is a property of the ESTIMATOR, not of the market. It also predicts the one
thing actually observed: the genuinely unseen days land near the true edge from
the start, because they were never used to pick the cell.

The numbers here are the real measured scale (+0.4%/day edge, 4.5%/day spread),
so the magnitude is comparable rather than decorative.
"""
from __future__ import annotations

import random
import statistics

#: measured per-day scale: the out-of-sample edge and the day-level spread
TRUE_EDGE = 0.004
TRUE_SPREAD = 0.045
#: how many candidate cells the sweep really considered
CELLS = 1260
#: set for reproducibility; change it and the qualitative result is unchanged
SEED = 20261008


def main() -> int:
    rng = random.Random(SEED)
    print("=" * 74)
    print("SELECTION BIAS: why the reported edge falls as days accumulate")
    print("=" * 74)
    print(f"  assumed TRUE edge      : {TRUE_EDGE*100:+.2f}%/day")
    print(f"  assumed day spread     : {TRUE_SPREAD*100:.2f}%/day")
    print(f"  candidate cells in sweep: {CELLS}")
    print()
    print("  Every cell is, by construction, the SAME edge. Any apparent")
    print("  difference between them, and any decline over time, is pure noise")
    print("  being selected on and then averaged out.")
    print()
    print(f"  {'sweep on':>10}{'best cell reported':>20}{'that cell on fresh days':>25}")
    print("-" * 74)

    trials = 300
    for days in (4, 11, 14, 24, 60):
        on_prefix = []
        on_fresh = []
        for _ in range(trials):
            # each candidate cell, measured over the prefix
            cells = [statistics.fmean(rng.gauss(TRUE_EDGE, TRUE_SPREAD)
                                      for _ in range(days))
                     for _ in range(CELLS)]
            best = max(cells)
            # the same cell, evaluated on days never used for the selection
            fresh = statistics.fmean(rng.gauss(TRUE_EDGE, TRUE_SPREAD)
                                     for _ in range(10))
            on_prefix.append(best)
            on_fresh.append(fresh)
        print(f"  {days:>10}{statistics.fmean(on_prefix)*100:>19.2f}%"
              f"{statistics.fmean(on_fresh)*100:>24.2f}%")

    print()
    print("  READ THIS: the middle column falls toward the true edge as the sweep")
    print("  period lengthens, while the right-hand column sits at the truth from")
    print("  the start. The decline is the estimator regressing, not the market")
    print("  changing.")
    print()
    print("  WHAT THIS MEANS FOR V4")
    print("  " + "-" * 70)
    print("  The measured history 4d +1.77% -> 14d +1.02% -> unseen +0.39% has the")
    print("  same shape. So the honest lesson is NOT 'the edge decayed' but:")
    print()
    print("    1. The early numbers were never a measurement of the edge - they")
    print("       were a measurement of how lucky the best cell was on 4 days.")
    print("    2. Only the out-of-sample number (+0.39%) estimates the edge.")
    print("    3. Deciding parameters on short samples guarantees this pattern.")
    print()
    print("  The fix is procedural, not parametric: decide parameters on a long")
    print("  sample, validate once, and never re-tune on either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

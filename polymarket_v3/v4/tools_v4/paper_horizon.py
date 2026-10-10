"""How long does a paper run need to be before the result means anything?

The daily edge is ~1.8 points against a ~2.4 point daily standard deviation, so
short runs are dominated by noise. This computes the probability that a paper run
of a given length shows a LOSS even when the strategy is genuinely profitable -
i.e. the false-negative rate - so a losing fortnight is not misread as failure.
"""
import math
import sys

MEAN = 0.0176          # measured per-share daily mean, 11 days, unfiltered
STD = 0.0239           # measured per-share daily standard deviation
MEAN_F = 0.0277        # with the 5 bps spot-move filter
STD_F = 0.0325


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> int:
    print("=" * 88)
    print("FALSE-NEGATIVE RISK: P(a genuinely profitable strategy shows a loss)")
    print("=" * 88)
    print("  Assumes the measured mean/std persist and returns are roughly normal.")
    print("  The second assumption is the weak one - noted, not hidden.")
    print()
    print(f"  {'run length':>14}{'expected':>12}{'std':>10}"
          f"{'P(loss) unfilt':>17}{'P(loss) filtered':>18}")
    print("-" * 88)
    for label, days in (("1 day", 1), ("3 days", 3), ("1 week", 5), ("2 weeks", 10),
                        ("1 month", 21), ("6 weeks", 42), ("3 months", 63)):
        cells = []
        for mean, std in ((MEAN, STD), (MEAN_F, STD_F)):
            mu = mean * days
            sd = std * math.sqrt(days)
            cells.append(norm_cdf(-mu / sd) if sd > 0 else 0.0)
        print(f"  {label:>14}{MEAN*days:>+12.4f}{STD*math.sqrt(days):>10.4f}"
              f"{cells[0]*100:>16.1f}%{cells[1]*100:>17.1f}%")
    print()
    print("  Read it as: if the edge is real, a 2-week run still loses money")
    print(f"  about {norm_cdf(-MEAN*10/(STD*math.sqrt(10)))*100:.0f}% of the time by chance alone.")
    print("  So do NOT read a losing fortnight as proof the strategy is broken - and")
    print("  equally, do not read a winning week as proof it works.")
    print()
    print("  A run long enough to distinguish the edge from zero at ~2 sigma needs")
    need = (2.0 * STD / MEAN) ** 2
    print(f"  roughly {need:.0f} trading days ({need/21:.1f} months) at the measured")
    print("  mean and volatility. That is the practical answer to 'how long do I test'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

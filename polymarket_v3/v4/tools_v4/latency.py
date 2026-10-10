"""Measure what latency actually costs this strategy, from the tape itself.

The useful question is not "what is my ping to the server". It is:

    at the moment the decision is made, how far does the price move before my
    order can land?

That is measurable: for every market, look at the quoted ask at T-240 and compare
it with the ask 250ms / 500ms / 1s / 2s later. The difference is the price
slippage that pure latency introduces, before any fee or spread.

A server's round trip (Amsterdam -> the matching engine -> back) is only part of
this; the quote stream has its own delay. But the tape bounds the whole thing.
"""
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from engine_v4.strategy import V4Params  # noqa: E402
from paths import resolve_data_root  # noqa: E402
sys.path.insert(0, "tools_v4")

def _resolve_default_root() -> str:
    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""
    return resolve_data_root(required=False) or 'micro_v3'

import backtest as B
from engine_v4 import hf

ROOT = resolve_data_root()
CHECKPOINT = V4Params().entry_t_minus
THRESHOLD = V4Params().threshold
#: latencies to price, in milliseconds
DELAYS = (100, 250, 500, 1000, 2000, 5000)


def ask_at_offset(market, entry_tick, side, delay_ms):
    """Ask for ``side`` roughly ``delay_ms`` after the entry tick."""
    target = entry_tick.to_deadline_ms - delay_ms
    best = None
    best_gap = None
    for tick in market.ticks:
        gap = abs(tick.to_deadline_ms - target)
        if gap <= 25.0 and (best_gap is None or gap < best_gap):
            best, best_gap = tick, gap
    if best is None:
        return None
    return best.ask(side)


def main() -> int:
    resolutions = hf.load_resolutions(ROOT)
    rows = []
    for market in hf.iter_markets(ROOT, horizons=["5m"], resolutions=resolutions):
        if market.up_won is None:
            continue
        entry_tick = market.tick_at_t_minus(CHECKPOINT, tolerance_s=1.5)
        open_tick = market.first_tick()
        if entry_tick is None or open_tick is None:
            continue
        picked = B.signal_side(open_tick.up_mid, entry_tick.up_mid, THRESHOLD, "follow")
        if picked is None:
            continue
        side, _ = picked
        base = entry_tick.ask(side)
        if base is None or not 0.01 < base < 0.99:
            continue
        record = {"base": base, "won": (side == "UP") == market.up_won}
        for delay in DELAYS:
            later = ask_at_offset(market, entry_tick, side, delay)
            record[delay] = None if later is None else later - base
        rows.append(record)

    print(f"trades with a measurable tape: {len(rows):,}")
    print(f"decision point T-{CHECKPOINT:.0f}s, mean entry ask "
          f"{statistics.fmean(r['base'] for r in rows):.4f}")
    print()
    print("=" * 92)
    print("PRICE DRIFT BETWEEN DECISION AND FILL (cents, on the side being bought)")
    print("=" * 92)
    print("  Positive = the price moved AWAY from you (you pay more). This is the")
    print("  pure latency cost, before fees, assuming the order fills at the touch.")
    print()
    print(f"  {'delay':>8}{'n':>7}{'median':>10}{'mean':>10}{'p75':>10}{'p90':>10}"
          f"{'share worse':>13}{'avg worse':>11}")
    print("-" * 92)
    summary = {}
    for delay in DELAYS:
        deltas = [r[delay] for r in rows if r[delay] is not None]
        if not deltas:
            continue
        deltas.sort()
        n = len(deltas)
        worse = [d for d in deltas if d > 0]
        mean_worse = statistics.fmean(worse) if worse else 0.0
        summary[delay] = {
            "median": statistics.median(deltas),
            "mean": statistics.fmean(deltas),
            "p75": deltas[3 * n // 4],
            "p90": deltas[9 * n // 10],
            "share_worse": len(worse) / n,
            "avg_worse": mean_worse,
        }
        print(f"  {delay:>6}ms{n:>7,}{statistics.median(deltas)*100:>+10.2f}"
              f"{statistics.fmean(deltas)*100:>+10.2f}{deltas[3*n//4]*100:>+10.2f}"
              f"{deltas[9*n//10]*100:>+10.2f}{len(worse)/n*100:>12.1f}%"
              f"{mean_worse*100:>+10.2f}")

    # ------------------------------------------------ effect on the trade's edge
    print()
    print("=" * 92)
    print("WHAT THAT DOES TO THE EDGE  (measured accuracy 77.19%, cap 0.76)")
    print("=" * 92)
    from engine_v4.signals import taker_fee_per_share
    accuracy = 0.7719
    base_entry = statistics.fmean(r["base"] for r in rows)
    print(f"  entry at the decision moment : {base_entry:.4f}")
    print()
    print(f"  {'total delay':>12}{'drift (c)':>11}{'paid entry':>12}{'break-even':>12}"
          f"{'edge/share':>12}{'verdict':>12}")
    print("-" * 92)
    for delay in DELAYS:
        if delay not in summary:
            continue
        drift = summary[delay]["median"]
        paid = base_entry + drift
        be = paid + taker_fee_per_share(paid)
        edge = accuracy - be
        print(f"  {delay:>9}ms{drift*100:>+11.2f}{paid:>12.4f}{be:>12.4f}"
              f"{edge:>+12.4f}{('profitable' if edge > 0 else 'LOSING'):>12}")

    # ------------------------------------------------------ profitability odds
    print()
    print("=" * 92)
    print("PROFITABILITY PROBABILITY  (from the 11 measured days)")
    print("=" * 92)
    daily = [0.0201, 0.0054, 0.0344, 0.0227, 0.0057, -0.0110,
             0.0778, -0.0062, 0.0262, 0.0239, -0.0056]
    filtered = [0.0361, 0.0042, 0.0146, 0.0057, 0.0233, 0.1077,
                0.0252, 0.0115, 0.0606, 0.0379, -0.0216]
    import math

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    for name, series in (("unfiltered", daily), ("5bps spot filter", filtered)):
        mu = statistics.fmean(series)
        sd = statistics.pstdev(series)
        print(f"\n  {name}: mean {mu:+.4f}/share, std {sd:.4f}, "
              f"{sum(1 for v in series if v > 0)}/{len(series)} days positive")
        print(f"    {'horizon':>12}{'expected':>12}{'std':>10}{'P(profit)':>12}")
        for label, days in (("1 day", 1), ("1 week", 5), ("1 month", 21), ("3 months", 63)):
            hmu = mu * days
            hsd = sd * math.sqrt(days)
            p = 1.0 - norm_cdf((0 - hmu) / hsd) if hsd > 0 else 1.0
            print(f"    {label:>12}{hmu:>+12.4f}{hsd:>10.4f}{p*100:>11.1f}%")

    print()
    print("  CAVEAT that matters more than the numbers: these are descriptive")
    print("  statistics from 11 days, fitted with a normal approximation. The daily")
    print("  returns are not normal (one +0.1077 outlier), and 11 observations cannot")
    print("  pin a tail. Treat P(profit) as indicative, not predictive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

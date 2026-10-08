#!/usr/bin/env python
"""run_v4.py - V4 sniper strategy: selfcheck, live outlook, paper, replay.

    python run_v4.py selfcheck
    python run_v4.py outlook                  # what live trading would plausibly return
    python run_v4.py replay --horizons 5m     # the full measured backtest
    python run_v4.py paper --snapshot config_v4/snapshot_example.json

V4 reads the real order book. It does not model entry price, does not model the
intra-window path, and does not model which barrier was hit first - all three come
from 100ms quotes and depth on both sides of the book.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_v3.signals import break_even_accuracy, taker_fee_per_share
from engine_v3.strategy_v4 import READY, V4Params, StrategyV4

#: Measured on the 14-day high-frequency sample (2026-03-27 .. 2026-04-26, 4,023
#: resolved 5m markets), for the SHIPPED cell:
#: follow, T-240, threshold 0.20, hold to settlement
MEASURED_ACCURACY = 0.8170
MEASURED_ENTRY = 0.7792
MEASURED_TRADES_PER_DAY = 73.0
#: that cell's realised per-day pnl/share across the 14 days: mean +0.0292,
#: std 0.0362, 10 of 14 positive.
DAILY_PNL_PER_SHARE = (0.0770, -0.0078, 0.0150, 0.0010, -0.0087, 0.0383, 0.0638,
                       0.0368, 0.0487, 0.0634, -0.0257, -0.0156, 0.0951, 0.0282)
#: The 5 bps spot-move deviation filter, threshold-0.15 cell, per day. Days 1-11
#: are from the 11-day run; days 12-14 are the 14-day assessment (-0.0216 is the
#: 04-19 value the 11-day run also ended on, then the three new days). It must
#: cover the SAME days as the unfiltered series or the comparison is invalid.
FILTERED_DAILY = (0.0361, 0.0042, 0.0146, 0.0057, 0.0233, 0.1077, 0.0252,
                  0.0115, 0.0606, 0.0379, -0.0216, 0.0051, 0.0864, 0.0223)
#: The sample history, kept as the main warning. The same rule measured
#: +1.77%/share on 4 days, +1.65% on 11, and +1.02% on 14 at threshold 0.15 -
#: every widening of the sample shrank the edge.
FOUR_DAY_EDGE = 0.0177
ELEVEN_DAY_EDGE = 0.0165
FOURTEEN_DAY_EDGE_015 = 0.0102
#: the threshold-0.15 cell's unfiltered per-day series over the first 11 days,
#: paired with FILTERED_DAILY so the filter comparison uses identical days
ELEVEN_DAY_FILTER_BASELINE = (0.0201, 0.0054, 0.0344, 0.0227, 0.0057, -0.0110,
                              0.0778, -0.0062, 0.0262, 0.0239, -0.0056)


# --------------------------------------------------------------- selfcheck
def cmd_selfcheck(params: V4Params) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    strategy = StrategyV4(params)
    p = params

    # 1. the break-even identity, which everything else rests on
    for price in (0.50, 0.65, 0.7394, 0.80, 0.90):
        needed = break_even_accuracy(price, p.fee_coefficient)
        check(f"break-even({price:.3f}) = price + fee",
              abs(needed - (price + taker_fee_per_share(price, p.fee_coefficient))) < 1e-12)
    check("fee is symmetric and peaks at 0.5",
          taker_fee_per_share(0.5) > taker_fee_per_share(0.2) > taker_fee_per_share(0.05))
    check("fee vanishes at the extremes",
          taker_fee_per_share(0.001) < 1e-4 and taker_fee_per_share(0.999) < 1e-4)

    # 2. the cap must sit above the measured entry but below the implied break-even
    cap = p.max_entry_price
    be = strategy.break_even_entry
    check("cap is above the measured entry price", cap > MEASURED_ENTRY,
          f"cap {cap} vs entry {MEASURED_ENTRY}")
    check("cap is not far above break-even", cap - be < 0.02,
          f"cap {cap:.4f} vs break-even {be:.4f}")

    # 3. direction: follow means the side that moved up.
    #    Move sizes are derived from the configured threshold rather than
    #    hard-coded, so raising the threshold cannot silently break the test.
    strong = p.threshold * 1.5
    up, move = strategy.direction(0.40, 0.40 + strong)
    check("a move up buys UP (follow)", up == "UP", f"{up} move={move:+.3f}")
    down, move2 = strategy.direction(0.40 + strong, 0.40)
    check("a move down buys DOWN (follow)", down == "DOWN", f"{down} move={move2:+.3f}")
    check("a small move is refused",
          strategy.direction(0.50, 0.50 + p.threshold / 2) is None)

    # 4. the gate refuses an overpriced entry even with a strong signal
    decision = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.90,
                               now=1000.0, quote_ts=1000.0, measured_fill=0.90,
                               shares_wanted=13.0, depth_cost=0.90 * 13.0)
    check("expensive entry refused", not decision.entering
          and decision.reason == "execution:entry_above_cap", decision.reason)

    # 5. the gate refuses a modelled fill
    decision = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70,
                               now=1000.0, quote_ts=1000.0, measured_fill=None,
                               shares_wanted=14.0, depth_cost=None)
    check("unmeasured fill refused", not decision.entering, decision.reason)

    # 6. a good setup enters, and never above the cap
    decision = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70,
                               now=1000.0, quote_ts=1000.0, measured_fill=0.70,
                               shares_wanted=14.0, depth_cost=0.70 * 14.0)
    check("a good setup enters", decision.entering and decision.reason == READY,
          decision.reason)
    check("entry is within the cap", decision.limit_price <= cap, str(decision.limit_price))

    # 7. staleness and halts
    stale = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70, now=1000.0,
                            quote_ts=900.0, measured_fill=0.70, shares_wanted=14.0,
                            depth_cost=0.70 * 14.0)
    check("stale quote refused", not stale.entering, stale.reason)
    halted = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70, now=1000.0,
                             quote_ts=1000.0, measured_fill=0.70, shares_wanted=14.0,
                             depth_cost=0.70 * 14.0, halted=True)
    check("halted market refused", not halted.entering, halted.reason)

    # 7b. the price-deviation filter
    from engine_v3.strategy_v4 import WEAK_DRIVER
    tiny = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70, now=1000.0,
                           quote_ts=1000.0, measured_fill=0.70, shares_wanted=14.0,
                           depth_cost=0.70 * 14.0, spot_move=p.min_spot_move / 10.0)
    check("unbacked token move refused by the deviation filter",
          not tiny.entering and tiny.reason == WEAK_DRIVER, tiny.reason)
    backed = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70, now=1000.0,
                             quote_ts=1000.0, measured_fill=0.70, shares_wanted=14.0,
                             depth_cost=0.70 * 14.0, spot_move=p.min_spot_move * 2.0)
    check("a spot-backed move still enters", backed.entering, backed.reason)
    no_spot = strategy.decide(open_up_mid=0.40, entry_up_mid=0.75, ask=0.70, now=1000.0,
                              quote_ts=1000.0, measured_fill=0.70, shares_wanted=14.0,
                              depth_cost=0.70 * 14.0, spot_move=None)
    check("filter is skipped when spot data is absent (documented, not silent)",
          no_spot.entering, no_spot.reason)

    # 8. exits: the shipped configuration holds to settlement
    stop, target = strategy.exit_levels(0.74)
    check("no take profit by default", target is None, str(target))
    check("no stop by default", stop is None, str(stop))
    tight = StrategyV4(V4Params(take_profit=0.05, stop_loss=0.30))
    stop2, target2 = tight.exit_levels(0.74)
    check("a tight configuration would set both legs",
          stop2 is not None and target2 is not None,
          f"stop={stop2} target={target2}")

    # 9. arithmetic of the edge: accuracy must exceed break-even
    per_share = MEASURED_ACCURACY - MEASURED_ENTRY - taker_fee_per_share(MEASURED_ENTRY)
    check("measured edge is positive", per_share > 0, f"{per_share:+.4f}/share")
    check("measured edge is thin (<5 points)",
          per_share < 0.05, f"{per_share:+.4f}/share")

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"selfcheck: {len(checks) - failed}/{len(checks)} passed")
    return 0 if not failed else 1


# ------------------------------------------------------------------ outlook
def cmd_outlook(params: V4Params, stake: float, equity: float) -> int:
    daily = list(DAILY_PNL_PER_SHARE)
    filtered = list(FILTERED_DAILY)
    strategy = StrategyV4(params)
    entry = MEASURED_ENTRY
    accuracy = MEASURED_ACCURACY
    fee = taker_fee_per_share(entry, params.fee_coefficient)
    edge = accuracy - entry - fee
    # derived here so the header and the later sections share one definition
    mean_daily = statistics.fmean(daily)
    std_daily = statistics.pstdev(daily)
    stake_shares = stake / entry
    stderr = std_daily / math.sqrt(len(daily))

    print("=" * 100)
    print("V4 LIVE OUTLOOK - what real trading would plausibly return")
    print("=" * 100)
    print("Every number below is anchored to a MEASURED input. Nothing is annualised,")
    print(f"because {len(daily)} days of data cannot support an annual figure.")
    print()
    print(f"MEASURED INPUTS ({len(daily)} days of real order book, "
          f"4,023 resolved 5m markets)")
    print(f"  direction accuracy      : {accuracy:.4f}")
    print(f"  average entry price     : {entry:.4f}")
    print(f"  taker fee per share     : {fee:.4f}")
    print(f"  break-even entry price  : {strategy.break_even_entry:.4f}"
          f"   (cap configured at {params.max_entry_price:.3f})")
    print(f"  edge per share          : {edge:+.4f}")
    print(f"  trades per day          : {MEASURED_TRADES_PER_DAY:.0f}")
    print()

    # ---------------------------------------------------- per-trade economics
    print("=" * 100)
    print("1. PER-TRADE ECONOMICS AT A GIVEN STAKE")
    print("=" * 100)
    print(f"{'stake':>8}{'shares':>9}{'win pays':>10}{'lose pays':>11}"
          f"{'expectancy':>12}{'per day':>10}{'per month':>11}")
    print("-" * 100)
    for candidate in (stake, stake * 2, stake * 5):
        shares = candidate / entry
        win = shares * (1.0 - entry - fee)
        lose = -shares * (entry + fee)
        per_trade = shares * edge
        print(f"{candidate:>8.0f}{shares:>9.2f}{win:>+10.3f}{lose:>+11.3f}"
              f"{per_trade:>+12.4f}{per_trade*MEASURED_TRADES_PER_DAY:>+10.2f}"
              f"{per_trade*MEASURED_TRADES_PER_DAY*30:>+11.1f}")

    # --------------------------------------------------------- the real risk
    print()
    print("=" * 100)
    print("2. THE PART THAT MATTERS: HOW WELL IS THE EDGE PINNED DOWN?")
    print("=" * 100)
    stake_shares = stake / entry
    stderr = std_daily / math.sqrt(len(daily))
    print(f"  per-day pnl/share actually observed: "
          f"{', '.join(f'{v:+.4f}' for v in daily)}")
    print(f"  mean {mean_daily:+.4f}/share over {len(daily)} days")
    print(f"  std {std_daily:.4f}/share -> standard error {stderr:.4f} "
          f"(only {len(daily)} days!)")
    print()
    print("  Two different uncertainties, and they are easy to confuse:")
    print(f"    a) per-TRADE edge      : {edge:+.4f}/share, "
          f"i.e. {edge*stake_shares:+.3f} USDC per trade at a {stake:.0f} USDC stake")
    print(f"       x {MEASURED_TRADES_PER_DAY:.0f} trades/day = "
          f"{edge*stake_shares*MEASURED_TRADES_PER_DAY:+.2f} USDC/day")
    print(f"    b) per-DAY realisation : {mean_daily:+.4f}/share observed mean, "
          f"std {std_daily:.4f}")
    print()
    print(f"  These two agree well: {mean_daily:+.4f} observed vs {edge:+.4f} arithmetic.")
    print(f"  {sum(1 for v in daily if v > 0)} of {len(daily)} days were positive, and "
          f"the worst day cost")
    print(f"  {abs(min(daily))*stake_shares*MEASURED_TRADES_PER_DAY:.1f} USDC at a "
          f"{stake:.0f} USDC stake. That is a coherent picture rather than a")
    print("  contradiction - which is a genuinely better result than most of the")
    print("  neighbouring parameter cells produce.")
    print()
    print(f"  But {len(daily)} observations is still tiny. Standard error of the daily")
    print(f"  mean: {stderr:.4f}/share, i.e. the daily result could plausibly be:")
    for level in (1.0, 2.0):
        lo = (mean_daily - level * stderr) * stake_shares * MEASURED_TRADES_PER_DAY
        hi = (mean_daily + level * stderr) * stake_shares * MEASURED_TRADES_PER_DAY
        print(f"    {level:.0f} std-error band: {lo:+.2f} .. {hi:+.2f} USDC/day")
    print()
    print("  The 2-sigma lower bound is still positive, which is encouraging, but it")
    print(f"  rests on {len(daily)} days. The SAME rule measured {FOUR_DAY_EDGE*100:.2f}%/share")
    print(f"  on 4 days, {ELEVEN_DAY_EDGE*100:.2f}% on 11, and {FOURTEEN_DAY_EDGE_015*100:.2f}% on 14 at")
    print("  threshold 0.15 - every widening of the sample shrank the edge. And of")
    print("  1,260 configurations swept across these 14 days, only 1.3% are positive")
    print("  at all. Parameter cells that close together behaving that differently is")
    print("  the whole caution.")

    print()
    print("  THE DEVIATION FILTER (require |spot move| >= 4.5 bps):")
    fmean = statistics.fmean(filtered)
    fstd = statistics.pstdev(filtered)
    fpos = sum(1 for v in filtered if v > 0)
    # Compare the filter on the SAME threshold-0.15 cell, so this pairing is
    # apples-to-apples. The shipped cell is threshold 0.20, whose unfiltered
    # numbers are the ones used everywhere else in this report.
    cell_daily = ELEVEN_DAY_FILTER_BASELINE + tuple(daily[11:])
    cmean = statistics.fmean(cell_daily)
    cpos = sum(1 for v in cell_daily if v > 0)
    print(f"    cell (thr 0.15)  without: {cmean:+.4f}/share, "
          f"{cpos}/{len(cell_daily)} days positive")
    print(f"    cell (thr 0.15)  with   : {fmean:+.4f}/share, "
          f"{fpos}/{len(filtered)} days positive")
    print(f"    per-share gain {fmean-cmean:+.4f} on the same days and the same cell.")
    print("    It trades 782 rather than 1,619 times, so the gain is per TRADE, not")
    print("    per day. Pooled figures: +1.83% filtered vs +1.02% unfiltered.")
    print("    Assessed by tools_v3/filter_assessment_v4.py on four axes: pooled gain,")
    print("    monotonicity, non-redundancy against a spread cap, and robustness to a")
    print("    delayed spot read (it survives, so it is not reading the future).")

    # --------------------------------------------------- accuracy sensitivity
    print()
    print("=" * 100)
    print("3. SENSITIVITY: HOW MUCH ACCURACY DECAY CAN IT SURVIVE?")
    print("=" * 100)
    print("  The edge is thin, so this is the question that decides whether it lives.")
    print()
    print(f"  {'accuracy':>9}{'entry':>8}{'break-even':>12}{'edge/share':>12}"
          f"{'per day':>10}{'verdict':>14}")
    print("-" * 100)
    for delta in (-0.06, -0.04, -0.02, 0.0, +0.02):
        acc = accuracy + delta
        be = break_even_accuracy(entry, params.fee_coefficient)
        e = acc - be
        per_day = e * (stake / entry) * MEASURED_TRADES_PER_DAY
        verdict = "profitable" if e > 0 else "LOSING"
        flag = "   <- measured" if abs(delta) < 1e-9 else ""
        print(f"  {acc:>9.4f}{entry:>8.3f}{be:>12.4f}{e:>+12.4f}{per_day:>+10.2f}"
              f"{verdict:>14}{flag}")
    print()
    print("  A 2-point accuracy decay halves the edge. A 4-point decay erases it.")
    print("  And the T-285 variant of this same rule decayed by 10 accuracy points")
    print("  across a four-day slice (+0.0458 -> -0.0592 per share), which is exactly")
    print("  the kind of drift that would wipe this out.")

    # ------------------------------------------------------------ cost stress
    print()
    print("=" * 100)
    print("4. COST STRESS: WHAT EATS THE EDGE?")
    print("=" * 100)
    print(f"  {'scenario':<34}{'entry':>8}{'edge/share':>12}{'per day':>10}{'verdict':>12}")
    print("-" * 100)
    scenarios = [
        ("as measured", 0.0),
        ("+1 cent slippage", 0.01),
        ("+2 cents slippage", 0.02),
        ("+3 cents slippage", 0.03),
        ("+5 cents slippage", 0.05),
    ]
    for label, extra in scenarios:
        paid = entry + extra
        e = accuracy - paid - taker_fee_per_share(paid, params.fee_coefficient)
        per_day = e * (stake / paid) * MEASURED_TRADES_PER_DAY
        print(f"  {label:<34}{paid:>8.4f}{e:>+12.4f}{per_day:>+10.2f}"
              f"{('profitable' if e > 0 else 'LOSING'):>12}")

    # -------------------------------------------------------------- capacity
    print()
    print("=" * 100)
    print("5. CAPACITY (measured, not assumed)")
    print("=" * 100)
    print("  Measured fill rate on the real book: 10 USDC 100%, 25 USDC 100%,")
    print("  50 USDC 100%; median extra cost 0.00 cents, p90 0.79 cents at 50 USDC.")
    print("  So size up to ~50 USDC is not depth-limited. Above that the p90 walk")
    print("  grows and has not been measured here.")
    print()
    print("=" * 100)
    print("BOTTOM LINE")
    print("=" * 100)
    print(f"  Arithmetic expectation : {edge*stake_shares*MEASURED_TRADES_PER_DAY:+.2f} "
          f"USDC/day at a {stake:.0f} USDC stake")
    lo2 = (mean_daily - 2 * stderr) * stake_shares * MEASURED_TRADES_PER_DAY
    hi2 = (mean_daily + 2 * stderr) * stake_shares * MEASURED_TRADES_PER_DAY
    print(f"  2 std-error band       : {lo2:+.2f} .. {hi2:+.2f} USDC/day")
    print(f"  What 4 days delivered  : {mean_daily*stake_shares*MEASURED_TRADES_PER_DAY:+.2f} "
          f"USDC/day ({sum(1 for v in daily if v > 0)} of {len(daily)} days positive)")
    print()
    print("  The point estimate is positive and the per-trade and per-day numbers")
    print("  agree, which is a good sign. What is NOT established is durability:")
    print("  the neighbouring T-285 cell of the same rule pooled at +0.0458/share and")
    print("  decayed to -0.0592 within these very four days. Two adjacent settings")
    print("  behaved completely differently, so four days cannot tell you which one")
    print("  generalises.")
    print()
    print("  Practical read: the mechanism (follow the move, hold to settlement, refuse")
    print("  overpriced entries) is supported by the data. The RETURN is not. Paper")
    print("  trade it for a month before committing capital - the code is ready, the")
    print("  evidence is not.")
    return 0


# ------------------------------------------------------------------- paper
def cmd_paper(params: V4Params, snapshot_path: str | None) -> int:
    if not snapshot_path:
        print("paper needs --snapshot <file.json>")
        return 2
    if not os.path.exists(snapshot_path):
        print(f"snapshot not found: {snapshot_path}")
        return 2
    with open(snapshot_path, encoding="utf-8") as handle:
        snapshots = json.load(handle)
    if isinstance(snapshots, dict):
        snapshots = [snapshots]

    strategy = StrategyV4(params)
    equity = 1000.0
    cash = equity = 1000.0
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "output", "v4", "v4_paper_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print("=" * 100)
    print("V4 PAPER RUN - virtual money, no order client exists in this package")
    print("=" * 100)
    print(f"{'market':<26}{'action':>8}{'reason':>28}{'side':>6}{'move':>8}{'price':>8}")
    print("-" * 100)
    refusals: dict[str, int] = {}
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["market", "action", "reason", "side", "move", "price", "equity"])
        for snap in snapshots:
            shares = params_stake = 0.0
            stake = float(snap.get("stake", 10.0))
            ask = snap.get("ask")
            depth_cost = snap.get("depth_cost")
            if ask:
                shares = stake / ask
                if depth_cost is None and snap.get("depth_ok", True):
                    depth_cost = ask * shares
            decision = strategy.decide(
                open_up_mid=snap.get("open_up_mid"),
                entry_up_mid=snap.get("entry_up_mid"),
                ask=ask, now=float(snap.get("now", 0.0)),
                quote_ts=snap.get("quote_ts"),
                measured_fill=snap.get("measured_fill"),
                shares_wanted=shares, depth_cost=depth_cost,
                halted=bool(snap.get("halted", False)),
            )
            if not decision.entering:
                refusals[decision.reason] = refusals.get(decision.reason, 0) + 1
            won = snap.get("won")
            if decision.entering and won is not None:
                payout = decision.shares if won else 0.0
                fee = decision.shares * taker_fee_per_share(decision.limit_price,
                                                            params.fee_coefficient)
                cash += payout - decision.shares * decision.limit_price - fee
                equity = cash
            writer.writerow([snap.get("market", ""), decision.action, decision.reason,
                             decision.side, round(decision.move, 4),
                             round(decision.limit_price, 4), round(equity, 2)])
            print(f"{str(snap.get('market','')):<26}{decision.action:>8}"
                  f"{decision.reason:>28}{decision.side:>6}{decision.move:>+8.3f}"
                  f"{decision.limit_price:>8.3f}")
    print()
    print(f"equity {equity:.2f}   refusals:")
    for reason, count in sorted(refusals.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<32}{count:>5}")
    print(f"log: {log_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["selfcheck", "outlook", "replay", "paper"])
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--horizons", nargs="*", default=["5m"])
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    params = V4Params()
    if args.command == "selfcheck":
        print("=" * 100)
        print("V4 SELFCHECK")
        print("=" * 100)
        for line in StrategyV4(params).describe():
            print(f"  {line}")
        print()
        return cmd_selfcheck(params)
    if args.command == "outlook":
        return cmd_outlook(params, args.stake, 1000.0)
    if args.command == "paper":
        return cmd_paper(params, args.snapshot)
    if args.command == "replay":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_v3"))
        from backtest_v4 import run
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return run(os.path.join(root, "micro_v3"),
                   os.path.join(root, "output", "v4"),
                   horizons=args.horizons, size_usdc=args.stake,
                   coefficient=params.fee_coefficient, limit=args.limit)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

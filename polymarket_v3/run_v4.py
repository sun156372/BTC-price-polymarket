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

#: measured on the 4-day high-frequency sample, for the SHIPPED cell
#: (follow, T-240, threshold 0.15, hold to settlement, 1,148 markets)
MEASURED_ACCURACY = 0.7701
MEASURED_ENTRY = 0.7394
MEASURED_TRADES_PER_DAY = 87.0
#: that cell's realised per-day pnl/share, recomputed from the tape per day:
#: 2026-03-27 +0.0210 (113 trades), 03-28 -0.0027 (90), 03-29 +0.0344 (102),
#: 03-31 +0.0159 (130). Mean +0.0171, std 0.0133, 3 of 4 days positive.
DAILY_PNL_PER_SHARE = (0.0210, -0.0027, 0.0344, 0.0159)
#: the neighbouring T-285 cell, kept as the cautionary counter-example: it looked
#: better in the pooled sweep (+0.0458/share) and then decayed to -0.0592.
DECAY_EXAMPLE = (0.0458, 0.0286, -0.0129, -0.0592)


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

    # 3. direction: follow means the side that moved up
    up, move = strategy.direction(0.40, 0.60)
    check("a move up buys UP (follow)", up == "UP", f"{up} move={move:+.3f}")
    down, move2 = strategy.direction(0.60, 0.40)
    check("a move down buys DOWN (follow)", down == "DOWN", f"{down} move={move2:+.3f}")
    quiet, _ = strategy.direction(0.50, 0.50 + p.threshold / 2), None
    check("a small move is refused", strategy.direction(0.50, 0.50 + p.threshold / 2) is None)

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
    strategy = StrategyV4(params)
    entry = MEASURED_ENTRY
    accuracy = MEASURED_ACCURACY
    fee = taker_fee_per_share(entry, params.fee_coefficient)
    edge = accuracy - entry - fee

    print("=" * 100)
    print("V4 LIVE OUTLOOK - what real trading would plausibly return")
    print("=" * 100)
    print("Every number below is anchored to a MEASURED input. Nothing is annualised,")
    print("because four days of data cannot support an annual figure.")
    print()
    print("MEASURED INPUTS (4 days of real order book, 1,148 resolved 5m markets)")
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
    daily = list(DAILY_PNL_PER_SHARE)
    mean_daily = statistics.fmean(daily)
    std_daily = statistics.pstdev(daily)
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
    print("  rests on four days. The cautionary counter-example is the neighbouring")
    print(f"  T-285 cell of the SAME rule, which looked better pooled "
          f"({DECAY_EXAMPLE[0]:+.4f}/share)")
    print(f"  and then decayed to {DECAY_EXAMPLE[-1]:+.4f}. Parameter cells this close")
    print("  together behaved completely differently, which is exactly why four days")
    print("  cannot settle it.")

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
    print("  And the T-285 variant of this same rule decayed by 10 points across the")
    print(f"  four days ({DECAY_EXAMPLE[0]:+.4f} -> {DECAY_EXAMPLE[-1]:+.4f} per share).")

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
    print(f"  2 std-error band       : +4.53 .. +35.83 USDC/day")
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

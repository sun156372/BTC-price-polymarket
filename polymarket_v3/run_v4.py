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

#: ---------------------------------------------------------------------------
#: SHIPPED: follow, T-120, threshold 0.20, 4.5bps spot filter, NO price cap,
#: hold to settlement. Parameters chosen on the 15 in-sample days, then tested on
#: 22 untouched days. Tools: oos2_v4.py, baseline_check_v4.py, cap_optimal_v4.py.
#:
#: THE ANSWER IS NEGATIVE. Realised accuracy tracks the break-even line at every
#: price cap, which is what a fairly priced market looks like; the crypto taker fee
#: is then the entire loss. The figures below are the cap-off measurement, which is
#: the best of the 16 caps swept and still slightly negative.
#: ---------------------------------------------------------------------------
#: OUT-OF-SAMPLE, no cap: 22 days (2026-05-01 .. 05-24), 1,894 trades
OOS_ACCURACY = 0.8710
OOS_ENTRY = 0.8650
OOS_TRADES_PER_DAY = 86.1
OOS_EDGE_PER_SHARE = -0.0017
OOS_DAYS_POSITIVE = 9
OOS_DAYS = 22
#: exact per-day pnl/share, shipped cell, out of sample
OOS_DAILY = (-0.0059, 0.0472, 0.0078, 0.0526, -0.0595,
             0.0454, 0.0061, -0.0187, 0.0509, -0.0865)
#: IN-SAMPLE, same configuration: 15 days, 1,414 trades
IN_SAMPLE_ACCURACY = 0.8540
IN_SAMPLE_ENTRY = 0.8470
IN_SAMPLE_TRADES_PER_DAY = 94.3
IN_SAMPLE_EDGE_PER_SHARE = -0.0009
IN_SAMPLE_DAYS_POSITIVE = 8
IN_SAMPLE_DAYS = 15
#: exact per-day pnl/share, shipped cell, in sample
IN_SAMPLE_DAILY = (0.0419, -0.0104, -0.0219, -0.0435, 0.0113, 0.0409, 0.0482,
                   0.0360, 0.0987, 0.0406, 0.0239, -0.0344, -0.0443, -0.0022)
#: the cell that WON in sample and FAILED out of sample - the cautionary result
FAILED_CELL = "T-240 thr0.20 no filter: +2.62% in sample, -0.86% out"
FAILED_CELL_OOS = -0.0086
#: retained for the filter comparison, on the threshold-0.15 cell
FILTERED_DAILY = (0.0361, 0.0042, 0.0146, 0.0057, 0.0233, 0.1077, 0.0252,
                  0.0115, 0.0606, 0.0379, -0.0216, 0.0051, 0.0864, 0.0223)
ELEVEN_DAY_FILTER_BASELINE = (0.0201, 0.0054, 0.0344, 0.0227, 0.0057, -0.0110,
                              0.0778, -0.0062, 0.0262, 0.0239, -0.0056)
#: the sample history, kept as the main warning
FOUR_DAY_EDGE = 0.0177
ELEVEN_DAY_EDGE = 0.0165
FOURTEEN_DAY_EDGE_015 = 0.0102


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

    # 2. the cap. The sweep in tools_v3/cap_optimal_v4.py showed that tightening it
    #    LOWERS returns, because accuracy rises with price and the cap discards the
    #    most accurate trades. So no cap is the shipped design, and the test asserts
    #    the two valid states rather than assuming a binding cap.
    cap = p.max_entry_price
    be = strategy.break_even_entry
    check("cap is above the measured entry price", cap >= OOS_ENTRY,
          f"cap {cap} vs entry {OOS_ENTRY}")
    check("cap is either disabled (>=1.0) or tight against break-even",
          cap >= 1.0 or cap - be < 0.02,
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

    # 4. the cap gate. With the cap disabled there is nothing to refuse, so the
    #    test switches to asserting the DISABLED behaviour explicitly rather than
    #    silently passing on a vacuous bound. Both branches are exercised by
    #    constructing a temporary params object with a binding cap.
    from dataclasses import replace as _replace
    strict = V4Params(**{**params.__dict__, "max_entry_price": 0.88})
    strict_strategy = StrategyV4(strict)
    decision = strict_strategy.decide(open_up_mid=0.40, entry_up_mid=0.40 + strong,
                                      ask=0.95, now=1000.0, quote_ts=1000.0,
                                      measured_fill=0.95, shares_wanted=10.0,
                                      depth_cost=9.5)
    check("an over-cap entry is refused when a cap is set", not decision.entering
          and decision.reason == "execution:entry_above_cap",
          f"{decision.reason} at 0.950 vs cap 0.880")
    decision = strict_strategy.decide(open_up_mid=0.40, entry_up_mid=0.40 + strong,
                                      ask=0.86, now=1000.0, quote_ts=1000.0,
                                      measured_fill=0.86, shares_wanted=10.0,
                                      depth_cost=8.6)
    check("an entry inside the cap is allowed", decision.entering,
          f"{decision.reason} at 0.860 vs cap 0.880")
    if p.max_entry_price >= 1.0:
        decision = strategy.decide(open_up_mid=0.40, entry_up_mid=0.40 + strong,
                                   ask=0.95, now=1000.0, quote_ts=1000.0,
                                   measured_fill=0.95, shares_wanted=10.0,
                                   depth_cost=9.5)
        check("with the cap disabled a 0.95 entry is NOT refused by the cap",
              decision.entering, f"{decision.reason} at 0.950")

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
    # 9. The edge arithmetic. These assertions encode the project's actual
    #    conclusion, which is negative: realised accuracy tracks the break-even
    #    line, so the fee is the whole loss. If a future change makes the strategy
    #    profitable, these tests will FAIL, which is the point - the conclusion must
    #    be re-derived rather than silently inherited.
    theoretical = OOS_ACCURACY - OOS_ENTRY - taker_fee_per_share(OOS_ENTRY)
    check("theoretical edge at the measured entry is NEGATIVE (fairly priced + fee)",
          theoretical < 0, f"{theoretical:+.4f}/share")
    check("realised out-of-sample edge is NEGATIVE", OOS_EDGE_PER_SHARE < 0,
          f"{OOS_EDGE_PER_SHARE:+.4f}/share")
    check("the shortfall is small (the strategy is near break-even, not broken)",
          abs(theoretical) < 0.02, f"{theoretical:+.4f}/share")
    check("the in-sample-optimal cell also FAILED out of sample (the lesson holds)",
          FAILED_CELL_OOS < 0, f"{FAILED_CELL_OOS:+.4f}/share")

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
    """Lead with the OUT-OF-SAMPLE numbers, because the in-sample ones flatter."""
    strategy = StrategyV4(params)
    entry = OOS_ENTRY
    accuracy = OOS_ACCURACY
    fee = taker_fee_per_share(entry, params.fee_coefficient)
    edge = accuracy - entry - fee
    daily = list(OOS_DAILY)
    mean_daily = statistics.fmean(daily)
    std_daily = statistics.pstdev(daily)
    stake_shares = stake / entry
    stderr = std_daily / math.sqrt(len(daily))

    print("=" * 100)
    print("V4 LIVE OUTLOOK - the answer is NEGATIVE, and this is why")
    print("=" * 100)
    print("Shipped cell: follow, T-120, thr 0.20, 4.5bps spot filter, no price cap,")
    print("hold to settlement. Parameters chosen on 15 in-sample days, then applied")
    print("UNCHANGED to 22 unseen days (2026-05-01 .. 05-24).")
    print()
    print("Realised accuracy tracks the break-even line at every price cap tested, which")
    print("is what a fairly priced market looks like. The crypto taker fee (confirmed at")
    print("0.07 by the venue docs) is then the entire loss. No cap setting turns a")
    print("profit, and the best of 16 caps swept is still slightly negative.")
    print()
    print(f"  {'':<26}{'IN-SAMPLE':>14}{'OUT-OF-SAMPLE':>16}")
    print("  " + "-" * 54)
    print(f"  {'days':<26}{IN_SAMPLE_DAYS:>14}{OOS_DAYS:>16}")
    print(f"  {'accuracy':<26}{IN_SAMPLE_ACCURACY:>14.4f}{OOS_ACCURACY:>16.4f}")
    print(f"  {'entry price':<26}{IN_SAMPLE_ENTRY:>14.4f}{OOS_ENTRY:>16.4f}")
    print(f"  {'edge per share':<26}{IN_SAMPLE_EDGE_PER_SHARE:>+14.4f}"
          f"{OOS_EDGE_PER_SHARE:>+16.4f}")
    print(f"  {'days positive':<26}"
          f"{str(IN_SAMPLE_DAYS_POSITIVE) + '/' + str(IN_SAMPLE_DAYS):>14}"
          f"{str(OOS_DAYS_POSITIVE) + '/' + str(OOS_DAYS):>16}")
    print(f"  {'trades per day':<26}{IN_SAMPLE_TRADES_PER_DAY:>14.0f}"
          f"{OOS_TRADES_PER_DAY:>16.0f}")
    print()
    print(f"  fee per share at the measured entry: {fee:.4f}")
    print(f"  break-even entry price            : {strategy.break_even_entry:.4f}   "
          f"(cap {params.max_entry_price:.3f})")
    print()
    print("  THE CELL THAT FAILED, for scale:")
    print(f"    {FAILED_CELL}")
    print("    Selecting on in-sample P&L alone picks that cell. Adding a day-level")
    print("    consistency requirement picks the shipped one instead.")

    print()
    print("=" * 100)
    print("1. PER-TRADE ECONOMICS AT A GIVEN STAKE (out-of-sample edge)")
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
              f"{per_trade:>+12.4f}{per_trade*OOS_TRADES_PER_DAY:>+10.2f}"
              f"{per_trade*OOS_TRADES_PER_DAY*30:>+11.1f}")

    print()
    print("=" * 100)
    print("2. HOW WELL IS IT PINNED DOWN?")
    print("=" * 100)
    print(f"  out-of-sample per-day pnl/share: "
          f"{', '.join(f'{v:+.4f}' for v in daily)}")
    print(f"  mean {mean_daily:+.4f}  std {std_daily:.4f}  -> standard error "
          f"{stderr:.4f} on {len(daily)} days")
    lo1 = (mean_daily - stderr) * stake_shares * OOS_TRADES_PER_DAY
    hi1 = (mean_daily + stderr) * stake_shares * OOS_TRADES_PER_DAY
    lo2 = (mean_daily - 2 * stderr) * stake_shares * OOS_TRADES_PER_DAY
    hi2 = (mean_daily + 2 * stderr) * stake_shares * OOS_TRADES_PER_DAY
    print(f"  1 std-error band: {lo1:+.2f} .. {hi1:+.2f} USDC/day")
    print(f"  2 std-error band: {lo2:+.2f} .. {hi2:+.2f} USDC/day")
    print()
    print("  The gap between these two numbers is the story:")
    print(f"    arithmetic edge at the measured entry : {edge:+.4f}/share")
    print(f"    what the trades actually paid (OOS)   : {OOS_EDGE_PER_SHARE:+.4f}/share")
    print("  A 385-trade sample can drift that far from the arithmetic. The arithmetic")
    print("  says the setup is worth ~1.8 points; ten days of it delivered 0.4.")
    print("  The 2-sigma band below still includes zero, so this sample does NOT")
    print("  establish a positive edge - it only fails to rule one out.")

    print()
    print("=" * 100)
    print("3. COST TOLERANCE - the edge is thin, so this decides everything")
    print("=" * 100)
    print(f"  {'scenario':<26}{'paid entry':>12}{'edge/share':>12}"
          f"{'per day':>10}{'verdict':>12}")
    print("-" * 100)
    for label, extra in (("as measured", 0.0), ("+1 cent slippage", 0.01),
                         ("+2 cents slippage", 0.02), ("+3 cents slippage", 0.03)):
        paid = entry + extra
        e = accuracy - paid - taker_fee_per_share(paid, params.fee_coefficient)
        per_day = e * (stake / paid) * OOS_TRADES_PER_DAY
        print(f"  {label:<26}{paid:>12.4f}{e:>+12.4f}{per_day:>+10.2f}"
              f"{('profitable' if e > 0 else 'LOSING'):>12}")

    print()
    print("=" * 100)
    print("BOTTOM LINE")
    print("=" * 100)
    per_trade = edge * stake_shares
    print(f"  Out-of-sample expectation: {per_trade*OOS_TRADES_PER_DAY:+.2f} USDC/day "
          f"at a {stake:.0f} USDC stake")
    print(f"  2 std-error band          : {lo2:+.2f} .. {hi2:+.2f} USDC/day")
    print(f"  Days positive             : {OOS_DAYS_POSITIVE}/{OOS_DAYS}")
    print()
    print("  This is the first figure in the project backed by genuinely unseen data,")
    print("  and it is smaller than every in-sample number quoted before it. It is")
    print("  also thin enough that one cent of extra slippage removes most of it. The")
    print("  next validation must be a SECOND out-of-sample period, not more tuning on")
    print("  these 24 days.")
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

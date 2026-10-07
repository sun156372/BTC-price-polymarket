#!/usr/bin/env python
"""run_v3.py - V3 sniper strategy: selfcheck, break-even ladder, evaluate, paper.

    python run_v3.py selfcheck
    python run_v3.py ladder
    python run_v3.py evaluate --dataset ../data/polymarket_btc_up_down_5m_251218_260909.csv
    python run_v3.py paper --snapshot snapshot_example.json

Read V3_README.md before pointing this at anything. Short version: the signal is
real but small (+3.0 points pooled out-of-sample), and the trade is only positive
EV below roughly 0.535 per share. The execution gate enforces that, so V3 will
usually answer "hold" - which is the point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_v3.config import ConfigError, describe, load_config
from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, load
from engine_v3.signals import Standardiser, break_even_accuracy, taker_fee_per_share
from engine_v3.strategy import (
    QUOTE_STALE,
    Quote,
    StrategyV3Runtime,
    required_accuracy_ladder,
)


# --------------------------------------------------------------- selfcheck
def cmd_selfcheck(config, args) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # 1. config loads and validates
    check("config loads", config is not None)
    check("entry cap is below 1", 0 < config.execution.max_entry_price < 1,
          str(config.execution.max_entry_price))

    # 2. break-even maths is self-consistent
    for price in (0.40, 0.50, 0.535, 0.60, 0.70):
        needed = break_even_accuracy(price, config.fees.taker_coefficient)
        check(f"break-even({price:.3f}) = price + fee",
              abs(needed - (price + taker_fee_per_share(price, config.fees.taker_coefficient))) < 1e-12,
              f"{needed:.6f}")
    check("fee vanishes at the extremes",
          taker_fee_per_share(0.001) < 1e-4 and taker_fee_per_share(0.999) < 1e-4)
    check("fee peaks at 0.5",
          taker_fee_per_share(0.5) > taker_fee_per_share(0.3) > taker_fee_per_share(0.1))

    # 3. the ladder is monotone: a higher entry price always needs more accuracy
    ladder = required_accuracy_ladder(config.fees.taker_coefficient)
    check("ladder is monotone increasing",
          all(b[1] > a[1] for a, b in zip(ladder, ladder[1:])))

    # 4. the gate refuses an entry above the cap, whatever the signal says
    runtime = StrategyV3Runtime(config)
    strong = {name: 0.0 for name in config.signal.features}
    quote_ok = Quote(bid=0.50, ask=0.51, ts=1000.0)
    decision = runtime.decide(features=strong, quote=quote_ok, now=1000.0,
                              equity=1000.0, cash=1000.0,
                              seconds_since_last_entry=10_000.0, open_positions=0,
                              measured_fill=0.51)
    check("no signal -> hold", not decision.entering, decision.reason)

    # 5. gate: price above cap must be refused even with a maximal signal
    runtime_no_scale = StrategyV3Runtime(config)
    # force a strong z by feeding an extreme feature deviation
    extreme = {name: 10.0 for name in config.signal.features}
    expensive = Quote(bid=0.70, ask=0.71, ts=1000.0)
    d = runtime_no_scale.decide(features=extreme, quote=expensive, now=1000.0,
                                equity=1000.0, cash=1000.0,
                                seconds_since_last_entry=10_000.0, open_positions=0,
                                measured_fill=0.71)
    check("expensive entry refused by the price cap",
          not d.entering and d.reason == "execution:entry_above_cap", d.reason)

    # 6. gate: a modelled (unmeasured) fill is refused when require_measured_fill
    if config.execution.require_measured_fill:
        cheap = Quote(bid=0.49, ask=0.50, ts=1000.0)
        d2 = runtime_no_scale.decide(features=extreme, quote=cheap, now=1000.0,
                                     equity=1000.0, cash=1000.0,
                                     seconds_since_last_entry=10_000.0,
                                     open_positions=0, measured_fill=None)
        check("unmeasured fill refused", not d2.entering, d2.reason)

    # 7. gate: a stale quote is refused
    stale = Quote(bid=0.49, ask=0.50, ts=0.0)
    d3 = runtime_no_scale.decide(features=extreme, quote=stale, now=10_000.0,
                                 equity=1000.0, cash=1000.0,
                                 seconds_since_last_entry=10_000.0, open_positions=0,
                                 measured_fill=0.50)
    check("stale quote refused", not d3.entering and d3.reason == QUOTE_STALE, d3.reason)

    # 8. gate: a wide spread is refused
    wide = Quote(bid=0.30, ask=0.60, ts=1000.0)
    d4 = runtime_no_scale.decide(features=extreme, quote=wide, now=1000.0,
                                 equity=1000.0, cash=1000.0,
                                 seconds_since_last_entry=10_000.0, open_positions=0,
                                 measured_fill=0.45)
    check("wide spread refused", not d4.entering, d4.reason)

    # 9. a good setup actually enters, and never above the cap
    good = Quote(bid=0.48, ask=0.49, ts=1000.0)
    d5 = runtime_no_scale.decide(features=extreme, quote=good, now=1000.0,
                                 equity=1000.0, cash=1000.0,
                                 seconds_since_last_entry=10_000.0, open_positions=0,
                                 measured_fill=0.49)
    check("good setup enters", d5.entering, d5.reason)
    check("entry respects the notional cap",
          d5.shares * d5.limit_price <= config.sizing.notional_usdc + 1e-9,
          f"{d5.shares * d5.limit_price:.4f}")

    # 10. cooldown and concurrency are enforced
    d6 = runtime_no_scale.decide(features=extreme, quote=good, now=1000.0,
                                 equity=1000.0, cash=1000.0,
                                 seconds_since_last_entry=0.0, open_positions=0,
                                 measured_fill=0.49)
    check("cooldown enforced", not d6.entering, d6.reason)
    d7 = runtime_no_scale.decide(features=extreme, quote=good, now=1000.0,
                                 equity=1000.0, cash=1000.0,
                                 seconds_since_last_entry=10_000.0, open_positions=1,
                                 measured_fill=0.49)
    check("one position at a time", not d7.entering, d7.reason)

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"selfcheck: {len(checks) - failed}/{len(checks)} passed")
    return 0 if not failed else 1


# ------------------------------------------------------------------ ladder
def cmd_ladder(config, args) -> int:
    print("=" * 78)
    print("V3 BREAK-EVEN LADDER - the gate this strategy enforces")
    print("=" * 78)
    print("A directional binary held to settlement breaks even at:")
    print("    accuracy_needed = entry_price + 0.07 * p * (1 - p)")
    print()
    print(f"{'entry price':>12}{'fee/share':>11}{'accuracy needed':>17}{'verdict':>14}")
    print("-" * 78)
    measured = 0.5519     # pooled out-of-sample accuracy from the 9-month walk-forward
    for price, needed in required_accuracy_ladder(config.fees.taker_coefficient):
        verdict = "positive EV" if measured > needed else "negative EV"
        marker = "  <-- cap" if abs(price - 0.535) < 1e-9 else ""
        print(f"{price:>12.3f}{taker_fee_per_share(price):>11.4f}"
              f"{needed:>17.4f}{verdict:>14}{marker}")
    print()
    print(f"Measured pooled out-of-sample accuracy : {measured:.4f}")
    print(f"Configured entry cap                   : {config.execution.max_entry_price:.3f}")
    print(f"Break-even accuracy at the cap         : "
          f"{break_even_accuracy(config.execution.max_entry_price):.4f}")
    print()
    print("Any fill above the cap is refused by execution.max_entry_price,")
    print("because no signal quality can rescue a price that already ate the edge.")
    return 0


def _fit_standardiser(config, dataset: str | None):
    """Fit the feature scaler on the dataset and persist it.

    The live path must standardise with the same statistics the walk-forward used,
    otherwise the z threshold means a different thing in production than it did in
    the analysis. The scaler is written next to the config and loaded by paper mode.
    """
    from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, load as load_windows
    from engine_v3.signals import Standardiser

    path = dataset or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DEFAULT_CSV)
    windows = load_windows(path, features=SIGNAL_FEATURES)
    standardiser = Standardiser(SIGNAL_FEATURES).fit(windows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "config_v3", "standardiser_v3.json")
    standardiser.save(out)
    return standardiser, out, len(windows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["selfcheck", "ladder", "evaluate", "fit", "paper"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--snapshot", default=None,
                        help="JSON feature snapshot for the paper command")
    parser.add_argument("--standardiser", default=None,
                        help="fitted scaler JSON (default: config_v3/standardiser_v3.json)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "selfcheck":
        print("=" * 78)
        print("V3 SELFCHECK")
        print("=" * 78)
        for line in describe(config):
            print(f"  {line}")
        print()
        return cmd_selfcheck(config, args)
    if args.command == "ladder":
        return cmd_ladder(config, args)
    if args.command == "fit":
        standardiser, out, count = _fit_standardiser(config, args.dataset)
        print(f"fitted scaler on {count:,} windows")
        for name in config.signal.features:
            print(f"  {name:<16} mean {standardiser.mean[name]:+.6g}  "
                  f"std {standardiser.std[name]:.6g}")
        print(f"written: {out}")
        return 0
    if args.command == "evaluate":
        from tools_v3.analyze_v3 import run as run_analysis
        dataset = args.dataset or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DEFAULT_CSV)
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output", "v3")
        return run_analysis(dataset, args.folds, out, quick=False, min_trades=50)
    if args.command == "paper":
        from engine_v3.paper import run_paper
        return run_paper(config, args.snapshot, args.standardiser)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

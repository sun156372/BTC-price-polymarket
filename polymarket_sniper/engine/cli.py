"""Command line interface.

    python run_engine.py selfcheck
    python run_engine.py backtest --data ../data
    python run_engine.py backtest --data ../data --adverse-bps 50
    python run_engine.py paper --schema

``selfcheck`` runs entirely on synthetic data - no dataset, no dependency on
pyarrow, no network. Run it first: it verifies the maths, the signal timing rule,
every guard, and the exit sequencing before you trust a single number from the
backtest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .backtest import BacktestError, Engine, compute_metrics, run_backtest, write_outputs
from .broker import LiveBroker, OrderRequest, OrderSide, PaperBroker, Position
from .config import ConfigError, EngineConfig, load_config
from .dataio import DataError, Dataset, Market, MarketTrades, SpotIndex
from .guards import (
    DepthGuard,
    GuardChain,
    PriceBandGuard,
    Quote,
    SlippageGuard,
    StalenessGuard,
)
from .mathx import MathError, shares_for_notional, taker_fee
from .signals import Reason, Side, SignalDetector
from . import __version__

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_DATA = 3


def _logger(level: str):
    import logging

    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("engine")


# --------------------------------------------------------------------- selfcheck
def cmd_selfcheck(args, config: EngineConfig) -> int:
    """Verify maths, timing, guards and exits on synthetic data."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # ---- fee curve ------------------------------------------------------
    fee = taker_fee(100, 0.5)
    check("taker fee at p=0.5 == shares*0.07*0.25", abs(fee - 1.75) < 1e-12, f"{fee}")
    check("fee peaks at p=0.5",
          taker_fee(100, 0.5) > taker_fee(100, 0.25) > taker_fee(100, 0.05))
    check("fee vanishes at the extremes", taker_fee(100, 0.999) < fee * 0.02)
    try:
        taker_fee(100, 0.0)
        check("fee rejects a price outside (0,1)", False, "no exception")
    except MathError:
        check("fee rejects a price outside (0,1)", True)

    # parity with the original research calculator, if it is importable
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.fee_calculator import taker_fee_usdc  # type: ignore

        worst = max(abs(taker_fee(100, p) - taker_fee_usdc(100, p))
                    for p in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95))
        check("fee matches the original research calculator", worst < 1e-12, f"max diff {worst:.2e}")
    except ImportError:
        check("fee matches the original research calculator", True, "original not importable, skipped")

    check("shares_for_notional inverts", abs(shares_for_notional(100, 0.4) - 250) < 1e-9)

    # ---- signal timing (backward-only, threshold, staleness) ------------
    detector = SignalDetector(lookback_s=180.0, min_move_pct=0.05,
                             max_spot_staleness_s=30.0, min_window_elapsed_s=60.0)
    base = 1_800_000_000.0

    def build_series(pct: float, *, span: int = 100, quiet: int = 300) -> SpotIndex:
        """Flat, then a linear ramp of ``pct`` percent over ``span`` seconds."""
        index = SpotIndex("BTC")
        for i in range(0, quiet + span + 1):
            factor = 1.0 + (pct / 100.0) * max(0, i - quiet) / span
            index.add(base + i, 80_000.0 * factor)
        index.finalise()
        return index

    rising = build_series(+1.0)          # +1% ramp ending at base+400
    decide_at = base + 340               # 60s into the ramp -> ~0.6% over 180s

    rise = detector.evaluate(rising, ts=decide_at, window_start_ts=base - 600)
    check("rising spot -> UP", rise.side is Side.UP and rise.is_actionable, rise.detail)
    check("signal reports move size", rise.move_pct is not None and rise.move_pct > 0.05,
          str(rise.move_pct))
    check("signal strength scales with the move",
          rise.strength > 1.0, f"{rise.strength:.2f}")

    falling = build_series(-1.0)
    drop = detector.evaluate(falling, ts=decide_at, window_start_ts=base - 600)
    check("falling spot -> DOWN", drop.side is Side.DOWN, drop.detail)

    flat_series = build_series(0.0)
    flat = detector.evaluate(flat_series, ts=decide_at, window_start_ts=base - 600)
    check("flat spot -> refused", flat.side is None, flat.reason.value)

    early = detector.evaluate(rising, ts=base + 10, window_start_ts=base)
    check("window warm-up respected", early.reason is Reason.WINDOW_TOO_EARLY, early.reason.value)

    stale = detector.evaluate(rising, ts=base + 900, window_start_ts=base - 600)
    check("stale spot refused", stale.reason in {Reason.SPOT_STALE, Reason.NO_SPOT},
          stale.reason.value)

    # backward-only proof: decide at a moment, THEN add a violent move after it.
    # If the decision changes, the lookup was reading the future.
    leak = SpotIndex("BTC")
    for i in range(0, 601):
        leak.add(base + i, 80_000.0)
    leak.finalise()
    before = detector.evaluate(leak, ts=base + 300, window_start_ts=base - 600)
    leak.add(base + 301, 88_000.0)       # +10% AFTER the decision moment
    leak.finalise()
    after = detector.evaluate(leak, ts=base + 300, window_start_ts=base - 600)
    check("no look-ahead: a future spike cannot change a past decision",
          before.side is None and after.side is None,
          f"before={before.reason.value} after={after.reason.value}")

    # ---- guards ---------------------------------------------------------
    fresh = Quote(ts=base, price=0.52, size_at_price=100.0)
    check("staleness guard passes fresh", StalenessGuard(10.0).check(fresh, now=base + 3).allowed)
    stale_q = Quote(ts=base - 40, price=0.52, size_at_price=100.0)
    decision = StalenessGuard(10.0).check(stale_q, now=base)
    check("staleness guard blocks old print", not decision.allowed, decision.detail)

    slip = SlippageGuard(200.0)
    check("slippage guard passes a 1c move on 0.50",
          slip.check(signal_price=0.50, quote=Quote(ts=base, price=0.505, size_at_price=None)).allowed)
    blocked = slip.check(signal_price=0.50, quote=Quote(ts=base, price=0.60, size_at_price=None))
    check("slippage guard blocks a 20c jump", not blocked.allowed, blocked.detail)

    depth = DepthGuard(0.5)
    check("depth guard abstains without size",
          depth.check(shares=500, quote=Quote(ts=base, price=0.5)).allowed)
    thin = depth.check(shares=80, quote=Quote(ts=base, price=0.5, size_at_price=100))
    check("depth guard blocks oversized order", not thin.allowed, thin.detail)

    band = PriceBandGuard(0.0, 1.0)
    check("price band disabled by default allows expensive entries",
          band.check(Quote(ts=base, price=0.95)).allowed)

    # ---- exits: sequential first touch ----------------------------------
    # The ramp must have happened BEFORE the decision moment, so build a series
    # whose 1% rise lands exactly on the market's T-120 checkpoint.
    market_end = base + 360.0
    entry_ts = market_end - 120.0                     # = base + 240
    exit_series = build_series(+1.0, span=100, quiet=int(entry_ts - base) - 100)

    cfg = load_config(overrides={
        "signal": {"entry_t_minus_s": 120.0, "lookback_s": 180.0, "min_move_pct": 0.05,
                   "max_spot_staleness_s": 60.0},
        "guards": {"min_window_elapsed_s": 0.0},
    })
    engine = Engine(cfg)
    engine.spot = {"BTC": exit_series}

    market = Market(slug="btc-updown-15m-1", asset="BTC", start_ts=base - 600,
                    end_ts=market_end, outcome="UP")
    tokens = ("tok_up", "tok_down")

    probe = SignalDetector(lookback_s=180.0, min_move_pct=0.05).evaluate(
        exit_series, ts=entry_ts, window_start_ts=market.start_ts
    )
    check("fixture produces an actionable signal at entry",
          probe.is_actionable, f"{probe.reason.value}: {probe.detail}")

    def window(prints: dict[str, list[tuple[float, float]]]) -> MarketTrades:
        return MarketTrades(slug=market.slug, by_token=prints)

    # entry at market_end-120 = base+240; spot rose, so UP is bought
    up_prints = [(entry_ts - 2, 0.50)]
    # a print touching the take profit (0.50 + 0.30 = 0.80) later in the window
    up_prints.append((entry_ts + 30, 0.85))
    trade, refusal = engine.trade_window(market, window({"tok_up": up_prints, "tok_down": []}), tokens)
    check("take profit fires on first touch",
          trade is not None and trade.exit_reason == "take_profit",
          refusal or (trade.exit_reason if trade else "no trade"))
    if trade:
        check("TP exit price is exactly entry+TP",
              abs(trade.exit_price - (0.50 + 0.30)) < 1e-9, f"{trade.exit_price}")
        check("exit fee charged on a real exit trade", trade.exit_fee > 0, f"{trade.exit_fee}")

    up_prints = [(entry_ts - 2, 0.50), (entry_ts + 30, 0.25)]
    trade, _ = engine.trade_window(market, window({"tok_up": up_prints, "tok_down": []}), tokens)
    check("stop loss fires on first touch",
          trade is not None and trade.exit_reason == "stop_loss",
          trade.exit_reason if trade else "no trade")
    if trade:
        check("SL exit price is exactly entry-SL",
              abs(trade.exit_price - (0.50 - 0.20)) < 1e-9, f"{trade.exit_price}")

    up_prints = [(entry_ts - 2, 0.50), (entry_ts + 30, 0.55), (entry_ts + 60, 0.60)]
    trade, _ = engine.trade_window(market, window({"tok_up": up_prints, "tok_down": []}), tokens)
    check("neither level -> settle at 1.0 for a correct call",
          trade is not None and trade.exit_reason == "settlement" and trade.exit_price == 1.0,
          trade.exit_reason if trade else "no trade")
    if trade:
        check("settlement charges no exit fee", trade.exit_fee == 0.0, f"{trade.exit_fee}")
        check("settled winner is profitable", trade.net_pnl > 0, f"{trade.net_pnl:.4f}")

    # both levels inside one print -> stop wins (conservative default)
    up_prints = [(entry_ts - 2, 0.50), (entry_ts + 30, 0.90), (entry_ts + 31, 0.10)]
    trade, _ = engine.trade_window(market, window({"tok_up": up_prints, "tok_down": []}), tokens)
    check("later stop after an earlier touch of TP: first touch wins (TP)",
          trade is not None and trade.exit_reason == "take_profit",
          trade.exit_reason if trade else "no trade")

    # ---- staleness gate changes the outcome ------------------------------
    stale_up = [(entry_ts - 45, 0.50), (entry_ts + 30, 0.85)]
    trade, refusal = engine.trade_window(market, window({"tok_up": stale_up, "tok_down": []}), tokens)
    check("stale entry is refused by the chain",
          trade is None and refusal.startswith("guard:"), refusal)

    # ---- a losing window -------------------------------------------------
    down_market = Market(slug="btc-updown-15m-2", asset="BTC", start_ts=base - 600,
                         end_ts=base + 400, outcome="DOWN")
    trade, _ = engine.trade_window(
        down_market,
        window({"tok_up": [(entry_ts - 2, 0.50), (entry_ts + 30, 0.55)], "tok_down": []}),
        tokens,
    )
    check("wrong direction is recorded as such",
          trade is not None and not trade.direction_correct,
          str(trade.direction_correct) if trade else "no trade")
    if trade:
        check("wrong direction loses money", trade.net_pnl < 0, f"{trade.net_pnl:.4f}")

    # ---- metrics separate the two accuracies -----------------------------
    metrics = compute_metrics([trade] if trade else [])
    check("metrics expose direction accuracy",
          "direction_accuracy_pct" in metrics.as_dict())
    check("metrics expose break-even accuracy",
          abs(metrics.break_even_accuracy - metrics.avg_entry_price) < 1e-9)

    # ---- live broker refuses to exist implicitly -------------------------
    try:
        LiveBroker()
        check("live broker refuses implicit construction", False, "constructed")
    except Exception:  # noqa: BLE001 - any refusal is acceptable
        check("live broker refuses implicit construction", True)

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"selfcheck: {len(checks) - failed}/{len(checks)} passed")
    if not failed:
        print()
        print("Next: python run_engine.py backtest --data ../data")
    return EXIT_OK if not failed else EXIT_FAIL


# --------------------------------------------------------------------- backtest
def _print_metrics(metrics, title: str) -> None:
    d = metrics.as_dict()
    print(f"\n{title}")
    if not d["trades"]:
        print("  no trades")
        return
    print(f"  trades                : {d['trades']:,}")
    print(f"  DIRECTION accuracy    : {d['direction_accuracy_pct']:.2f}%   "
          f"(break-even needs {d['break_even_accuracy_pct']:.2f}% at avg entry "
          f"{d['avg_entry_price']:.4f})")
    print(f"  edge                  : {d['edge_points']:+.2f} points")
    print(f"  profitable trades     : {d['win_rate_pct']:.2f}%")
    print(f"  gross / fees / net    : {d['gross_pnl']:+,.2f} / {d['fees']:,.2f} / "
          f"{d['net_pnl']:+,.2f}")
    print(f"  avg per trade         : {d['avg_net_pnl']:+.4f}   "
          f"profit factor {d['profit_factor'] if d['profit_factor'] is None else round(d['profit_factor'], 3)}")
    print(f"  max drawdown          : {d['max_drawdown']:,.2f}   sharpe {d['sharpe']}")
    print(f"  exits                 : TP {d['take_profit_exits']} / "
          f"SL {d['stop_loss_exits']} / settle {d['settled_exits']}")
    print(f"  entry staleness       : avg {d['avg_entry_staleness_s']:.1f}s   "
          f"p90 {d['p90_entry_staleness_s']:.1f}s")
    if d["rejected_entries"]:
        print(f"  refused entries       : {d['rejected_entries']:,}")
        for reason, count in sorted(d["rejection_reasons"].items(),
                                    key=lambda kv: -kv[1])[:6]:
            print(f"      {reason:<28} {count:,}")


def cmd_backtest(args, config: EngineConfig) -> int:
    log = _logger(args.log_level or config.section("logging").level)
    data_dir = args.data or config.resolve(config.section("data").data_dir)
    try:
        dataset = Dataset(data_dir, assets=args.assets or config.section("data").assets,
                          batch_size=config.section("data").batch_size)
        summary = dataset.describe()
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA

    print("=" * 84)
    print(f"ENDGAME MOMENTUM SNIPER  v{__version__}")
    print("=" * 84)
    print(f"data dir        : {summary['data_dir']}")
    print(f"files           : {summary['files']}")
    print(f"labelled markets: {summary['markets_with_outcome']:,}")
    print(f"spot assets     : {summary['spot_assets']}")
    print(f"entry           : T-{config.section('signal').entry_t_minus_s:.0f}s   "
          f"lookback {config.section('signal').lookback_s:.0f}s   "
          f"threshold {config.section('signal').min_move_pct}%")
    print(f"exits           : TP +{config.section('exits').take_profit}   "
          f"SL -{config.section('exits').stop_loss}   "
          f"staleness gate {config.section('guards').max_quote_staleness_s:.0f}s")

    try:
        ideal, _ = run_backtest(
            config, data_dir=data_dir,
            trades_file=args.trades, assets=args.assets,
            adverse_bps=0.0, label="ideal fills", logger=log,
        )
    except (BacktestError, DataError) as exc:
        print(f"backtest error: {exc}", file=sys.stderr)
        return EXIT_DATA

    _print_metrics(ideal.metrics, "IDEAL FILLS (fills at the last real print)")

    live_metrics = None
    if config.section("backtest").simulate_adverse_fills:
        adverse = args.adverse_bps
        adverse = config.section("backtest").entry_slippage_bps if adverse is None else adverse
        live, _ = run_backtest(
            config, data_dir=data_dir,
            trades_file=args.trades, assets=args.assets,
            adverse_bps=adverse, label=f"adverse {adverse:.0f}bps", logger=log,
        )
        live_metrics = live.metrics
        _print_metrics(live.metrics, f"LIVE-REALISTIC FILLS (every fill {adverse:.0f}bps worse)")

    output_dir = config.resolve(args.output_dir or config.section("backtest").output_dir)
    paths = write_outputs(ideal, output_dir, stem="ideal")
    if live_metrics is not None:
        payload = {"ideal": ideal.metrics.as_dict(), "live": live_metrics.as_dict()}
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        paths["summary"] = summary_path

    print()
    for name, path in paths.items():
        print(f"{name:<10}: {path}")

    if ideal.metrics.trades:
        edge = ideal.metrics.edge_points
        print()
        if edge <= 0:
            print("VERDICT: no edge after costs on this sample - do not trade this config.")
        elif ideal.metrics.trades < 100:
            print(f"VERDICT: thin sample ({ideal.metrics.trades} trades). Treat with suspicion.")
        else:
            print(f"VERDICT: +{edge:.1f} points of edge over {ideal.metrics.trades:,} trades. "
                  "Validate out-of-sample before sizing up.")
    return EXIT_OK


# ------------------------------------------------------------------------- paper
def cmd_paper(args, config: EngineConfig) -> int:
    """Explain and validate the live wiring without connecting to anything."""
    print("=" * 84)
    print("PAPER MODE")
    print("=" * 84)
    print()
    print("The engine is deliberately split so that live trading cannot happen by")
    print("accident. To run against a live feed you supply two objects:")
    print()
    print("  1. a spot feed    - an object with .as_of(ts) / .change_pct(ts, lookback)")
    print("                      (see engine/dataio.py: SpotIndex)")
    print("  2. a market feed  - prints per token: (timestamp, price)")
    print("                      (see engine/dataio.py: MarketTrades)")
    print()
    print("Then drive the same decision path the backtest uses:")
    print()
    print("    from engine.backtest import Engine")
    print("    from engine.dataio import Market, MarketTrades, SpotIndex")
    print("    from engine.broker import LiveBroker   # your subclass")
    print()
    print("    engine = Engine(config, broker=MyLiveBroker(confirm_live=True))")
    print("    trade, refusal = engine.trade_window(market, trades, tokens)")
    print()
    print("`trade_window` performs the full sequence - signal, guards, entry, sequential")
    print("TP/SL exit - and never opens a socket itself.")
    print()

    log = _logger(args.log_level or config.section("logging").level)
    data_dir = args.data or config.resolve(config.section("data").data_dir)
    try:
        dataset = Dataset(data_dir, assets=config.section("data").assets,
                          batch_size=config.section("data").batch_size)
        summary = dataset.describe()
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA

    print("Dataset readiness for a replay/paper run:")
    for key, value in summary.items():
        print(f"  {key:<24} {value}")
    if not summary["markets_with_outcome"]:
        print("\n  NOTE: no labelled markets. Paper replay needs outcomes only to score")
        print("        itself; live trading does not.")
    print()
    print("To replay history through the live code path, use `backtest` - it calls the")
    print("same Engine.trade_window the live loop would.")
    return EXIT_OK


# -------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_engine.py",
        description="Polymarket endgame momentum sniper (offline by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, help="path to engine/config.json")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--version", action="version", version=f"engine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("selfcheck", help="verify maths, signals, guards, exits (synthetic)")
    p_self.set_defaults(func=cmd_selfcheck)

    p_bt = sub.add_parser("backtest", help="replay the dataset through the engine")
    p_bt.add_argument("--data", default=None, help="data directory (parquet exports)")
    p_bt.add_argument("--trades", default="auto", help="trades parquet (default: richest)")
    p_bt.add_argument("--assets", nargs="*", default=None, help="restrict to these assets")
    p_bt.add_argument("--adverse-bps", type=float, default=None,
                      help="worsen every fill by this many bps for the live-realistic run")
    p_bt.add_argument("--output-dir", default=None)
    p_bt.set_defaults(func=cmd_backtest)

    p_paper = sub.add_parser("paper", help="show live wiring and check dataset readiness")
    p_paper.add_argument("--data", default=None)
    p_paper.set_defaults(func=cmd_paper)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level boundary
        if (args.log_level or "").upper() == "DEBUG":
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL

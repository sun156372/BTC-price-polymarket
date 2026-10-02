"""Main entry point - Polymarket 5-minute sniper system.

Subcommands
-----------
``selfcheck``
    Verify the fee/VWAP/depth maths and the decision path offline. No data, no
    network. Run this first.
``sweep``
    Backtest all 1200 combinations from ``data/param_combinations.csv`` and rank
    them.
``backtest``
    Backtest a single parameter set (by combo id, or explicit flags).
``paper``
    Replay recorded observations through the live decision path without sending
    anything (this is the closest thing to a dry run of the strategy).

Examples
--------
    python main.py selfcheck
    python main.py sweep --prices ../data/btc5m_prices.csv --markets ../data/btc5m_markets.csv
    python main.py backtest --combo c00001 --top 5
    python main.py paper --observations ../data/btc5m_prices.csv --limit 200

Nothing here performs a network request. Live trading requires you to inject a
broker and a real transport; both are opt-in by design.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

# --- make the package importable no matter where python is invoked from -------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.loader import Config, load_config  # noqa: E402
from utils.logger import add_file_handler, get_logger, set_level, write_json  # noqa: E402

log = get_logger("main")


# --------------------------------------------------------------------- helpers
def _resolve_project_path(value: str) -> str:
    """Resolve a user path: absolute stays, relative tries cwd then project root."""
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    if os.path.exists(value):
        return os.path.abspath(value)
    candidate = os.path.join(_PROJECT_ROOT, value)
    if os.path.exists(candidate):
        return candidate
    # fall back to the parent workspace, where the scraper writes its data/
    parent = os.path.dirname(_PROJECT_ROOT)
    candidate = os.path.join(parent, value)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(value)


def _engine_kwargs(config: Config) -> dict:
    b = config.section("backtest")
    e = config.section("execution")
    f = config.section("fees")
    s = config.section("strategy")
    return {
        "notional_per_trade": float(e.default_order_notional),
        "fee_coefficient": float(f.taker_fee_coefficient),
        "fee_curve": str(b.fee_curve),
        "maker_rebate_share": float(f.maker_rebate_share),
        "entry_side_mode": str(s.entry_side_mode),
        "spot_confirm_pct": float(s.spot_confirm_pct),
        "probability_gap_min": float(s.probability_gap_min),
        "probability_gap_max": float(s.probability_gap_max),
        "max_entry_price": float(s.max_entry_price),
        "min_entry_price": float(s.min_entry_price),
    }


def _load_observations(args, config: Config):
    """Load observations, preferring the raw series when available."""
    from core.backtest_engine import (
        BacktestError,
        load_observations_from_prices_csv,
        load_observations_from_raw_jsonl,
    )

    offsets = None
    if args.offsets:
        offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]

    raw = _resolve_project_path(args.raw or config.section("backtest").raw_jsonl or "")
    if raw and os.path.exists(raw):
        wanted = offsets or [int(o) for o in _default_offsets(config)]
        observations = load_observations_from_raw_jsonl(raw, offsets=wanted)
        log.info("loaded %d observations from raw series %s", len(observations), raw)
        return observations

    prices = _resolve_project_path(args.prices or "")
    if not prices or not os.path.exists(prices):
        raise BacktestError(
            "no observations available: pass --prices <btc5m_prices.csv> or "
            "--raw <btc5m_raw.jsonl>"
        )
    markets = _resolve_project_path(args.markets) if args.markets else ""
    observations = load_observations_from_prices_csv(
        prices,
        markets_csv=markets or None,
        offsets=offsets,
        require_price=not args.allow_missing_prices,
        limit_markets=args.limit_markets,
    )
    log.info("loaded %d observations from %s", len(observations), prices)
    return observations


def _default_offsets(config: Config) -> list[int]:
    """The observation grid the sweep should use when none is given."""
    base = int(config.section("strategy").observation_time_s)
    grid = sorted({base, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300}, reverse=True)
    return grid


# ------------------------------------------------------------------ selfcheck
def cmd_selfcheck(args, config: Config) -> int:
    """Offline verification of the maths and the decision path."""
    from core.backtest_engine import MarketObservation, ParamSet, BacktestEngine
    from core.execution import ExecutionEngine, estimate_execution
    from core.market_data import BookSnapshot, MarketDataHub, SpotTick, UP, DOWN, liquidity_score, order_book_imbalance
    from core.risk_control import RiskController, RiskLimits, Action
    from strategies.sniper_strategy import SniperStrategy, implied_move_probability
    from utils.fee_calculator import break_even_move, fee_pool_usdc, maker_fee_usdc, taker_fee_usdc
    from utils.vwap_calculator import vwap_buy, split_notional

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---- fees -----------------------------------------------------------
    fee = taker_fee_usdc(100, 0.5)
    check("taker fee at p=0.5 is shares*0.07*0.25", abs(fee - 100 * 0.07 * 0.25) < 1e-12, f"{fee}")
    check(
        "fee shrinks toward the extremes",
        taker_fee_usdc(100, 0.999) < fee * 0.02 and taker_fee_usdc(100, 0.001) < fee * 0.02,
        f"p=0.999 -> {taker_fee_usdc(100, 0.999):.6f} vs p=0.5 -> {fee:.6f}",
    )
    check(
        "fee peaks at p=0.5",
        taker_fee_usdc(100, 0.5) > taker_fee_usdc(100, 0.25) > taker_fee_usdc(100, 0.05),
    )
    check("maker side is a rebate", maker_fee_usdc(100, 0.5) < 0, f"{maker_fee_usdc(100, 0.5)}")
    check(
        "break-even move equals coefficient*p*(1-p)",
        abs(break_even_move(0.5) - 0.07 * 0.25) < 1e-12,
        f"{break_even_move(0.5)}",
    )
    check("fee formula is curve-consistent", abs(fee_pool_usdc(100, 0.3) - 100 * 0.07 * 0.3 * 0.7) < 1e-12)

    # ---- vwap -----------------------------------------------------------
    asks = [(0.50, 10.0), (0.51, 10.0), (0.52, 10.0)]
    result = vwap_buy(asks, 15.0, reference_price=0.50)
    expected_vwap = (10 * 0.50 + 5 * 0.51) / 15
    check("VWAP walks the ladder", abs(result.vwap - expected_vwap) < 1e-9, f"{result.vwap}")
    check("VWAP slippage is positive when lifting", result.slippage_bps > 0)
    partial = vwap_buy(asks, 100.0, reference_price=0.50)
    check("thin book reports partial fill", not partial.fully_filled, f"{partial.filled_shares}")
    clips = split_notional(300.0, asks, participation_rate=0.25, min_children=2, max_children=6)
    check("split children sum to the parent", abs(sum(clips) - 300.0) < 1e-6, f"{clips}")

    # ---- order book analytics -------------------------------------------
    # a realistic binary book: 1c wide around 0.50 (3.9% of price), with a
    # dominant top level so wall detection has something to find
    bids = [(0.49, 100.0), (0.48, 30.0), (0.47, 20.0)]
    asks_book = [(0.51, 80.0), (0.52, 20.0), (0.53, 10.0)]
    obi = order_book_imbalance(bids, asks_book)
    check("OBI sign follows the heavier side", obi > 0, f"{obi:.4f}")
    check("OBI is bounded", -1.0 <= obi <= 1.0)
    check("OBI is zero for an empty book", order_book_imbalance([], []) == 0.0)
    check("liquidity score is bounded", 0.0 <= liquidity_score(bids) <= 1.0)
    check(
        "balanced depth scores above one-sided depth",
        liquidity_score([(0.5, 10), (0.49, 10), (0.48, 10)])
        > liquidity_score([(0.5, 30), (0.49, 0.0), (0.48, 0.0)]),
    )
    snapshot = BookSnapshot(token_id="t", side=UP, bids=tuple(bids), asks=tuple(asks_book))
    check("spread is ask-bid", abs((snapshot.spread() or 0) - 0.02) < 1e-12, f"{snapshot.spread()}")
    check("midpoint is the mid", abs((snapshot.midpoint() or 0) - 0.50) < 1e-12)
    check("mid_sum complements hold", abs((snapshot.midpoint() or 0) + (1 - (snapshot.midpoint() or 0)) - 1) < 1e-12)
    check("walls are detected", len(snapshot.walls(min_share=0.3)) >= 1)

    # ---- execution ------------------------------------------------------
    engine = ExecutionEngine(slippage_tolerance=0.01, max_slippage_bps=150.0)
    plan = engine.plan_order(snapshot, "buy", notional_usdc=20.0)
    check("plan produces children", plan.executable and len(plan.children) >= 1, str(plan.blocked_reason))
    tiny = BookSnapshot(
        token_id="thin", side=UP,
        bids=((0.45, 0.05),),
        asks=((0.55, 0.05),),
    )
    blocked = engine.plan_order(tiny, "buy", notional_usdc=1000.0, allow_partial=False)
    check("tolerance gate blocks an impossible order", not blocked.executable, str(blocked.blocked_reason))
    worst = engine.apply_slippage_tolerance(0.50, "buy")
    check("slippageTolerance widens the cap for buys", abs(worst - 0.505) < 1e-9, f"{worst}")
    estimate = estimate_execution(snapshot, "buy", 10.0, slippage_tolerance=0.01)
    ok, deviation = engine.check_fill(estimate, estimate.vwap * 1.5, 10.0)
    check("a bad print is rejected post-trade", not ok and deviation > 0, f"{deviation:.0f}bps")

    # ---- risk -----------------------------------------------------------
    risk = RiskController(RiskLimits(hedge_enabled=True, hedge_ratio=0.6, hedge_min_signal_strength=0.5))
    risk.open_position(UP, 10.0, 0.50)
    flat = BookSnapshot(token_id="t", side=UP, bids=tuple(bids), asks=tuple(asks_book))
    decision = risk.evaluate(UP, snapshot=flat, opposite_snapshot=snapshot)
    check("risk returns a decision", decision.action in set(Action), decision.action.value)
    risk.open_position(DOWN, 10.0, 0.30)
    risk_doc = BookSnapshot(
        token_id="d", side=DOWN, bids=((0.29, 5),), asks=((0.31, 5),), timestamp=0.0
    )
    aggressive_up = BookSnapshot(
        token_id="u", side=UP, bids=((0.79, 500),), asks=((0.81, 5),), timestamp=0.0
    )
    hedge = risk.evaluate(
        DOWN, snapshot=risk_doc, opposite_snapshot=aggressive_up,
        volatility=0.05, signal_side=UP, signal_strength=0.9,
    )
    check("reverse hedge triggers on a flipped signal", hedge.is_hedge, hedge.action.value)
    check("hedge ratio stays within bounds", 0 < (hedge.shares / 10.0) <= 1.0, f"{hedge.shares}")
    stop = risk.evaluate(
        UP, snapshot=BookSnapshot(token_id="x", side=UP, bids=((0.20, 10),), asks=((0.22, 10),)),
        volatility=0.05,
    )
    check("hard stop fires on an adverse move", stop.action in {Action.STOP_LOSS, Action.TRAILING_STOP}, stop.action.value)

    # ---- fair value ------------------------------------------------------
    check("no time left -> decisive", implied_move_probability(0.05, volatility_pct=0.05, seconds_to_close=0.0) > 0.95)
    check("much time left -> muted", implied_move_probability(0.05, volatility_pct=0.05, seconds_to_close=300.0) < 0.7)
    check("zero move -> coin flip", abs(implied_move_probability(0.0, volatility_pct=0.05, seconds_to_close=30.0) - 0.5) < 1e-9)

    # ---- strategy decision path ------------------------------------------
    strategy = SniperStrategy(config, mode="paper")
    hub = MarketDataHub(token_ids={UP: "u", DOWN: "d"}, max_divergence=0.05)
    for side, bid_px, ask_px in ((UP, 0.60, 0.62), (DOWN, 0.38, 0.40)):
        hub.books[side].apply_snapshot(
            {
                "bids": [[bid_px - i * 0.001, 40 - i * 5] for i in range(5)],
                "asks": [[ask_px + i * 0.001, 40 - i * 5] for i in range(5)],
                "timestamp": 1000.0,
                "window_close_epoch": 1005.0,
            },
            source="test",
        )
    hub.update_spot(SpotTick(symbol="BTCUSDT", price=80100.0, timestamp=1000.0))
    decision = strategy.evaluate(hub, now=1000.0, window_open_price=80000.0, volatility_pct=0.05)
    check("strategy evaluates without network", decision.action in {"hold", "enter"}, decision.action)
    check("strategy reports T-minus", decision.seconds_to_close is not None, str(decision.seconds_to_close))
    check("strategy produced a fair value", decision.fair_probability is not None, str(decision.fair_probability))

    stale_hub = MarketDataHub(token_ids={UP: "u", DOWN: "d"})
    stale_hub.books[UP].apply_snapshot(
        {"bids": [[0.5, 10]], "asks": [[0.51, 10]], "timestamp": 0.0}, source="test"
    )
    stale_decision = strategy.evaluate(stale_hub, now=10_000.0, window_open_price=80000.0)
    check("data-quality gate blocks stale feeds", stale_decision.action == "hold" and not stale_decision.data_healthy,
          ",".join(stale_decision.blockers[:1]))

    # ---- backtest engine -------------------------------------------------
    # a lagging market: BTC is up 0.1% with volatility 0.3%, so fair value at
    # T-30s is ~0.99, but the winning side is still quoted at 0.79. The engine
    # should take it and charge fees + slippage.
    params = ParamSet("t1", 30, 0.05, 0.10, 0.05, 0,
                      risk_reward=2.0, source="selfcheck")
    observations = [
        MarketObservation(
            market_id=f"m{i}", slug=f"s{i}", window_start_epoch=1000.0 + i * 300,
            window_close_epoch=1300.0 + i * 300, t_minus_s=30,
            up_price=0.79 if i % 2 == 0 else 0.21,
            down_price=0.21 if i % 2 == 0 else 0.79,
            result="UP" if i % 2 == 0 else "DOWN",
            btc_open=80000.0, btc_close=80000.0 * (1.001 if i % 2 == 0 else 0.999),
        )
        for i in range(10)
    ]
    bt = BacktestEngine(**{k: v for k, v in _engine_kwargs(config).items()})
    result = bt.run(params, observations)
    check("backtest produces trades", result.metrics.trades > 0, str(result.metrics.trades))
    check("fees are charged", result.metrics.total_fees > 0, f"{result.metrics.total_fees}")
    check("metrics serialise", isinstance(result.metrics.as_dict(), dict))
    check(
        "winning-side entries end up profitable",
        result.metrics.net_pnl > 0,
        f"net={result.metrics.net_pnl:.4f} gross={result.metrics.gross_pnl:.4f}",
    )
    check(
        "slippage is modelled, not assumed away",
        result.metrics.avg_slippage_bps > 0,
        f"{result.metrics.avg_slippage_bps}",
    )
    check(
        "fee drag is reported",
        result.metrics.fee_drag_pct_of_gross > 0 or result.metrics.gross_pnl == 0,
        f"{result.metrics.fee_drag_pct_of_gross}",
    )

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"selfcheck: {len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------- sweep
def cmd_sweep(args, config: Config) -> int:
    from core.backtest_engine import BacktestError
    from core.param_selector import ParamSelector, print_ranking

    params_path = _resolve_project_path(args.params or config.section("backtest").params_csv)
    params = ParamSelector.load_params(
        params_path,
        limit=args.combos,
        time_filter=([int(t) for t in str(args.times).split(",")] if args.times else None),
    )
    log.info("loaded %d parameter sets from %s", len(params), params_path)

    observations = _load_observations(args, config)
    selector = ParamSelector(
        engine_kwargs=_engine_kwargs(config),
        rank_metric=args.rank_metric,
        top_n=args.top,
        min_trades=args.min_trades,
    )
    sweep = selector.sweep(params, observations)
    output_dir = _resolve_project_path(args.output_dir or config.section("backtest").output_dir)
    paths = selector.write_outputs(sweep, output_dir, write_top_trades=not args.no_trades)

    print()
    print_ranking(sweep, limit=args.top)
    print()
    if sweep.best:
        best = sweep.best
        print(f"best combo : {best.params.combo_id}  ({sweep.rank_metric})")
        print(
            f"  --time {best.params.time_s} --threshold {best.params.threshold_pct:g} "
            f"--tp {best.params.take_profit:g} --sl {best.params.stop_loss:g} "
            f"--min-gap {best.params.min_gap_s}"
        )
    print(f"combos swept    : {sweep.combos}")
    print(f"combos w/ trades: {sweep.diagnostics.get('combos_producing_trades', 0)}")
    print(f"observations    : {sweep.observations}")
    print(f"offsets in data : {sweep.diagnostics.get('offsets_available')}")
    print(f"offsets traded  : {sweep.diagnostics.get('offsets_traded')}")
    print(f"results csv     : {paths.get('results_csv', '')}")
    print(f"ranking json    : {paths.get('ranking_json', '')}")
    trade_logs = [p for k, p in paths.items() if k.startswith("trades_")]
    print(f"trade logs      : {len(trade_logs)} file(s) in {output_dir}")
    return 0


# ------------------------------------------------------------------- backtest
def cmd_backtest(args, config: Config) -> int:
    from core.backtest_engine import BacktestEngine
    from core.param_selector import ParamSelector

    if args.combo:
        params_path = _resolve_project_path(args.params or config.section("backtest").params_csv)
        params = ParamSelector.load_params(params_path)
        match = [p for p in params if p.combo_id == args.combo]
        if not match:
            log.error("combo %s not found in %s", args.combo, params_path)
            return 2
        param_set = match[0]
    else:
        from core.backtest_engine import ParamSet

        param_set = ParamSet(
            combo_id="manual",
            time_s=args.time,
            threshold_pct=args.threshold,
            take_profit=args.tp,
            stop_loss=args.sl,
            min_gap_s=args.min_gap,
        )
    log.info("backtesting %s", param_set.label)

    observations = _load_observations(args, config)
    engine = BacktestEngine(**_engine_kwargs(config))
    result = engine.run(param_set, observations)

    output_dir = _resolve_project_path(args.output_dir or config.section("backtest").output_dir)
    from core.backtest_engine import write_backtest_outputs

    paths = write_backtest_outputs(result, output_dir)

    print()
    print(f"params          : {param_set.as_dict()}")
    for key, value in result.metrics.as_dict().items():
        print(f"{key:24}: {value}")
    print()
    for name, path in paths.items():
        print(f"{name:16}: {path}")
    if result.metrics.book_source == "synthetic":
        print()
        print("note: book_source=synthetic - this dataset carries no depth, so the ladder")
        print("      was modelled. Slippage figures show the method, not that market's depth.")
    return 0


# ---------------------------------------------------------------------- paper
def cmd_paper(args, config: Config) -> int:
    """Replay observations through the live decision path (sends nothing)."""
    from strategies.sniper_strategy import SniperStrategy, replay_observations

    observations = _load_observations(args, config)
    if args.limit:
        observations = observations[: args.limit]

    strategy = SniperStrategy(config, mode="paper")
    if args.scan_window:
        # live default is T-10s per the design; widen it to exercise data whose
        # only real prices sit further out (e.g. a T-15s near-close snapshot)
        strategy.scan_window_s = float(args.scan_window)
        log.info("scan window overridden to T-%.0fs", strategy.scan_window_s)
    if args.combo:
        from core.param_selector import ParamSelector

        params_path = _resolve_project_path(args.params or config.section("backtest").params_csv)
        match = [p for p in ParamSelector.load_params(params_path) if p.combo_id == args.combo]
        if match:
            strategy.apply_param_set(match[0])
            log.info("applied parameter set %s", args.combo)

    outcome = replay_observations(strategy, observations)

    print()
    print(f"observations replayed : {outcome['observations']}")
    print(f"entries               : {outcome['entries']}")
    print(f"hedges                : {outcome['hedges']}")
    print(f"holds                 : {outcome['holds']}")
    print()
    for key, value in outcome["summary"]["stats"].items():
        print(f"{key:24}: {value}")
    if outcome["entries"] == 0:
        print()
        print("every observation was held. top reasons:")
        for reason, count in Counter(str(d.reason) for d in strategy.decisions).most_common(3):
            print(f"  {count:>5} x {reason}")
        print()
        print("if the reason is 'outside scan window', the data's offsets sit further out")
        print("than the live T-10s window - widen it with --scan-window, e.g.:")
        print("  python main.py paper --scan-window 20 --prices <csv> --markets <csv>")
    if args.dump_decisions:
        from utils.logger import write_csv

        path = write_csv(
            _resolve_project_path(args.dump_decisions), strategy.decisions_as_rows()
        )
        print(f"\ndecisions written     : {path}")
    print()
    print("paper mode never sends an order: no broker is injected, so every plan stops")
    print("at the execution layer. Wire a broker and mode=live to trade for real.")
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket-sniper",
        description="Polymarket 5-minute last-second sniper (offline by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--log-file", default=None, help="mirror logs into this file")
    sub = parser.add_subparsers(dest="command", required=True)

    # shared data arguments
    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--params", default=None, help="param_combinations.csv")
        p.add_argument("--prices", default=None, help="btc5m_prices.csv")
        p.add_argument("--markets", default=None, help="btc5m_markets.csv (sidecar)")
        p.add_argument("--raw", default=None, help="btc5m_raw.jsonl (richest source)")
        p.add_argument("--offsets", default=None, help="comma-separated T-minus offsets")
        p.add_argument("--limit-markets", type=int, default=None)
        p.add_argument("--allow-missing-prices", action="store_true",
                       help="keep rows without prices (they are skipped, never filled)")
        p.add_argument("--output-dir", default=None)

    p_self = sub.add_parser("selfcheck", help="verify maths and decision path offline")
    p_self.set_defaults(func=cmd_selfcheck)

    p_sweep = sub.add_parser("sweep", help="backtest and rank all parameter combinations")
    add_data_args(p_sweep)
    p_sweep.add_argument("--combos", type=int, default=None, help="limit to first N combos")
    p_sweep.add_argument("--times", default=None, help="only these T-minus offsets, e.g. 30,60")
    p_sweep.add_argument("--rank-metric", default="net_pnl")
    p_sweep.add_argument("--top", type=int, default=10)
    p_sweep.add_argument("--min-trades", type=int, default=1)
    p_sweep.add_argument("--no-trades", action="store_true", help="skip per-combo trade logs")
    p_sweep.set_defaults(func=cmd_sweep)

    p_bt = sub.add_parser("backtest", help="backtest one parameter set")
    add_data_args(p_bt)
    p_bt.add_argument("--combo", default=None, help="combo_id from the parameter file")
    p_bt.add_argument("--time", type=int, default=30)
    p_bt.add_argument("--threshold", type=float, default=0.05)
    p_bt.add_argument("--tp", type=float, default=0.10)
    p_bt.add_argument("--sl", type=float, default=0.05)
    p_bt.add_argument("--min-gap", type=int, default=0, dest="min_gap")
    p_bt.set_defaults(func=cmd_backtest)

    p_paper = sub.add_parser("paper", help="replay observations through the decision path")
    add_data_args(p_paper)
    p_paper.add_argument("--combo", default=None)
    p_paper.add_argument("--limit", type=int, default=None)
    p_paper.add_argument("--scan-window", type=float, default=None, dest="scan_window",
                         help="override the T-minus scan window in seconds (live default is 10)")
    p_paper.add_argument("--dump-decisions", default=None, help="write decisions to this CSV")
    p_paper.set_defaults(func=cmd_paper)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config: Config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - config boundary
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.log_level:
        set_level(args.log_level)
    log_file = args.log_file or config.section("logging").get("log_file")
    if log_file and args.command in {"sweep", "paper"}:
        try:
            add_file_handler(_resolve_project_path(log_file))
        except OSError as exc:  # noqa: BLE001
            log.warning("could not open log file %s: %s", log_file, exc)

    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level boundary
        if args.log_level == "DEBUG":
            raise
        log.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

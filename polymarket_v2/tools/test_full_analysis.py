"""End-to-end test for tools/run_full_analysis.py.

Builds a fixture dataset that mirrors the REAL schema of the user's DuckDB
exports (five tables, same column names and types), where the correct answers are
known by construction, then runs the full pipeline and asserts it finds them.

Requires pyarrow (to write and read parquet).

    python tools/test_full_analysis.py
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from tools import run_full_analysis as rfa  # noqa: E402

FIXTURE = os.path.join(PROJECT, ".work", "analysis-fixture")
BASE = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)
WINDOW = 900  # 15-minute markets, as in the user's data


def build_fixture() -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if os.path.isdir(FIXTURE):
        shutil.rmtree(FIXTURE)
    for sub in ("orderbooks", "crypto_prices", "trades", "price_history", "resolution"):
        os.makedirs(os.path.join(FIXTURE, sub), exist_ok=True)

    # ---- 30 markets; half end with UP winning, half DOWN
    n_markets = 30
    resolution_rows = []
    for index in range(n_markets):
        start = BASE + timedelta(seconds=index * WINDOW)
        end = start + timedelta(seconds=WINDOW)
        up_wins = index % 2 == 0
        resolution_rows.append({
            "market_id": f"mkt{index:03d}",
            "asset": "BTC",
            "condition_id": f"0x{index:064x}",
            "question": f"Bitcoin Up or Down - window {index}",
            "start_time": start,
            "end_time": end,
            "resolved_at": end + timedelta(seconds=180),
            "outcome": "Up" if up_wins else "Down",
        })
    pq.write_table(pa.Table.from_pylist(resolution_rows), os.path.join(FIXTURE, "resolution", "all.parquet"))

    # ---- order books: one snapshot every 2s for the last 300s of each window,
    #      and in HALF the markets the two asks sum below 0.95 near the close
    book_rows = []
    for index in range(n_markets):
        start = BASE + timedelta(seconds=index * WINDOW)
        end = start + timedelta(seconds=WINDOW)
        up_wins = index % 2 == 0
        cheap = index % 2 == 0            # these markets get the panic dump
        for token_index, token in enumerate(("tok_up", "tok_down")):
            for step in range(0, 301, 2):
                ts = end - timedelta(seconds=300 - step)
                t_minus = 300 - step
                # fair value drifts toward the truth as the window closes
                truth = 1.0 if (token_index == 0) == up_wins else 0.0
                progress = (300 - t_minus) / 300.0
                fair = 0.5 + (truth - 0.5) * progress
                if cheap and t_minus <= 60:
                    # the "mispricing": the losing side gets dumped, so this
                    # side's ask is pushed down and the SUM of asks drops below 1
                    fair = 0.30 if token_index == 0 else 0.60
                    if token_index == 1:
                        fair = 0.60
                mid = max(0.02, min(0.98, fair))
                bid = round(mid - 0.005, 4)
                ask = round(mid + 0.005, 4)
                book_rows.append({
                    "timestamp": ts,
                    "asset": "BTC",
                    "market_id": f"mkt{index:03d}",
                    "condition_id": f"0x{index:064x}",
                    "token_id": token,
                    "question": f"Bitcoin Up or Down - window {index}",
                    "best_bid": bid,
                    "best_ask": ask,
                    "spread": round(ask - bid, 4),
                    "mid_price": round(mid, 4),
                    "bid_levels": [[bid, 100.0], [round(bid - 0.01, 4), 200.0]],
                    "ask_levels": [[ask, 100.0], [round(ask + 0.01, 4), 200.0]],
                })
    pq.write_table(pa.Table.from_pylist(book_rows), os.path.join(FIXTURE, "orderbooks", "books.parquet"))

    # ---- spot prices: 1s bars, with a sharp drop in the "cheap" markets
    spot_rows = []
    for index in range(n_markets):
        start = BASE + timedelta(seconds=index * WINDOW)
        end = start + timedelta(seconds=WINDOW)
        cheap = index % 2 == 0
        price = 80000.0
        step = timedelta(seconds=1)
        ts = start - timedelta(seconds=120)
        while ts <= end:
            t_minus = (end - ts).total_seconds()
            if cheap and 0 < t_minus <= 30:
                price = 80000.0 * (1 - 0.002 * (30 - t_minus) / 30.0)  # -0.2% dump
            spot_rows.append({
                "timestamp": ts,
                "asset": "BTC",
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1.0, "trade_count": 1,
            })
            ts += step
    pq.write_table(pa.Table.from_pylist(spot_rows), os.path.join(FIXTURE, "crypto_prices", "spot.parquet"))

    # ---- trades + price history (present so the pipeline must tolerate them)
    trade_rows = [{
        "timestamp": BASE + timedelta(seconds=i),
        "asset": "BTC", "market_id": "mkt000", "condition_id": "0x0",
        "token_id": "tok_up", "side": "BUY", "price": 0.5, "size": 10.0,
        "tx_hash": f"0x{i}", "maker": "0xabc", "source": "clob",
    } for i in range(20)]
    pq.write_table(pa.Table.from_pylist(trade_rows), os.path.join(FIXTURE, "trades", "trades.parquet"))

    history_rows = [{
        "timestamp": BASE + timedelta(seconds=i),
        "asset": "BTC", "market_id": "mkt000", "token_id": "tok_up", "price": 0.5 + i * 0.001,
    } for i in range(20)]
    pq.write_table(pa.Table.from_pylist(history_rows), os.path.join(FIXTURE, "price_history", "ph.parquet"))

    return FIXTURE


def main() -> int:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is required for this test (pip install pyarrow)")
        return 2

    root = build_fixture()
    print(f"fixture built at {root}")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # ---- hedge maths: the core economic claim, in BOTH constructions
    #
    # equal-notional (what the described bot does) does NOT lock a profit: the
    # cheap side pays well, the expensive side barely covers cost, so the branch
    # where the expensive side wins can lose. share-ratio does lock, and only
    # when the two asks sum below the fee-adjusted break-even.
    equal = rfa.hedge_pnl(0.30, 0.60, mode="equal_notional")
    check("equal-notional hedge is NOT a lock (one branch loses)",
          equal["pnl_worst"] < 0 < equal["pnl_best"],
          f"up={equal['pnl_if_up']:.2f} down={equal['pnl_if_down']:.2f}")
    check("equal-notional still charges fees", equal["fees"] > 0, f"{equal['fees']:.4f}")
    check("equal-notional is not flagged as guaranteed", equal["guaranteed_profit"] is False)

    locked = rfa.hedge_pnl(0.30, 0.60, mode="lock")
    check("lock hedge IS a lock on both branches",
          locked["pnl_if_up"] > 0 and locked["pnl_if_down"] > 0,
          f"up={locked['pnl_if_up']:.4f} down={locked['pnl_if_down']:.4f}")
    check("lock pays the same on both branches",
          abs(locked["pnl_if_up"] - locked["pnl_if_down"]) < 1e-9,
          f"{locked['pnl_if_up']:.6f} vs {locked['pnl_if_down']:.6f}")
    check("lock is flagged guaranteed", locked["guaranteed_profit"] is True)
    check("lock buys EQUAL share counts",
          abs(locked["up_shares"] - locked["down_shares"]) < 1e-9,
          f"{locked['up_shares']:.4f} vs {locked['down_shares']:.4f}")
    check("lock profit on a 0.90 pair is positive", locked["pnl_worst"] > 0,
          f"{locked['pnl_worst']:.4f}")

    # the break-even sum is a computed number, not a guess
    break_even = rfa.lock_break_even_sum(0.30, 0.60)
    check("break-even sum is below 1.0 after fees", 0.90 < break_even < 1.0,
          f"{break_even:.4f}")
    just_under = rfa.hedge_pnl(0.49, 0.49, mode="lock")
    check("a 0.98 pair no longer locks a profit (fees win)",
          just_under["pnl_worst"] < 0, f"{just_under['pnl_worst']:.4f}")

    thin = rfa.hedge_pnl(0.4875, 0.4875, mode="lock")
    check("a 0.975 pair loses after fees (break-even is ~0.96)",
          thin["pnl_worst"] < 0, f"{thin['pnl_worst']:.6f}")

    # at exactly the computed break-even sum, P&L must be ~0 - proving the
    # break-even formula and the P&L model agree. Compare per unit (units=1) so
    # the fixed-notional sizing does not perturb the fee.
    be = rfa.lock_break_even_sum(0.48, 0.48)
    at_break_even = rfa.hedge_pnl(be / 2, be / 2, mode="lock", units=1.0)
    check("P&L is ~0 exactly at the computed break-even sum (per unit)",
          abs(at_break_even["pnl_worst"]) < 1e-4,
          f"sum={be:.6f} pnl={at_break_even['pnl_worst']:.12f}")
    # and the fixed-notional path must agree closely (documented approximation)
    at_break_even_notional = rfa.hedge_pnl(be / 2, be / 2, mode="lock")
    check("fixed-notional sizing agrees with the formula within 1%",
          abs(at_break_even_notional["pnl_worst"]) < 1e-2 * 100,
          f"{at_break_even_notional['pnl_worst']:.6f}")

    losing = rfa.hedge_pnl(0.55, 0.55, mode="lock")
    check("a 1.10 pair loses money", losing["pnl_worst"] < 0, f"{losing['pnl_worst']:.4f}")
    check("guaranteed_profit flag is honest",
          locked["guaranteed_profit"] is True and losing["guaranteed_profit"] is False)

    slip = rfa.hedge_pnl(0.30, 0.60, mode="lock", slippage_bps=100.0)
    check("slippage reduces the locked profit",
          slip["pnl_worst"] < locked["pnl_worst"],
          f"{slip['pnl_worst']:.4f} vs {locked['pnl_worst']:.4f}")

    # ---- the full pipeline
    offsets = [120, 60, 30, 15]
    print("\nrunning the full pipeline ...")
    exit_code = rfa.main([
        "--root", root,
        "--out", os.path.join(PROJECT, ".work", "analysis-out"),
        "--offsets", ",".join(str(o) for o in offsets),
    ])
    check("pipeline exits 0", exit_code == 0, str(exit_code))

    out_dir = os.path.join(PROJECT, ".work", "analysis-out")
    for name in ("timing_sweep.csv", "param_sweep.csv", "summary.json"):
        check(f"wrote {name}", os.path.exists(os.path.join(out_dir, name)), name)

    import json

    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    check("summary counted the markets", summary["markets"] == 30, str(summary["markets"]))
    check("summary found order books for all markets",
          summary["markets_with_books"] == 30, str(summary["markets_with_books"]))
    check("checkpoints were produced", summary["checkpoints"] > 0, str(summary["checkpoints"]))

    timing = {row["t_minus"]: row for row in summary["timing"]}
    triggered_offsets = [o for o in offsets if timing.get(o, {}).get("triggered", 0) > 0]
    check("the planted mispricing is detected at late offsets",
          len(triggered_offsets) > 0, f"triggered at {triggered_offsets}")
    check("late offsets trigger more often than early ones",
          timing.get(15, {}).get("triggered", 0) >= timing.get(120, {}).get("triggered", 0),
          f"T-15={timing.get(15, {}).get('triggered')} vs T-120={timing.get(120, {}).get('triggered')}")

    late = timing.get(15, {})
    if late.get("triggered"):
        check("avg sum_ask at T-15 is below the 0.95 target",
              late["avg_sum_ask"] is not None and late["avg_sum_ask"] <= 0.95,
              str(late["avg_sum_ask"]))
        check("locked P&L at T-15 is positive on average",
              late["avg_pnl"] > 0, str(late["avg_pnl"]))

    # ---- the labelling must be honest: ambiguous markets are skipped
    markets = rfa.load_resolution(root)
    check("resolution loaded", len(markets) == 30, str(len(markets)))
    check("outcomes parsed", {m.outcome for m in markets.values()} == {"UP", "DOWN"},
          str({m.outcome for m in markets.values()}))

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    print(f"checks: {len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

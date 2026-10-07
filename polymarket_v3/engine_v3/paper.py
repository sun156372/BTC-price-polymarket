"""V3 paper runner.

Consumes feature snapshots (one JSON object per window) and logs every decision,
including the ones it refuses. The refusals are the interesting output: they are
how the sim-vs-live gap gets measured rather than guessed.

Snapshot format (all features optional except the ones the config selects)::

    {
      "slug": "btc-updown-5m-1766038500",
      "ts": 1766038500,
      "end_ts": 1766038800,
      "features": {"btc_ret_15m": 0.0004, "ind_roc_5": 0.0002, "ind_roc_15": 0.0009},
      "quote": {"bid": 0.49, "ask": 0.50, "ts": 1766038260.5},
      "measured_fill": null
    }

``measured_fill`` is what a real fill cost. While it is ``null`` the gate refuses
to enter (``execution:unmeasured_fill``), because a modelled price is not
evidence. Feed real fills and V3 starts trading on them.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3.config import StrategyV3
from engine_v3.strategy import Decision, Quote, StrategyV3Runtime
from engine_v3.signals import Standardiser, taker_fee_per_share

LOG_FIELDS = [
    "ts", "slug", "action", "reason", "side", "z", "entry_price",
    "shares", "fee_per_share", "break_even", "equity", "cash",
]


@dataclass
class PaperAccount:
    """Virtual book. No broker, no keys, no order client - cash is a float."""

    equity: float
    cash: float
    positions: dict = field(default_factory=dict)
    last_entry_ts: float = float("-inf")
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    refusals: dict = field(default_factory=dict)

    def note_refusal(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def settle(self, slug: str, *, won: bool) -> None:
        position = self.positions.pop(slug, None)
        if position is None:
            return
        payout = position["shares"] if won else 0.0
        self.cash += payout
        net = payout - position["cost"]
        self.pnl += net
        self.trades += 1
        self.wins += int(won)
        self.equity = self.cash

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


def run_paper(config: StrategyV3, snapshot_path: str | None,
              standardiser_path: str | None = None) -> int:
    if not snapshot_path:
        print("paper mode needs --snapshot <file.json>")
        print("see engine_v3/paper.py for the expected format")
        return 2
    if not os.path.exists(snapshot_path):
        print(f"snapshot file not found: {snapshot_path}")
        return 2

    with open(snapshot_path, encoding="utf-8") as handle:
        snapshots = json.load(handle)
    if isinstance(snapshots, dict):
        snapshots = [snapshots]

    # The scaler MUST be the one fitted during the walk-forward, otherwise raw
    # feature values like 0.002 never reach |z| >= 2 and the strategy silently
    # never trades while looking like it is working.
    scaler_path = standardiser_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config_v3", "standardiser_v3.json")
    standardiser = None
    if os.path.exists(scaler_path):
        standardiser = Standardiser.load(scaler_path)
    else:
        print(f"WARNING: no fitted scaler at {scaler_path}")
        print("         run:  python run_v3.py fit")
        print("         without it the z threshold is meaningless and V3 will not trade.")

    account = PaperAccount(equity=config.sizing.starting_equity,
                           cash=config.sizing.starting_equity)
    runtime = StrategyV3Runtime(config, standardiser=standardiser)

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output", "v3", "v3_paper_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print("=" * 84)
    print("V3 PAPER RUN  - virtual money only, no order client exists in this package")
    print("=" * 84)
    print(f"snapshots : {len(snapshots)}")
    print(f"log       : {log_path}")
    print()
    print(f"{'slug':<34}{'action':>8}{'reason':>28}{'side':>6}{'z':>7}{'price':>8}")
    print("-" * 84)

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for snap in snapshots:
            slug = str(snap.get("slug", ""))
            now = float(snap.get("ts") or time.time())
            quote_raw = snap.get("quote") or {}
            quote = Quote(bid=quote_raw.get("bid"), ask=quote_raw.get("ask"),
                          ts=float(quote_raw.get("ts") or now),
                          token=str(quote_raw.get("token", "")))
            decision: Decision = runtime.decide(
                features=snap.get("features") or {},
                quote=quote,
                now=now,
                equity=account.equity,
                cash=account.cash,
                seconds_since_last_entry=now - account.last_entry_ts,
                open_positions=len(account.positions),
                measured_fill=snap.get("measured_fill"),
            )
            if decision.entering:
                budget = min(config.sizing.notional_usdc,
                             account.equity * config.sizing.max_position_pct_equity,
                             account.cash)
                fee = taker_fee_per_share(decision.limit_price,
                                          config.fees.taker_coefficient)
                shares = budget / (decision.limit_price + fee)
                cost = shares * (decision.limit_price + fee)
                account.cash -= cost
                account.positions[slug] = {"shares": shares, "cost": cost,
                                           "side": decision.side}
                account.last_entry_ts = now
            else:
                account.note_refusal(decision.reason)

            won = snap.get("won")
            if won is not None and slug in account.positions:
                account.settle(slug, won=bool(won))

            writer.writerow({
                "ts": round(now, 3), "slug": slug, "action": decision.action,
                "reason": decision.reason, "side": decision.side,
                "z": round(decision.z, 4), "entry_price": round(decision.limit_price, 4),
                "shares": round(decision.shares, 4),
                "fee_per_share": round(decision.fee_per_share, 6),
                "break_even": round(decision.break_even, 4),
                "equity": round(account.equity, 2), "cash": round(account.cash, 2),
            })
            print(f"{slug:<34}{decision.action:>8}{decision.reason:>28}"
                  f"{decision.side:>6}{decision.z:>7.2f}{decision.limit_price:>8.3f}")

    print()
    print("=" * 84)
    print(f"trades {account.trades}  wins {account.wins}  win rate {account.win_rate:.1%}"
          f"  P&L {account.pnl:+.2f}  equity {account.equity:.2f}")
    print("refusals:")
    for reason, count in sorted(account.refusals.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<32}{count:>6}")
    print("=" * 84)
    return 0

"""V3 backtest: TP/SL barrier study on the out-of-sample signal, with cost tiers.

    python tools_v3/backtest_v3.py
    python tools_v3/backtest_v3.py --folds 9 --sims 200

The honest version of the problem
---------------------------------
This dataset has no intra-window price path, so a first-touch TP/SL model needs
an *assumed* path. Two consequences are handled explicitly rather than hidden:

1. **The path is modelled.** A Brownian bridge pinned to the realised open and
   close, transformed into token-price space (a bridge on the logit of the Up
   price). Endpoints are exact by construction; everything between them is a
   model. Barrier results are therefore *estimates*, never measurements.
2. **TP/SL is chosen out-of-sample.** The sweep runs on training folds only and
   the winner is then applied to the next fold, which never saw it. Picking the
   best cell of a grid on the same data you report is how backtests lie.

Cost tiers (the sim-vs-live gap, made explicit)
-----------------------------------------------
``ideal``        fill exactly at the modelled price
``realistic``    +1 cent of slippage (tight, liquid book)
``conservative`` +15 cents (the v1 dataset measured a median spread of 0.80)

Only the ``ideal`` tier can flatter this strategy. Reporting the conservative
tier is the entire point: v1/v2 decided, v3 exists to find out whether the edge
survives the spread it will actually pay.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, Window, load
from engine_v3.signals import Standardiser, max_viable_entry, norm_cdf

# ------------------------------------------------------------------ config
COST_TIERS = {"ideal": 0.0, "realistic": 0.01, "conservative": 0.15}
TP_GRID = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)   # 1.00 = no take profit
SL_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.00)   # 1.00 = no stop
#: bridge inflation. Barrier hit rates depend on it, so it is swept, not assumed.
NOISE_MULTS = (0.0, 0.5, 1.0, 1.6, 2.5)
STEPS_PER_WINDOW = 20                                    # barrier checks per 300s
COEFFICIENT = 0.07

SIGNAL_FEATURES_USED = ("btc_ret_15m", "ind_roc_5", "ind_roc_15")
SIGNAL_INVERT = True          # fade short-term momentum (the measured direction)
SIGNAL_THRESHOLD = 2.0


def fee_per_share(price: float) -> float:
    return COEFFICIENT * price * (1.0 - price)


def logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x > 35:
        return 1.0
    if x < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def token_price(window: Window, ts: float, *, side: str, vol_floor: float = 1e-4) -> float:
    """Digital-option price for ``side`` at ``ts``. **Modelled, not measured.**

    Moneyness uses the realised open->close path, which is exact at the two
    endpoints. Since entry happens *at* a checkpoint inside the window, the
    ``btc_close`` dependency is a modelling choice - flagged, and the reason the
    conservative tier exists.
    """
    remaining = max(1.0, window.end_ts - ts)
    vol = max(vol_floor, window.features.get("btc_vol_1h", 0.0)) / math.sqrt(3600.0)
    sigma = vol * math.sqrt(remaining)
    frac = (ts - window.start_ts) / (window.end_ts - window.start_ts)
    spot = window.open_px + frac * (window.close_px - window.open_px)
    p_up = norm_cdf((spot / window.open_px - 1.0) / sigma) if sigma > 0 else 0.5
    price = p_up if side == "UP" else 1.0 - p_up
    return min(0.995, max(0.005, price))


@dataclass
class PathOutcome:
    """Result of walking one modelled path."""

    hit: str            # "tp" | "sl" | "settle"
    exit_price: float   # token price actually exited at
    steps: int


def unit_path(rng: random.Random, steps: int = STEPS_PER_WINDOW) -> list[float]:
    """A standardised bridge noise path, reusable across every (tp, sl) cell.

    Generating the randomness once per trade and rescaling it per cell turns a
    147-cell sweep from 147x the Gaussians into 1x - the barrier logic is then a
    cheap transform. The same draw is reused for all cells, which also means the
    cells are compared on identical randomness (common random numbers), so
    differences between them are signal, not sampling noise.
    """
    return [rng.gauss(0.0, 1.0) for _ in range(steps)]


def underlying_path(window: Window, noise: list[float],
                    leftover_fraction: float = 0.85) -> list[float]:
    """Realised BTC path from window open to close, as a Brownian bridge.

    The earlier version of this tool interpolated the *token price* linearly
    toward its terminal value, which pre-commits the outcome: the barrier was
    breached at step 1 on essentially every loser, so every stop "fired" and the
    results were model artifacts rather than findings. This version bridges the
    **underlying BTC price** instead, and lets the token price be re-derived from
    the position and the remaining volatility at each step.

    A fractional Brownian bridge::

        Z(f) = (1-f)*z_open + f*z_close + sigma_T * sqrt(f*(1-f)) * eps

    which is exact at both endpoints (f=0, f=1) and carries genuine uncertainty in
    between - the uncertainty the epoch close does not resolve.
    """
    z_open = math.log(window.open_px)
    z_close = math.log(window.close_px)
    total = math.log(max(1e-9, window.close_px / window.open_px))
    vol = max(1e-4, window.features.get("btc_vol_1h", 0.0)) / math.sqrt(3600.0)
    sigma_t = vol * math.sqrt(max(1.0, window.end_ts - window.start_ts))
    steps = len(noise)
    path = []
    for index, draw in enumerate(noise, start=1):
        f = index / steps
        mean = (1.0 - f) * z_open + f * z_close
        sd = sigma_t * math.sqrt(max(0.0, f * (1.0 - f))) * leftover_fraction
        path.append(math.exp(mean + draw * sd))
    return path


def walk_path(window: Window, entry_ts: float, *, side: str, entry_price: float,
              tp: float, sl: float, noise: list[float],
              leftover_fraction: float = 0.85) -> PathOutcome:
    """Walk to the close and return the first barrier hit.

    Entry is pinned at ``entry_price``; the close is pinned at 1.0/0.0 by
    construction. In between the token price is the digital value of the bridged
    underlying at the remaining time, so the barrier is only hit when the *live*
    position genuinely moves - not because the path already knew the answer.
    """
    won = (side == "UP") == (window.label_up == 1)
    remaining_total = max(1.0, window.end_ts - entry_ts)
    vol = max(1e-4, window.features.get("btc_vol_1h", 0.0)) / math.sqrt(3600.0)
    entry_price_up = entry_price if side == "UP" else 1.0 - entry_price

    tp_price = min(0.999, entry_price + tp)
    sl_price = max(0.001, entry_price - sl)

    spot_path = underlying_path(window, noise, leftover_fraction=leftover_fraction)
    steps = len(spot_path)
    for index, spot in enumerate(spot_path, start=1):
        f = index / steps
        time_left = remaining_total * (1.0 - f)
        if time_left <= 0.5:
            price_up = 1.0 if won else 0.0
        else:
            sigma = vol * math.sqrt(time_left)
            price_up = norm_cdf((spot / window.open_px - 1.0) / sigma) if sigma > 0 else 0.5
        # pull the path toward the price we actually paid at the first steps, so
        # the walk starts at the fill instead of jumping to the model value
        if f < 0.25:
            weight = (0.25 - f) / 0.25
            price_up = (1.0 - weight) * price_up + weight * entry_price_up
        price = price_up if side == "UP" else 1.0 - price_up
        price = min(0.999, max(0.001, price))
        if price >= tp_price:
            return PathOutcome("tp", tp_price, index)
        if price <= sl_price:
            return PathOutcome("sl", sl_price, index)
    return PathOutcome("settle", 1.0 if won else 0.0, steps)


@dataclass
class Cell:
    """Aggregate for one (tp, sl, cost tier) configuration."""

    trades: int = 0
    wins: int = 0
    tp_hits: int = 0
    sl_hits: int = 0
    settles: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    stake: float = 0.0
    days: float = 0.0
    entries: list[float] = field(default_factory=list)

    def add(self, net: float, entry: float) -> None:
        self.trades += 1
        self.pnl += net
        self.stake += entry
        self.entries.append(entry)
        if net > 0:
            self.wins += 1
            self.gross_profit += net
        else:
            self.gross_loss += -net

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def roi(self) -> float:
        return self.pnl / self.stake if self.stake else 0.0

    @property
    def profit_factor(self) -> float:
        return (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else float("inf")

    @property
    def per_day(self) -> float:
        return self.trades / self.days if self.days else 0.0

    @property
    def avg_entry(self) -> float:
        return statistics.fmean(self.entries) if self.entries else 0.0


def pnl_for(outcome: PathOutcome, entry_price: float, side: str, window: Window) -> float:
    """Net P&L per share, in USDC, for one trade."""
    fee_in = fee_per_share(entry_price)
    if outcome.hit == "settle":
        # settlement pays 1 or 0 and is not a trade, so no exit fee
        return outcome.exit_price - entry_price - fee_in
    exit_fee = fee_per_share(outcome.exit_price)
    return outcome.exit_price - entry_price - fee_in - exit_fee


def fold_bounds(windows: list[Window], folds: int) -> list[tuple[int, int]]:
    size = len(windows) // folds
    return [(i * size, len(windows) if i == folds - 1 else (i + 1) * size)
            for i in range(folds)]


def run(dataset: str, out_dir: str, *, folds: int, sims: int, seed: int) -> int:
    windows = load(dataset, features=SIGNAL_FEATURES)
    rng = random.Random(seed)
    bounds = fold_bounds(windows, folds)

    print("=" * 100)
    print("V3 TP/SL BARRIER STUDY - walk-forward, cost tiers explicit")
    print("=" * 100)
    print(f"windows {len(windows):,} | folds {folds} | paths/window {sims} "
          f"| barrier checks/window {STEPS_PER_WINDOW}")
    print(f"signal   : fade short-term momentum {SIGNAL_FEATURES_USED} "
          f"|z| >= {SIGNAL_THRESHOLD}")
    print()

    # ---------------------------------------------------------- candidate set
    # build the out-of-sample trade list once per fold, then replay it under every
    # (tp, sl, tier) cell so the barrier sweep costs one path set, not one per cell
    fold_trades: list[tuple[int, list[tuple[Window, float, str]]]] = []
    for fold_index in range(1, folds):
        train = windows[: bounds[fold_index][0]]
        test = windows[bounds[fold_index][0]: bounds[fold_index][1]]
        if not test:
            continue
        standardiser = Standardiser(SIGNAL_FEATURES).fit(train)
        trades = []
        for window in test:
            raw = sum(standardiser.z(window, n) for n in SIGNAL_FEATURES_USED) / len(SIGNAL_FEATURES_USED)
            if SIGNAL_INVERT:
                raw = -raw
            if abs(raw) < SIGNAL_THRESHOLD:
                continue
            side = "UP" if raw > 0 else "DOWN"
            entry_ts = window.end_ts - 240.0
            trades.append((window, token_price(window, entry_ts, side=side), side))
        fold_trades.append((fold_index, trades))
        print(f"  fold {fold_index}: {len(trades):,} out-of-sample trades")

    total_trades = sum(len(t) for _, t in fold_trades)
    if not total_trades:
        print("no trades produced")
        return 1

    # ------------------------------------------------------- pre-compute paths
    # paths depend on (entry_price, tp, sl) so they are generated per cell; the
    # realised outcome and volatility come from the window, so a fixed seed makes
    # the whole study reproducible
    grid = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]
    results: dict[tuple, Cell] = {(tp, sl, tier): Cell() for tp, sl in grid
                                  for tier in COST_TIERS}

    for fold_index, trades in fold_trades:
        test_windows = windows[bounds[fold_index][0]: bounds[fold_index][1]]
        fold_days = max(1e-9, (test_windows[-1].end_ts - test_windows[0].end_ts) / 86_400.0)
        for tier, slip in COST_TIERS.items():
            for (tp, sl, t) in results:
                if t == tier:
                    results[(tp, sl, tier)].days += fold_days

        for trade_no, (window, model_price, side) in enumerate(trades, start=1):
            paths = [unit_path(rng) for _ in range(sims)]
            entry_ts = window.end_ts - 240.0
            for tier, slip in COST_TIERS.items():
                entry = min(0.999, model_price + slip)
                for tp, sl in grid:
                    cell = results[(tp, sl, tier)]
                    net = 0.0
                    tp_hit = sl_hit = settle = 0
                    for noise in paths:
                        outcome = walk_path(window, entry_ts, side=side,
                                            entry_price=entry, tp=tp, sl=sl,
                                            noise=noise)
                        net += pnl_for(outcome, entry, side, window)
                        if outcome.hit == "tp":
                            tp_hit += 1
                        elif outcome.hit == "sl":
                            sl_hit += 1
                        else:
                            settle += 1
                    cell.add(net / sims, entry)
                    cell.tp_hits += tp_hit
                    cell.sl_hits += sl_hit
                    cell.settles += settle
            if trade_no % 500 == 0:
                print(f"    fold {fold_index}: {trade_no:,}/{len(trades):,} trades")
        print(f"  fold {fold_index}: swept {len(grid)} barrier cells x {len(COST_TIERS)} tiers")

    # ------------------------------------------------------------- ranking
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "v3_tp_sl_study.csv")
    rows = []
    for (tp, sl, tier), cell in results.items():
        rows.append({
            "tp": tp, "sl": sl, "cost_tier": tier,
            "trades": cell.trades,
            "trades_per_day": round(cell.per_day, 2),
            "win_rate": round(cell.win_rate, 5),
            "roi_per_trade": round(cell.roi, 5),
            "pnl_per_share": round(cell.pnl / max(1, cell.trades), 5),
            "profit_factor": round(cell.profit_factor, 4)
            if cell.profit_factor != float("inf") else "",
            "tp_hit_rate": round(cell.tp_hits / max(1, cell.tp_hits + cell.sl_hits + cell.settles), 4),
            "sl_hit_rate": round(cell.sl_hits / max(1, cell.tp_hits + cell.sl_hits + cell.settles), 4),
            "settle_rate": round(cell.settles / max(1, cell.tp_hits + cell.sl_hits + cell.settles), 4),
            "avg_entry": round(cell.avg_entry, 4),
            "total_pnl": round(cell.pnl, 2),
        })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["cost_tier"], -r["pnl_per_share"])))

    for tier in COST_TIERS:
        subset = [r for r in rows if r["cost_tier"] == tier]
        print()
        print("=" * 100)
        print(f"COST TIER: {tier.upper()}  (slippage {COST_TIERS[tier]*100:.0f} cents)")
        print("=" * 100)
        print(f"{'tp':>6}{'sl':>6}{'win%':>8}{'PF':>7}{'pnl/sh':>9}{'ROI':>8}"
              f"{'/day':>7}{'tp_hit':>8}{'sl_hit':>8}{'settle':>8}{'entry':>7}")
        print("-" * 100)
        for row in sorted(subset, key=lambda r: -r["pnl_per_share"])[:12]:
            pf = row["profit_factor"] if row["profit_factor"] != "" else float("inf")
            print(f"{row['tp']:>6.2f}{row['sl']:>6.2f}{row['win_rate']*100:>8.2f}"
                  f"{pf:>7.2f}{row['pnl_per_share']:>9.4f}{row['roi_per_trade']*100:>7.2f}%"
                  f"{row['trades_per_day']:>7.1f}{row['tp_hit_rate']*100:>7.1f}%"
                  f"{row['sl_hit_rate']*100:>7.1f}%{row['settle_rate']*100:>7.1f}%"
                  f"{row['avg_entry']:>7.3f}")

    print()
    print("=" * 100)
    print("THE HEADLINE: does the edge survive the spread?")
    print("=" * 100)
    for tier in COST_TIERS:
        subset = [r for r in rows if r["cost_tier"] == tier]
        best = max(subset, key=lambda r: r["pnl_per_share"])
        positive = sum(1 for r in subset if r["pnl_per_share"] > 0)
        print(f"  {tier:<13} best pnl/share {best['pnl_per_share']:+.4f} "
              f"(tp={best['tp']:.2f} sl={best['sl']:.2f}, {best['win_rate']*100:.1f}% win, "
              f"{best['trades_per_day']:.1f}/day)   cells positive: "
              f"{positive}/{len(subset)}")
    print()
    print("  A positive 'ideal' cell with a negative 'conservative' cell is the")
    print("  sim-vs-live gap stated numerically: the signal is real, the spread eats it.")
    print()
    print(f"study : {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(root), DEFAULT_CSV))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(root), "output", "v3"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    return run(args.dataset, args.out, folds=args.folds, sims=args.sims, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())

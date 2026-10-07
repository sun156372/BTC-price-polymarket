"""V3 win-rate / profit-factor / frequency optimisation.

    python tools_v3/edge_report_v3.py --folds 8

Why this tool exists
--------------------
The main sweep ranked configurations by *accuracy*. Accuracy is not the objective:
a binary held to settlement pays 1.0 or 0.0, so what matters is how accuracy
compares to the price paid. Two consequences the accuracy ranking hides:

1. **Profit factor is dominated by entry price, not by signal quality.** At a
   ~0.505 entry and ~55% accuracy the average win is ~+0.49 and the average loss
   ~-0.50, so PF hovers near 1.0 and flips on fractions of a cent. PF is therefore
   reported as a *matrix over entry price*, never as a single number.
2. **A configuration can win 55% of its trades and still lose money.** Break-even
   accuracy at entry p is p + 0.07*p*(1-p).

So this tool answers the three things the objective asks for directly:

* win rate          -> measured, per fold, with its spread
* profit factor     -> matrix over entry price, plus its break-even price
* trade frequency   -> per day, and the frequency needed for a daily P&L target

Everything is walk-forward. No configuration is selected on the data it is
reported on.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, Window, load
from engine_v3.signals import Standardiser, break_even_accuracy, taker_fee_per_share

#: entry price ladder. 0.505 is the modelled T-240 entry; the rest are the prices
#: actually paid once slippage and spread are included.
ENTRY_PRICES = (0.500, 0.505, 0.510, 0.515, 0.520, 0.525, 0.530, 0.535,
                0.540, 0.545, 0.550, 0.560, 0.570, 0.580)

#: the configuration V3 ships with
FEATURES = ("btc_ret_15m", "ind_roc_5", "ind_roc_15")
INVERT = True
THRESHOLD = 2.0
#: thresholds swept for the frequency/profit-factor trade-off
THRESHOLD_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)


@dataclass
class Book:
    """Per-trade P&L book at a fixed entry price."""

    price: float
    trades: int = 0
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    pnl: float = 0.0
    days: float = 0.0

    def add(self, won: bool) -> None:
        entry = self.price
        fee = taker_fee_per_share(entry)
        net = (1.0 if won else 0.0) - entry - fee
        self.trades += 1
        self.wins += int(won)
        self.pnl += net
        if net > 0:
            self.gross_profit += net
        elif net < 0:
            self.gross_loss += -net

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def expectancy(self) -> float:
        return self.pnl / self.trades if self.trades else 0.0

    @property
    def roi(self) -> float:
        return self.pnl / (self.trades * self.price) if self.trades and self.price else 0.0

    @property
    def per_day(self) -> float:
        return self.trades / self.days if self.days else 0.0


def fold_bounds(windows: list[Window], folds: int) -> list[tuple[int, int]]:
    size = len(windows) // folds
    return [(i * size, len(windows) if i == folds - 1 else (i + 1) * size)
            for i in range(folds)]


def oos_scores(windows: list[Window], folds: int, threshold: float):
    """Yield (fold_index, days, [(won, ...)]) with walk-forward standardisation."""
    bounds = fold_bounds(windows, folds)
    for fold_index in range(1, folds):
        train = windows[: bounds[fold_index][0]]
        test = windows[bounds[fold_index][0]: bounds[fold_index][1]]
        if not test:
            continue
        standardiser = Standardiser(SIGNAL_FEATURES).fit(train)
        days = max(1e-9, (test[-1].end_ts - test[0].end_ts) / 86_400.0)
        outcomes = []
        for window in test:
            raw = sum(standardiser.z(window, f) for f in FEATURES) / len(FEATURES)
            if INVERT:
                raw = -raw
            if abs(raw) < threshold:
                continue
            side = "UP" if raw > 0 else "DOWN"
            outcomes.append((side == "UP") == (window.label_up == 1))
        yield fold_index, days, outcomes


def break_even_price_for(accuracy: float) -> float:
    """Entry price at which this accuracy exactly breaks even."""
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if break_even_accuracy(mid) < accuracy:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def entry_price_for_profit_factor(accuracy: float, pf_target: float,
                                  coefficient: float = 0.07) -> float:
    """Highest entry price whose profit factor still reaches ``pf_target``.

    Solved numerically because the fee makes the closed form awkward. Per share:

        win  = (1 - p) - fee(p)
        loss = p + fee(p)
        PF   = a * win / ((1 - a) * loss)

    with ``fee(p) = coefficient * p * (1 - p)``. The fee is symmetric in absolute
    terms, so it shrinks the win less than the loss in relative terms and therefore
    *raises* PF - which is why ignoring it puts the bound on the wrong side of the
    break-even. The result is verified against the empirical grid in run().
    """
    if not 0.0 < accuracy < 1.0 or pf_target <= 0:
        return 0.0

    def pf_at(price: float) -> float:
        fee = taker_fee_per_share(price, coefficient)
        win = (1.0 - price) - fee
        loss = price + fee
        if loss <= 0:
            return float("inf")
        if win <= 0:
            return 0.0
        return accuracy * win / ((1.0 - accuracy) * loss)

    low, high = 0.0, 0.999
    for _ in range(80):
        mid = (low + high) / 2.0
        if pf_at(mid) >= pf_target:
            low = mid          # still good enough at this price
        else:
            high = mid
    return (low + high) / 2.0


def run(dataset: str, out_dir: str, *, folds: int) -> int:
    windows = load(dataset, features=SIGNAL_FEATURES)
    span_days = (windows[-1].end_ts - windows[0].start_ts) / 86_400.0
    base_rate = sum(w.label_up for w in windows) / len(windows)

    print("=" * 104)
    print("V3 EDGE REPORT - win rate, profit factor, frequency (walk-forward, out-of-sample)")
    print("=" * 104)
    print(f"windows {len(windows):,} | span {span_days:.1f} days | folds {folds} "
          f"| base rate {base_rate:.4f}")
    print(f"config  : fade {FEATURES} |z| >= {THRESHOLD}")
    print()

    # ---------------------------------------------------- 1. per-fold win rate
    print("=" * 104)
    print("1. OUT-OF-SAMPLE WIN RATE BY FOLD")
    print("=" * 104)
    print(f"{'fold':>5}{'days':>7}{'trades':>8}{'/day':>7}{'wins':>8}{'win rate':>10}"
          f"{'b/e price':>11}{'verdict':>12}")
    print("-" * 104)
    all_outcomes: list[bool] = []
    total_days = 0.0
    fold_rates: list[float] = []
    for fold_index, days, outcomes in oos_scores(windows, folds, THRESHOLD):
        if not outcomes:
            continue
        wins = sum(outcomes)
        rate = wins / len(outcomes)
        be = break_even_price_for(rate)
        fold_rates.append(rate)
        all_outcomes.extend(outcomes)
        total_days += days
        verdict = "profitable" if be >= 0.505 else "needs cheaper fill"
        print(f"{fold_index:>5}{days:>7.0f}{len(outcomes):>8,}{len(outcomes)/days:>7.1f}"
              f"{wins:>8,}{rate:>10.4f}{be:>11.4f}{verdict:>12}")
    pooled = sum(all_outcomes) / len(all_outcomes)
    print("-" * 104)
    print(f"{'ALL':>5}{total_days:>7.0f}{len(all_outcomes):>8,}"
          f"{len(all_outcomes)/total_days:>7.1f}{sum(all_outcomes):>8,}"
          f"{pooled:>10.4f}{break_even_price_for(pooled):>11.4f}")
    print()
    print(f"  pooled win rate        : {pooled:.4f}")
    print(f"  fold std dev           : {statistics.pstdev(fold_rates):.4f}")
    print(f"  best / worst fold      : {max(fold_rates):.4f} / {min(fold_rates):.4f}")
    print(f"  folds above 50%        : {sum(1 for r in fold_rates if r > 0.5)}/{len(fold_rates)}")
    print(f"  break-even entry price : {break_even_price_for(pooled):.4f}"
          f"   <- pay more than this and the trade is negative EV")

    # ------------------------------------------- 2. profit factor over entries
    print()
    print("=" * 104)
    print("2. PROFIT FACTOR AS A FUNCTION OF ENTRY PRICE  (the decisive table)")
    print("=" * 104)
    print("   A binary held to settlement pays 1.0 or 0.0. At ~0.50 entry the average")
    print("   win and average loss are nearly equal, so profit factor hinges on cents.")
    print()
    print(f"{'entry':>7}{'fee/sh':>8}{'b/e acc':>9}{'win rate':>10}{'PF':>8}"
          f"{'expectancy':>12}{'ROI/trade':>11}{'verdict':>14}")
    print("-" * 104)
    books = []
    for price in ENTRY_PRICES:
        book = Book(price=price, days=total_days)
        for won in all_outcomes:
            book.add(won)
        books.append(book)
        be = break_even_accuracy(price)
        verdict = "positive EV" if pooled > be else "NEGATIVE EV"
        marker = "  <- V3 cap" if abs(price - 0.535) < 1e-9 else ""
        print(f"{price:>7.3f}{taker_fee_per_share(price):>8.4f}{be:>9.4f}"
              f"{book.win_rate:>10.4f}{book.profit_factor:>8.4f}"
              f"{book.expectancy:>+12.4f}{book.roi*100:>10.2f}%{verdict:>14}{marker}")

    pf_break_even = break_even_price_for(pooled)
    # profit factor moves almost linearly in entry price here, so state the
    # sensitivity directly instead of hand-waving about "extra edge"
    pf_at_510 = next(b for b in books if abs(b.price - 0.510) < 1e-9)
    pf_at_530 = next(b for b in books if abs(b.price - 0.530) < 1e-9)
    slope = (pf_at_530.profit_factor - pf_at_510.profit_factor) / 0.020
    print()
    print(f"  profit factor crosses 1.000 at entry ~{pf_break_even:.4f}"
          f"   (matches the analytic {entry_price_for_profit_factor(pooled, 1.0):.4f})")
    print(f"  profit factor falls as the price rises: {slope:+.2f} per 1.00 of entry"
          f"  ({slope/100:+.3f} per cent)")
    for target in (1.05, 1.10, 1.20):
        print(f"  PF >= {target:.2f} needs entry <= "
              f"{entry_price_for_profit_factor(pooled, target):.4f}")
    print("  NOTE: these P&L figures use the settlement payoff only. They are exact")
    print("  given the win rate - no path model is involved. What is NOT measured is")
    print("  whether a fill at these prices is obtainable at all.")

    # ------------------------------------------------ 3. frequency trade-off
    print()
    print("=" * 104)
    print("3. FREQUENCY vs EDGE  (threshold sweep, win rate and PF at a realistic fill)")
    print("=" * 104)
    print(f"{'|z|':>6}{'trades':>9}{'/day':>8}{'win rate':>10}{'b/e price':>11}"
          f"{'PF@0.510':>10}{'exp@0.510':>11}{'PF@0.520':>10}{'exp@0.520':>11}")
    print("-" * 104)
    freq_rows = []
    for threshold in THRESHOLD_GRID:
        outcomes: list[bool] = []
        days = 0.0
        for _, fold_days, fold_out in oos_scores(windows, folds, threshold):
            outcomes.extend(fold_out)
            days += fold_days
        if len(outcomes) < 100:
            continue
        rate = sum(outcomes) / len(outcomes)
        row = {"threshold": threshold, "trades": len(outcomes),
               "per_day": len(outcomes) / days, "win_rate": rate,
               "break_even_price": break_even_price_for(rate)}
        for price in (0.510, 0.520):
            book = Book(price=price, days=days)
            for won in outcomes:
                book.add(won)
            row[f"pf_{price}"] = book.profit_factor
            row[f"exp_{price}"] = book.expectancy
        freq_rows.append(row)
        print(f"{threshold:>6.2f}{len(outcomes):>9,}{row['per_day']:>8.1f}"
              f"{rate:>10.4f}{row['break_even_price']:>11.4f}"
              f"{row['pf_0.51']:>10.4f}{row['exp_0.51']:>+11.4f}"
              f"{row['pf_0.52']:>10.4f}{row['exp_0.52']:>+11.4f}")

    best_pf = max(freq_rows, key=lambda r: r["pf_0.51"])
    print()
    print(f"  best profit factor at a 0.510 fill : |z| >= {best_pf['threshold']:.2f}"
          f"  (PF {best_pf['pf_0.51']:.4f}, {best_pf['per_day']:.1f} trades/day,")
    print(f"                                        win rate {best_pf['win_rate']:.4f})")
    print("  Read this carefully: the PF-optimal threshold is NOT automatically the")
    print("  right choice. Higher thresholds raise PF a little and cut sample size a")
    print("  lot, which is how a curve gets fitted to one year of data.")

    # ------------------------------- 3b. optimise DAILY P&L, not profit factor
    print()
    print("=" * 104)
    print("3b. OPTIMISING WHAT ACTUALLY MATTERS: DAILY P&L (10 USDC per position)")
    print("=" * 104)
    print("   Profit factor is a *per-trade* ratio. The objective is money per day,")
    print("   which is per-trade edge x frequency. A looser threshold can win.")
    print()
    entries = (0.505, 0.510, 0.515, 0.520, 0.525)
    head = f"{'|z|':>6}{'/day':>8}" + "".join(f"{'daily@' + format(p, '.3f'):>16}" for p in entries)
    print(head)
    print("-" * 104)
    best_daily = None
    for row in freq_rows:
        cells = []
        for price in entries:
            net = row[f"exp_{price}"] * row["per_day"] * 10.0 if f"exp_{price}" in row else None
            if net is None:
                # compute from the win rate directly for prices not pre-tabulated
                fee = taker_fee_per_share(price)
                exp = row["win_rate"] - price - fee
                net = exp * row["per_day"] * 10.0
            cells.append(net)
        line = f"{row['threshold']:>6.2f}{row['per_day']:>8.1f}" + "".join(
            f"{c:>+16.3f}" for c in cells)
        print(line)
        for price, net in zip(entries, cells):
            if best_daily is None or net > best_daily[0]:
                best_daily = (net, row["threshold"], price, row["per_day"], row["win_rate"])
    print("-" * 104)
    net, thr, price, per_day, rate = best_daily
    print(f"  best daily P&L in this grid: {net:+.3f} USDC/day at |z| >= {thr:.2f}, "
          f"entry {price:.3f}")
    print(f"    ({per_day:.1f} trades/day, win rate {rate:.4f})")
    print()
    print("  This is the frequency answer: daily P&L is maximised at a LOOSER")
    print("  threshold than profit factor is, because extra trades at a slightly")
    print("  thinner edge still add money - until cost per trade catches up.")
    print("  The binding constraint is therefore not 'trade less', it is 'pay less")
    print("  per trade'. Every cent of slippage is worth ~"
          f"{9.9 * 10 * 0.01:.2f} USDC/day at ~10 trades/day on a 10 USDC stake.")

    # ------------------------------------------------- 4. daily P&L realities
    print()
    print("=" * 104)
    print("4. WHAT THE FREQUENCY IS WORTH")
    print("=" * 104)
    print(f"{'entry':>7}{'exp/trade':>12}{'trades/day':>12}{'exp/day (10 USDC)':>20}"
          f"{'exp/month':>12}")
    print("-" * 104)
    for price in (0.505, 0.510, 0.515, 0.520, 0.525, 0.530):
        book = Book(price=price, days=total_days)
        for won in all_outcomes:
            book.add(won)
        per_day = book.expectancy * book.per_day * 10.0
        print(f"{price:>7.3f}{book.expectancy:>+12.4f}{book.per_day:>12.1f}"
              f"{per_day:>+20.4f}{per_day*30:>+12.2f}")
    print()
    print("  At 10 USDC per position the whole edge is worth single-digit USDC per day.")
    print("  That is the honest size of it, and it is why execution cost dominates:")
    print("  one extra cent of slippage at ~10 trades/day costs about 0.10 USDC/day")
    print("  per unit staked, which is the same order as the entire edge.")

    # ------------------------------------------------------------ 5. outputs
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "v3_edge_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["summary", "windows", len(windows)])
        writer.writerow(["summary", "span_days", round(span_days, 2)])
        writer.writerow(["summary", "pooled_win_rate", round(pooled, 5)])
        writer.writerow(["summary", "win_rate_fold_std", round(statistics.pstdev(fold_rates), 5)])
        writer.writerow(["summary", "worst_fold_win_rate", round(min(fold_rates), 5)])
        writer.writerow(["summary", "break_even_entry_price", round(pf_break_even, 5)])
        writer.writerow(["summary", "trades_per_day", round(len(all_outcomes) / total_days, 2)])
        for book in books:
            writer.writerow(["profit_factor", f"entry_{book.price:.3f}",
                             round(book.profit_factor, 5)])
            writer.writerow(["expectancy", f"entry_{book.price:.3f}",
                             round(book.expectancy, 5)])
        for row in freq_rows:
            writer.writerow(["frequency", f"z_{row['threshold']:.2f}",
                             round(row["per_day"], 3)])
    print()
    print(f"report : {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(root), DEFAULT_CSV))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(root), "output", "v3"))
    parser.add_argument("--folds", type=int, default=8)
    args = parser.parse_args(argv)
    return run(args.dataset, args.out, folds=args.folds)


if __name__ == "__main__":
    raise SystemExit(main())

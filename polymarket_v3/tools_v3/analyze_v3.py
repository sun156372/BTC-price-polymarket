"""Walk-forward analysis for V3: accuracy, frequency, and the entry-price frontier.

    python tools_v3/analyze_v3.py                 # full 9-month sweep
    python tools_v3/analyze_v3.py --folds 6 --quick

Why there is no "intra-window" signal here
------------------------------------------
v1/v2 measured the spot move *inside* the window from tick data. This dataset has
no intra-window BTC ticks - only ``btc_open`` and ``btc_close`` per window. So any
attempt to reconstruct the price at the decision moment has to interpolate
between open and close, and the **close is the outcome**. A signal built that way
is a linear function of the label.

It looks spectacular (98% accuracy on the first sweep) and it is worthless: it
cannot be computed live, because at the decision moment the close does not exist
yet. That family is therefore excluded by construction - see
:data:`SIGNAL_FAMILIES`. What remains is the only thing genuinely observable at
entry: the pre-window features.

Method
------
Walk-forward, no lookahead anywhere:

* windows sorted by time, split into ``folds`` equal **contiguous** blocks
* fold 0 is training only; every later fold is evaluated with a standardiser
  fitted on *all earlier folds*
* ``implied_up_last`` (the window's final implied price) and every outcome column
  are never inputs

Frequency is reported next to accuracy because it decides whether a per-trade
edge survives: 280 trades/day is a fee generator, not a strategy.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v3.data import DEFAULT_CSV, SIGNAL_FEATURES, Window, intraday_frequency, load
from engine_v3.signals import (
    PRE_FEATURES_MOMENTUM,
    SignalDef,
    Standardiser,
    break_even_accuracy,
    edge_points,
    max_viable_entry,
    modelled_entry_price,
)

#: only observable-at-entry families are allowed
SIGNAL_FAMILIES = ("pre",)

#: entry checkpoints (seconds before close). With no intra-window data these are
#: all equivalent for the *signal*; they matter for the entry price and for the
#: remaining-time risk, which is exactly what the TP/SL study uses them for.
CHECKPOINTS = (240.0, 180.0, 120.0, 60.0, 30.0)

#: standardised-score thresholds. A higher bar means fewer, more selective trades.
Z_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)

#: candidate pre-window composites (each is a tuple of dataset features)
COMPOSITES = {
    "mom_short": ("btc_ret_15m", "ind_roc_5", "ind_roc_15"),
    "mom_long": ("btc_ret_1h", "btc_ret_4h", "ind_roc_60"),
    "trend": ("ind_ema_9_dist", "ind_ema_26_dist", "ind_macd_hist"),
    "blend_all": PRE_FEATURES_MOMENTUM,
    "roc_only": ("ind_roc_5", "ind_roc_15", "ind_roc_60"),
    "reversion": ("ind_rsi_14", "ind_bb_pctb", "ind_vwap_24h_dist"),
}


@dataclass
class Tally:
    trades: int = 0
    wins: int = 0
    days: float = 0.0
    entry_prices: list[float] = field(default_factory=list)
    pnl_modelled: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def per_day(self) -> float:
        return self.trades / self.days if self.days else 0.0

    @property
    def avg_entry(self) -> float:
        return statistics.fmean(self.entry_prices) if self.entry_prices else 0.0


def composite_score(window: Window, standardiser: Standardiser,
                    names: tuple[str, ...]) -> float:
    return sum(standardiser.z(window, n) for n in names) / len(names)


def evaluate(windows: list[Window], names: tuple[str, ...], checkpoint: float,
             threshold: float, standardiser: Standardiser, days: float,
             *, coefficient: float = 0.07, invert: bool = False) -> Tally:
    """One configuration over one fold.

    ``invert`` flips the sign: momentum composites are tested both ways, because
    "fade the move" is as plausible a prior as "follow the move" and the data
    should decide, not the author.
    """
    tally = Tally(days=days)
    for window in windows:
        entry_ts = window.end_ts - checkpoint
        raw = composite_score(window, standardiser, names)
        if invert:
            raw = -raw
        if abs(raw) < threshold:
            continue
        side = "UP" if raw > 0 else "DOWN"
        won = (side == "UP") == (window.label_up == 1)
        tally.trades += 1
        tally.wins += int(won)
        entry = modelled_entry_price(window, entry_ts, threshold=threshold, side=side)
        tally.entry_prices.append(entry)
        tally.pnl_modelled += (1.0 if won else 0.0) - entry \
            - coefficient * entry * (1.0 - entry)
    return tally


def fold_bounds(windows: list[Window], folds: int) -> list[tuple[int, int]]:
    size = len(windows) // folds
    return [(i * size, len(windows) if i == folds - 1 else (i + 1) * size)
            for i in range(folds)]


def run(dataset: str, folds: int, out_dir: str, *, quick: bool,
        min_trades: int) -> int:
    windows = load(dataset, features=SIGNAL_FEATURES)
    info = intraday_frequency(windows)
    base_rate = sum(w.label_up for w in windows) / len(windows)

    print("=" * 104)
    print("V3 WALK-FORWARD ANALYSIS - observable-at-entry features only")
    print("=" * 104)
    print(f"dataset        : {dataset}")
    print(f"windows        : {len(windows):,}   span {info['days']:.1f} days"
          f"   ({info['windows_per_day']:.1f} windows/day)")
    print(f"folds          : {folds}   (fold 0 trains, later folds are out-of-sample)")
    print(f"label base rate: {base_rate:.4f}   (a coin flip is 0.5000)")
    print()

    if folds < 2:
        print("need --folds >= 2 for any out-of-sample number")
        return 2

    bounds = fold_bounds(windows, folds)
    composites = COMPOSITES
    if quick:
        composites = {k: COMPOSITES[k] for k in ("blend_all", "mom_short", "reversion")}
    thresholds = Z_THRESHOLDS[::2] if quick else Z_THRESHOLDS
    checkpoints = CHECKPOINTS[:3] if quick else CHECKPOINTS

    per_config: dict[tuple, Tally] = defaultdict(Tally)
    fold_acc: dict[tuple, list[float]] = defaultdict(list)

    for fold_index in range(1, folds):
        train = windows[: bounds[fold_index][0]]
        test = windows[bounds[fold_index][0]: bounds[fold_index][1]]
        if not test:
            continue
        standardiser = Standardiser(SIGNAL_FEATURES).fit(train)
        test_days = max(1e-9, (test[-1].end_ts - test[0].start_ts) / 86_400.0)

        for name, names in composites.items():
            for invert in (False, True):
                tag = f"{name}{'_inv' if invert else ''}"
                for checkpoint in checkpoints:
                    for threshold in thresholds:
                        key = (tag, names, checkpoint, threshold, invert)
                        tally = evaluate(test, names, checkpoint, threshold,
                                         standardiser, test_days, invert=invert)
                        if tally.trades < min_trades:
                            continue
                        agg = per_config[key]
                        agg.trades += tally.trades
                        agg.wins += tally.wins
                        agg.days += tally.days
                        agg.entry_prices.extend(tally.entry_prices)
                        agg.pnl_modelled += tally.pnl_modelled
                        fold_acc[key].append(tally.accuracy)
        print(f"  fold {fold_index}/{folds - 1} done  ({len(test):,} test windows, "
              f"{test_days:.0f} days)")

    if not per_config:
        print("\nno configuration produced enough trades")
        return 1

    rows = []
    for key, tally in per_config.items():
        tag, names, checkpoint, threshold, invert = key
        accuracy = tally.accuracy
        avg_entry = tally.avg_entry
        rows.append({
            "signal": tag,
            "features": "|".join(names),
            "checkpoint_s": checkpoint,
            "threshold": threshold,
            "trades": tally.trades,
            "trades_per_day": round(tally.per_day, 2),
            "accuracy": round(accuracy, 5),
            "wins": tally.wins,
            "avg_entry_modelled": round(avg_entry, 4),
            "break_even_at_entry": round(break_even_accuracy(avg_entry), 4),
            "edge_points_modelled": round(edge_points(accuracy, avg_entry), 3),
            "max_viable_entry": round(max_viable_entry(accuracy), 4),
            "pnl_modelled_per_share": round(tally.pnl_modelled / max(1, tally.trades), 5),
            "fold_accuracies": [round(x, 4) for x in fold_acc[key]],
            "fold_min_accuracy": round(min(fold_acc[key]), 4) if fold_acc[key] else 0.0,
            "fold_max_accuracy": round(max(fold_acc[key]), 4) if fold_acc[key] else 0.0,
            "fold_std": round(statistics.pstdev(fold_acc[key]), 4)
            if len(fold_acc[key]) > 1 else 0.0,
            "beats_base_rate": accuracy > base_rate,
        })

    os.makedirs(out_dir, exist_ok=True)
    sweep_path = os.path.join(out_dir, "v3_sweep.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: -r["accuracy"]):
            writer.writerow(row)

    def show(title: str, key, count: int = 18) -> None:
        print()
        print("=" * 104)
        print(title)
        print("=" * 104)
        print(f"{'signal':<16}{'T-k':>5}{'thr':>7}{'trades':>9}{'/day':>8}{'acc':>8}"
              f"{'entry~':>8}{'b/e':>7}{'edge':>7}{'maxP':>7}{'minFold':>9}{'maxFold':>9}")
        print("-" * 104)
        for row in sorted(rows, key=key)[:count]:
            print(f"{row['signal']:<16}{row['checkpoint_s']:>5.0f}{row['threshold']:>7.2f}"
                  f"{row['trades']:>9,}{row['trades_per_day']:>8.1f}{row['accuracy']:>8.4f}"
                  f"{row['avg_entry_modelled']:>8.3f}{row['break_even_at_entry']:>7.3f}"
                  f"{row['edge_points_modelled']:>7.2f}{row['max_viable_entry']:>7.3f}"
                  f"{row['fold_min_accuracy']:>9.4f}{row['fold_max_accuracy']:>9.4f}")

    eligible = [r for r in rows if r["trades"] >= 500]
    show("TOP BY ACCURACY  (>= 500 trades, all folds pooled)", lambda r: -r["accuracy"])
    show("MOST ROBUST  (highest worst-fold accuracy)", lambda r: -r["fold_min_accuracy"])
    show("BEST MODELLED P&L PER SHARE", lambda r: -r["pnl_modelled_per_share"])

    # ---------------------------------------------------- honesty scoreboard
    print()
    print("=" * 104)
    print("HOW MANY CONFIGURATIONS ACTUALLY BEAT A COIN FLIP OUT OF SAMPLE?")
    print("=" * 104)
    better = [r for r in eligible if r["accuracy"] > base_rate]
    robust = [r for r in eligible if r["fold_min_accuracy"] > base_rate]
    profitable = [r for r in eligible if r["pnl_modelled_per_share"] > 0]
    print(f"  evaluated configurations   : {len(rows):,}")
    print(f"  eligible (>= 500 trades)   : {len(eligible):,}")
    print(f"  pooled accuracy > base rate: {len(better):,}  ({len(better)/max(1,len(eligible))*100:.1f}%)")
    print(f"  EVERY fold > base rate     : {len(robust):,}  ({len(robust)/max(1,len(eligible))*100:.1f}%)")
    print(f"  modelled P&L > 0           : {len(profitable):,}  ({len(profitable)/max(1,len(eligible))*100:.1f}%)")
    print()
    print("  A pure-noise sweep still puts ~50% of configs above the base rate and")
    print("  scatters a few 'profitable' ones, so read the EVERY-fold column, not the")
    print("  pooled one. Nothing here is a result until it survives that filter.")

    print()
    print(f"sweep : {sweep_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(root), DEFAULT_CSV))
    parser.add_argument("--folds", type=int, default=9)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(root), "output", "v3"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--min-trades", type=int, default=50)
    args = parser.parse_args(argv)
    return run(args.dataset, args.folds, args.out, quick=args.quick,
               min_trades=args.min_trades)


if __name__ == "__main__":
    raise SystemExit(main())

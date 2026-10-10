"""V4 backtest on the real order book: measured entry, measured exit.

    python tools_v4/backtest.py --horizons 5m
    python tools_v4/backtest.py --horizons 5m 15m --cash 1000

What changed versus every earlier backtest
------------------------------------------
Earlier versions had to *model* two things because the data did not contain them:

1. **entry price** - the aggregate dataset had no quotes at all
2. **which barrier was touched first** - no intra-window path existed, so a
   Brownian bridge was invented, and TP/SL conclusions swung between +0.09 and
   -0.20 per share depending on that one assumption

This dataset has both, measured: ``up_best_ask`` / ``down_best_ask`` plus full L2
depth at 100ms, from the open to the close, with both sides of the book priced.
So every exit result below is read off the tape. No bridge, no diffusion
assumption, no "modelled" label.

Signal under test
-----------------
Intra-window token momentum, faded. At T-240 the Up token has already moved some
distance from its value at the open; V3's evidence says fade that move, i.e. buy
the side that has gone DOWN. The signal is computed from the Up mid at the open
versus at T-240, which is observable at the decision moment.

Accounting
----------
* entry is a taker fill: walk the real asks for the requested size
* exit at a barrier is a taker fill against the real bids (adverse by design)
* settlement pays 1.0/0.0 with **no exit fee** (settlement is not a trade)
* taker fee per share = 0.07 * p * (1 - p)

The fills are deliberately pessimistic: a real maker could do better, a real
taker could do worse, and no attempt is made to claim queue priority.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import resolve_data_root  # noqa: E402

def _resolve_default_root() -> str:
    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""
    return resolve_data_root(required=False) or 'micro_v3'


from engine_v4 import hf
from engine_v4.signals import taker_fee_per_share

#: entry checkpoints in seconds before the close. V3 chose T-240; earlier is
#: cheaper but the move is smaller, and the tape should decide which wins.
CHECKPOINTS = (60.0, 120.0, 180.0, 240.0, 270.0, 285.0, 292.0)
#: the entry the descriptive stats use
DEFAULT_CHECKPOINT = 240.0
#: conviction thresholds on |intra-window move| in probability points
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
#: (take profit, stop loss) in probability points; 1.0 on a leg = disabled
EXITS = ((1.00, 1.00), (0.30, 1.00), (0.20, 1.00), (0.10, 1.00), (0.05, 1.00),
         (0.30, 0.30), (0.20, 0.30), (0.20, 0.20), (0.15, 0.30))
#: order sizes in USDC to test fills against real depth
SIZES = (10.0, 25.0, 50.0)


@dataclass
class Trade:
    market_id: str
    horizon: str
    entry_ts: float
    side: str
    entry_price: float
    entry_slip: float
    shares: float
    exit_kind: str
    exit_price: float
    pnl: float
    won: bool
    fillable: bool


@dataclass
class Book:
    """Accumulator for one (checkpoint, threshold, exit, size) configuration."""

    label: str
    trades: list[Trade] = field(default_factory=list)
    skipped_unfillable: int = 0
    skipped_no_quote: int = 0

    def add(self, trade: Trade) -> None:
        self.trades.append(trade)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def per_share(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl / max(1e-9, t.shares) for t in self.trades) / self.n

    @property
    def avg_entry(self) -> float:
        return statistics.fmean(t.entry_price for t in self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if gross_loss <= 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def direction_accuracy(self) -> float:
        return sum(1 for t in self.trades if t.won) / self.n if self.n else 0.0

    def exit_mix(self) -> dict:
        out: dict[str, int] = {}
        for trade in self.trades:
            out[trade.exit_kind] = out.get(trade.exit_kind, 0) + 1
        return out


def signal_side(open_up_mid: float | None, entry_up_mid: float | None,
                threshold: float, mode: str = "fade") -> tuple[str, float] | None:
    """Direction from the intra-window move of the Up token.

    ``mode="fade"`` buys the side that has gone DOWN; ``mode="follow"`` buys the
    side that has gone UP. Both are tested because V3's evidence came from a
    different dataset and the data should decide, not the earlier conclusion.

    Returns (side, move) or None when the move is inside the threshold.
    """
    if open_up_mid is None or entry_up_mid is None:
        return None
    move = entry_up_mid - open_up_mid
    if abs(move) < threshold:
        return None
    up_side = move > 0
    if mode == "fade":
        up_side = not up_side
    return ("UP" if up_side else "DOWN"), move


def simulate_exit(market: hf.Market, entry_tick: hf.Tick, side: str,
                  entry_price: float, shares: float,
                  tp: float, sl: float) -> tuple[str, float]:
    """Read the real tape to find the first barrier touch, else settle.

    This is the part that was previously modelled. The tape is 100ms so "first
    touch" is decided by actual quoted prices, and the exit fill is taken from
    the real bids at that moment rather than assumed to be the barrier level.
    """
    stop_price = entry_price - sl if sl < 1.0 else None
    target_price = entry_price + tp if tp < 1.0 else None
    deadline_ms = entry_tick.to_deadline_ms

    for tick in market.ticks:
        if tick.to_deadline_ms >= deadline_ms:
            continue
        if tick.to_deadline_ms <= 0:
            break
        bid = tick.bid(side)
        if bid is None:
            continue
        if stop_price is not None and bid <= stop_price:
            proceeds = tick.proceeds_to_sell(side, shares)
            price = (proceeds / shares) if proceeds is not None else bid
            return "stop", price
        if target_price is not None and bid >= target_price:
            proceeds = tick.proceeds_to_sell(side, shares)
            price = (proceeds / shares) if proceeds is not None else bid
            return "target", price

    won = market.up_won if side == "UP" else (not market.up_won)
    return "settle", (1.0 if won else 0.0)


def run_market(market: hf.Market, *, checkpoint: float, threshold: float,
               tp: float, sl: float, size_usdc: float,
               coefficient: float, mode: str = "fade") -> Trade | None:
    """One trade attempt. Returns None when the gate refuses."""
    if market.up_won is None:
        return None
    window = market.session_seconds
    entry_tick = market.tick_at_t_minus(checkpoint, tolerance_s=1.5)
    if entry_tick is None:
        return None

    # value of the Up token at the open, for the intra-window move
    open_tick = market.tick_at_t_minus(window, tolerance_s=2.0)
    if open_tick is None:
        in_window = market.in_window()
        if not in_window:
            return None
        open_tick = in_window[0]

    picked = signal_side(open_tick.up_mid, entry_tick.up_mid, threshold, mode)
    if picked is None:
        return None
    side, _move = picked

    ask = entry_tick.ask(side)
    if ask is None or not 0.01 < ask < 0.99:
        return None

    shares = size_usdc / ask
    cost = entry_tick.cost_to_buy(side, shares)
    if cost is None:
        return None                      # book too thin: a real refusal, counted
    entry_price = cost / shares
    slip = entry_price - ask

    exit_kind, exit_price = simulate_exit(market, entry_tick, side, entry_price,
                                          shares, tp, sl)
    fee_in = shares * taker_fee_per_share(entry_price, coefficient)
    if exit_kind == "settle":
        proceeds = shares * exit_price
        fee_out = 0.0
    else:
        fee_out = shares * taker_fee_per_share(exit_price, coefficient)
        proceeds = shares * exit_price

    pnl = proceeds - shares * entry_price - fee_in - fee_out
    won = (side == "UP") == market.up_won
    return Trade(
        market_id=market.market_id, horizon=market.horizon,
        entry_ts=entry_tick.ts, side=side, entry_price=entry_price,
        entry_slip=slip, shares=shares, exit_kind=exit_kind,
        exit_price=exit_price, pnl=pnl, won=won, fillable=True,
    )


def run(root: str, out_dir: str, *, horizons: list[str], size_usdc: float,
        coefficient: float, limit: int | None) -> int:
    resolutions = hf.load_resolutions(root)
    print("=" * 104)
    print("V4 REAL-ORDERBOOK BACKTEST - entry and exit both measured from the tape")
    print("=" * 104)
    print(f"horizons   : {horizons}")
    print(f"order size : {size_usdc:.0f} USDC")
    print(f"fee model  : taker {coefficient} * p * (1-p) per share")
    print()

    # ---------------------------------------------------------- load markets
    markets: list[hf.Market] = []
    import time
    started = time.time()

    def progress(index, total, kept):
        print(f"  loaded {index}/{total} files ({kept} kept)  {time.time()-started:.0f}s")

    for market in hf.iter_markets(root, horizons=horizons, resolutions=resolutions,
                                 limit=limit, progress=progress):
        if market.up_won is None:
            continue
        markets.append(market)

    print(f"\nmarkets with resolution: {len(markets):,}  ({time.time()-started:.0f}s)")
    if not markets:
        print("nothing to test")
        return 1
    by_horizon: dict[str, int] = {}
    for market in markets:
        by_horizon[market.horizon] = by_horizon.get(market.horizon, 0) + 1
    print(f"by horizon: {by_horizon}")
    ups = sum(1 for m in markets if m.up_won)
    print(f"UP base rate: {ups/len(markets):.4f}")

    # -------------------------------------------------- tape order self-check
    # A first-touch exit model is only valid if ticks run FORWARD in time. The
    # countdown descends over the window, so an ascending sort silently reverses
    # the tape and makes every exit read the LAST touch. That bug produced
    # 0.69 targets "exiting" at 0.99 - one of the most expensive kind of error,
    # because the results still look plausible.
    bad_order = [m.market_id for m in markets[:400]
                 if any(m.ticks[i].to_deadline_ms < m.ticks[i + 1].to_deadline_ms
                        for i in range(len(m.ticks) - 1))]
    print(f"tape order check: {len(bad_order)} of first 400 markets NOT chronological")
    if bad_order:
        print("  ABORT: exits would read the tape backwards")
        return 1
    print("  ticks run open -> close, so first-touch exits are valid")

    # how much does the token actually move inside a window?
    moves = []
    for market in markets:
        window = market.session_seconds
        first = market.in_window()
        entry = market.tick_at_t_minus(DEFAULT_CHECKPOINT, tolerance_s=1.5)
        if first and entry and first[0].up_mid is not None and entry.up_mid is not None:
            moves.append(abs(entry.up_mid - first[0].up_mid))
    stats = hf.summarise(moves)
    print(f"\n|Up mid move| from open to T-240: median {stats['median']:.4f} "
          f"p75 {stats['p75']:.4f} p90 {stats['p90']:.4f} max {stats['max']:.4f}")
    print(f"  -> {sum(1 for m in moves if m >= 0.02)/len(moves)*100:.1f}% of windows move "
          f">= 0.02 by T-240")

    # ------------------------------------------------------------- the sweep
    print()
    print("=" * 104)
    print("SWEEP: checkpoint x threshold x exit rule")
    print("=" * 104)
    results: list[Book] = []
    modes = ("fade", "follow")
    total_configs = len(modes) * len(CHECKPOINTS) * len(THRESHOLDS) * len(EXITS)
    done = 0
    for mode in modes:
        for checkpoint in CHECKPOINTS:
            for threshold in THRESHOLDS:
                for tp, sl in EXITS:
                    book = Book(label=f"{mode}|T-{checkpoint:.0f}|thr{threshold:.2f}"
                                      f"|tp{tp:.2f}/sl{sl:.2f}")
                    for market in markets:
                        trade = run_market(market, checkpoint=checkpoint,
                                           threshold=threshold, tp=tp, sl=sl,
                                           size_usdc=size_usdc, coefficient=coefficient,
                                           mode=mode)
                        if trade is None:
                            book.skipped_no_quote += 1
                        else:
                            book.add(trade)
                    results.append(book)
                    done += 1
            print(f"  {mode} T-{checkpoint:.0f} done ({done}/{total_configs})")

    # ------------------------------------------------------------- reporting
    eligible = [b for b in results if b.n >= 100]
    print()
    print("=" * 104)
    print("TOP BY WIN RATE  (>= 100 trades)")
    print("=" * 104)
    print(f"{'config':<34}{'n':>7}{'win%':>8}{'dir%':>8}{'entry':>8}{'PF':>7}"
          f"{'pnl/sh':>9}{'exit mix':>26}")
    print("-" * 104)
    for book in sorted(eligible, key=lambda b: -b.win_rate)[:15]:
        mix = ",".join(f"{k}:{v}" for k, v in sorted(book.exit_mix().items()))
        print(f"{book.label:<34}{book.n:>7,}{book.win_rate*100:>8.2f}{book.direction_accuracy*100:>8.2f}"
              f"{book.avg_entry:>8.3f}{book.profit_factor:>7.2f}{book.per_share:>+9.4f}{mix:>26}")

    print()
    print("=" * 104)
    print("TOP BY P&L PER SHARE  (>= 100 trades)")
    print("=" * 104)
    print(f"{'config':<34}{'n':>7}{'win%':>8}{'dir%':>8}{'entry':>8}{'PF':>7}"
          f"{'pnl/sh':>9}{'exit mix':>26}")
    print("-" * 104)
    for book in sorted(eligible, key=lambda b: -b.per_share)[:15]:
        mix = ",".join(f"{k}:{v}" for k, v in sorted(book.exit_mix().items()))
        print(f"{book.label:<34}{book.n:>7,}{book.win_rate*100:>8.2f}{book.direction_accuracy*100:>8.2f}"
              f"{book.avg_entry:>8.3f}{book.profit_factor:>7.2f}{book.per_share:>+9.4f}{mix:>26}")

    # ------------------------------------------- does the signal actually work?
    print()
    print("=" * 104)
    print("IS THE SIGNAL REAL?  direction accuracy at T-240, hold to settlement")
    print("=" * 104)
    print(f"{'mode':>7}{'threshold':>10}{'n':>7}{'trades/day':>12}{'direction acc':>15}"
          f"{'avg entry':>11}{'break-even':>12}{'edge pts':>10}")
    print("-" * 104)
    days = max(1e-9, sum(m.session_seconds for m in markets) / 86_400.0)
    for book in results:
        if "T-240|" not in book.label or "tp1.00/sl1.00" not in book.label:
            continue
        mode = book.label.split("|")[0]
        threshold = float(book.label.split("thr")[1].split("|")[0])
        acc = book.direction_accuracy
        entry = book.avg_entry
        break_even = entry + taker_fee_per_share(entry, coefficient)
        print(f"{mode:>7}{threshold:>10.2f}{book.n:>7,}{book.n/days:>12.1f}{acc:>15.4f}"
              f"{entry:>11.3f}{break_even:>12.4f}{(acc-break_even)*100:>+10.2f}")

    # ------------------------------------------------ fill rate at each size
    print()
    print("=" * 104)
    print("CAN THE ORDER ACTUALLY FILL?  share of attempts the real book absorbs")
    print("=" * 104)
    print("  A refusal means the visible asks could not cover the size. That is a real")
    print("  constraint read off the depth arrays, not a modelling assumption.")
    print()
    print(f"  {'size USDC':>10}{'attempts':>10}{'filled':>9}{'fill rate':>11}"
          f"{'median slip':>13}{'p90 slip':>10}")
    print("-" * 104)
    fill_rows = []
    lead_mode, lead_checkpoint, lead_threshold = "follow", 285.0, 0.02
    for size in SIZES:
        attempts = filled = 0
        slips = []
        for market in markets:
            entry_tick = market.tick_at_t_minus(lead_checkpoint, tolerance_s=1.5)
            if entry_tick is None:
                continue
            picked = signal_side(market.first_mid(), entry_tick.up_mid,
                                 lead_threshold, lead_mode)
            if picked is None:
                continue
            side, _ = picked
            ask = entry_tick.ask(side)
            if ask is None or not 0.01 < ask < 0.99:
                continue
            attempts += 1
            shares = size / ask
            cost = entry_tick.cost_to_buy(side, shares)
            if cost is None:
                continue
            filled += 1
            slips.append(cost / shares - ask)
        rate = filled / attempts if attempts else 0.0
        slip_stats = hf.summarise(slips)
        fill_rows.append({"size": size, "attempts": attempts, "filled": filled,
                          "rate": rate, "median_slip": slip_stats.get("median", 0.0),
                          "p90_slip": slip_stats.get("p90", 0.0)})
        print(f"  {size:>10.0f}{attempts:>10,}{filled:>9,}{rate*100:>10.1f}%"
              f"{slip_stats.get('median', 0.0)*100:>12.2f}c"
              f"{slip_stats.get('p90', 0.0)*100:>9.2f}c")

    # ----------------------------------------------- consistency across days
    print()
    print("=" * 104)
    print("IS IT CONSISTENT ACROSS DAYS?  the same rule, day by day")
    print("=" * 104)
    print("  This is the check that matters most. Four days is a small sample, but a")
    print("  rule that only works on one of them is a coincidence, not an edge.")
    print()
    candidates = [
        ("follow", 285.0, 0.02, 1.00, 1.00),
        ("follow", 120.0, 0.05, 1.00, 1.00),
        ("follow", 240.0, 0.05, 1.00, 1.00),
        ("follow", 285.0, 0.02, 1.00, 0.35),
    ]
    print(f"  {'rule':<38}{'day':>12}{'n':>6}{'win%':>8}{'dir%':>8}{'entry':>8}"
          f"{'PF':>7}{'pnl/sh':>9}")
    print("-" * 104)
    days_seen = sorted({m.day for m in markets if m.day})
    day_rows = []
    for mode, checkpoint, threshold, tp, sl in candidates:
        label = f"{mode} T-{checkpoint:.0f} thr{threshold:.2f} tp{tp:.2f}/sl{sl:.2f}"
        for day in days_seen:
            subset = [m for m in markets if m.day == day]
            if not subset:
                continue
            book = Book(label=label)
            for market in subset:
                trade = run_market(market, checkpoint=checkpoint, threshold=threshold,
                                   tp=tp, sl=sl, size_usdc=size_usdc,
                                   coefficient=coefficient, mode=mode)
                if trade is not None:
                    book.add(trade)
            if book.n == 0:
                continue
            day_rows.append({"rule": label, "day": day, "n": book.n,
                             "win_rate": book.win_rate, "dir": book.direction_accuracy,
                             "entry": book.avg_entry, "pf": book.profit_factor,
                             "per_share": book.per_share})
            pf = book.profit_factor
            print(f"  {label:<38}{day:>12}{book.n:>6,}{book.win_rate*100:>8.2f}"
                  f"{book.direction_accuracy*100:>8.2f}{book.avg_entry:>8.3f}"
                  f"{(pf if pf != float('inf') else 99):>7.2f}{book.per_share:>+9.4f}")
        print()
    print("  Per-day verdict for the leading rule:")
    lead = [r for r in day_rows if r["rule"].startswith("follow T-285")]
    positives = sum(1 for r in lead if r["per_share"] > 0)
    print(f"    days tested {len(lead)}, days with positive pnl/share: {positives}")

    # ------------------------------------------------------------- outputs
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "v4_backtest.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "mode", "checkpoint", "threshold", "tp", "sl", "trades",
                         "win_rate", "direction_accuracy", "avg_entry", "profit_factor",
                         "pnl_per_share", "total_pnl", "skipped_no_quote",
                         "exit_target", "exit_stop", "exit_settle"])
        for book in results:
            parts = book.label.split("|")
            mode = parts[0]
            checkpoint = parts[1].replace("T-", "")
            threshold = parts[2].replace("thr", "")
            tp, sl = parts[3].replace("tp", "").split("/sl")
            mix = book.exit_mix()
            writer.writerow([book.label, mode, checkpoint, threshold, tp, sl, book.n,
                             round(book.win_rate, 5), round(book.direction_accuracy, 5),
                             round(book.avg_entry, 5),
                             round(book.profit_factor, 5) if book.profit_factor != float("inf") else "",
                             round(book.per_share, 6), round(book.pnl, 4),
                             book.skipped_no_quote,
                             mix.get("target", 0), mix.get("stop", 0), mix.get("settle", 0)])
    print()
    print(f"sweep : {path}")
    return 0


def main(argv=None) -> int:
    root = _resolve_default_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=resolve_data_root(required=False) or 'micro_v3')
    parser.add_argument("--out", default=os.path.join(root, "output", "v4"))
    parser.add_argument("--horizons", nargs="*", default=["5m"])
    parser.add_argument("--size", type=float, default=10.0)
    parser.add_argument("--fee", type=float, default=0.07)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    return run(args.root, args.out, horizons=args.horizons, size_usdc=args.size,
               coefficient=args.fee, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

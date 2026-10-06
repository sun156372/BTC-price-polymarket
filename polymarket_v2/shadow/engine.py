"""Shadow engine: 1200 parameter sets, one thread, one loop.

Architecture
------------
::

    market update  ->  pre-parse ONCE (float timestamps, sorted print arrays,
                       per-checkpoint spot moves, per-window ATR)
                   ->  for account in accounts:      <- the single hot loop
                           entry gate -> fill -> sequential TP/SL -> hedge/settle

There is exactly one thread. Each account keeps its own cash, equity, positions
and trade counters; the market data is shared by reference. No timestamp
conversion, JSON parsing, regex or string formatting happens inside the loop -
those all live in :mod:`shadow.markets` and run once per window.

Exit logic (sequential, first touch wins)
-----------------------------------------
For the primary leg, walk prints strictly after entry:

* price <= stop  -> hard stop, unless ``hedge_enabled``; then hedge instead of
  closing: buy ``hedge_ratio`` of the position on the opposite token and let both
  legs settle at the window's payoff.
* price >= target -> take profit, close at the target.
* neither -> settle at 1.0/0.0.

Dynamic TP/SL (``dynamic_tp_sl``)
---------------------------------
Instead of fixed distances, both exits scale with realised ATR (mean absolute 1s
return over the last 60s) relative to a reference ATR of 0.0002 (2 bps/s, close
to a calm BTC minute). A quiet window tightens the exits; a violent one widens
them - which is what stops a fixed 0.02 stop from being swept by noise.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .grid import Leg, OpenPosition, ShadowParams, VirtualAccount, build_accounts, entry_times
from .markets import K, MarketBook, MarketFrame, SpotRing, now_utc

REFERENCE_ATR = 0.0002        # ~2 bps per second: a calm market
MIN_SCALE, MAX_SCALE = 0.5, 2.5
#: how long a checkpoint stays "due" so a slow cycle cannot miss it
CHECKPOINT_GRACE = 3.0

OUTPUT_FIELDS = [
    "param_id", "equity", "return_pct", "trades", "wins", "win_rate", "hedges",
    "stops", "take_profits", "settlements", "fees_paid", "gross_pnl",
    "max_drawdown", "open_positions", "time", "threshold", "tp", "sl", "min_gap",
    "dynamic_tp_sl", "hedge_enabled", "hedge_ratio", "updated_ts",
]


@dataclass
class ExitPlan:
    """Resolved exit levels for one account in one window."""

    tp_price: float
    sl_price: float
    scale: float
    atr: float | None


@dataclass
class ShadowStats:
    """Global counters for the terminal summary."""

    cycles: int = 0
    windows_seen: int = 0
    windows_settled: int = 0
    entries: int = 0
    stops: int = 0
    hedges: int = 0
    take_profits: int = 0
    settlements: int = 0
    blocked_no_cash: int = 0
    blocked_min_gap: int = 0
    blocked_threshold: int = 0
    blocked_no_print: int = 0
    archived: int = 0
    last_archive_ts: float = 0.0

    def as_dict(self) -> dict:
        return {
            "cycles": self.cycles,
            "windows_seen": self.windows_seen,
            "windows_settled": self.windows_settled,
            "entries": self.entries,
            "stops": self.stops,
            "hedges": self.hedges,
            "take_profits": self.take_profits,
            "settlements": self.settlements,
            "blocked_no_cash": self.blocked_no_cash,
            "blocked_min_gap": self.blocked_min_gap,
            "blocked_threshold": self.blocked_threshold,
            "blocked_no_print": self.blocked_no_print,
        }


class ShadowEngine:
    """Drives every virtual account over a shared market book."""

    def __init__(
        self,
        accounts: Sequence[VirtualAccount],
        *,
        lookback_s: float = 180.0,
        notional: float = 100.0,
        coefficient: float = 0.07,
        max_staleness_s: float = 10.0,
        max_slippage_bps: float = 200.0,
        archive_path: str = "shadow_metrics.csv",
        archive_interval_s: float = 300.0,
        top_n: int = 10,
        logger=print,
    ) -> None:
        self.accounts = list(accounts)
        self.by_time: dict[float, list[int]] = {}
        for index, account in enumerate(self.accounts):
            self.by_time.setdefault(account.params.time_s, []).append(index)
        self.times = sorted(self.by_time)

        self.lookback_s = float(lookback_s)
        self.notional = float(notional)
        self.coefficient = float(coefficient)
        self.max_staleness_s = float(max_staleness_s)
        self.max_slippage_bps = float(max_slippage_bps)
        self.archive_path = archive_path
        self.archive_interval_s = float(archive_interval_s)
        self.top_n = int(top_n)
        self.log = logger
        self.stats = ShadowStats()

    # ------------------------------------------------------------- sizing
    def _scale_for(self, params: ShadowParams, atr: float | None) -> float:
        if not params.dynamic_tp_sl or not atr or atr <= 0:
            return 1.0
        return min(MAX_SCALE, max(MIN_SCALE, atr / REFERENCE_ATR))

    def _exit_plan(self, params: ShadowParams, entry_price: float,
                   atr: float | None) -> ExitPlan:
        scale = self._scale_for(params, atr)
        tp = params.tp * scale
        sl = params.sl * scale
        return ExitPlan(
            tp_price=min(0.999, entry_price + tp),
            sl_price=max(0.001, entry_price - sl),
            scale=scale,
            atr=atr,
        )

    # --------------------------------------------------------------- cycle
    def cycle(self, book: MarketBook, spots: dict[str, SpotRing],
              now: float | None = None) -> None:
        """One iteration over every account. This is the whole engine."""
        moment = time.time() if now is None else now
        self.stats.cycles += 1

        for frame in book:
            self._prepare(frame, spots, moment)
            self._run_frame(frame, moment)

    def _prepare(self, frame: MarketFrame, spots: dict[str, SpotRing],
                 now: float) -> None:
        """Per-window work done once, outside the account loop."""
        if frame.cutoff is not None:
            # replay: hide bars and prints after the simulated moment
            for token, series in list(frame.prints.items()):
                frame.prints[token] = series.with_cutoff(frame.cutoff)
        frame.refresh_mid()
        spot = spots.get(frame.asset)
        if spot is not None:
            frame.atr = spot.atr(now, 60.0)
            frame.prepare_signal(spot, self.times, self.lookback_s)

    def _run_frame(self, frame: MarketFrame, now: float) -> None:
        """The single hot loop: one pass over 1200 accounts for this window."""
        remaining = frame.end_ts - now
        if remaining > max(self.times) + 1.0:
            return                                   # no checkpoint reached yet

        for t_minus in self.times:
            # Fire each cadence exactly once: the checkpoint becomes due when the
            # window has `t_minus` seconds left, and stays due for a short grace
            # period so a slow cycle cannot skip it. `min_gap` and the per-window
            # position slot then make a duplicate entry impossible.
            if not (t_minus - CHECKPOINT_GRACE <= remaining <= t_minus):
                continue
            indices = self.by_time[t_minus]
            move = frame.spot_move.get(t_minus)
            entry_ts = frame.entry_ts_for(t_minus)

            for index in indices:
                account = self.accounts[index]
                self._decide(account, frame, entry_ts, move, now)

    # ---------------------------------------------------------------- decide
    def _decide(self, account: VirtualAccount, frame: MarketFrame,
                entry_ts: float, move: float | None, now: float) -> None:
        params = account.params
        if frame.slug in account.decided:
            return                                   # this window is already spent
        if frame.slug in account.positions:
            return                                   # already holding this window
        account.decided.add(frame.slug)
        if entry_ts - account.last_entry_ts < params.min_gap_s:
            self.stats.blocked_min_gap += 1
            return
        if move is None or abs(move) < params.threshold_pct:
            self.stats.blocked_threshold += 1
            return

        side = "UP" if move > 0 else "DOWN"
        token = frame.tokens[0] if side == "UP" else frame.tokens[1]
        series = frame.series(token)
        last = series.last_at_or_before(entry_ts)
        if last is None:
            self.stats.blocked_no_print += 1
            return
        print_ts, price = last
        if not 0.02 < price < 0.98:
            self.stats.blocked_no_print += 1
            return

        staleness = entry_ts - print_ts
        if staleness > self.max_staleness_s:
            self.stats.blocked_no_print += 1
            return

        shares = self.notional / price
        cost = shares * price
        fee = shares * self.coefficient * price * (1.0 - price)
        if not account.can_enter(cost + fee):
            self.stats.blocked_no_cash += 1
            return

        plan = self._exit_plan(params, price, frame.atr)
        account.cash -= cost + fee
        account.positions[frame.slug] = OpenPosition(
            slug=frame.slug, asset=frame.asset, window_end_ts=frame.end_ts,
            side=side, token=token, shares=shares, entry_price=price,
            entry_fee=fee, entry_ts=entry_ts, tp_price=plan.tp_price,
            sl_price=plan.sl_price, atr=plan.atr,
        )
        account.last_entry_ts = entry_ts
        account.fees_paid += fee
        self.stats.entries += 1

    # ---------------------------------------------------------------- exits
    def resolve(self, book: MarketBook, now: float) -> int:
        """Walk every open position to its exit or settlement. Returns closes."""
        closed = 0
        for frame in list(book):
            if frame.end_ts > now:
                continue
            retired = book.retire(frame.slug)
            if retired is None:
                continue
            self.stats.windows_settled += 1
            for account in self.accounts:
                position = account.positions.pop(retired.slug, None)
                if position is None:
                    continue
                self._close_position(account, retired, position)
                closed += 1
        return closed

    def _close_position(self, account: VirtualAccount, frame: MarketFrame,
                        position: OpenPosition) -> None:
        """Sequential first-touch over prints after entry, then settle."""
        params = account.params
        series = frame.series(position.token)
        start = series.first_strictly_after(position.entry_ts)

        for _, price in series.walk_from(start):
            if price <= position.sl_price:
                if params.hedge_enabled and params.hedge_ratio > 0:
                    # Buy the opposite leg, then let the code below book exactly
                    # one settlement for the whole round trip. (Do not increment
                    # or return here: that is what made the settlement counters
                    # disagree with the position count.)
                    if self._hedge(account, frame, position, price):
                        account.hedges += 1
                        self.stats.hedges += 1
                    break
                self._realise(account, position, position.sl_price,
                              "stop_loss", settled=False)
                account.stops += 1
                self.stats.stops += 1
                return
            if price >= position.tp_price:
                self._realise(account, position, position.tp_price,
                              "take_profit", settled=False)
                account.take_profits += 1
                self.stats.take_profits += 1
                return

        # ---- exactly one settlement per position (hedged or not)
        winner = frame.winner or self._infer_winner(frame)
        self._realise(account, position, None,
                      "settlement_hedged" if position.hedged else "settlement",
                      settled=True, winner=winner)
        account.settlements += 1
        self.stats.settlements += 1

    def _hedge(self, account: VirtualAccount, frame: MarketFrame,
               position: OpenPosition, trigger_price: float) -> bool:
        """Buy the opposite token instead of realising the stop.

        Returns True when the hedge leg was opened. The caller then books the
        whole round trip at settlement - a hedged position is one trade, not two.
        """
        opposite_token = (frame.tokens[1] if position.token == frame.tokens[0]
                          else frame.tokens[0])
        opposite_side = "DOWN" if position.side == "UP" else "UP"
        series = frame.series(opposite_token)
        last = series.last_at_or_before(frame.end_ts) or series.last_at_or_before(trigger_price)
        price = last[1] if last else max(0.001, 1.0 - trigger_price)

        shares = position.shares * account.params.hedge_ratio
        cost = shares * price
        fee = shares * self.coefficient * price * (1.0 - price)
        if account.cash < cost + fee:
            # Cannot fund the hedge. Report False and book NOTHING here: the
            # caller falls through to exactly one settlement. Booking the stop
            # here as well would realise the same position twice.
            return False

        account.cash -= cost + fee
        account.fees_paid += fee
        position.hedge_leg = Leg(token=opposite_token, side=opposite_side,
                                 shares=shares, price=price, fee=fee,
                                 ts=frame.end_ts)
        position.hedged = True
        return True

    @staticmethod
    def _infer_winner(frame: MarketFrame) -> str:
        up = frame.mid.get(frame.tokens[0])
        down = frame.mid.get(frame.tokens[1])
        if up is None or down is None:
            return ""
        if abs(up - down) < 0.05:
            return ""
        return "UP" if up > down else "DOWN"

    def _realise(self, account: VirtualAccount, position: OpenPosition,
                 exit_price: float | None, reason: str, *, settled: bool,
                 winner: str = "") -> None:
        """Book an exit (sell at a level) or a settlement (payoff, no fee).

        ``pnl`` is always ``proceeds - total_cost`` where ``total_cost`` covers
        every leg bought for this window, so a hedged position is accounted for
        as one round trip rather than two unrelated trades.
        """
        if settled:
            payout = position.payout(winner) if winner else 0.0
            proceeds, exit_fee = payout, 0.0     # settlement is not a trade
            account.cash += payout
        else:
            price = position.entry_price if exit_price is None else exit_price
            proceeds = position.shares * price
            exit_fee = position.shares * self.coefficient * price * (1.0 - price)
            account.cash += proceeds - exit_fee

        pnl = proceeds - position.cost
        account.apply_pnl(
            pnl,
            gross=proceeds,
            fees=position.entry_fee + exit_fee,
            reason=reason,
        )

    # ------------------------------------------------------------- archiving
    def archive(self, path: str | None = None, *, now: float | None = None,
                force: bool = False) -> str | None:
        """Atomically write every account's state to shadow_metrics.csv."""
        moment = time.time() if now is None else now
        if not force and (moment - self.stats.last_archive_ts) < self.archive_interval_s:
            return None
        target = path or self.archive_path
        directory = os.path.dirname(os.path.abspath(target))
        os.makedirs(directory, exist_ok=True)

        rows = [account.as_row(now=moment) for account in self.accounts]
        rows.sort(key=lambda row: row["equity"], reverse=True)

        handle = tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", delete=False,
            dir=directory, prefix=".shadow_", suffix=".tmp",
        )
        try:
            import csv
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                # fsync is a durability nicety, not a correctness requirement:
                # os.replace below is what makes the write atomic. On some
                # filesystems/handles (notably redirected output on Windows) fsync
                # can block indefinitely, which would wedge the trading loop.
                pass
        finally:
            handle.close()
        os.replace(handle.name, target)          # atomic: no half-written file
        self.stats.archived += 1
        self.stats.last_archive_ts = moment
        return target

    # -------------------------------------------------------------- reporting
    def top(self, n: int | None = None) -> list[VirtualAccount]:
        ranked = sorted(self.accounts, key=lambda a: a.equity, reverse=True)
        return ranked[: (n or self.top_n)]

    def summary_lines(self, *, spot_price: float | None = None,
                      active_windows: int = 0) -> list[str]:
        best = self.top(self.top_n)
        active = sum(1 for a in self.accounts if a.positions)
        lines = [
            f"[{now_utc()}] BTC {spot_price if spot_price is None else f'{spot_price:,.1f}'}"
            f" | accounts {len(self.accounts)} | windows {active_windows}"
            f" | active {active} | entries {self.stats.entries}"
            f" | TP {self.stats.take_profits} SL {self.stats.stops}"
            f" | hedge {self.stats.hedges} | settle {self.stats.settlements}",
        ]
        for rank, account in enumerate(best, start=1):
            lines.append(
                f"    #{rank:<2} {account.params.combo_id} "
                f"equity {account.equity:>9,.2f} ({account.return_pct:+6.2f}%) "
                f"trades {account.trades:>4} win {account.win_rate * 100:>5.1f}% "
                f"hedge {account.hedges:>3} "
                f"[T-{account.params.time_s:g} thr{account.params.threshold_pct:g} "
                f"tp{account.params.tp:g} sl{account.params.sl:g}"
                f"{' dyn' if account.params.dynamic_tp_sl else ''}"
                f"{' h' + str(account.params.hedge_ratio) if account.params.hedge_enabled else ''}]"
            )
        return lines

    def log_summary(self, *, spot_price: float | None = None,
                    active_windows: int = 0) -> None:
        for line in self.summary_lines(spot_price=spot_price,
                                       active_windows=active_windows):
            self.log(line)


def build_engine(grid, *, starting_cash: float = 1_000.0,
                 notional: float | None = None, **kwargs) -> ShadowEngine:
    """Convenience: grid -> accounts -> engine.

    ``notional`` defaults to 1% of the per-account cash. The earlier default
    (100 on 1,000) was 10% of capital per window, which bankrupts most accounts
    after a few hundred trades purely through stake sizing - it looks like the
    strategy failing when it is really the bet size failing.
    """
    accounts = build_accounts(grid, starting_cash=starting_cash)
    if notional is None:
        notional = starting_cash * 0.01
    return ShadowEngine(accounts, notional=notional, **kwargs)

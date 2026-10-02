"""Module 2b - parameter selection: sweep the 1200-combination grid.

Reads ``data/param_combinations.csv``, backtests every row against the loaded
observations, ranks the results, and writes:

* ``sweep_results.csv``  - one row per combination (all metrics)
* ``sweep_ranking.json`` - ranked table + the winner's diagnostics
* ``sweep_trades_<id>.csv`` - trade log for the top N combinations

Ranking is metric-driven and explicit; there is no hidden "best" - the caller
picks the metric (``--rank-metric``), because a good parameter set for a
latency-sensitive sniper is not the same as a good one for a patient maker.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from core.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    ParamSet,
    load_param_sets,
    write_backtest_outputs,
)
from core.backtest_engine import MarketObservation
from utils.logger import get_logger, write_csv, write_json

log = get_logger("param_selector")

#: Metrics exposed for ranking, with the direction that counts as better.
RANKABLE: dict[str, str] = {
    "net_pnl": "max",
    "total_pnl": "max",
    "expectancy_per_trade": "max",
    "sharpe": "max",
    "sortino": "max",
    "profit_factor": "max",
    "win_rate": "max",
    "max_drawdown": "min",
    "total_fees": "min",
    "avg_slippage_bps": "min",
    "trades": "max",
}


class ParamSelectorError(RuntimeError):
    """Raised for bad sweep configuration."""


@dataclass
class SweepRow:
    """One parameter set's outcome, flattened for tabular output."""

    params: ParamSet
    metrics: dict[str, Any]

    def as_row(self) -> dict:
        row = self.params.as_dict()
        row.update({k: v for k, v in self.metrics.items() if k not in row})
        return row

    def as_dict(self) -> dict:
        return {"params": self.params.as_dict(), "metrics": self.metrics}


@dataclass
class SweepResult:
    """Everything a sweep produced."""

    rows: list[SweepRow] = field(default_factory=list)
    best: SweepRow | None = None
    rank_metric: str = "net_pnl"
    observations: int = 0
    combos: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trades_by_combo: dict[str, list] = field(default_factory=dict)

    def top(self, n: int = 10) -> list[SweepRow]:
        return self.rows[:n]

    def as_dict(self) -> dict:
        return {
            "rank_metric": self.rank_metric,
            "combos": self.combos,
            "observations": self.observations,
            "best": self.best.as_dict() if self.best else None,
            "top": [row.as_dict() for row in self.rows[:25]],
            "diagnostics": self.diagnostics,
        }

    def to_csv_rows(self) -> list[dict]:
        return [row.as_row() for row in self.rows]


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def rank_rows(rows: Iterable[SweepRow], metric: str) -> list[SweepRow]:
    """Sort rows best-first on ``metric``.

    ``None``/NaN values always sort last: a combination that produced no metric
    (usually because it never traded) must not win by accident.
    """
    if metric not in RANKABLE:
        raise ParamSelectorError(
            f"unknown rank metric {metric!r}; choose from {sorted(RANKABLE)}"
        )
    direction = RANKABLE[metric]
    rows = list(rows)

    def key(row: SweepRow) -> tuple[int, float]:
        value = _metric_value(row.metrics, metric)
        if value is None:
            return (1, 0.0)
        return (0, -value if direction == "max" else value)

    return sorted(rows, key=key)


def filter_rows(
    rows: Sequence[SweepRow],
    *,
    min_trades: int = 1,
    min_win_rate: float | None = None,
    max_drawdown: float | None = None,
) -> list[SweepRow]:
    """Apply the practical guards a raw metric ranking needs.

    A parameter set with 3 trades and a 100% win rate is noise, not an edge;
    ``min_trades`` is the cheapest way to say so.
    """
    kept: list[SweepRow] = []
    for row in rows:
        trades = _metric_value(row.metrics, "trades") or 0.0
        if trades < min_trades:
            continue
        if min_win_rate is not None:
            win_rate = _metric_value(row.metrics, "win_rate")
            if win_rate is None or win_rate < min_win_rate:
                continue
        if max_drawdown is not None:
            dd = _metric_value(row.metrics, "max_drawdown")
            if dd is not None and dd > max_drawdown:
                continue
        kept.append(row)
    return kept


class ParamSelector:
    """Runs the grid sweep and packages the result."""

    def __init__(
        self,
        engine: BacktestEngine | None = None,
        *,
        engine_kwargs: dict[str, Any] | None = None,
        rank_metric: str = "net_pnl",
        top_n: int = 25,
        min_trades: int = 1,
        progress_every: int = 100,
    ) -> None:
        if rank_metric not in RANKABLE:
            raise ParamSelectorError(
                f"unknown rank metric {rank_metric!r}; choose from {sorted(RANKABLE)}"
            )
        self.engine = engine or BacktestEngine(**(engine_kwargs or {}))
        self.rank_metric = rank_metric
        self.top_n = int(top_n)
        self.min_trades = int(min_trades)
        self.progress_every = int(progress_every)

    # -- loading ----------------------------------------------------------
    @staticmethod
    def load_params(
        path: str, *, limit: int | None = None, time_filter: Iterable[int] | None = None
    ) -> list[ParamSet]:
        params = load_param_sets(path, limit=limit)
        if time_filter is not None:
            wanted = {int(t) for t in time_filter}
            params = [p for p in params if p.time_s in wanted]
            if not params:
                raise ParamSelectorError(
                    f"no parameter rows left after filtering to times {sorted(wanted)}"
                )
        return params

    # -- sweep ------------------------------------------------------------
    def sweep(
        self,
        params: Sequence[ParamSet],
        observations: Sequence[MarketObservation],
        *,
        keep_trades_for_top: bool = True,
        on_progress: Callable[[int, int, SweepRow], None] | None = None,
    ) -> SweepResult:
        if not params:
            raise ParamSelectorError("no parameter sets to sweep")
        if not observations:
            raise ParamSelectorError("no observations to sweep over")

        rows: list[SweepRow] = []
        trades_by_combo: dict[str, list] = {}
        total = len(params)

        for index, param_set in enumerate(params, start=1):
            result: BacktestResult = self.engine.run(param_set, observations)
            row = SweepRow(params=param_set, metrics=result.metrics.as_dict())
            rows.append(row)
            if keep_trades_for_top:
                trades_by_combo[param_set.combo_id] = result.trades

            if on_progress is not None:
                on_progress(index, total, row)
            elif self.progress_every and index % self.progress_every == 0:
                log.info("swept %d/%d combinations", index, total)

        eligible = filter_rows(rows, min_trades=self.min_trades)
        traded = [row for row in rows if (_metric_value(row.metrics, "trades") or 0) > 0]
        ranked = rank_rows(eligible or rows, self.rank_metric)
        ranked_ids = {row.params.combo_id for row in ranked[: self.top_n]}

        if keep_trades_for_top:
            trades_by_combo = {
                combo: trades
                for combo, trades in trades_by_combo.items()
                if combo in ranked_ids
            }

        offsets_seen = sorted({o.t_minus_s for o in observations})
        offsets_traded = sorted({p.time_s for p in params if p.time_s in set(offsets_seen)})

        return SweepResult(
            rows=ranked,
            best=ranked[0] if ranked and traded else None,
            rank_metric=self.rank_metric,
            observations=len(observations),
            combos=total,
            diagnostics={
                "eligible_after_min_trades": len(eligible),
                "min_trades": self.min_trades,
                "offsets_available": offsets_seen,
                "offsets_traded": offsets_traded,
                "combos_producing_trades": len(traded),
                "markets": len({o.market_id for o in observations}),
                "all_zero_reason": (
                    None
                    if traded
                    else (
                        "no parameter set produced a single trade. The most common "
                        "cause is that the dataset only carries prices at "
                        f"{offsets_seen} while the grid trades "
                        f"{sorted({p.time_s for p in params})}. Point --prices at a "
                        "dataset with real prices at the offsets you want to test, "
                        "or restrict the grid with --times."
                    )
                ),
            },
            trades_by_combo=trades_by_combo,
        )

    # -- output -----------------------------------------------------------
    def write_outputs(
        self,
        sweep: SweepResult,
        output_dir: str,
        *,
        write_top_trades: bool = True,
    ) -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths: dict[str, str] = {}

        paths["results_csv"] = write_csv(
            os.path.join(output_dir, "sweep_results.csv"), sweep.to_csv_rows()
        )
        paths["ranking_json"] = write_json(
            os.path.join(output_dir, "sweep_ranking.json"), sweep.as_dict()
        )

        if write_top_trades:
            for row in sweep.rows[: self.top_n]:
                combo = row.params.combo_id
                trades = sweep.trades_by_combo.get(combo) or []
                if not trades:
                    continue
                paths[f"trades_{combo}"] = write_csv(
                    os.path.join(output_dir, f"sweep_trades_{combo}.csv"),
                    [trade.as_dict() for trade in trades],
                )
        return paths


def print_ranking(sweep: SweepResult, limit: int = 10) -> None:
    """Human-readable leaderboard."""
    if not sweep.rows:
        print("no combinations produced results")
        return
    if not sweep.best:
        print("no combination produced a single trade - nothing to rank.")
        reason = sweep.diagnostics.get("all_zero_reason")
        if reason:
            print()
            print(textwrap.fill(reason, width=96))
        return
    header = (
        f"{'combo':>8} {'time':>5} {'thr':>6} {'tp':>5} {'sl':>5} {'gap':>4} "
        f"{'trades':>6} {'win%':>6} {'net$':>10} {'fees$':>8} {'maxDD':>8} {'sharpe':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in sweep.rows[:limit]:
        m = row.metrics
        sharpe = m.get("sharpe")
        print(
            f"{row.params.combo_id:>8} {row.params.time_s:>5} "
            f"{row.params.threshold_pct:>6.3f} {row.params.take_profit:>5.2f} "
            f"{row.params.stop_loss:>5.2f} {row.params.min_gap_s:>4} "
            f"{m.get('trades', 0):>6} {(m.get('win_rate') or 0) * 100:>6.1f} "
            f"{m.get('net_pnl', 0):>10.3f} {m.get('total_fees', 0):>8.3f} "
            f"{m.get('max_drawdown', 0):>8.3f} "
            f"{('-' if sharpe is None else f'{sharpe:.2f}'):>7}"
        )

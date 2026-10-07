"""Column contract for the V3 dataset, and the single point where time is parsed.

Dataset: ``polymarket_btc_up_down_5m_251218_260909.csv``
9 months of BTC 5-minute Up/Down windows, one row per window, 71,602 rows.

What this dataset has
---------------------
* ``btc_open`` / ``btc_close`` per window -> the realised return and therefore the
  label, plus a usable intra-window price path proxy
* pre-entry BTC features (multi-horizon returns, EMAs, RSI, MACD, Bollinger,
  ATR, stochastics, ROC, relative volume, VWAP distance)
* ``implied_up_last``: the market's final implied Up price
* ``trade_count``: how many trades the window saw

What it does **not** have
-------------------------
No token-level bid/ask and no per-trade token prints. That means entry price,
slippage and the intra-window TP/SL path cannot be *measured* - they can only be
*modelled*. Everything downstream flags which numbers are measured and which are
modelled.

Leakage rules enforced here
---------------------------
``implied_up_last`` is the **last** implied price of the window, i.e. close to
settlement, so it is post-decision information. It is loaded (for diagnostics and
for the efficiency benchmark) but it is never a signal input: :data:`SIGNAL_FEATURES`
is the only feature set the strategy may read, and it excludes it.

Likewise ``btc_return``, ``btc_close``, ``label_up``, ``winning_outcome`` and
``final_price`` describe the *outcome* of the window. They are labels or
diagnostics, never inputs.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

WINDOW_SECONDS = 300
DEFAULT_CSV = os.path.join("data", "polymarket_btc_up_down_5m_251218_260909.csv")

#: features that are genuinely observable BEFORE the window opens.
#: These are the only inputs the strategy is allowed to use.
SIGNAL_FEATURES = (
    "btc_ret_15m", "btc_ret_1h", "btc_ret_4h", "btc_vol_1h",
    "ind_ema_9_dist", "ind_ema_12_dist", "ind_ema_26_dist", "ind_ema_50_dist",
    "ind_ema_100_dist", "ind_ema_200_dist",
    "ind_rsi_14", "ind_macd", "ind_macd_signal", "ind_macd_hist",
    "ind_bb_pctb", "ind_bb_bw", "ind_atr_pct", "ind_stoch_k", "ind_stoch_d",
    "ind_roc_5", "ind_roc_15", "ind_roc_60", "ind_rel_vol_60",
    "ind_vwap_24h_dist",
)

#: columns that describe the OUTCOME. Never feed these to a signal.
OUTCOME_COLUMNS = (
    "btc_close", "btc_return", "chainlink_return", "winning_outcome", "label_up",
    "final_price",
)

#: post-decision market information: diagnostic only
POST_DECISION_COLUMNS = ("implied_up_last", "price_to_beat")


@dataclass(frozen=True)
class Window:
    """One 5-minute market, with only numbers the strategy can justify using.

    ``open_px``/``close_px`` are the *realised* window endpoints. ``close_px`` is
    a label-side value, so it is only read when building labels - never as an
    input to a decision.
    """

    slug: str
    start_ts: float
    end_ts: float
    open_px: float
    close_px: float
    label_up: int
    trade_count: int
    implied_last: float | None
    features: dict[str, float]

    @property
    def ret(self) -> float:
        return (self.close_px / self.open_px - 1.0) if self.open_px else 0.0


def _to_epoch(value: str) -> float:
    """Parse an ISO timestamp as UTC.

    ``datetime.fromisoformat`` keeps the offset when one is present; the fallback
    pins UTC explicitly. Naive timestamps are the classic source of an 8-hour skew
    on a UTC+8 machine (this bit the v2 replay), so they are never left naive.
    """
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _f(value, default: float | None = None) -> float | None:
    if value is None or value == "" or value == "None":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load(path: str, *, limit: int | None = None,
         features: tuple[str, ...] = SIGNAL_FEATURES) -> list[Window]:
    """Read and validate the dataset, sorted by window start."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            "expected data/polymarket_btc_up_down_5m_251218_260909.csv"
        )
    windows: list[Window] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("slug", "start_ts", "end_ts", "label_up",
                               "btc_open", "btc_close") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            open_px = _f(row.get("btc_open"))
            close_px = _f(row.get("btc_close"))
            if not open_px or not close_px:
                continue
            windows.append(Window(
                slug=row["slug"],
                start_ts=_f(row.get("start_ts")) or _to_epoch(row["start_time"]),
                end_ts=_f(row.get("end_ts")) or _to_epoch(row["end_time"]),
                open_px=open_px,
                close_px=close_px,
                label_up=int(_f(row.get("label_up"), 0) or 0),
                trade_count=int(_f(row.get("trade_count"), 0) or 0),
                implied_last=_f(row.get("implied_up_last")),
                features={name: (_f(row.get(name), 0.0) or 0.0) for name in features},
            ))
    windows.sort(key=lambda w: w.start_ts)
    return windows


def iter_batches(windows: list[Window], size: int) -> Iterator[list[Window]]:
    for start in range(0, len(windows), size):
        yield windows[start:start + size]


def intraday_frequency(windows: list[Window]) -> dict:
    """How many windows exist per day - the ceiling on trade frequency.

    A 5-minute market means at most 288 windows/day. Any real strategy trades a
    fraction of them, and that fraction is a first-class output of the analysis
    because frequency is what turns a per-trade edge into (or out of) a daily one.
    """
    if not windows:
        return {"days": 0.0, "windows": 0, "windows_per_day": 0.0}
    span_days = max(1e-9, (windows[-1].end_ts - windows[0].start_ts) / 86_400.0)
    return {
        "days": span_days,
        "windows": len(windows),
        "windows_per_day": len(windows) / span_days,
    }

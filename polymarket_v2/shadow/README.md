# Shadow book — 1200 parameter sets, one thread, virtual money

Runs every row of `param_combinations.csv` as an independent virtual account over
one shared market feed. Single-threaded. No coroutine or thread is created per
parameter set, and nothing here can reach a funded wallet.

```
run_shadow.py               entry point (replay | live | selftest)
shadow/
├── markets.py              pre-parsed data: SpotRing, PrintSeries, MarketFrame
├── grid.py                 parameter grid + VirtualAccount / OpenPosition
├── engine.py               the single 1200-account loop, exits, hedging, archive
├── feeds.py                Binance spot + Polymarket websocket + parquet replay
└── selftest.py             20 assertions on the core behaviours
```

## Commands

```bash
# 1. prove the engine before trusting any number
python run_shadow.py --mode selftest

# 2. offline replay over your parquet history (no network)
python run_shadow.py --mode replay --data ../data --trades "2026-03-11 (1).parquet"

# 3. live shadow book (Binance + Polymarket, virtual money only)
python run_shadow.py --mode live --assets BTC ETH
```

Watch the archive:

```powershell
# PowerShell
Import-Csv shadow_metrics.csv | Sort-Object {[double]$_.equity} -Descending |
  Select-Object -First 15 param_id, equity, return_pct, trades, win_rate, hedges
```

```bash
# bash
sort -t, -k2 -gr shadow_metrics.csv | head -15
```

## How the 1200 accounts share one thread

Per cycle the engine does exactly this, with no branching per account beyond its
own parameters:

```
for frame in active_windows:
    _prepare(frame)                 # once per window, for all accounts:
        #   - spot move per entry checkpoint (8 lookups)
        #   - ATR over the last 60s
        #   - cutoff applied so a replay cannot read the future
    for t_minus in entry_checkpoints:
        for account in accounts_at(t_minus):     # <- the hot loop
            decide -> fill -> hedge/TP/SL/settle
```

Design rules that keep it fast and honest:

* **Everything expensive happens before the loop.** Timestamps are already
  floats; print series are sorted tuples with bisect lookup; token ids are
  resolved once per window; ATR and spot moves are computed per window, not per
  account. There is no `datetime`, `json`, `re` or f-string work in the loop.
* **Pre-parsed arrays, not objects.** `PrintSeries` holds two tuples; `SpotRing`
  is a fixed-capacity float ring with a cached bisect view.
* **Positions are keyed by slug.** One position per account per window is
  structural, not a policy check.
* **`cutoff` hides the future.** In replay, every read (spot and prints) stops at
  the simulated moment. `move_pct(future_ts)` returns exactly what
  `move_pct(cutoff_ts)` returns - asserted in the selftest.

## The two new dimensions

### `dynamic_tp_sl`

When true, both exits scale with realised ATR (mean absolute 1s return over the
last 60s) against a reference of `0.0002` (2 bps/s, a calm BTC minute):

```
scale = clamp(atr / 0.0002, 0.5, 2.5)
tp_price = entry + tp * scale
sl_price = entry - sl * scale
```

Quiet market → tighter exits; violent market → wider exits. The clamp stops a
single wild minute from producing absurd levels.

### `hedge_enabled` / `hedge_ratio`

On a stop touch, instead of closing at the stop:

* buy `hedge_ratio × shares` of the **opposite** token at the prevailing print,
* keep both legs to settlement, paying 1.0 on whichever side wins.

A hedged position is booked as **one** round trip at settlement, so trade counts
and win rates stay comparable with non-hedged accounts. If the account cannot
fund the hedge it falls back to the plain stop and records `stop_loss_unfunded`
rather than opening a phantom leg.

## What the replay actually found (2026-03-11, 1482 windows, 1200 params)

```
entries 211,043   TP 74,339   SL 17,533   hedge 57,918   settle 119,171
median account  -87.0%      profitable 16 / 1200      avg equity 272.59
hedge OFF  avg equity 354.65   |  hedge ON  avg equity 253.03
dynamic ON avg equity 275.41   |  dynamic OFF avg equity 269.82

best: p00491  +21.4%  (229 trades, 27% win, T-60, thr 0.04, tp 0.20, sl 0.02)
```

Ledger invariants hold on the full run: `entries == closed + open` and
`trades == entries - open`, both exact (0 discrepancy).

Read that carefully:

* **Most parameter sets lose.** At 1% of equity per trade, a few hundred trades
  is enough for the median account to bleed out. The profitable 16 are a tail,
  not a typical outcome - treat any single winner as overfitted until it
  survives an out-of-sample split.
* **Hedging did not help here.** Hedge-off accounts averaged *higher* equity than
  hedge-on ones (354.65 vs 253.03). Hedging converts a clean stop into two legs
  that both pay fees, and the sweep says the fee drag outweighed the saved stop.
  That is a measured result on this data, not a claim about hedging in general.
* **Dynamic TP/SL was roughly neutral** (275.41 vs 269.82). It changes *which*
  window gets which exit, not the average outcome.

## Tuning the experiment

| Flag | Default | Why you would change it |
|---|---|---|
| `--cash` | 1000 | virtual starting cash per account |
| `--notional` | 1% of cash | **the most consequential knob.** 10% per window bankrupts most accounts through stake sizing alone |
| `--archive-every` | 300 | archive interval, seconds (atomic rewrite) |
| `--limit` | all | run a subset of the grid for a quick look |

`shadow_metrics.csv` is rewritten **atomically** (temp file + `os.replace`), so an
interrupted run never leaves a half-written file. `os.fsync` is best-effort: it is
a durability nicety, and on some handles it can block indefinitely, which would
wedge the trading loop.

## Safety

There is no broker, no API key, no signing code and no order client in this
package. Cash is a Python float in a dataclass. `--mode live` subscribes to public
market data and records decisions; it cannot place anything.

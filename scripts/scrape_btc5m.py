"""Scrape the last N Polymarket BTC 5-minute ("Bitcoin Up or Down") markets.

For every market it collects:
  * timestamps (window start / close, UTC + epoch)
  * UP and DOWN token prices at a grid of observation points before the close
  * the settled result (UP won / DOWN won)

Outputs (into --outdir, default ./data):
  btc5m_markets.csv      one row per market (metadata + outcome)
  btc5m_prices.csv       one row per market x observation time (the main dataset)
  btc5m_raw.jsonl        the full per-second price series, one market per line
  btc5m_failed.json      markets that could not be collected (retry candidates)

Discovery modes
  slug    (default) build deterministic slugs: btc-updown-5m-{t_start}, t_start
                    floored to 300s. Walks back in time from --start.
  events  page through gamma /events, keep events whose slug matches
                    btc-updown-5m-*, then walk back with --before.

Rate limits: public endpoints, paced by --rps (default 6/s) with exponential
backoff on 429/5xx. Responses are cached under ./.cache so re-runs are cheap and
the scrape is resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from http_util import HAVE_REQUESTS, HttpClient, HttpError  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com"

WINDOW_SECONDS = 300
SLUG_RE = re.compile(r"^btc-updown-5m-(\d{10})$")

# observation offsets: seconds before the window close
DEFAULT_OFFSETS = [300, 270, 240, 210, 180, 150, 120, 90, 60, 45, 30, 15]

MARKET_FIELDS = [
    "market_id",
    "slug",
    "event_id",
    "question",
    "window_start_utc",
    "window_close_utc",
    "window_start_epoch",
    "window_close_epoch",
    "up_token_id",
    "down_token_id",
    "result",
    "outcome_up_price",
    "outcome_down_price",
    "volume_usd",
    "liquidity_usd",
    "btc_open",
    "btc_close",
    "btc_change_pct",
    "clob_points",
]

PRICE_FIELDS = [
    "market_id",
    "slug",
    "window_close_epoch",
    "window_close_utc",
    "t_minus_s",
    "ts",
    "ts_utc",
    "up_price",
    "down_price",
    "mid_sum",
    "up_bid",
    "up_ask",
    "down_bid",
    "down_ask",
    "spread",
    "btc_spot",
    "btc_offset_pct",
    "result",
]


# --------------------------------------------------------------------- helpers
def to_epoch(value) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonish(value):
    """Gamma sometimes returns JSON-encoded strings for list fields."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value]
    return value


class Cache:
    def __init__(self, root: str | None):
        self.root = root
        if root:
            os.makedirs(root, exist_ok=True)

    def path(self, url: str, params: dict | None) -> str | None:
        if not self.root:
            return None
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, digest + ".json")

    def get(self, url: str, params: dict | None):
        path = self.path(url, params)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, url: str, params: dict | None, payload) -> None:
        path = self.path(url, params)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError:
            pass


# ------------------------------------------------------------------- discovery
def fetch_event(client: HttpClient, cache: Cache, slug: str):
    params = {"slug": slug}
    cached = cache.get(f"{GAMMA}/events", params)
    if cached is not None:
        return cached
    data = client.get_json(f"{GAMMA}/events", params, allow_404=True)
    cache.put(f"{GAMMA}/events", params, data)
    return data


def discover_by_slug(client, cache, start_epoch: int, count: int, max_back: int, log):
    """Walk backwards one 5-minute window at a time."""
    found: list[tuple[int, dict]] = []
    cursor = (start_epoch // WINDOW_SECONDS) * WINDOW_SECONDS
    checked = 0
    while len(found) < count and checked < max_back:
        slug = f"btc-updown-5m-{cursor}"
        event = fetch_event(client, cache, slug)
        checked += 1
        if event:
            ev = event[0] if isinstance(event, list) else event
            found.append((cursor, ev))
            if len(found) % 25 == 0:
                log(f"  discovered {len(found)}/{count} (checked {checked} windows)")
        cursor -= WINDOW_SECONDS
    log(f"  discovery done: {len(found)} events from {checked} candidate windows")
    return found


def discover_by_events(client, cache, before_epoch: int, count: int, log):
    """Page gamma /events and keep btc-updown-5m-* events."""
    found: list[tuple[int, dict]] = []
    offset = 0
    while len(found) < count:
        params = {
            "limit": 100,
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
            "closed": "true",
        }
        if before_epoch:
            params["end_date_max"] = iso(before_epoch)
        key = f"{GAMMA}/events"
        batch = cache.get(key, params)
        if batch is None:
            batch = client.get_json(key, params)
            cache.put(key, params, batch)
        if not batch:
            break
        for ev in batch:
            match = SLUG_RE.match(str(ev.get("slug", "")))
            if not match:
                continue
            start = int(match.group(1))
            if before_epoch and start >= before_epoch:
                continue
            found.append((start, ev))
            if len(found) >= count:
                break
        offset += 100
        log(f"  paged offset={offset}, kept {len(found)}/{count}")
        if len(batch) < 100:
            break
    return found


# ---------------------------------------------------------------------- market
def pick_market(event: dict):
    markets = event.get("markets") or []
    for market in markets:
        outcomes = [str(o).lower() for o in jsonish(market.get("outcomes")) or []]
        if "up" in outcomes and "down" in outcomes:
            return market
    return markets[0] if markets else None


def parse_market(event: dict, slug: str, start_epoch: int):
    market = pick_market(event)
    if not market:
        return None
    outcomes = [str(o) for o in jsonish(market.get("outcomes")) or []]
    tokens = [str(t) for t in jsonish(market.get("clobTokenIds")) or []]
    prices = [as_float(p) for p in jsonish(market.get("outcomePrices")) or []]

    up_token = down_token = None
    up_price = down_price = None
    for idx, name in enumerate(outcomes):
        key = name.strip().lower()
        if idx < len(tokens):
            if key == "up":
                up_token = tokens[idx]
            elif key == "down":
                down_token = tokens[idx]
        if idx < len(prices):
            if key == "up":
                up_price = prices[idx]
            elif key == "down":
                down_price = prices[idx]

    if not up_token or not down_token:
        # fall back to positional assumption: first token = Up
        if len(tokens) >= 2:
            up_token = up_token or tokens[0]
            down_token = down_token or tokens[1]

    close_epoch = to_epoch(market.get("endDate")) or (start_epoch + WINDOW_SECONDS)
    return {
        "market_id": str(market.get("id") or market.get("conditionId") or slug),
        "slug": slug,
        "event_id": str(event.get("id") or ""),
        "question": str(market.get("question") or event.get("title") or ""),
        "window_start_epoch": start_epoch,
        "window_close_epoch": close_epoch,
        "up_token_id": up_token,
        "down_token_id": down_token,
        "closed": bool(market.get("closed")),
        "outcome_up_price": up_price,
        "outcome_down_price": down_price,
        "volume_usd": as_float(market.get("volumeNum") or market.get("volume"), 0.0),
        "liquidity_usd": as_float(market.get("liquidityNum") or market.get("liquidity"), 0.0),
    }


def resolve_result(meta: dict) -> str:
    up, down = meta.get("outcome_up_price"), meta.get("outcome_down_price")
    if up is None and down is None:
        return "UNKNOWN"
    up = up or 0.0
    down = down or 0.0
    if up >= 0.99 and down <= 0.01:
        return "UP"
    if down >= 0.99 and up <= 0.01:
        return "DOWN"
    if up > down:
        return "UP"
    if down > up:
        return "DOWN"
    return "UNKNOWN"


def price_history(client, cache, token_id: str, start_epoch: int, end_epoch: int):
    params = {"market": token_id, "startTs": start_epoch, "endTs": end_epoch, "fidelity": 1}
    key = f"{CLOB}/prices-history"
    data = cache.get(key, params)
    if data is None:
        try:
            data = client.get_json(key, params, allow_404=True)
        except HttpError:
            data = None
        cache.put(key, params, data)
    if not data:
        return []
    points = data.get("history") if isinstance(data, dict) else data
    out = []
    for point in points or []:
        ts = to_epoch(point.get("t"))
        price = as_float(point.get("p"))
        if ts is None or price is None:
            continue
        out.append((ts, price))
    out.sort(key=lambda item: item[0])
    return out


def nearest(points, target_epoch: int, tolerance: int):
    """Closest point to target within tolerance, else None."""
    best = None
    best_gap = None
    for ts, price in points:
        gap = abs(ts - target_epoch)
        if best_gap is None or gap < best_gap:
            best, best_gap = (ts, price), gap
    if best is None or best_gap is None or best_gap > tolerance:
        return None
    return best


def btc_spot_series(client, cache, open_epoch: int, close_epoch: int):
    params = {
        "symbol": "BTCUSDT",
        "interval": "1s" if close_epoch - open_epoch <= 3600 else "1m",
        "startTime": (open_epoch - 5) * 1000,
        "endTime": (close_epoch + 5) * 1000,
        "limit": 1000,
    }
    key = f"{BINANCE}/api/v3/klines"
    data = cache.get(key, params)
    if data is None:
        try:
            data = client.get_json(key, params, allow_404=True)
        except Exception:  # noqa: BLE001 - enrichment is best effort
            data = None
        cache.put(key, params, data)
    series = []
    for candle in data or []:
        try:
            series.append((int(candle[0]) // 1000, float(candle[4])))
        except (TypeError, ValueError, IndexError):
            continue
    series.sort()
    return series


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=500, help="how many markets to collect")
    ap.add_argument("--mode", choices=["slug", "events"], default="slug")
    ap.add_argument("--start", type=int, default=0, help="start epoch (default: last closed window)")
    ap.add_argument("--before", type=int, default=0, help="events mode: only windows starting before this epoch")
    ap.add_argument("--max-back", type=int, default=0, help="slug mode: max windows to probe (default count*4)")
    ap.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rps", type=float, default=6.0)
    ap.add_argument("--tolerance", type=int, default=12, help="seconds of slack when matching price points")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-btc", action="store_true", help="skip Binance spot enrichment")
    ap.add_argument("--env-file", default="", help="unused placeholder for future auth")
    args = ap.parse_args()

    lock = threading.Lock()

    def log(message: str) -> None:
        with lock:
            print(message, flush=True)

    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()}, reverse=True)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    client = HttpClient(rps=args.rps)
    cache = Cache(None if args.no_cache else args.cache_dir)

    log(f"http backend: {'requests' if HAVE_REQUESTS else 'stdlib urllib'}")
    log(f"offsets     : {offsets}")

    now = int(time.time())
    if args.start:
        start_epoch = args.start
    else:
        start_epoch = (now // WINDOW_SECONDS) * WINDOW_SECONDS - WINDOW_SECONDS

    log(f"discovery mode: {args.mode} (start={start_epoch} {iso(start_epoch)})")
    if args.mode == "slug":
        max_back = args.max_back or args.count * 4
        events = discover_by_slug(client, cache, start_epoch, args.count, max_back, log)
    else:
        events = discover_by_events(client, cache, args.before, args.count, log)

    if not events:
        log("no events discovered - check connectivity or --start")
        return 2

    events.sort(key=lambda item: item[0], reverse=True)
    log(f"parsing {len(events)} events")

    market_rows: list[dict] = []
    price_rows: list[dict] = []
    raw_lines: list[dict] = []
    failures: list[dict] = []

    def work(item):
        start, event = item
        slug = str(event.get("slug") or f"btc-updown-5m-{start}")
        meta = parse_market(event, slug, start)
        if not meta:
            return {"error": "no market in event", "slug": slug}
        result = resolve_result(meta)
        close_epoch = meta["window_close_epoch"]

        up_hist = price_history(client, cache, meta["up_token_id"], start, close_epoch)
        down_hist = price_history(client, cache, meta["down_token_id"], start, close_epoch)
        if not up_hist and not down_hist:
            return {"error": "empty price history", "slug": slug, "meta": meta}

        btc_series = [] if args.no_btc else btc_spot_series(client, cache, start, close_epoch)
        btc_open = nearest(btc_series, start, 90)
        btc_close = nearest(btc_series, close_epoch, 90)

        market_row = {
            "market_id": meta["market_id"],
            "slug": slug,
            "event_id": meta["event_id"],
            "question": meta["question"],
            "window_start_utc": iso(start),
            "window_close_utc": iso(close_epoch),
            "window_start_epoch": start,
            "window_close_epoch": close_epoch,
            "up_token_id": meta["up_token_id"],
            "down_token_id": meta["down_token_id"],
            "result": result,
            "outcome_up_price": meta["outcome_up_price"],
            "outcome_down_price": meta["outcome_down_price"],
            "volume_usd": meta["volume_usd"],
            "liquidity_usd": meta["liquidity_usd"],
            "btc_open": btc_open[1] if btc_open else "",
            "btc_close": btc_close[1] if btc_close else "",
            "btc_change_pct": (
                round((btc_close[1] / btc_open[1] - 1) * 100, 5)
                if btc_open and btc_close and btc_open[1]
                else ""
            ),
            "clob_points": len(up_hist),
        }

        rows = []
        for offset in offsets:
            target = close_epoch - offset
            up = nearest(up_hist, target, args.tolerance)
            down = nearest(down_hist, target, args.tolerance)
            if up is None and down is None:
                continue
            up_px = up[1] if up else None
            down_px = down[1] if down else None
            if up_px is not None and down_px is None:
                down_px = round(1.0 - up_px, 6)
            if down_px is not None and up_px is None:
                up_px = round(1.0 - down_px, 6)
            btc = nearest(btc_series, target, 90)
            spot = btc[1] if btc else None
            rows.append(
                {
                    "market_id": meta["market_id"],
                    "slug": slug,
                    "window_close_epoch": close_epoch,
                    "window_close_utc": iso(close_epoch),
                    "t_minus_s": offset,
                    "ts": up[0] if up else (down[0] if down else ""),
                    "ts_utc": iso(up[0] if up else down[0]) if (up or down) else "",
                    "up_price": up_px,
                    "down_price": down_px,
                    "mid_sum": round((up_px or 0) + (down_px or 0), 6),
                    "spread": "",
                    "btc_spot": spot if spot is not None else "",
                    "btc_offset_pct": (
                        round((spot / market_row["btc_open"] - 1) * 100, 5)
                        if spot is not None and market_row["btc_open"]
                        else ""
                    ),
                    "result": result,
                }
            )

        raw = {
            "slug": slug,
            "window_start_epoch": start,
            "window_close_epoch": close_epoch,
            "result": result,
            "up_token_id": meta["up_token_id"],
            "down_token_id": meta["down_token_id"],
            "up_history": up_hist,
            "down_history": down_hist,
            "btc_history": btc_series,
        }
        return {"market": market_row, "rows": rows, "raw": raw}

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, item): item for item in events}
        for future in as_completed(futures):
            item = futures[future]
            slug = str(item[1].get("slug") or "")
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"slug": slug, "error": f"{type(exc).__name__}: {exc}"})
                result = None
            if result is None:
                pass  # already recorded in `failures` above
            elif result.get("error"):
                failures.append({"slug": slug, "error": result["error"]})
            else:
                market_rows.append(result["market"])
                price_rows.extend(result["rows"])
                raw_lines.append(result["raw"])
            done += 1
            if done % 25 == 0 or done == len(events):
                log(f"  processed {done}/{len(events)} (ok={len(market_rows)}, failed={len(failures)})")

    market_rows.sort(key=lambda row: row["window_close_epoch"], reverse=True)
    price_rows.sort(key=lambda row: (row["window_close_epoch"], -row["t_minus_s"]), reverse=True)

    markets_path = os.path.join(outdir, "btc5m_markets.csv")
    with open(markets_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MARKET_FIELDS)
        writer.writeheader()
        writer.writerows(market_rows)

    prices_path = os.path.join(outdir, "btc5m_prices.csv")
    with open(prices_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRICE_FIELDS)
        writer.writeheader()
        writer.writerows(price_rows)

    raw_path = os.path.join(outdir, "btc5m_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fh:
        for line in raw_lines:
            fh.write(json.dumps(line) + "\n")

    failed_path = os.path.join(outdir, "btc5m_failed.json")
    with open(failed_path, "w", encoding="utf-8") as fh:
        json.dump(failures, fh, indent=2)

    up_count = sum(1 for row in market_rows if row["result"] == "UP")
    down_count = sum(1 for row in market_rows if row["result"] == "DOWN")
    unknown = sum(1 for row in market_rows if row["result"] == "UNKNOWN")

    print()
    print(f"markets collected : {len(market_rows)} (target {args.count})")
    print(f"  UP / DOWN       : {up_count} / {down_count} (unknown {unknown})")
    print(f"price rows        : {len(price_rows)}")
    print(f"failures          : {len(failures)}")
    print(f"markets CSV       : {markets_path}")
    print(f"prices CSV        : {prices_path}")
    print(f"raw JSONL         : {raw_path}")
    print(f"failures JSON     : {failed_path}")
    return 0 if market_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

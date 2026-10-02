"""Offline end-to-end test: run the scraper against a local mock of the three
upstream APIs, then validate the CSV output.

The mock mirrors the real payload shapes:
  GET /events?slug=...            gamma event with one Up/Down market
  GET /prices-history?market=...  clob history {"history": [{"t":..,"p":..}]}
  GET /api/v3/klines              binance 1s klines
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WINDOW = 300
TEST_START = 1_780_320_000  # divisible by 300; fixed for reproducibility
MISSING_WINDOWS = {TEST_START - 300 * 5, TEST_START - 300 * 11}  # simulated 404s


def make_market(start_epoch: int) -> dict:
    rng = random.Random(f"underlying-{start_epoch}")  # same seed as history_for
    up_wins = rng.random() < 0.5
    up_token = f"up{start_epoch}"
    down_token = f"down{start_epoch}"
    title = f"Bitcoin Up or Down - window {start_epoch}"
    return {
        "id": f"ev{start_epoch}",
        "slug": f"btc-updown-5m-{start_epoch}",
        "title": title,
        "markets": [
            {
                "id": f"mkt{start_epoch}",
                "question": title,
                "slug": f"btc-updown-5m-{start_epoch}",
                "outcomes": json.dumps(["Up", "Down"]),
                "outcomePrices": json.dumps(["1", "0"] if up_wins else ["0", "1"]),
                "clobTokenIds": json.dumps([up_token, down_token]),
                "endDate": datetime.fromtimestamp(
                    start_epoch + WINDOW, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "closed": True,
                "volumeNum": round(rng.uniform(1000, 90000), 2),
                "liquidityNum": round(rng.uniform(500, 20000), 2),
            }
        ],
    }


def history_for(token: str, start: int):
    """Settling series. Real Up/Down tokens are complementary (Up + Down = 1),
    so the Up series is authoritative and the Down series is its complement with
    a small, realistic bid/ask shave."""
    rng = random.Random(f"underlying-{start}")
    up_wins = rng.random() < 0.5
    settle = 1.0 if up_wins else 0.0
    points = []
    for step in range(WINDOW + 1):
        frac = step / WINDOW
        # drift converges on the settled value; noise shrinks as the close nears,
        # so the T-30s price reliably agrees with the outcome (as in a real market)
        drift = (settle - 0.5) * (frac**1.2)
        noise = rng.gauss(0, 0.035) * (1 - frac) ** 1.5
        up_price = min(0.998, max(0.002, 0.5 + drift + noise))
        if not token.startswith("up"):
            up_price = 1.0 - up_price
        points.append({"t": start + step, "p": round(up_price, 4)})
    return {"history": points}


def klines_for(start: int):
    rng = random.Random(f"btc-{start}")
    price = 80000 + rng.uniform(-2000, 2000)
    rows = []
    for step in range(WINDOW + 1):
        price *= 1 + rng.gauss(0, 0.00004)
        ts_ms = (start + step) * 1000
        rows.append([ts_ms, str(price), str(price), str(price), str(price), "1", ts_ms + 999, "0", 1, "0", "0", "0"])
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/events":
            slug = (query.get("slug") or [""])[0]
            try:
                start = int(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                return self._send(404, {"error": "bad slug"})
            if start in MISSING_WINDOWS:
                return self._send(404, {"error": "not found"})
            return self._send(200, [make_market(start)])

        if parsed.path == "/prices-history":
            token = (query.get("market") or [""])[0]
            start = int(query.get("startTs", ["0"])[0])
            if not token:
                return self._send(400, {"error": "missing market"})
            return self._send(200, history_for(token, start))

        if parsed.path == "/api/v3/klines":
            start = int(query.get("startTime", ["0"])[0]) // 1000
            return self._send(200, klines_for(start))

        return self._send(404, {"error": "unknown path"})


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    print(f"mock API listening on {base}")

    import http_util
    import scrape_btc5m

    scrape_btc5m.GAMMA = base
    scrape_btc5m.CLOB = base
    scrape_btc5m.BINANCE = base

    # keep all scratch paths inside the workspace: %TEMP% is outside the sandbox
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(root, ".work", "test-cache")
    outdir = os.path.join(root, ".work", "test-out")
    for path in (cache_dir, outdir):
        if os.path.isdir(path):
            for name in os.listdir(path):
                os.remove(os.path.join(path, name))
        os.makedirs(path, exist_ok=True)
    argv = [
        "scrape_btc5m.py",
        "--count", "24",
        "--start", str(TEST_START),
        "--max-back", "40",
        "--workers", "4",
        "--rps", "0",
        "--outdir", outdir,
        "--cache-dir", cache_dir,
    ]
    sys.argv = argv
    rc = scrape_btc5m.main()
    print(f"scraper exit code: {rc}")
    if rc != 0:
        return 1

    markets_path = os.path.join(outdir, "btc5m_markets.csv")
    prices_path = os.path.join(outdir, "btc5m_prices.csv")
    raw_path = os.path.join(outdir, "btc5m_raw.jsonl")

    with open(markets_path, encoding="utf-8") as fh:
        markets = list(csv.DictReader(fh))
    with open(prices_path, encoding="utf-8") as fh:
        prices = list(csv.DictReader(fh))
    with open(raw_path, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    check("markets collected == 24", len(markets) == 24, f"got {len(markets)}")
    check("no UNKNOWN results", all(m["result"] in ("UP", "DOWN") for m in markets),
          str(sorted({m["result"] for m in markets})))
    check("all rows have both token ids",
          all(m["up_token_id"] and m["down_token_id"] for m in markets))
    check("raw jsonl has one line per market", len(raw) == len(markets), f"{len(raw)} vs {len(markets)}")

    expected_rows = len(markets) * 12
    check("price rows == markets x 12 offsets", len(prices) == expected_rows,
          f"got {len(prices)}, expected {expected_rows}")

    offsets = sorted({int(r["t_minus_s"]) for r in prices}, reverse=True)
    check("offsets are the requested grid",
          offsets == [300, 270, 240, 210, 180, 150, 120, 90, 60, 45, 30, 15], str(offsets))

    bad_sum = [r for r in prices if abs(float(r["mid_sum"]) - 1.0) > 0.02]
    check("up+down prices sum to ~1", not bad_sum, f"{len(bad_sum)} outliers")

    # the settled result must match which side converged to 1 near the close
    late = [r for r in prices if int(r["t_minus_s"]) <= 30]
    mismatches = []
    for row in late:
        up = float(row["up_price"])
        implied = "UP" if up > 0.5 else "DOWN"
        if implied != row["result"]:
            mismatches.append((row["slug"], row["t_minus_s"], up, row["result"]))
    check("late price agrees with settled result", len(mismatches) <= len(late) * 0.15,
          f"{len(mismatches)}/{len(late)} disagree (expected some noise)")

    check("btc spot enrichment present",
          all(r["btc_spot"] not in ("", None) for r in prices[:50]),
          "missing btc_spot in first 50 rows")

    check("missing windows were skipped, not fatal", len(markets) == 24)

    failures = json.load(open(os.path.join(outdir, "btc5m_failed.json"), encoding="utf-8"))
    check("failures recorded separately", isinstance(failures, list), f"{len(failures)} entries")

    print()
    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))

    print()
    print(f"output dir: {outdir}")
    print(f"checks: {len(checks) - failed}/{len(checks)} passed")
    server.shutdown()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

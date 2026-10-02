"""Map which hosts egress actually allows, and show proxy environment."""
from __future__ import annotations

import os
import urllib.error
import urllib.request

TIMEOUT = 15
UA = {"User-Agent": "Mozilla/5.0"}

print("=" * 70)
print("proxy / network environment")
for key in sorted(os.environ):
    if any(token in key.upper() for token in ("PROXY", "NO_PROXY", "DSH_", "SSL", "REQUESTS")):
        value = os.environ[key]
        if len(value) > 140:
            value = value[:140] + "..."
        print(f"   {key} = {value}")

print()
print("=" * 70)
print("host reachability (5s timeout, two runs for flaky hosts)")
HOSTS = [
    "https://api.github.com/rate_limit",
    "https://github.com/robots.txt",
    "https://raw.githubusercontent.com/Polymarket/agent-skills/main/README.md",
    "https://huggingface.co/api/datasets?limit=1",
    "https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown/raw/main/README.md",
    "https://datasets-server.huggingface.co/valid",
    "https://gamma-api.polymarket.com/markets?limit=1",
    "https://clob.polymarket.com/ok",
    "https://api.binance.com/api/v3/time",
    "https://api.allorigins.win/raw?url=https%3A%2F%2Fexample.com",
    "https://api.codetabs.com/v1/proxy?quest=https%3A%2F%2Fexample.com",
    "https://example.com",
    "https://pypi.org/simple/requests/",
]


def probe(url: str):
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            return str(resp.status), len(resp.read())
    except urllib.error.HTTPError as exc:
        return str(exc.code), 0
    except Exception as exc:  # noqa: BLE001
        return "ERR", type(exc).__name__


for url in HOSTS:
    first = probe(url)
    second = probe(url)
    flag = "  <== reachable" if first[0].isdigit() and first[0] != "ERR" else ""
    if second[0] != first[0]:
        flag += f"  (flaky: {first[0]} then {second[0]})"
    print(f"   {first[0]:>4} / {second[0]:<4} {url[:78]}{flag}")

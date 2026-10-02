"""Probe read-only HTTPS relays that might be able to reach Polymarket's API
from this machine (direct sockets are blocked by the sandbox)."""
from __future__ import annotations

import re
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 30
TARGET = "https://gamma-api.polymarket.com/markets?limit=2&closed=true&order=endDate&ascending=false"
ENC = urllib.parse.quote(TARGET, safe="")

CANDIDATES = [
    ("allorigins-raw", f"https://api.allorigins.win/raw?url={ENC}"),
    ("allorigins-get", "https://api.allorigins.win/get?url=" + ENC),
    ("codetabs", f"https://api.codetabs.com/v1/proxy?quest={ENC}"),
    ("corsproxy.io", f"https://corsproxy.io/?url={ENC}"),
    ("thingproxy", "https://thingproxy.freeboard.io/fetch/" + TARGET),
    ("whateverorigin", "http://www.whateverorigin.org/get?url=" + ENC),
    ("textance", "https://r.jina.ai/" + TARGET),
    ("direct", TARGET),
]


def main() -> int:
    for name, url in CANDIDATES:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8", "replace")
            looks_like_data = bool(re.search(r'"question"|"clobTokenIds"|"slug"', body))
            print(f"[OK]   {name:16} HTTP {resp.status} {len(body):>8} chars  data={looks_like_data}")
            if looks_like_data:
                print("       sample: " + body[:300].replace("\n", " "))
        except urllib.error.HTTPError as exc:
            print(f"[HTTP] {name:16} {exc.code} {exc.reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name:16} {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

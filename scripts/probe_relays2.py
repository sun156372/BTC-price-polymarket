"""Inspect what the allorigins relay actually returned, and test whether public
data hosts (Hugging Face, Kaggle) are reachable from this machine."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 30
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str):
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, (exc.read().decode("utf-8", "replace") if exc.fp else "")
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


TARGET = "https://gamma-api.polymarket.com/markets?limit=2"
relay = "https://api.allorigins.win/get?url=" + urllib.parse.quote(TARGET, safe="")

print("=" * 70)
print("1) allorigins body inspection")
status, body = fetch(relay)
print(f"   HTTP {status}, {len(body)} chars")
try:
    wrapper = json.loads(body)
    contents = wrapper.get("contents", "")
    print(f"   wrapper keys: {list(wrapper)[:6]}")
    print(f"   contents length: {len(contents)}")
    print("   contents head: " + contents[:400].replace("\n", " "))
except json.JSONDecodeError:
    print("   not JSON: " + body[:400].replace("\n", " "))

print()
print("=" * 70)
print("2) host reachability from the sandbox")
hosts = [
    ("huggingface.co README", "https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown/raw/main/README.md"),
    ("hf datasets-server", "https://datasets-server.huggingface.co/valid"),
    ("kaggle api", "https://www.kaggle.com/api/v1/datasets/list?search=polymarket"),
    ("github api", "https://api.github.com/rate_limit"),
]
for label, url in hosts:
    code, text = fetch(url)
    preview = text[:150].replace("\n", " ") if text else ""
    print(f"   [{str(code):>5}] {label:22} {preview}")

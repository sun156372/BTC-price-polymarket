"""Search hf-mirror.com for public Polymarket / BTC 5-minute datasets."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://hf-mirror.com"


def get_json(url: str):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def search(query: str, limit: int = 20):
    url = f"{BASE}/api/datasets?" + urllib.parse.urlencode(
        {"search": query, "limit": limit, "full": "false", "sort": "downloads", "direction": "-1"}
    )
    try:
        return get_json(url)
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code} for {query}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {exc}")
    return []


QUERIES = [
    "polymarket",
    "polymarket btc",
    "polymarket updown",
    "polymarket 5m",
    "crypto updown binary",
    "btc 5 minute polymarket",
    "prediction market orderbook",
]

seen: dict[str, dict] = {}
for query in QUERIES:
    results = search(query)
    print(f"query {query!r}: {len(results)} hits")
    for item in results:
        repo_id = item.get("id", "")
        if repo_id and repo_id not in seen:
            seen[repo_id] = item

print()
print("=" * 100)
print(f"{len(seen)} unique datasets")
print("=" * 100)
rows = sorted(seen.values(), key=lambda item: item.get("downloads") or 0, reverse=True)
for item in rows:
    repo_id = item.get("id", "")
    downloads = item.get("downloads") or 0
    likes = item.get("likes") or 0
    modified = (item.get("lastModified") or "")[:10]
    description = " ".join((item.get("description") or "").split())[:110]
    print(f"{downloads:>7} dl {likes:>3}L {modified}  {repo_id}")
    if description:
        print(f"{'':>22}  {description}")

with open("hf_search_results.json", "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)
print()
print("saved: hf_search_results.json")

"""List files in candidate hf-mirror datasets (size, name) so we can pick a source."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://hf-mirror.com"

CANDIDATES = [
    "Sigrex/polymarket_btc_up_down_5m_251218_260909",
    "Sigrex/polymarket_btc_up_down_1d_250314_260909",
    "whodisidk/polymarket-btc-updown-market-data",
    "polyorderbooks/polymarket-btc-5min-historical-l2",
    "bmoney1321/polymarket-crypto-5m-15m",
    "Mindbyte-89/polymarket-btc-updown",
    "marketlens/polymarket-btc-5m-l2-depth",
    "polyorderbooks/polymarket-crypto-5min-l2",
]


def tree(repo_id: str):
    url = f"{BASE}/api/datasets/{repo_id}/tree/main?recursive=true"
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=40) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


for repo_id in CANDIDATES:
    print("=" * 100)
    print(repo_id)
    data = tree(repo_id)
    if isinstance(data, dict) and data.get("error"):
        print("   " + data["error"])
        continue
    files = [item for item in data if item.get("type") == "file"]
    total = sum(item.get("size") or 0 for item in files)
    print(f"   {len(files)} files, {human(total)} total")
    for item in sorted(files, key=lambda x: -(x.get("size") or 0))[:14]:
        print(f"   {human(item.get('size') or 0):>9}  {item['path']}")

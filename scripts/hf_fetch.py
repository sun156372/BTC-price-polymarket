"""Resumable downloader for hf-mirror.com (supports HTTP Range, retries stalls).

The mirror throttles and occasionally drops the connection, so this keeps a .part
file, resumes with Range requests, and retries indefinitely until the byte count
matches Content-Length.

Usage:
  python scripts/hf_fetch.py <repo_id> <path_in_repo> <local_path>
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
CHUNK = 256 * 1024
READ_TIMEOUT = 30
MAX_ATTEMPTS = 200


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def total_size(url: str) -> int:
    request = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(request, timeout=READ_TIMEOUT) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length and length.isdigit() else 0


def main() -> int:
    if len(sys.argv) < 4:
        return sys.exit(__doc__)
    repo_id, path_in_repo, local_path = sys.argv[1:4]
    url = f"{BASE}/datasets/{repo_id}/resolve/main/{path_in_repo}"
    part = local_path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)

    expected = total_size(url)
    print(f"url      : {url}")
    print(f"expected : {human(expected) if expected else 'unknown'}")
    print(f"target   : {local_path}")
    print(flush=True)

    started = time.time()
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expected and have >= expected:
            break
        attempt += 1
        headers = dict(UA)
        mode = "wb"
        if have:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=READ_TIMEOUT) as resp:
                if have and resp.status != 206:
                    # server ignored Range: restart cleanly
                    have = 0
                    mode = "wb"
                with open(part, mode) as fh:
                    while True:
                        try:
                            chunk = resp.read(CHUNK)
                        except (TimeoutError, OSError):
                            break  # let the outer loop resume
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if have % (8 * 1024 * 1024) < CHUNK:
                            elapsed = max(0.001, time.time() - started)
                            pct = f"{have / expected * 100:5.1f}%" if expected else "  ?  "
                            print(f"  {pct}  {human(have)}  ({human(have / elapsed)}/s)  "
                                  f"attempt {attempt}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  attempt {attempt}: {type(exc).__name__} - retrying in 2s", flush=True)
            time.sleep(2)
            continue

        if expected and os.path.getsize(part) >= expected:
            break
        time.sleep(1)

    final = os.path.getsize(part) if os.path.exists(part) else 0
    elapsed = max(0.001, time.time() - started)
    if expected and final < expected:
        print(f"INCOMPLETE: {human(final)} of {human(expected)} after {attempt} attempts")
        return 3

    os.replace(part, local_path)
    with open(local_path, "rb") as fh:
        head = fh.read(200)
    print()
    print(f"downloaded : {human(final)} in {elapsed:.0f}s ({human(final / elapsed)}/s)")
    print(f"saved      : {local_path}")
    print(f"first bytes: {head[:120]!r}")
    if head.lstrip()[:1] == b"<":
        print("VERDICT: HTML, not data -> FAILED")
        return 2
    print("VERDICT: real data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

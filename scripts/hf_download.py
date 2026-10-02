"""Download a hf-mirror file with streaming + progress, verifying it is real data.

Usage:
  python scripts/hf_download.py <repo_id> <path_in_repo> <local_path>
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20  # 1 MiB
BASE = "https://hf-mirror.com"


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main() -> int:
    repo_id, path_in_repo, local_path = sys.argv[1], sys.argv[2], sys.argv[3]
    url = f"{BASE}/datasets/{repo_id}/resolve/main/{path_in_repo}"
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    started = time.time()
    written = 0
    head = b""
    last_report = 0.0

    with urllib.request.urlopen(request, timeout=60) as resp:
        declared = resp.headers.get("Content-Length")
        expect = int(declared) if declared and declared.isdigit() else 0
        final_url = resp.geturl()
        ctype = resp.headers.get("Content-Type", "")
        print(f"url      : {url}")
        print(f"resolved : {final_url}")
        print(f"type     : {ctype}")
        print(f"size     : {human(expect) if expect else 'unknown'}")
        print()

        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                if len(head) < 512:
                    head += chunk[: 512 - len(head)]
                fh.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report > 3:
                    last_report = now
                    speed = written / max(0.001, now - started)
                    pct = f"{written / expect * 100:5.1f}%" if expect else "  ?  "
                    print(f"  {pct}  {human(written)} / {human(expect) if expect else '?'}  "
                          f"({human(speed)}/s)", flush=True)
        os.replace(tmp_path, local_path)

    elapsed = time.time() - started
    print()
    print(f"downloaded : {human(written)} in {elapsed:.1f}s")
    print(f"saved      : {local_path}")

    # sanity: real data, not an HTML error page
    sample = head.lstrip()
    looks_html = sample[:1] == b"<"
    print(f"first bytes: {sample[:120]!r}")
    if looks_html:
        print("VERDICT: looks like HTML, not data -> FAILED")
        return 2
    print("VERDICT: looks like real data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

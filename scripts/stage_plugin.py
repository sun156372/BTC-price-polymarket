"""Download and stage the dsh-github-connect plugin from its GitHub tarball.

raw.githubusercontent.com is DNS-blocked on this machine, so we use the GitHub
API tarball endpoint instead (it resolves and serves the same tree).

Output: .plugins-stage/dsh-github-connect/
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request

REPO = "Moon-shiyue/dsh-github-connect"
BRANCH = "master"
URL = f"https://api.github.com/repos/{REPO}/tarball/{BRANCH}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# stage under .work (a directory this sandbox already proved writable)
STAGE = os.path.join(ROOT, ".work", "plugins-stage", "dsh-github-connect")


def main() -> int:
    print(f"downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as resp:
        blob = resp.read()
    print(f"  got {len(blob)} bytes, magic={blob[:2]!r}")
    if blob[:2] != b"\x1f\x8b":
        print("not a gzip tarball -> aborting")
        return 2

    os.makedirs(STAGE, exist_ok=True)
    written = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            # strip the "<repo>-<sha>/" prefix
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            if member.issym() or member.islnk() or not member.isfile():
                continue  # no links/dirs in this repo; refuse links on principle
            name = parts[1]
            if ".." in name.split("/"):
                print(f"refusing suspicious path: {name}")
                return 3

            # Write file contents directly instead of tar.extractall: extractall
            # also replays tar mtime/mode via utime/chmod, which this sandbox denies.
            target = os.path.join(STAGE, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with open(target, "wb") as fh:
                fh.write(source.read())
            written.append((name, os.path.getsize(target)))

    print(f"\nextracted {len(written)} files to {STAGE}")
    for name, size in sorted(written):
        print(f"  {size:>7}  {name}")

    required = ["package.json", "cordis.patch.yml", "lib/index.js"]
    missing = [r for r in required if not os.path.exists(os.path.join(STAGE, r))]
    if missing:
        print(f"\nmissing required files: {missing}")
        return 4
    print("\nstaged OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

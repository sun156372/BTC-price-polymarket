"""Extract a wheel into a clean directory and report whether it imports.

pip's own install cannot complete in this environment (the file sandbox denies the
metadata/attribute writes pip performs), but a plain zipfile extraction succeeds,
so this script is the reliable way to vendor a package here.

    python tools/vendor_wheel.py <wheel-path> <dest-dir>
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import zipfile


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    pattern, dest = argv[1], argv[2]
    wheels = glob.glob(pattern)
    if not wheels:
        print(f"no wheel matched: {pattern}")
        return 1
    wheel = wheels[0]
    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        archive.extractall(dest)

    package_root = None
    for candidate in ("pyarrow", "pandas"):
        path = os.path.join(dest, candidate)
        if os.path.isdir(path):
            package_root = candidate
            break

    print(f"wheel      : {os.path.basename(wheel)}")
    print(f"dest       : {dest}")
    print(f"members    : {len(members)}")
    if package_root:
        files = sum(len(f) for _root, _dirs, f in os.walk(os.path.join(dest, package_root)))
        print(f"package    : {package_root} ({files} files)")
    else:
        print("package    : could not identify a top-level package directory")

    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import pyarrow, pyarrow.parquet as pq; "
        "print('import OK:', pyarrow.__file__); print('ParquetFile:', pq.ParquetFile)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, dest], capture_output=True, text=True
    )
    print("--- import probe ---")
    print(result.stdout.strip() or result.stderr.strip()[-500:])
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

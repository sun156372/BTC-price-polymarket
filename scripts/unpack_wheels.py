"""Unpack downloaded wheels into ./.vendor without pip's install step.

pip's install phase rewrites file attributes in ways the DSH file sandbox denies,
so we unpack the wheel archives ourselves with zipfile.
"""
import glob
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEEL_DIR = os.path.join(ROOT, "_wheels")
VENDOR = os.path.join(ROOT, "vendor")

os.makedirs(VENDOR, exist_ok=True)
count = 0
for wheel in sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl"))):
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(VENDOR)
    count += 1
    print("unpacked", os.path.basename(wheel))

print(f"total wheels unpacked: {count} -> {VENDOR}")
sys.exit(0 if count else 1)

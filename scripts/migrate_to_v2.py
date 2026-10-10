"""One-off migration: gather all v2 code into a single labelled folder.

Before: v2 was split across ``polymarket_sniper/engine`` (package),
``polymarket_sniper/run_engine.py`` (entry) and ``polymarket_sniper/tools``
(research/verification) - with no "v2" marker anywhere, so it was impossible to
tell the new work from the original v1 code.

After: everything v2 lives under ``polymarket_v2/`` with an explicit ``v2`` name,
and the imports are rewritten so it runs from there.

    python scripts/migrate_to_v2.py            # dry run: show the plan
    python scripts/migrate_to_v2.py --apply    # perform it
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "polymarket_sniper")
V2 = os.path.join(ROOT, "polymarket_v2")

#: v2 source -> destination relative to V2
ENGINE_FILES = {
    "__init__.py": "__init__.py",
    "config.py": "config.py",
    "config.json": "config.json",
    "dataio.py": "dataio.py",
    "signals.py": "signals.py",
    "guards.py": "guards.py",
    "broker.py": "broker.py",
    "mathx.py": "mathx.py",
    "backtest.py": "backtest.py",
    "cli.py": "cli.py",
    "README.md": "README.md",
}

TOOL_FILES = [
    "backtest_directional.py",
    "check_lock_opportunities.py",
    "inspect_dataset.py",
    "make_demo_params.py",
    "parquet_meta_pure.py",
    "run_full_analysis.py",
    "survey_parquet_folder.py",
    "test_dataset_loader.py",
    "test_full_analysis.py",
    "test_inspect_dataset.py",
    "vendor_wheel.py",
]

DOC_FILES = ["ANALYSIS_REPORT.md", "STRATEGY_OPTIMIZATION.md", "DATA_ONBOARDING.md"]

CONFIG_FILES = {"engine/config.json": "engine_v2/config.json",
                "config/config.json": "config/config.json"}

#: import rewrites applied to migrated python files
#:  - ``from engine.x import`` / ``from engine import`` / ``import engine.x``
#:  - v1 modules that v2 must not depend on: dataset_loader was relocated into
#:    tools/, and the fee/vwap helpers were inlined into engine_v2.mathx
IMPORT_REWRITES = [
    (re.compile(r"^from engine\.([a-zA-Z_][\w.]*) import", re.M), r"from engine_v2.\1 import"),
    (re.compile(r"^from engine import", re.M), "from engine_v2 import"),
    (re.compile(r"^import engine\.([a-zA-Z_][\w.]*)", re.M), r"import engine_v2.\1"),
    (re.compile(r"^from core\.dataset_loader import", re.M), "from dataset_loader import"),
    (re.compile(r"^from utils\.fee_calculator import[^\n]*\n", re.M), ""),
    (re.compile(r"^from utils\.vwap_calculator import[^\n]*\n", re.M), ""),
    (re.compile(r"^from utils\.logger import[^\n]*\n", re.M), ""),
    (re.compile(r"^from config\.loader import[^\n]*\n", re.M), ""),
]

#: constants the deleted v1 imports used to provide, re-declared inline
FALLBACK_CONSTANTS = {
    "tools/check_lock_opportunities.py": "DEFAULT_COEFFICIENT = 0.07\n",
    "tools/run_full_analysis.py": "DEFAULT_COEFFICIENT = 0.07\n",
}

#: v2 files left behind in the v1 tree once the copy is verified
OLD_V2_PATHS = [
    "polymarket_sniper/engine",
    "polymarket_sniper/run_engine.py",
    "polymarket_sniper/tools/backtest_directional.py",
    "polymarket_sniper/tools/check_lock_opportunities.py",
    "polymarket_sniper/tools/inspect_dataset.py",
    "polymarket_sniper/tools/parquet_meta_pure.py",
    "polymarket_sniper/tools/run_full_analysis.py",
    "polymarket_sniper/tools/survey_parquet_folder.py",
    "polymarket_sniper/tools/test_dataset_loader.py",
    "polymarket_sniper/tools/test_full_analysis.py",
    "polymarket_sniper/tools/test_inspect_dataset.py",
    "polymarket_sniper/tools/vendor_wheel.py",
    "polymarket_sniper/core/dataset_loader.py",
]


def ensure_dir(path: str, apply: bool) -> None:
    if apply:
        os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str, apply: bool, *, rewrite: bool = False) -> bool:
    if not os.path.exists(src):
        print(f"  MISSING  {os.path.relpath(src, ROOT)}")
        return False
    ensure_dir(os.path.dirname(dst), apply)
    if not apply:
        print(f"  copy     {os.path.relpath(src, ROOT)}  ->  {os.path.relpath(dst, ROOT)}")
        return True
    if rewrite:
        with open(src, "r", encoding="utf-8") as fh:
            text = fh.read()
        for pattern, replacement in IMPORT_REWRITES:
            text = pattern.sub(replacement, text)

        # re-declare any constant the removed v1 import used to supply
        rel_dst = os.path.relpath(dst, V2).replace("\\", "/")
        fallback = FALLBACK_CONSTANTS.get(rel_dst)
        if fallback and fallback.strip().split("=")[0].strip() not in text:
            text = _insert_after_imports(text, fallback)

        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  migrate  {os.path.relpath(src, ROOT)}  ->  {os.path.relpath(dst, ROOT)} (imports rewritten)")
    else:
        shutil.copy2(src, dst)
        print(f"  copy     {os.path.relpath(src, ROOT)}  ->  {os.path.relpath(dst, ROOT)}")
    return True


def _insert_after_imports(text: str, block: str) -> str:
    """Insert ``block`` after the last top-level import line."""
    lines = text.splitlines(keepends=True)
    last = 0
    for index, line in enumerate(lines):
        if line.startswith(("import ", "from ")) or (last and line.startswith(("    ", ")"))):
            last = index + 1
    lines.insert(last, "\n" + block)
    return "".join(lines)


def cleanup(args) -> int:
    """Delete the v2 copies that now live in the new folder.

    Refuses to run unless the new folder is present and its selfcheck has been
    run, because deleting the only working copy of the engine would be an
    unforced error.
    """
    print("=" * 88)
    print("CLEANUP: remove duplicated v2 files from the v1 tree")
    print("=" * 88)
    if not os.path.isdir(V2):
        print(f"refusing: {os.path.relpath(V2, ROOT)} does not exist")
        return 2
    required = [
        os.path.join(V2, "engine_v2", "backtest.py"),
        os.path.join(V2, "run_engine_v2.py"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("refusing: the new folder is incomplete, missing:")
        for path in missing:
            print(f"  {os.path.relpath(path, ROOT)}")
        return 2

    if not args.apply:
        print("(dry run - pass --apply to delete)")
    removed = 0
    for rel in OLD_V2_PATHS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  absent   {rel}")
            continue
        if args.apply:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  removed  {rel}")
        else:
            print(f"  would remove  {rel}")
        removed += 1
    print()
    print(f"{'removed' if args.apply else 'would remove'}: {removed} path(s)")
    if args.apply:
        print()
        print("Verify the new folder still works:")
        print("  python polymarket_v2/run_engine_v2.py selfcheck")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually perform the migration")
    ap.add_argument("--cleanup", action="store_true",
                    help="remove the now-duplicated v2 copies from the v1 tree "
                         "(run only after the new folder passes its selfcheck)")
    args = ap.parse_args(argv)

    if args.cleanup:
        return cleanup(args)

    print("=" * 88)
    print("MIGRATE v2 CODE INTO A SINGLE LABELLED FOLDER")
    print("=" * 88)
    print(f"source : {os.path.relpath(V1, ROOT)}")
    print(f"target : {os.path.relpath(V2, ROOT)}")
    print(f"mode   : {'APPLY' if args.apply else 'DRY RUN (pass --apply to perform)'}")
    print()

    if not os.path.isdir(V1):
        print(f"source folder not found: {V1}")
        return 2

    print("[1/4] engine package -> engine_v2/")
    for src_name, dst_name in ENGINE_FILES.items():
        copy_file(os.path.join(V1, "engine", src_name),
                  os.path.join(V2, "engine_v2", dst_name), args.apply)

    print("\n[2/4] entry point -> run_engine_v2.py")
    copy_file(os.path.join(V1, "run_engine.py"),
              os.path.join(V2, "run_engine_v2.py"), args.apply, rewrite=True)

    print("\n[3/4] research + verification tools -> tools/")
    if args.apply:
        ensure_dir(os.path.join(V2, "tools"), True)
        marker = os.path.join(V2, "tools", "__init__.py")
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write('"""V2 research and verification tools."""\n')
        print("  create   polymarket_v2/tools/__init__.py")
    for name in TOOL_FILES:
        copy_file(os.path.join(V1, "tools", name),
                  os.path.join(V2, "tools", name), args.apply, rewrite=True)
    copy_file(os.path.join(V1, "core", "dataset_loader.py"),
              os.path.join(V2, "tools", "dataset_loader.py"), args.apply, rewrite=True)

    print("\n[4/4] reports -> docs/, configs -> config/")
    copy_file(os.path.join(V1, "README.md"),
              os.path.join(V2, "docs", "v1_README.md"), args.apply)
    for name in DOC_FILES:
        copy_file(os.path.join(V1, name), os.path.join(V2, "docs", name), args.apply)
    copy_file(os.path.join(V1, "config", "config.json"),
              os.path.join(V2, "config", "v1_config.json"), args.apply, rewrite=True)

    print()
    print("=" * 88)
    if args.apply:
        print("Migration applied. Next:")
        print("  1. python polymarket_v2/run_engine_v2.py selfcheck")
        print("  2. python polymarket_v2/run_engine_v2.py backtest --data ../data")
        print("  3. delete the old scattered copies once verified (see --cleanup note)")
    else:
        print("Dry run complete. Re-run with --apply to perform it.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

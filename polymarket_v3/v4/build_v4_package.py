"""Build the standalone V4 package from the research tree.

    python build_v4_package.py

Produces ./v4/ containing everything needed to run, verify and re-validate the
strategy: the engine, the live paper loop, the offline replay backtest, the
verification tools, the configuration, and the reports.

What is NOT copied: the 1.3 GB dataset, the caches built from it, the V3 strategy and
its research scripts, and the parameter-search scripts that were used to form the
hypothesis (those are deliberately excluded - re-running a search on the same data is
how a validated result gets destroyed).

Two files are GENERATED rather than copied:

* ``engine_v4/signals.py``   - a dependency-free replacement for the V3 signals
  module, which otherwise drags in ``Standardiser``/``score`` and a CSV reader.
* ``engine_v4/imbalance.py`` - extracted from a research probe script so the feature
  lives in the engine instead of in a test file.
"""
from __future__ import annotations

import io
import os
import re
import sys

#: this script sits INSIDE v4/, so the research tree is one level up
PKG = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(PKG)

#: (destination relative to PKG, source relative to HERE)
FILES = [
    # ---- engine -----------------------------------------------------------
    ("engine_v4/strategy.py", "engine_v3/strategy_v4.py"),
    ("engine_v4/feed.py", "engine_v4/feed.py"),
    ("engine_v4/hf.py", "engine_v3/hf.py"),
    # ---- entry points -----------------------------------------------------
    ("run_strategy_v4.py", "run_v4.py"),
    ("run_live_v4.py", "run_live_v4.py"),
    # ---- tools: data access ----------------------------------------------
    ("tools_v4/slim.py", "tools_v3/slim.py"),
    ("tools_v4/build_cache.py", "tools_v3/build_slim_cache_v4.py"),
    ("tools_v4/parity_check.py", "tools_v3/parity_v4.py"),
    # ---- tools: verification ---------------------------------------------
    ("tools_v4/review.py", "tools_v3/review_v4.py"),
    ("tools_v4/checkpoint_sweep.py", "tools_v3/checkpoint_sweep_v4.py"),
    ("tools_v4/imbalance_validate.py", "tools_v3/imbalance_validate_v4.py"),
    ("tools_v4/imbalance_final.py", "tools_v3/imbalance_final_v4.py"),
    ("tools_v4/cap_sweep.py", "tools_v3/cap_optimal_v4.py"),
    # ---- tools: diagnostics ----------------------------------------------
    ("tools_v4/latency.py", "tools_v3/latency_v4.py"),
    ("tools_v4/paper_horizon.py", "tools_v3/paper_horizon_v4.py"),
    ("tools_v4/baseline_check.py", "tools_v3/baseline_check_v4.py"),
    ("tools_v4/fee_sensitivity.py", "tools_v3/fee_sensitivity_v4.py"),
    # ---- tools: offline replay -------------------------------------------
    ("tools_v4/backtest.py", "tools_v3/backtest_v4.py"),
    # ---- docs -------------------------------------------------------------
    ("README.md", None),
    ("STRATEGY_EXPLAINED.md", "V4_STRATEGY_EXPLAINED.md"),
    ("VALIDATION_REPORT.md", "V4_OOS_REPORT.md"),
]

#: generated files that must already exist (never overwritten by this script)
GENERATED = ("engine_v4/signals.py", "engine_v4/imbalance.py")

#: import and path rewrites, applied to every copied text file
REWRITES = [
    (r"from engine_v3\.strategy_v4 import", "from engine_v4.strategy import"),
    (r"from engine_v3\.signals import", "from engine_v4.signals import"),
    (r"from engine_v3 import hf", "from engine_v4 import hf"),
    (r"from imbalance_probe_v4 import token_imbalance",
     "from engine_v4.imbalance import token_imbalance"),
    (r"import backtest_v4 as B", "import backtest as B"),
    (r"\bbacktest_v4\b", "backtest"),
    (r"import slim\b", "import slim"),
    (r"import engine_v3\.", "import engine_v4."),
    (r"\bengine_v3\.", "engine_v4."),
    (r"\btools_v3\.", "tools_v4."),
    (r"tools_v3/", "tools_v4/"),
    (r"tools_v3\\\\", "tools_v4\\\\"),
    (r'"tools_v3"', '"tools_v4"'),
    (r"'tools_v3'", "'tools_v4'"),
    (r"/tools_v3\b", "/tools_v4"),
    # renamed during packaging
    (r"tools_v4/build_slim_cache_v4\.py", "tools_v4/build_cache.py"),
    (r"tools_v4/parity_v4\.py", "tools_v4/parity_check.py"),
    (r"tools_v4/review_v4\.py", "tools_v4/review.py"),
    (r"tools_v4/checkpoint_sweep_v4\.py", "tools_v4/checkpoint_sweep.py"),
    (r"tools_v4/imbalance_validate_v4\.py", "tools_v4/imbalance_validate.py"),
    (r"tools_v4/imbalance_final_v4\.py", "tools_v4/imbalance_final.py"),
    (r"tools_v4/imbalance_filter_v4\.py", "tools_v4/imbalance_validate.py"),
    (r"tools_v4/cap_optimal_v4\.py", "tools_v4/cap_sweep.py"),
    (r"tools_v4/latency_v4\.py", "tools_v4/latency.py"),
    (r"tools_v4/paper_horizon_v4\.py", "tools_v4/paper_horizon.py"),
    (r"tools_v4/baseline_check_v4\.py", "tools_v4/baseline_check.py"),
    (r"tools_v4/fee_sensitivity_v4\.py", "tools_v4/fee_sensitivity.py"),
    (r"tools_v4/backtest_v4\.py", "tools_v4/backtest.py"),
    (r"tools_v4/oos\w*\.py", "tools_v4/checkpoint_sweep.py"),
    (r"tools_v4/imbalance_probe_v4\.py", "engine_v4/imbalance.py"),
    (r"V4_OOS_REPORT\.md", "VALIDATION_REPORT.md"),
    (r"V4_STRATEGY_EXPLAINED\.md", "STRATEGY_EXPLAINED.md"),
    (r"V3_ANALYSIS\.md", "the research tree (not shipped)"),
    (r"V3_README\.md", "the research tree (not shipped)"),
]


def rewrite(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


#: The research tools each located the dataset from their own file depth:
#:     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: That is correct at <workspace>/tools_v3/ and wrong at <v4>/tools_v4/, and it fails
#: QUIETLY - the tool starts, finds nothing, and prints "no data". Replace the whole
#: expression with the shared resolver instead.
ROOT_DERIVATION = re.compile(
    r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(\s*"
    r"os\.path\.abspath\(__file__\)\s*\)\)\)")

#: One tool hardcoded the dataset path as a raw Windows string, which makes the
#: package non-portable (and leaks a local username). Match any such literal.
HARDCODED_ROOT = re.compile(
    r'^ROOT\s*=\s*r?["\'][^"\']*[\\/]micro_v3["\']\s*$', re.M)

PATHS_IMPORT = "from paths import resolve_data_root  # noqa: E402"

#: The validation tools hardcoded the checkpoint they were written against. When the
#: shipped entry time later moved from T-120 to T-180, those tools kept validating
#: T-120 and printed numbers that no longer described the strategy. Bind each
#: hardcoded constant to the shipped parameters so the two cannot drift again.
PARAM_BINDINGS = [
    (re.compile(r"^CHECKPOINT\s*=\s*[\d.]+", re.M),
     "CHECKPOINT = V4Params().entry_t_minus"),
    (re.compile(r"^THRESHOLD\s*=\s*[\d.]+", re.M),
     "THRESHOLD = V4Params().threshold"),
    (re.compile(r"^DECLARED_FLOOR\s*=\s*[\d.]+", re.M),
     "DECLARED_FLOOR = V4Params().min_book_imbalance"),
    (re.compile(r"^SPOT_FLOOR\s*=\s*[\d.]+", re.M),
     "SPOT_FLOOR = V4Params().min_spot_move"),
    (re.compile(r"^CAP\s*=\s*[\d.]+", re.M),
     "CAP = V4Params().max_entry_price"),
]
PARAMS_IMPORT = "from engine_v4.strategy import V4Params  # noqa: E402"


def patch_params(text: str, dst_rel: str) -> str:
    """Bind hardcoded validation constants to the shipped V4Params.

    Note the guard is on the CONSTANT assignment, not on whether V4Params already
    appears in the file: a tool can import V4Params for its own use while still
    hardcoding the checkpoint it was written against, which is exactly the drift this
    patch exists to prevent.
    """
    if not dst_rel.startswith("tools_v4/"):
        return text
    if not any(pattern.search(text) for pattern, _ in PARAM_BINDINGS):
        return text
    for pattern, replacement in PARAM_BINDINGS:
        text = pattern.sub(replacement, text)
    # guard on the IMPORT STATEMENT, not on the name: the constants above now
    # reference V4Params(), so a name check would always pass and the import would
    # never be added.
    if "from engine_v4.strategy import V4Params" not in text:
        lines = text.split("\n")
        for i, line in enumerate(lines[:70]):
            if line.startswith("import slim") or line.startswith("from paths import"):
                lines.insert(i, PARAMS_IMPORT)
                break
        text = "\n".join(lines)
    return text


#: After the shim replaces the dirname expression, tools still do
#: ``os.path.join(<var>, "micro_v3")``, which nests the name twice
#: (".../micro_v3/micro_v3") because the resolver already returns the dataset
#: directory. Collapse any single-identifier join to a direct resolution.
DOUBLE_NEST = re.compile(
    r'os\.path\.join\(\s*[A-Za-z_][A-Za-z_0-9]*\s*,\s*["\']micro_v3["\']\s*\)')


def patch_data_root(text: str, dst_rel: str) -> str:
    """Route data-root discovery through tools_v4/paths.py.

    Scoped to tools_v4/, which is where ``paths`` is importable from: these scripts
    already put their own directory on sys.path. Applying it to a top-level entry
    point would inject an import that cannot resolve.
    """
    if not dst_rel.startswith("tools_v4/") or dst_rel.endswith("paths.py"):
        return text
    changed = False
    if ROOT_DERIVATION.search(text):
        text = ROOT_DERIVATION.sub("_resolve_default_root()", text)
        changed = True
    if HARDCODED_ROOT.search(text):
        text = HARDCODED_ROOT.sub('ROOT = resolve_data_root()', text)
        changed = True
    if DOUBLE_NEST.search(text):
        text = DOUBLE_NEST.sub("resolve_data_root(required=False) or 'micro_v3'",
                               text)
        changed = True
    if not changed:
        return text
    lines = text.split("\n")
    insert_at = 0
    for i, line in enumerate(lines[:60]):
        if "sys.path.insert" in line:
            insert_at = i + 1
    shim = [
        "",
        "def _resolve_default_root() -> str:",
        "    \"\"\"Dataset location, resolved by tools_v4/paths.py (see its docstring).\"\"\"",
        "    return resolve_data_root(required=False) or 'micro_v3'",
        "",
    ]
    lines[insert_at:insert_at] = shim
    text = "\n".join(lines)
    if "from paths import resolve_data_root" not in text:
        lines = text.split("\n")
        for i, line in enumerate(lines[:60]):
            if "sys.path.insert" in line:
                lines.insert(i + 1, PATHS_IMPORT)
                break
        text = "\n".join(lines)
    return text


def main() -> int:
    for sub in ("engine_v4", "tools_v4", "config_v4", "paper_v4"):
        os.makedirs(os.path.join(PKG, sub), exist_ok=True)
        if sub in ("engine_v4", "tools_v4"):
            init = os.path.join(PKG, sub, "__init__.py")
            if not os.path.exists(init):
                io.open(init, "w", encoding="utf-8").write("")

    copied = 0
    missing = 0
    for dst_rel, src_rel in FILES:
        dst = os.path.join(PKG, dst_rel)
        if src_rel is None:
            print(f"  {'(edited in place)':<44} -> v4/{dst_rel}")
            continue
        src = os.path.join(HERE, src_rel)
        if not os.path.exists(src):
            print(f"  MISSING SOURCE: {src_rel}")
            missing += 1
            continue
        text = io.open(src, encoding="utf-8").read()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        text = patch_data_root(rewrite(text), dst_rel)
        text = patch_params(text, dst_rel)
        io.open(dst, "w", encoding="utf-8", newline="\n").write(text)
        copied += 1
        print(f"  {src_rel:<44} -> v4/{dst_rel}")

    print()
    for rel in GENERATED:
        path = os.path.join(PKG, rel)
        state = "present" if os.path.exists(path) else "MISSING"
        print(f"  generated: v4/{rel:<32} {state}")
        if state == "MISSING":
            missing += 1

    # ------------------------------------------------ self-check the package
    print()
    print("  checking the package for leftover references...")
    # this script legitimately names the research tree, since that is what it copies
    exempt = {os.path.abspath(__file__)}
    leftovers = []
    for dirpath, _dirnames, filenames in os.walk(PKG):
        if "__pycache__" in dirpath:
            continue
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if os.path.abspath(path) in exempt:
                continue
            text = io.open(path, encoding="utf-8").read()
            for pattern in ("engine_v3", "tools_v3", "backtest_v4",
                            "imbalance_probe_v4"):
                if pattern in text:
                    leftovers.append(f"{os.path.relpath(path, PKG)}: {pattern}")
    if leftovers:
        print("  LEFTOVER REFERENCES (imports may fail):")
        for item in leftovers:
            print(f"    {item}")
        missing += len(leftovers)
    else:
        print("  clean - no references to the research tree remain")

    print()
    print(f"  copied {copied} files, {len(GENERATED)} generated, "
          f"{len(FILES) - copied - sum(1 for d, s in FILES if s is None)} skipped")
    if missing:
        print(f"  {missing} problem(s) - package is INCOMPLETE")
        return 1
    print("  package build OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

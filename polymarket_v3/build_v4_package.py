"""Build the standalone v4 package from the research tree.

    python build_v4_package.py            # run from anywhere

Produces ``<research tree>/v4/`` containing everything needed to run, verify and
re-validate the strategy: the engine, the live paper loop, the offline replay
backtest, the verification tools, the configuration, and the reports.

The script locates the research tree by searching upward for ``engine_v3``, so it
keeps working whether it sits at the tree root or inside ``v4/``. An earlier version
assumed one specific depth and silently produced a half-empty package when moved.

Files are produced three ways:

* **copied** from the research tree, with imports and docstring paths rewritten
* **generated** inline below, for modules that must NOT be copies: ``signals.py``
  replaces the V3 signals module (which drags in Standardiser/score and a CSV
  reader) and ``imbalance.py`` lifts the core feature out of a research probe
* **left alone**: the README and the config, which are hand-maintained

Deliberately NOT copied: the 1.3 GB dataset, the caches built from it, the V3
strategy, and the parameter-search scripts used to form the hypothesis. Re-running a
parameter search on the same data is how a validated result gets destroyed.
"""
from __future__ import annotations

import io
import os
import re
import sys

#: how far up to look for the research tree
MAX_SEARCH_DEPTH = 4


def find_research_tree(start: str) -> str:
    """Walk up from ``start`` until a directory containing ``engine_v3`` is found."""
    current = os.path.abspath(start)
    for _ in range(MAX_SEARCH_DEPTH):
        if os.path.isdir(os.path.join(current, "engine_v3")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise SystemExit(
        f"cannot find the research tree (a directory containing engine_v3) within "
        f"{MAX_SEARCH_DEPTH} levels above {start}")


HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = find_research_tree(HERE)
PKG = os.path.join(RESEARCH, "v4")

#: (destination relative to PKG, source relative to RESEARCH)
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
    # ---- docs (hand-maintained; copied only if the source exists) ---------
    ("STRATEGY_EXPLAINED.md", "V4_STRATEGY_EXPLAINED.md"),
    ("VALIDATION_REPORT.md", "V4_OOS_REPORT.md"),
]

#: Hand-maintained sources. These live in ``v4_src/`` in the research tree so a
#: rebuild cannot destroy them - an earlier version kept them only inside v4/, and
#: deleting v4/ to test a clean build silently lost all four.
HAND_MAINTAINED = [
    ("README.md", "v4_src/README.md"),
    ("tools_v4/paths.py", "v4_src/paths.py"),
    ("tools_v4/static_check.py", "v4_src/static_check.py"),
    ("tools_v4/live_path_test.py", "v4_src/live_path_test.py"),
    ("config_v4/strategy_v4.json", "v4_src/strategy_v4.json"),
]

#: import and path rewrites applied to every copied text file
REWRITES = [
    (r"from engine_v3\.strategy_v4 import", "from engine_v4.strategy import"),
    (r"from engine_v3\.signals import", "from engine_v4.signals import"),
    (r"from engine_v3 import hf", "from engine_v4 import hf"),
    (r"from imbalance_probe_v4 import token_imbalance",
     "from engine_v4.imbalance import token_imbalance"),
    (r"import backtest_v4 as B", "import backtest as B"),
    (r"\bbacktest_v4\b", "backtest"),
    (r"import engine_v3\.", "import engine_v4."),
    (r"\bengine_v3\.", "engine_v4."),
    (r"\btools_v3\.", "tools_v4."),
    (r"tools_v3/", "tools_v4/"),
    (r'"tools_v3"', '"tools_v4"'),
    (r"'tools_v3'", "'tools_v4'"),
    (r"/tools_v3\b", "/tools_v4"),
    # renames that happened during packaging
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

#: The research tools located the dataset from their own file depth, which was
#: correct beside the dataset and wrong inside v4/tools_v4/. The failure was silent:
#: the tool started, found nothing, and printed "no data".
ROOT_DERIVATION = re.compile(
    r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(\s*"
    r"os\.path\.abspath\(__file__\)\s*\)\)\)")

#: One tool hardcoded the dataset as a raw Windows string, which is non-portable and
#: leaks a local username.
HARDCODED_ROOT = re.compile(
    r'^ROOT\s*=\s*r?["\'][^"\']*[\\/]micro_v3["\']\s*$', re.M)

#: After the shim, tools still join the dataset name onto an already-resolved root,
#: nesting it twice (".../micro_v3/micro_v3").
DOUBLE_NEST = re.compile(
    r'os\.path\.join\(\s*[A-Za-z_][A-Za-z_0-9]*\s*,\s*["\']micro_v3["\']\s*\)')

PATHS_IMPORT = "from paths import resolve_data_root  # noqa: E402"
PARAMS_IMPORT = "from engine_v4.strategy import V4Params  # noqa: E402"

#: Validation tools hardcoded the checkpoint they were written against. When the
#: shipped entry time moved from T-120 to T-180 they kept validating T-120 and
#: printed numbers that no longer described the strategy.
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


def rewrite(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def _insert_after_sys_path(text: str, lines_to_add: list[str]) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines[:70]):
        if "sys.path.insert" in line:
            lines[i + 1:i + 1] = lines_to_add
            return "\n".join(lines)
    return text


def patch_data_root(text: str, dst_rel: str) -> str:
    """Route data-root discovery through tools_v4/paths.py.

    Scoped to tools_v4/, which is where ``paths`` is importable from: those scripts
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
        text = HARDCODED_ROOT.sub("ROOT = resolve_data_root()", text)
        changed = True
    if DOUBLE_NEST.search(text):
        text = DOUBLE_NEST.sub("resolve_data_root(required=False) or 'micro_v3'",
                               text)
        changed = True
    if not changed:
        return text
    text = _insert_after_sys_path(text, [
        "",
        "def _resolve_default_root() -> str:",
        '    """Dataset location, resolved by tools_v4/paths.py (see its docstring)."""',
        "    return resolve_data_root(required=False) or 'micro_v3'",
        "",
    ])
    if "from paths import resolve_data_root" not in text:
        text = _insert_after_sys_path(text, [PATHS_IMPORT])
    return text


def patch_params(text: str, dst_rel: str) -> str:
    """Bind hardcoded validation constants to the shipped V4Params."""
    if not dst_rel.startswith("tools_v4/"):
        return text
    if not any(pattern.search(text) for pattern, _ in PARAM_BINDINGS):
        return text
    for pattern, replacement in PARAM_BINDINGS:
        text = pattern.sub(replacement, text)
    # guard on the IMPORT STATEMENT, not the name: the constants above now reference
    # V4Params(), so a name check would always pass and the import would never be added
    if "from engine_v4.strategy import V4Params" not in text:
        text = _insert_after_sys_path(text, [PARAMS_IMPORT])
    return text


GENERATED: dict[str, str] = {}

GENERATED["engine_v4/signals.py"] = '''"""Fee and break-even arithmetic for the shipped V4 strategy.

Deliberately tiny, with no dependencies - not even on the rest of this package. It
replaces the older ``signals`` module from the research tree, which carried the V3
signal machinery and a CSV data reader for classes the shipped strategy never
touches. Shipping dead code into a live trading package is how a live package
acquires bugs that nothing exercises.

Fee model, confirmed against the venue documentation::

    fee = C x feeRate x p x (1 - p)

with ``feeRate = 0.07`` for the Crypto category, charged to takers only. The fee is
symmetric in price (a 0.30 trade and a 0.70 trade cost the same dollars) and peaks
at p = 0.50. Settlement is not a trade, so no exit fee is charged.
"""
from __future__ import annotations

#: Crypto-category taker rate, as documented by the venue
DEFAULT_FEE_COEFFICIENT = 0.07


def taker_fee_per_share(price: float,
                        coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Taker fee per share: ``coefficient * p * (1 - p)``."""
    return coefficient * price * (1.0 - price)


def break_even_accuracy(entry_price: float,
                        coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Accuracy needed to break even at a given entry price.

    For a hold-to-settlement directional binary, ``accuracy * 1 = entry + fee``.
    Settlement is not a trade, so no exit fee is charged.
    """
    return min(1.0, entry_price + taker_fee_per_share(entry_price, coefficient))


def max_viable_entry(accuracy: float,
                     coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Highest entry price that still breaks even at this accuracy.

    Solves ``accuracy = p + c*p*(1-p)`` for p by bisection.
    """
    if accuracy >= 1.0:
        return 1.0
    low, high = 0.0, 0.9999
    for _ in range(80):
        mid = (low + high) / 2.0
        if break_even_accuracy(mid, coefficient) < accuracy:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def edge_points(accuracy: float, entry_price: float,
                coefficient: float = DEFAULT_FEE_COEFFICIENT) -> float:
    """Per-share edge of a directional bet, after fees, in probability points."""
    return accuracy - break_even_accuracy(entry_price, coefficient)
'''

GENERATED["engine_v4/imbalance.py"] = '''"""Token book imbalance - the feature carrying the V4 edge.

    from engine_v4.imbalance import token_imbalance, book_imbalance, orient

WHAT IT IS
----------
``full_depth_bid_notional`` and ``full_depth_ask_notional`` are ready-made scalar
columns in the dataset: the total notional resting on each side of the token book.
Their normalised difference is the feature::

    imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)

``+1`` means every dollar of resting liquidity is a bid; ``-1`` means every dollar is
an ask. Measured on 44 days / 12,670 markets, its relationship to winning is U-shaped::

    imbalance decile   1     2     3     4     5     6     7     8
    win rate (%)     93.6  93.1  79.5  67.9  70.3  75.1  93.3  94.4

Extremes are strong, the middle is weak - the mid buckets LOSE money badly, because
the entry is around 0.79 while the win rate is only about 70%. Hence the gate
``|imbalance| >= 0.30``, which removes the balanced-book trades.

ORIENTATION, WHICH IS EASY TO GET WRONG
---------------------------------------
The raw feature is in UP-token space. When the strategy buys DOWN the sign must be
FLIPPED, because a book that is bid-heavy on the UP token is ask-heavy on the DOWN
token. ``orient`` does that and is the only sanctioned way to do it. A gate on
``|imbalance|`` looks symmetric and would silently accept a sign error, so the
convention is asserted in ``tools_v4/review.py``.
"""
from __future__ import annotations

import pyarrow.parquet as pq

#: columns needed from the feature files
_NEEDED = ("market_id", "time_to_deadline_ms",
           "full_depth_bid_notional", "full_depth_ask_notional")


def book_imbalance(bid_notional: float | None,
                   ask_notional: float | None) -> float | None:
    """(bid - ask) / (bid + ask); None when the book is empty or unusable."""
    if bid_notional is None or ask_notional is None:
        return None
    total = bid_notional + ask_notional
    if total <= 0:
        return None
    return (bid_notional - ask_notional) / total


def orient(imbalance: float | None, side: str) -> float | None:
    """Flip the sign so it is always measured in favour of the side being bought."""
    if imbalance is None:
        return None
    return imbalance if side == "UP" else -imbalance


def token_imbalance(root: str, checkpoint_s: float = 120.0,
                    tolerance_ms: float = 1500.0) -> dict[str, float]:
    """Map ``market_id`` -> imbalance at the decision tick.

    Indexed by ``market_id`` because ``window_key`` is corrupted in this dataset (it
    holds the literal string ``"et"``), so joining on the obvious-looking key yields
    zero matches. The decision tick is the one nearest ``checkpoint_s`` seconds before
    the close, matching ``Market.tick_at_t_minus`` in the loader.

    Reads only four scalar columns, so it is far cheaper than a full market load.
    """
    from engine_v4 import hf

    out: dict[str, float] = {}
    target = checkpoint_s * 1000.0
    for path in hf.day_files(root, hf.FEATURES):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not all(c in names for c in _NEEDED):
            continue
        data = pq.read_table(path, columns=list(_NEEDED)).to_pydict()
        ids = data.get("market_id") or []
        if not ids:
            continue
        market_id = str(ids[0])
        deadline = data["time_to_deadline_ms"]
        bids = data["full_depth_bid_notional"]
        asks = data["full_depth_ask_notional"]
        best_i = None
        best_gap = None
        for i, value in enumerate(deadline):
            if value is None:
                continue
            gap = abs(value - target)
            if gap <= tolerance_ms and (best_gap is None or gap < best_gap):
                best_i, best_gap = i, gap
        if best_i is None:
            continue
        value = book_imbalance(bids[best_i], asks[best_i])
        if value is not None:
            out[market_id] = value
    return out
'''


def main() -> int:
    print(f"  research tree : {RESEARCH}")
    print(f"  package       : {PKG}")
    print()
    for sub in ("engine_v4", "tools_v4", "config_v4", "paper_v4"):
        os.makedirs(os.path.join(PKG, sub), exist_ok=True)
        if sub in ("engine_v4", "tools_v4"):
            init = os.path.join(PKG, sub, "__init__.py")
            if not os.path.exists(init):
                io.open(init, "w", encoding="utf-8").write("")

    problems = 0
    for dst_rel, src_rel in FILES:
        src = os.path.join(RESEARCH, src_rel)
        dst = os.path.join(PKG, dst_rel)
        if not os.path.exists(src):
            print(f"  MISSING SOURCE: {src_rel}")
            problems += 1
            continue
        text = io.open(src, encoding="utf-8").read()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        text = patch_data_root(rewrite(text), dst_rel)
        text = patch_params(text, dst_rel)
        io.open(dst, "w", encoding="utf-8", newline="\n").write(text)
        print(f"  {src_rel:<44} -> v4/{dst_rel}")

    print()
    for rel, body in GENERATED.items():
        path = os.path.join(PKG, rel)
        io.open(path, "w", encoding="utf-8", newline="\n").write(body)
        print(f"  {'(generated)':<44} -> v4/{rel}")

    print()
    for dst_rel, src_rel in HAND_MAINTAINED:
        src = os.path.join(RESEARCH, src_rel)
        dst = os.path.join(PKG, dst_rel)
        if not os.path.exists(src):
            print(f"  MISSING HAND-MAINTAINED SOURCE: {src_rel}")
            problems += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        text = io.open(src, encoding="utf-8").read()
        if dst_rel.endswith(".py"):
            text = patch_data_root(rewrite(text), dst_rel)
            text = patch_params(text, dst_rel)
        io.open(dst, "w", encoding="utf-8", newline="\n").write(text)
        print(f"  {src_rel:<44} -> v4/{dst_rel}")

    # ------------------------------------------------ self-check the package
    print()
    print("  checking for leftover references to the research tree...")
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
        print("  LEFTOVER REFERENCES:")
        for item in leftovers:
            print(f"    {item}")
        problems += len(leftovers)
    else:
        print("  clean")

    print()
    if problems:
        print(f"  {problems} problem(s) - package is INCOMPLETE")
        return 1
    print("  package build OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

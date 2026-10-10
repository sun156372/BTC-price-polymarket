"""Build the standalone v4 package from the working tree.

    python build_v4_package.py

Copies the modules the strategy actually needs into ./v4/ and rewrites their imports
so the package runs on its own. Deliberately NOT copied: the research tools, the
backtest sweepers, and the 1.3 GB data. What ships is the strategy, its data loader,
the live paper loop, and the config.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "v4")

#: (destination relative to PKG, source relative to HERE)
FILES = [
    ("engine_v4/strategy.py", "engine_v3/strategy_v4.py"),
    ("engine_v4/feed.py", "engine_v4/feed.py"),
    ("engine_v4/hf.py", "engine_v3/hf.py"),
    ("tools_v4/slim.py", "tools_v3/slim.py"),
    ("tools_v4/build_cache.py", "tools_v3/build_slim_cache_v4.py"),
    ("tools_v4/parity_check.py", "tools_v3/parity_v4.py"),
    ("run_strategy_v4.py", "run_v4.py"),
    ("run_live_v4.py", "run_live_v4.py"),
]
#: generated fresh rather than copied, to keep V3 machinery out of the shipped
#: package. See the file's docstring.
GENERATED = ("engine_v4/signals.py",)

#: import rewrites, applied to every copied file. The last few also fix docstring
#: paths so the shipped package does not point at the research tree it came from.
REWRITES = [
    (r"from engine_v3\.strategy_v4 import", "from engine_v4.strategy import"),
    (r"from engine_v3\.signals import", "from engine_v4.signals import"),
    (r"from engine_v3 import hf", "from engine_v4 import hf"),
    (r"from engine_v4\.feed import", "from engine_v4.feed import"),
    (r"import engine_v3\.", "import engine_v4."),
    (r"\bengine_v3\.", "engine_v4."),
    (r"\btools_v3\.", "tools_v4."),
    (r"tools_v3/", "tools_v4/"),
    (r'"tools_v3"', '"tools_v4"'),
    (r"'tools_v3'", "'tools_v4'"),
    (r"/tools_v3\b", "/tools_v4"),
    # file renames that happened during packaging
    (r"tools_v4/cap_optimal_v4\.py", "the research tree (not shipped)"),
    (r"tools_v4/checkpoint_sweep_v4\.py", "the research tree (not shipped)"),
    (r"tools_v4/imbalance_\w+\.py", "the research tree (not shipped)"),
    (r"tools_v4/oos\w*\.py", "the research tree (not shipped)"),
    (r"tools_v4/build_slim_cache_v4\.py", "tools_v4/build_cache.py"),
    (r"tools_v4/parity_v4\.py", "tools_v4/parity_check.py"),
]


def rewrite(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def main() -> int:
    os.makedirs(PKG, exist_ok=True)
    for sub in ("engine_v4", "tools_v4", "config_v4", "paper_v4"):
        os.makedirs(os.path.join(PKG, sub), exist_ok=True)
        init = os.path.join(PKG, sub, "__init__.py")
        if sub in ("engine_v4", "tools_v4") and not os.path.exists(init):
            io.open(init, "w", encoding="utf-8").write("")

    missing = 0
    for dst_rel, src_rel in FILES:
        src = os.path.join(HERE, src_rel)
        dst = os.path.join(PKG, dst_rel)
        if not os.path.exists(src):
            print(f"  MISSING SOURCE: {src_rel}")
            missing += 1
            continue
        text = io.open(src, encoding="utf-8").read()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        io.open(dst, "w", encoding="utf-8", newline="\n").write(rewrite(text))
        print(f"  {src_rel:<42} -> v4/{dst_rel}")

    for rel in GENERATED:
        path = os.path.join(PKG, rel)
        if os.path.exists(path):
            print(f"  {'(generated, preserved)':<42} -> v4/{rel}")
        else:
            print(f"  {'(generated)':<42} -> v4/{rel}  [MISSING - create it]")
            missing += 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

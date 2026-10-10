#!/usr/bin/env python
"""Entry point for the endgame momentum sniper engine.

    python run_engine.py selfcheck
    python run_engine.py backtest --data ../data

Kept at the project root so the engine runs from a clean checkout with no
installation step and no packaging ceremony.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_v2.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

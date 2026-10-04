"""Polymarket endgame momentum sniper - production engine.

Self-contained: no imports from the research tooling, no network access unless you
explicitly inject a live feed and broker.

    python run_engine.py selfcheck
    python run_engine.py backtest --data ../data
"""

__version__ = "2.0.0"

__all__ = ["__version__"]

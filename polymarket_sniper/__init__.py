"""Polymarket 5-minute last-second sniper system.

Pure local code: every module is offline by default. Live market access is
available only through transports that the caller injects explicitly (see
``core.market_data.RestSnapshotSource`` / ``OrderBookStream``).
"""

__version__ = "1.0.0"
__all__ = ["__version__"]

"""Shadow engine: 1200 parameter sets simulated in one thread over live data."""
from __future__ import annotations

from .engine import ShadowEngine, ShadowStats, build_engine
from .grid import (
    GridError,
    OpenPosition,
    ShadowParams,
    VirtualAccount,
    build_accounts,
    entry_times,
    load_grid,
)
from .markets import MarketBook, MarketFrame, PrintSeries, SpotRing

__all__ = [
    "ShadowEngine",
    "ShadowStats",
    "build_engine",
    "GridError",
    "OpenPosition",
    "ShadowParams",
    "VirtualAccount",
    "build_accounts",
    "entry_times",
    "load_grid",
    "MarketBook",
    "MarketFrame",
    "PrintSeries",
    "SpotRing",
]

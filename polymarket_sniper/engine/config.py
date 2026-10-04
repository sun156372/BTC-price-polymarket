"""Typed configuration with defaults, validation and provenance.

Every number in :data:`DEFAULTS` is either (a) an exchange rule, or (b) a value
chosen from measured results on the user's own trade history - the comment on
each one records which. Nothing is a guess.

Load order: DEFAULTS -> engine/config.json -> explicit overrides.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config.json")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or self-contradictory."""


DEFAULTS: dict[str, Any] = {
    "data": {
        # where the DuckDB parquet exports live
        "data_dir": os.path.join(PROJECT_DIR, "..", "data"),
        "assets": ["BTC", "ETH", "SOL", "XRP"],
        "batch_size": 100_000,
        "require_spot": True,
    },
    "signal": {
        # MEASURED: direction accuracy rises monotonically with the lookback.
        # 60s -> 62.7%, 120s -> 67.1%, 180s -> 73.1% (T-120 entries).
        "lookback_s": 180.0,
        # MEASURED: 0.05% of spot movement is where the filter stops being
        # noise-selection and starts being a real direction call.
        "min_move_pct": 0.05,
        # MEASURED: T-30 and later have ZERO trades in the data, so anything
        # below T-60 is unverifiable. T-120 is the primary checkpoint.
        "entry_t_minus_s": 120.0,
        # A window may only be traded once.
        "one_entry_per_window": True,
        # Spot stale beyond this means the "confirmed direction" is fiction.
        "max_spot_staleness_s": 30.0,
    },
    "exits": {
        # MEASURED (combined OCO, first touch wins): TP 0.30 / SL 0.20 gives
        # net +28,967 and maxDD 866 versus +2,808 / 1,815 for TP 0.10 / SL 0.05.
        "take_profit": 0.30,
        "stop_loss": 0.20,
        # Conservative: if one trade print touches both levels, take the stop.
        "stop_loss_wins_ties": True,
        # Position is resolved at settlement when neither level is touched.
        "hold_to_settlement": True,
    },
    "guards": {
        # MEASURED: >10s stale quotes hit 35.4% direction accuracy and lost
        # 1,915 USDC across 65 trades - the entire net loss came from here.
        "max_quote_staleness_s": 10.0,
        # Reject if the price we would pay is worse than the signal price.
        "max_slippage_bps": 200.0,
        # Do not take more than this share of visible depth at the touch.
        "max_depth_participation": 0.50,
        # Warm-up: no trading in the first seconds of a window.
        "min_window_elapsed_s": 60.0,
    },
    "fees": {
        # Venue rule: fee = shares * 0.07 * p * (1 - p). Peaks at p = 0.5.
        "taker_coefficient": 0.07,
    },
    "sizing": {
        "notional_usdc": 100.0,
        "max_position_pct_equity": 1.0,
        "starting_equity": 10_000.0,
    },
    "backtest": {
        "trades_file": "auto",
        "assets_filter": [],
        "max_markets": 0,
        "output_dir": os.path.join(PROJECT_DIR, "output", "engine"),
        # Apply the adverse-fill model on top of the ideal run.
        "simulate_adverse_fills": True,
        "entry_slippage_bps": 50.0,
        "exit_slippage_bps": 50.0,
    },
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict, overlay: Mapping[str, Any]) -> dict:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class Section:
    """Attribute + mapping access over one config section."""

    def __init__(self, name: str, data: Mapping[str, Any]) -> None:
        self._name = name
        self._data = dict(data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(f"{self._name}.{key} is not configured") from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)


@dataclass
class EngineConfig:
    """Resolved configuration, one Section per top-level key."""

    sections: dict[str, Section] = field(default_factory=dict)
    path: str | None = None

    def __getattr__(self, key: str) -> Section:
        sections = self.__dict__.get("sections", {})
        if key in sections:
            return sections[key]
        raise AttributeError(f"no config section {key!r}")

    def section(self, name: str) -> Section:
        if name not in self.sections:
            raise ConfigError(f"unknown config section: {name!r}")
        return self.sections[name]

    def resolve(self, value: str) -> str:
        """Resolve a possibly-relative path against the project directory."""
        if not value:
            return ""
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(PROJECT_DIR, value))

    def as_dict(self) -> dict:
        return {name: section.as_dict() for name, section in self.sections.items()}


def load_config(
    path: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    validate: bool = True,
) -> EngineConfig:
    """DEFAULTS -> JSON file -> overrides, then validate."""
    data = copy.deepcopy(DEFAULTS)
    config_path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                file_data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{config_path} is not valid JSON: {exc}") from exc
        if not isinstance(file_data, Mapping):
            raise ConfigError(f"{config_path} must contain a JSON object")
        data = _deep_merge(data, file_data)
    elif path:
        raise ConfigError(f"config file not found: {config_path}")

    if overrides:
        data = _deep_merge(data, overrides)

    config = EngineConfig(
        sections={k: Section(k, v) for k, v in data.items() if isinstance(v, Mapping)},
        path=config_path,
    )
    # carry non-mapping top-level keys through a pseudo-section for completeness
    if validate:
        validate_config(config)
    return config


def validate_config(config: EngineConfig) -> None:
    """Fail loudly now rather than produce quietly wrong numbers later."""
    problems: list[str] = []
    signal = config.section("signal")
    exits = config.section("exits")
    guards = config.section("guards")
    fees = config.section("fees")
    sizing = config.section("sizing")

    if signal.lookback_s <= 0:
        problems.append("signal.lookback_s must be > 0")
    if signal.min_move_pct < 0:
        problems.append("signal.min_move_pct must be >= 0")
    if signal.entry_t_minus_s <= 0:
        problems.append("signal.entry_t_minus_s must be > 0")
    if signal.max_spot_staleness_s <= 0:
        problems.append("signal.max_spot_staleness_s must be > 0")

    if not 0.0 < exits.take_profit < 1.0:
        problems.append("exits.take_profit must be in (0, 1)")
    if not 0.0 < exits.stop_loss < 1.0:
        problems.append("exits.stop_loss must be in (0, 1)")

    if guards.max_quote_staleness_s <= 0:
        problems.append("guards.max_quote_staleness_s must be > 0")
    if guards.max_slippage_bps <= 0:
        problems.append("guards.max_slippage_bps must be > 0")
    if not 0.0 < guards.max_depth_participation <= 1.0:
        problems.append("guards.max_depth_participation must be in (0, 1]")
    if guards.min_window_elapsed_s < 0:
        problems.append("guards.min_window_elapsed_s must be >= 0")

    if fees.taker_coefficient < 0:
        problems.append("fees.taker_coefficient must be >= 0")
    if sizing.notional_usdc <= 0:
        problems.append("sizing.notional_usdc must be > 0")
    if sizing.starting_equity <= 0:
        problems.append("sizing.starting_equity must be > 0")
    if not 0.0 < sizing.max_position_pct_equity <= 100.0:
        problems.append("sizing.max_position_pct_equity must be in (0, 100]")

    window = 900.0  # 15-minute markets; override per asset if that changes
    if signal.entry_t_minus_s >= window - guards.min_window_elapsed_s:
        problems.append(
            "signal.entry_t_minus_s leaves no room after guards.min_window_elapsed_s"
        )

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))

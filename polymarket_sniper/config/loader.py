"""Typed configuration loading.

The system keeps every tunable in ``config/config.json`` so behaviour can be
changed without touching code. This module turns that JSON into nested
attribute-style objects with defaults, and validates the values that would
otherwise fail silently (negative fees, inverted thresholds, bad hedge ratios).

    cfg = load_config()                     # config/config.json
    cfg = load_config("path/to/other.json") # explicit file
    cfg = load_config(overrides={"strategy": {"threshold_pct": 0.02}})
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable, Mapping

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, "config", "config.json")

#: Fallback values, used when a key is absent from the JSON file.
DEFAULTS: dict[str, Any] = {
    "profile": "backtest",
    "market": {
        "symbol": "BTCUSDT",
        "window_seconds": 300,
        "polymarket_series_slug": "btc-updown-5m",
        "gamma_rest_url": "https://gamma-api.polymarket.com",
        "clob_rest_url": "https://clob.polymarket.com",
        "clob_ws_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        "binance_rest_url": "https://api.binance.com",
        "binance_ws_url": "wss://stream.binance.com:9443/ws",
    },
    "credentials": {
        "polymarket_api_key": "",
        "polymarket_api_secret": "",
        "polymarket_api_passphrase": "",
        "polymarket_private_key": "",
        "polygon_rpc_url": "",
    },
    "strategy": {
        "scan_window_seconds": 10,
        "observation_time_s": 30,
        "threshold_pct": 0.05,
        "take_profit": 0.10,
        "stop_loss": 0.05,
        "min_entry_gap_s": 30,
        "entry_side_mode": "both",
        "spot_confirm_pct": 0.02,
        "probability_gap_min": 0.03,
        "probability_gap_max": 0.25,
        "max_entry_price": 0.92,
        "min_entry_price": 0.08,
    },
    "execution": {
        "default_order_notional": 100.0,
        "slippage_tolerance": 0.01,
        "max_slippage_bps": 150.0,
        "maker_first": True,
        "maker_timeout_s": 2.0,
        "split_order_threshold_notional": 50.0,
        "split_participation_rate": 0.25,
        "min_split_children": 2,
        "max_split_children": 6,
        "passive_price_improve_ticks": 1,
        "tick_size": 0.001,
    },
    "risk": {
        "max_position_notional": 500.0,
        "max_window_notional": 1500.0,
        "stop_loss_prob": 0.05,
        "trailing_stop_prob": 0.04,
        "obi_extreme_abs": 0.85,
        "obi_flip_abs": 0.60,
        "volatility_stop_mult": 2.5,
        "hedge_enabled": True,
        "hedge_ratio": 0.60,
        "max_hedge_ratio": 1.0,
        "hedge_min_signal_strength": 0.55,
    },
    "fees": {"taker_fee_coefficient": 0.07, "maker_rebate_share": 0.25, "min_tick": 0.001},
    "backtest": {
        "params_csv": "data/param_combinations.csv",
        "observations_csv": "",
        "raw_jsonl": "",
        "output_dir": "output",
        "sweep_workers": 1,
        "rank_metric": "total_pnl",
        "top_n": 25,
        "use_full_book": False,
        "book_levels": 5,
        "fee_curve": "documented",
        "slippage_model": "vwap",
    },
    "logging": {"level": "INFO", "log_file": "output/sniper.log"},
}


class ConfigError(ValueError):
    """Raised when configuration values are impossible or contradictory."""


class Section(Mapping):
    """Read-only attribute/mapping view over one config section."""

    def __init__(self, name: str, data: Mapping[str, Any]) -> None:
        self._name = name
        self._data = dict(data)

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access -------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"{self._name}.{key} is not configured") from exc

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Section({self._name!r}, {len(self._data)} keys)"


class Config:
    """Root config object with attribute access to sections."""

    def __init__(self, data: Mapping[str, Any], path: str | None = None) -> None:
        self._sections: dict[str, Section] = {}
        self._raw = copy.deepcopy(dict(data))
        self.path = path
        for key, value in self._raw.items():
            if isinstance(value, Mapping):
                self._sections[key] = Section(key, value)

    # -- section access ---------------------------------------------------
    def section(self, name: str) -> Section:
        if name not in self._sections:
            raise ConfigError(f"unknown config section: {name!r}")
        return self._sections[name]

    def __getattr__(self, key: str) -> Any:
        sections = self.__dict__.get("_sections", {})
        if key in sections:
            return sections[key]
        raw = self.__dict__.get("_raw", {})
        if key in raw:
            return raw[key]
        raise AttributeError(f"no such config key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    def resolved_path(self, value: str) -> str:
        """Resolve a config path relative to the package root."""
        if not value:
            return ""
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(PACKAGE_ROOT, value))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._raw, indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config(profile={self._raw.get('profile')!r}, path={self.path!r})"


# --------------------------------------------------------------------- loading
def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _coerce_like(default: Any, value: Any) -> Any:
    """Keep JSON types honest (e.g. "10" -> 10.0 for a float field)."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, str) and value.strip():
            return int(float(value))
        return int(value)
    if isinstance(default, float):
        if isinstance(value, str) and value.strip():
            return float(value)
        return float(value)
    return value


def load_config(
    path: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    """Load config from JSON, merged over DEFAULTS, then validate."""
    config_path = path or DEFAULT_CONFIG_PATH
    data = copy.deepcopy(DEFAULTS)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            file_data = json.load(fh)
        if not isinstance(file_data, Mapping):
            raise ConfigError(f"{config_path} must contain a JSON object")
        # credentials block carries a documentation comment key; keep it harmless
        data = _deep_merge(data, file_data)
    elif path is not None:
        raise ConfigError(f"config file not found: {config_path}")

    if overrides:
        data = _deep_merge(data, overrides)

    # type-coerce against the defaults so "0.05" from a hand-edited file works
    for section, defaults in DEFAULTS.items():
        if section not in data or not isinstance(defaults, Mapping):
            continue
        if not isinstance(data[section], Mapping):
            raise ConfigError(f"config section {section!r} must be an object")
        for key, default in defaults.items():
            if key in data[section]:
                try:
                    data[section][key] = _coerce_like(default, data[section][key])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"{section}.{key} = {data[section][key]!r} is not a valid "
                        f"{type(default).__name__}"
                    ) from exc

    config = Config(data, config_path)
    if validate:
        validate_config(config)
    return config


# ------------------------------------------------------------------ validation
def validate_config(config: Config) -> None:
    """Fail fast on settings that would produce silently wrong results."""
    problems: list[str] = []

    strategy = config.section("strategy")
    if strategy.threshold_pct < 0:
        problems.append("strategy.threshold_pct must be >= 0")
    if not 0.0 < strategy.scan_window_seconds:
        problems.append("strategy.scan_window_seconds must be > 0")
    if strategy.observation_time_s <= 0:
        problems.append("strategy.observation_time_s must be > 0")
    if strategy.take_profit <= 0:
        problems.append("strategy.take_profit must be > 0")
    if strategy.stop_loss <= 0:
        problems.append("strategy.stop_loss must be > 0")
    if strategy.min_entry_gap_s < 0:
        problems.append("strategy.min_entry_gap_s must be >= 0")
    if strategy.entry_side_mode not in {"up", "down", "both"}:
        problems.append("strategy.entry_side_mode must be one of up/down/both")
    if not 0.0 < strategy.min_entry_price < strategy.max_entry_price < 1.0:
        problems.append(
            "strategy entry price bounds must satisfy 0 < min_entry_price < "
            "max_entry_price < 1"
        )
    if strategy.probability_gap_min < 0:
        problems.append("strategy.probability_gap_min must be >= 0")
    if strategy.probability_gap_max <= strategy.probability_gap_min:
        problems.append(
            "strategy.probability_gap_max must be greater than probability_gap_min"
        )

    execution = config.section("execution")
    if execution.slippage_tolerance < 0:
        problems.append("execution.slippage_tolerance must be >= 0 (it is a fraction)")
    if execution.max_slippage_bps <= 0:
        problems.append("execution.max_slippage_bps must be > 0")
    if execution.tick_size <= 0:
        problems.append("execution.tick_size must be > 0")
    if not 0.0 < execution.split_participation_rate <= 1.0:
        problems.append("execution.split_participation_rate must be in (0, 1]")
    if execution.min_split_children < 1:
        problems.append("execution.min_split_children must be >= 1")
    if execution.max_split_children < execution.min_split_children:
        problems.append(
            "execution.max_split_children must be >= execution.min_split_children"
        )
    if execution.default_order_notional <= 0:
        problems.append("execution.default_order_notional must be > 0")

    risk = config.section("risk")
    if not 0.0 <= risk.hedge_ratio <= risk.max_hedge_ratio <= 1.0:
        problems.append(
            "risk hedge ratios must satisfy 0 <= hedge_ratio <= max_hedge_ratio <= 1"
        )
    if not 0.0 <= risk.hedge_min_signal_strength <= 1.0:
        problems.append("risk.hedge_min_signal_strength must be in [0, 1]")
    if not 0.0 < risk.obi_extreme_abs <= 1.0:
        problems.append("risk.obi_extreme_abs must be in (0, 1]")
    if risk.max_position_notional <= 0:
        problems.append("risk.max_position_notional must be > 0")
    if risk.trailing_stop_prob < 0 or risk.stop_loss_prob < 0:
        problems.append("risk stop distances must be >= 0")

    fees = config.section("fees")
    if fees.taker_fee_coefficient < 0:
        problems.append("fees.taker_fee_coefficient must be >= 0")
    if not 0.0 <= fees.maker_rebate_share <= 1.0:
        problems.append("fees.maker_rebate_share must be in [0, 1]")

    market = config.section("market")
    if market.window_seconds <= 0:
        problems.append("market.window_seconds must be > 0")

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


def iter_missing_credentials(config: Config) -> Iterable[str]:
    """Names of credential fields that are empty (for paper/live gating)."""
    creds = config.section("credentials")
    for key in ("polymarket_api_key", "polymarket_api_secret", "polymarket_api_passphrase"):
        if not creds.get(key):
            yield key

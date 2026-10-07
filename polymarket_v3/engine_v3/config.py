"""Typed configuration for V3, with validation that refuses incoherent setups."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config_v3", "strategy_v3.json")


class ConfigError(ValueError):
    """Raised when the configuration cannot be traded coherently."""


@dataclass
class SignalConfig:
    features: tuple[str, ...] = ("btc_ret_15m", "ind_roc_5", "ind_roc_15")
    threshold_z: float = 2.0
    invert: bool = True
    entry_t_minus_s: float = 240.0
    min_trades_per_day: float = 1.0


@dataclass
class RiskConfig:
    mode: str = "hold_to_settlement"
    catastrophe_stop: float | None = 0.35
    take_profit: float | None = None
    max_hold_s: float = 300.0
    max_concurrent_positions: int = 1
    min_seconds_between_entries: float = 300.0


@dataclass
class SizingConfig:
    notional_usdc: float = 10.0
    max_position_pct_equity: float = 0.02
    starting_equity: float = 1000.0


@dataclass
class ExecutionConfig:
    max_entry_price: float = 0.535
    max_spread_cents: float = 4.0
    max_slippage_cents: float = 3.0
    require_measured_fill: bool = True
    max_quote_staleness_s: float = 5.0


@dataclass
class FeeConfig:
    taker_coefficient: float = 0.07


@dataclass
class SafetyConfig:
    live_trading_enabled: bool = False
    paper_only: bool = True


@dataclass
class StrategyV3:
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"section '{name}' must be an object")
    return {k: v for k, v in value.items() if not k.startswith("_")}


def load_config(path: str | None = None) -> StrategyV3:
    target = path or DEFAULT_CONFIG
    if not os.path.exists(target):
        raise ConfigError(f"config not found: {target}")
    with open(target, encoding="utf-8") as handle:
        raw = json.load(handle)

    config = StrategyV3(
        signal=SignalConfig(**_coerce(_section(raw, "signal"), SignalConfig)),
        risk=RiskConfig(**_coerce(_section(raw, "risk"), RiskConfig)),
        sizing=SizingConfig(**_coerce(_section(raw, "sizing"), SizingConfig)),
        execution=ExecutionConfig(**_coerce(_section(raw, "execution"), ExecutionConfig)),
        fees=FeeConfig(**_coerce(_section(raw, "fees"), FeeConfig)),
        safety=SafetyConfig(**_coerce(_section(raw, "safety"), SafetyConfig)),
    )
    validate(config)
    return config


def _coerce(values: dict, cls) -> dict:
    """Keep only fields the dataclass declares, and shape tuples/lists right."""
    allowed = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "features" and isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


def validate(config: StrategyV3) -> None:
    """Refuse configurations that cannot be traded coherently."""
    if config.signal.threshold_z <= 0:
        raise ConfigError("signal.threshold_z must be > 0; a zero threshold trades every window")
    if not config.signal.features:
        raise ConfigError("signal.features must not be empty")
    if config.signal.entry_t_minus_s <= 0 or config.signal.entry_t_minus_s > 300:
        raise ConfigError("signal.entry_t_minus_s must be in (0, 300] seconds for a 5m window")

    if config.risk.mode not in ("hold_to_settlement", "barrier"):
        raise ConfigError(f"unknown risk.mode: {config.risk.mode}")
    if config.risk.mode == "barrier":
        if config.risk.catastrophe_stop is None and config.risk.take_profit is None:
            raise ConfigError("risk.mode=barrier needs a stop or a take profit")
    if config.risk.catastrophe_stop is not None and not 0.0 < config.risk.catastrophe_stop < 1.0:
        raise ConfigError("risk.catastrophe_stop must be in (0,1) as a loss fraction")

    if config.sizing.notional_usdc <= 0:
        raise ConfigError("sizing.notional_usdc must be > 0")
    if not 0.0 < config.sizing.max_position_pct_equity <= 1.0:
        raise ConfigError("sizing.max_position_pct_equity must be in (0,1]")
    if config.sizing.notional_usdc > config.sizing.starting_equity * config.sizing.max_position_pct_equity + 1e-9:
        raise ConfigError(
            f"notional {config.sizing.notional_usdc} exceeds "
            f"{config.sizing.max_position_pct_equity:.1%} of equity "
            f"({config.sizing.starting_equity * config.sizing.max_position_pct_equity:.2f})"
        )

    if not 0.0 < config.execution.max_entry_price < 1.0:
        raise ConfigError("execution.max_entry_price must be in (0,1)")
    if config.execution.max_spread_cents < 0 or config.execution.max_slippage_cents < 0:
        raise ConfigError("spread and slippage limits cannot be negative")
    if config.execution.max_quote_staleness_s <= 0:
        raise ConfigError("execution.max_quote_staleness_s must be > 0")

    if config.fees.taker_coefficient < 0:
        raise ConfigError("fees.taker_coefficient cannot be negative")
    if config.safety.live_trading_enabled and config.safety.paper_only:
        raise ConfigError("safety: live_trading_enabled and paper_only cannot both be set")


def describe(config: StrategyV3) -> list[str]:
    """Human-readable summary, used by the CLI and the selfcheck."""
    direction = "fade" if config.signal.invert else "follow"
    return [
        f"signal        : {direction} |z| >= {config.signal.threshold_z} of "
        f"{', '.join(config.signal.features)}",
        f"entry         : T-{config.signal.entry_t_minus_s:.0f}s in a 5m window",
        f"risk          : {config.risk.mode}"
        + (f", catastrophe stop {config.risk.catastrophe_stop:.2f}"
           if config.risk.catastrophe_stop else ""),
        f"sizing        : {config.sizing.notional_usdc:.2f} USDC "
        f"(<= {config.sizing.max_position_pct_equity:.1%} of equity)",
        f"entry cap     : {config.execution.max_entry_price:.3f} "
        f"(break-even accuracy at this price = the signal's measured accuracy)",
        f"cost limits   : spread <= {config.execution.max_spread_cents:.1f}c, "
        f"slippage <= {config.execution.max_slippage_cents:.1f}c",
        f"fills         : {'measured only' if config.execution.require_measured_fill else 'modelled allowed'}",
        f"safety        : {'PAPER ONLY' if config.safety.paper_only else 'live enabled'}",
    ]

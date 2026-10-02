"""Logging + serialisation helpers shared by every module."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

_LOGGER_NAME = "polymarket_sniper"
_configured = False


def get_logger(name: str | None = None, level: str = "INFO") -> logging.Logger:
    """Package logger; configures a single stdout handler on first use."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")
    if not _configured:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger(_LOGGER_NAME)
        root.addHandler(handler)
        root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        root.propagate = False
        _configured = True
    return logger


def set_level(level: str) -> None:
    logging.getLogger(_LOGGER_NAME).setLevel(
        getattr(logging, str(level).upper(), logging.INFO)
    )


def add_file_handler(path: str) -> None:
    """Mirror logs into a file (used by main.py when --log-file is given)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    logging.getLogger(_LOGGER_NAME).addHandler(handler)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def iso(ts: float | datetime | None) -> str:
    if ts is None:
        return ""
    if isinstance(ts, datetime):
        moment = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        moment = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/floats into JSON-serialisable values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str, payload: Any, indent: int = 2) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=indent, ensure_ascii=False)
    return path


def write_jsonl(path: str, rows: Iterable[Any]) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return path


def write_csv(path: str, rows: Iterable[dict], fieldnames: list[str] | None = None) -> str:
    import csv

    rows = list(rows)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError("cannot write an empty table without fieldnames")
    columns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(v) for k, v in row.items()})
    return path


def fmt_prob(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_usdc(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"

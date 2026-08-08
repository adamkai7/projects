"""Shared helpers: config loading, logging, path resolution."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level: str = "INFO") -> None:
    """Configure a simple console logger for research runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


def load_config(config_path: Path | str = "configs/default.yaml") -> dict[str, Any]:
    """Load default config and resolve relative paths against the project root."""
    cfg = load_yaml(config_path)
    data = cfg.setdefault("data", {})
    for key in ("cache_dir", "db_path"):
        if key in data and data[key] is not None:
            p = Path(data[key])
            data[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

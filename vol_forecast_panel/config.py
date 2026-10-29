"""Shared paths and config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    data = cfg.setdefault("data", {})
    cache = Path(data.get("cache_dir", "data/cache"))
    data["cache_dir"] = str(cache if cache.is_absolute() else PROJECT_ROOT / cache)
    uni = Path(data.get("universe_path", "universe.txt"))
    data["universe_path"] = str(uni if uni.is_absolute() else PROJECT_ROOT / uni)
    return cfg


def load_universe(path: Path | str, limit: int = 0) -> list[str]:
    symbols: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line.upper())
    symbols = list(dict.fromkeys(symbols))
    return symbols[:limit] if limit else symbols

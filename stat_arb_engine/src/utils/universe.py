"""Universe helpers and ticker metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, load_yaml


def load_universe(
    name: str = "liquid50",
    custom_tickers: list[str] | None = None,
    universes_path: Path | str = "configs/universes.yaml",
) -> dict[str, str]:
    """
    Return mapping ticker -> sector.

    Parameters
    ----------
    name :
        qqq | liquid50 | sp100 | custom
    custom_tickers :
        Required when name == 'custom'.
    """
    path = Path(universes_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = load_yaml(path)

    if name == "custom":
        if not custom_tickers:
            raise ValueError("custom universe requires custom_tickers")
        return {t.upper(): "Unknown" for t in custom_tickers}

    if name == "qqq":
        if "qqq" not in raw:
            raise ValueError("universes.yaml is missing a 'qqq' section")
        return {k: v for k, v in raw["qqq"].items()}

    if name == "liquid50":
        return {k: v for k, v in raw["liquid50"].items()}

    if name == "sp100":
        merged: dict[str, str] = dict(raw["liquid50"])
        merged.update(raw.get("sp100_extra", {}))
        return merged

    raise ValueError(f"Unknown universe '{name}'. Use qqq, liquid50, sp100, or custom.")


def tickers_from_config(cfg: dict[str, Any]) -> dict[str, str]:
    data = cfg.get("data", {})
    return load_universe(
        name=data.get("universe", "liquid50"),
        custom_tickers=data.get("custom_tickers") or None,
    )

"""Utility helpers for configuration, logging, and reproducibility."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_json(obj: Dict[str, Any], path: str | pathlib.Path) -> None:
    """Save dictionary as JSON with pretty formatting."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def configure_logging(name: str = "ai_workflow") -> logging.Logger:
    """Configure and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_global_seed(seed: int) -> None:
    """Set deterministic seed for numpy and lightgbm where applicable."""
    np.random.seed(seed)
    try:
        import lightgbm as lgb

        lgb.basic._config.set_config(seed=seed)  # type: ignore[attr-defined]
    except Exception:
        # LightGBM not installed or API changed; ignore.
        pass


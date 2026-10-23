"""Config loading, hashing, seeding, logging and results.json helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (default: configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Short stable hash of the config (sorted-key JSON, sha256, first 12 hex chars)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    """Seed python, numpy and PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def project_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve cfg['paths'][key] against the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def update_results(cfg: dict[str, Any], section: str, payload: dict[str, Any], merge: bool = False) -> Path:
    """Write `payload` under `section` in results/results.json (atomic write).

    merge=False replaces the section; merge=True updates its top-level keys instead.
    Every number used in a table or in the report must be written through here.
    """
    path = project_path(cfg, "results_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    if merge and isinstance(data.get(section), dict):
        data[section].update(payload)
    else:
        data[section] = payload
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path

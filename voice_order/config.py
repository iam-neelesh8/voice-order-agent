"""Typed config loading from configs/*.yaml.

Stage 0. Config is data, not code -- nothing here decides behaviour, it only
makes the YAML reachable and validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
DEFAULT_DB = DATA_DIR / "voice_order.db"


@dataclass(frozen=True)
class Config:
    """A loaded YAML config, with dotted-path access."""

    name: str
    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get("dense.top_k")` -> 50."""
        raise NotImplementedError("stage 0")


def load(name: str) -> Config:
    """Load `configs/{name}.yaml`. Raises if the file is missing."""
    raise NotImplementedError("stage 0")


def database_path() -> Path:
    """SQLite file, overridable with $VOICE_ORDER_DB.

    One file, no server. Deleting it and rerunning `voice-order catalog` is a
    complete reset, which is the property that makes the eval reproducible for
    someone who just cloned the repo.
    """
    return Path(os.environ.get("VOICE_ORDER_DB", DEFAULT_DB))


def index_dir() -> Path:
    """Where the on-disk retrieval indexes live. Derived data, gitignored.

    Nothing verifies these are in sync with the database -- rebuild them
    whenever the catalog changes. `voice-order db health` reports what exists.
    """
    return Path(os.environ.get("VOICE_ORDER_INDEX_DIR", INDEX_DIR))

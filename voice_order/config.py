"""Typed config loading from configs/*.yaml.

Stage 0. Config is data, not code -- nothing here decides behaviour, it only
makes the YAML reachable and validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Read .env files into the environment, without a dependency.

    Looks in the project root and in voice_order/, and never overwrites a
    variable already set in the real environment -- an explicit `export` wins
    over a file. KEY=VALUE per line, # comments and blank lines ignored.
    """
    for path in (ROOT / ".env", ROOT / "voice_order" / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
DEFAULT_DB = DATA_DIR / "voice_order.db"

_MISSING = object()


@dataclass(frozen=True)
class Config:
    """A loaded YAML config, with dotted-path access."""

    name: str
    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """`cfg.get("dense.top_k")` -> 50.

        Raises KeyError when the path is absent and no default was given. A
        silent None here would surface much later as a mystifying retrieval
        result, so missing config is a loud failure.
        """
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"{self.name}.yaml has no key {dotted!r}")
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)


@lru_cache(maxsize=None)
def load(name: str) -> Config:
    """Load `configs/{name}.yaml`. Raises if the file is missing."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no config at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return Config(name=name, raw=raw)


def data_dir() -> Path:
    """Root for every generated artefact, redirectable as a whole.

    Individual overrides below still win, but this is the one switch that
    isolates an entire run -- which is what a test needs. Without it, anything
    writing under data/ reaches straight into a real catalog, audio set or
    export; that has already caused one silent data loss.
    """
    return Path(os.environ.get("VOICE_ORDER_DATA_DIR") or DATA_DIR)


def database_path() -> Path:
    """SQLite file, overridable with $VOICE_ORDER_DB.

    One file, no server. Deleting it and rerunning `voice-order catalog` is a
    complete reset, which is the property that makes the eval reproducible for
    someone who just cloned the repo.
    """
    return Path(os.environ.get("VOICE_ORDER_DB") or data_dir() / "voice_order.db")


def index_dir() -> Path:
    """Where the on-disk retrieval indexes live. Derived data, gitignored.

    Nothing verifies these are in sync with the database -- rebuild them
    whenever the catalog changes. `voice-order db health` reports what exists.
    """
    return Path(os.environ.get("VOICE_ORDER_INDEX_DIR") or data_dir() / "index")

"""Stage 1 — pull a bounded slice of Amazon Reviews 2023 item metadata.

Raw JSONL over HTTP, not `datasets` (script loaders were dropped in 4.x).
Streamed and capped per category: the full corpus is ~48M items and we want
~100k. fitment-rag already has a working version of this loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def download_category(category: str, limit: int, dest: Path) -> Path:
    """Stream `meta_{category}.jsonl`, stop at `limit` items, write to `dest`.

    Resumable: if `dest` already holds >= limit lines, do nothing.
    """
    raise NotImplementedError("stage 1")


def iter_raw_items(path: Path) -> Iterator[dict]:
    """Yield decoded JSONL records, skipping malformed lines with a warning."""
    raise NotImplementedError("stage 1")


def download_all() -> dict[str, Path]:
    """Download every category in `configs/catalog.yaml`. Returns name -> path."""
    raise NotImplementedError("stage 1")

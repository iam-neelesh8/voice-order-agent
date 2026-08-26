"""Stage 1 — download, normalise, write to SQLite. The stage 1 entry point."""

from __future__ import annotations


def build_catalog(force: bool = False) -> dict[str, int]:
    """Run the whole stage: stream -> normalise -> upsert.

    Returns per-category row counts. Stage 1 is done when these match the
    limits in `configs/catalog.yaml`.
    """
    raise NotImplementedError("stage 1")

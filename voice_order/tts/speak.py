"""Stages 4 and 7 — Piper.

Stage 4 uses it to *build the test set* (several voices, so the eval does not
measure one speaker). Stage 7 uses it to answer the caller.
"""

from __future__ import annotations

from pathlib import Path


class Speaker:
    def __init__(self, voice: str | None = None) -> None:
        raise NotImplementedError("stage 4")

    def synthesize(self, text: str, out_path: Path) -> Path:
        raise NotImplementedError("stage 4")

    def say(self, text: str) -> None:
        """Synthesize and play. Stage 7."""
        raise NotImplementedError("stage 7")

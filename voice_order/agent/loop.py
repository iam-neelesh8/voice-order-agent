"""Stages 6 and 7 — the dialogue loop.

    audio in -> ASR -> intent -> retrieval -> policy -> reply -> audio out

Every pass writes a traced `Turn`. That trace is the whole point: without it
a wrong order is just a shrug.
"""

from __future__ import annotations

from pathlib import Path

from voice_order.types import Transcript, Turn


class OrderAgent:
    def __init__(self, call_id: str | None = None) -> None:
        raise NotImplementedError("stage 6")

    def handle_utterance(self, transcript: Transcript) -> Turn:
        """One full turn, text in. Stage 6 — testable without any audio."""
        raise NotImplementedError("stage 6")

    def run_file(self, audio_path: Path) -> list[Turn]:
        """Batch: a recorded call in, traced turns out. Stage 6."""
        raise NotImplementedError("stage 6")

    def run_live(self) -> None:
        """Microphone in, speaker out, with barge-in. Stage 7."""
        raise NotImplementedError("stage 7")

    def close(self) -> list[str]:
        """Commit the cart to `orders`. Returns order ids."""
        raise NotImplementedError("stage 6")

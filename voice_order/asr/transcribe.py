"""Stage 4 — faster-whisper, returning n-best rather than one transcript.

One transcript is not enough. The whole robustness argument depends on
retrieval seeing several hypotheses, so this wrapper never collapses to a
single string.
"""

from __future__ import annotations

from pathlib import Path

from voice_order.types import Transcript


class Transcriber:
    """Loads the model once. Instantiating this per call is the obvious
    performance mistake and the reason it is a class."""

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        raise NotImplementedError("stage 4")

    def transcribe_file(self, path: Path) -> Transcript:
        raise NotImplementedError("stage 4")

    def transcribe_stream(self, audio_chunks) -> Transcript:
        """For the live loop (stage 7). Batch path is stage 4."""
        raise NotImplementedError("stage 7")


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER. Reported clean vs phone-degraded — the gap is a result."""
    raise NotImplementedError("stage 4")

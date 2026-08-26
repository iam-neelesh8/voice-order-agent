"""Stage 6 — commit, confirm, or re-ask.

This is the mechanism that turns a low-confidence retrieval into either a
correct order or a cheap re-ask, instead of a silently wrong one. It is the
difference between a demo and something you would let take real orders.
"""

from __future__ import annotations

from voice_order.types import Candidate


def decide(candidates: list[Candidate], confidence: float) -> str:
    """-> "commit" | "confirm" | "clarify" | "reask".

    Thresholds come from `configs/agent.yaml`. A thin margin between the top
    two candidates forces a confirmation even at high absolute confidence —
    two plausible part numbers is exactly the dangerous case.
    """
    raise NotImplementedError("stage 6")


def readback(candidate: Candidate, quantity: int) -> str:
    """Speakable confirmation text.

    Never read the full Amazon title — they are unspeakable. Brand plus
    identifier plus a short noun phrase.
    """
    raise NotImplementedError("stage 6")


def clarifying_question(candidates: list[Candidate]) -> str:
    """Ask about the attribute that actually separates the top candidates."""
    raise NotImplementedError("stage 6")

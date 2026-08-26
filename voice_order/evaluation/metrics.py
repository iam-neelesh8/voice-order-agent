"""Metrics, shared by every stage that reports a number.

Reported per category, never as one average — Automotive and
Health_and_Household are in the slice precisely because they differ, and an
average hides the only interesting result.
"""

from __future__ import annotations

from voice_order.types import Candidate


def recall_at_k(candidates: list[Candidate], relevant: list[str], k: int) -> float:
    raise NotImplementedError("stage 2")


def mrr(candidates: list[Candidate], relevant: list[str]) -> float:
    raise NotImplementedError("stage 2")


def aggregate(per_query: list[dict], by: str = "category") -> dict[str, dict]:
    """Group per-query results and summarise.

    `by` also accepts "has_part_number" — that split is the stage 5 story.
    """
    raise NotImplementedError("stage 2")

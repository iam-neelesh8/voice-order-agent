"""Shared record types.

These are the contracts between stages. They exist now, before any stage is
implemented, so that stage N+1 can be written against stage N's output
without waiting for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Product:
    """One catalog item, normalised from Amazon Reviews 2023 metadata."""

    parent_asin: str
    title: str
    category: str
    store: str | None = None
    price: float | None = None
    average_rating: float | None = None
    features: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    # Extracted at normalise time (stage 1), not at query time — part numbers
    # are what retrieval hinges on, so they get their own indexed column.
    part_numbers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Hypothesis:
    """One ASR transcript candidate. n-best means several of these per turn."""

    text: str
    score: float  # log-prob or confidence, higher is better
    rank: int


@dataclass(frozen=True)
class Transcript:
    """Everything the ASR produced for one utterance."""

    audio_path: str | None
    hypotheses: list[Hypothesis]
    duration_s: float
    latency_ms: float

    @property
    def best(self) -> str:
        """The 1-best text. Convenient, but never the only thing retrieval sees."""
        return self.hypotheses[0].text if self.hypotheses else ""


@dataclass(frozen=True)
class Candidate:
    """One retrieval result, with the trail that produced it.

    Retrievers return these carrying only `parent_asin`; `product` is filled
    in by a single hydration pass after fusion, so the row lookup happens once
    per query instead of once per retriever.
    """

    parent_asin: str
    score: float
    # Which retrievers fired and what they scored — this is what makes a bad
    # order debuggable rather than mysterious.
    component_scores: dict[str, float] = field(default_factory=dict)
    product: Product | None = None
    matched_hypothesis: str | None = None

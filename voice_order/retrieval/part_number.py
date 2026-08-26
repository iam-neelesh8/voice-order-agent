"""Stage 5 -- the module this project exists for.

Speech destroys identifiers. "41-993" comes back as "forty one dash nine
ninety three", "41-99 3", "4199 3". Character matching fails on all three.
So: normalise spoken digits back to figures, strip separators, and fall back
to a phonetic key when the digits themselves came through wrong.

The index is a plain dict of normalised form -> product ids, built from
`product_part_numbers` and written to `data/index/part_numbers.json`. It is
small enough to hold in memory and cheap enough to rebuild.

This whole retriever is switched off by default so that stages 2 and 4
measure a clean baseline. Stage 5 turns each of its three pieces on
separately -- see `Retriever` in fusion.py.
"""

from __future__ import annotations

from pathlib import Path

from voice_order.types import Candidate

INDEX_FILE = "part_numbers.json"


def spoken_digits_to_figures(text: str) -> str:
    """"forty one dash nine ninety three" -> "41-993".

    Handles the ways ASR writes numbers out: word digits, tens compounds
    ("ninety three"), "double seven", "oh" and "o" for zero, and
    "dash"/"hyphen"/"slash" for separators.

    This is the single highest-leverage function in the project. Get it wrong
    and stage 5 shows no improvement for reasons that have nothing to do with
    retrieval.
    """
    raise NotImplementedError("stage 5")


def candidate_part_numbers(text: str) -> list[str]:
    """Every substring of an utterance that could plausibly be an identifier.

    Over-generates on purpose. A false candidate costs one dict lookup; a
    missed one costs the turn.
    """
    raise NotImplementedError("stage 5")


def phonetic_key(token: str) -> str:
    """Metaphone key, so B/V/P and M/N confusions still collide."""
    raise NotImplementedError("stage 5")


class PartNumberIndex:
    """Normalised identifier -> product ids, plus a phonetic bucket map."""

    def __init__(self, exact: dict[str, list[str]], phonetic: dict[str, list[str]]) -> None:
        raise NotImplementedError("stage 5")

    @classmethod
    def build(cls, index_dir: Path | None = None) -> "PartNumberIndex":
        raise NotImplementedError("stage 5")

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "PartNumberIndex":
        raise NotImplementedError("stage 5")

    def search(self, query: str, top_k: int = 50) -> list[Candidate]:
        """Match extracted identifiers against the index, in tiers.

        Exact normalised match scores highest, then edit-distance 1, then
        phonetic. Which tier fired is recorded in `Candidate.component_scores`
        so the stage 5 ablation can attribute wins to a specific mechanism
        rather than to "the part number thing".
        """
        raise NotImplementedError("stage 5")

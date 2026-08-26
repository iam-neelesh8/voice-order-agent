"""Stage 1 — raw metadata record -> `Product`.

The interesting part is `extract_part_numbers`. Doing it at load time rather
than query time means retrieval can index identifiers directly, which is what
makes the Automotive-vs-Health_and_Household split measurable.
"""

from __future__ import annotations

from voice_order.types import Product


def normalize_item(raw: dict, category: str) -> Product | None:
    """Map one raw record to a `Product`, or None if required fields are missing."""
    raise NotImplementedError("stage 1")


def extract_part_numbers(title: str, details: dict, features: list[str]) -> list[str]:
    """Pull identifier-shaped tokens out of an item.

    Identifier-shaped means: contains a digit, is not a pure quantity/measure
    ("12v", "3 pack", "10mm"), and is long enough to be discriminative.
    Returns the *normalised* forms — see `normalize_part_number`.
    """
    raise NotImplementedError("stage 1")


def normalize_part_number(token: str) -> str:
    """Collapse an identifier to its comparison form.

    `41-993`, `41 993`, `AC41993` and `41993` must not all collapse together —
    separators are stripped but the alphanumeric core is preserved, so
    `41-993` -> `41993` while `AC41993` stays distinct.
    """
    raise NotImplementedError("stage 1")

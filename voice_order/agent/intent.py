"""Stage 6 — utterance -> `OrderIntent`.

Pulls quantity, brand and identifier out of the text so retrieval can be
given structure rather than a blob. "I need two AC Delco 41-993 spark plugs"
-> qty 2, brand "AC Delco", part "41-993", words "spark plugs".
"""

from __future__ import annotations

from voice_order.types import OrderIntent


def parse(text: str) -> OrderIntent:
    raise NotImplementedError("stage 6")


def parse_quantity(text: str) -> int:
    """"two", "a couple of", "2", "a" -> int. Defaults to 1."""
    raise NotImplementedError("stage 6")


def is_affirmative(text: str) -> bool | None:
    """yes / no / neither. `None` means the caller said something else and the
    confirmation has to be re-asked rather than guessed at."""
    raise NotImplementedError("stage 6")

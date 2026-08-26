"""Reads and writes. No SQL escapes this module.

This is the seam. Everything above it -- retrieval, the agent, the eval
harness -- talks in `Product` / `Cart` / `Candidate`, never in rows. Adding a
Postgres backend later means writing one more implementation of these
functions, not touching anything that calls them.

Stage 1 for products; stage 6 for calls, carts and orders.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from voice_order.types import Cart, Candidate, Product, Transcript, Turn

# ------------------------------------------------------------------ stage 1 --


def upsert_products(products: Iterable[Product], batch_size: int = 1000) -> int:
    """Bulk-insert normalised items and their part numbers. Returns rows written.

    Writes both `products` and `product_part_numbers` in one transaction per
    batch -- a product whose identifiers failed to land is worse than one that
    is missing entirely, because it looks fine and silently cannot be found.
    """
    raise NotImplementedError("stage 1")


def iter_products(category: str | None = None) -> Iterator[Product]:
    """Stream every product. Used to build the on-disk indexes (stage 2).

    Streams rather than returns a list: 100k hydrated products is enough
    memory to matter on a laptop, and the index builders only need one at a time.
    """
    raise NotImplementedError("stage 1")


def get_products(parent_asins: Sequence[str]) -> dict[str, Product]:
    """Hydrate retrieval hits back into full products.

    Retrieval returns ids and scores; this is what turns them into something
    the agent can read aloud. Batched because a fused candidate list is 20-50
    ids and 50 round trips is silly.
    """
    raise NotImplementedError("stage 2")


def count_by_category() -> dict[str, int]:
    """Per-category row counts -- the stage 1 done-check."""
    raise NotImplementedError("stage 1")


def count_with_part_numbers() -> dict[str, int]:
    """Per-category count of items carrying at least one identifier.

    This is the number that says whether the category mix did its job:
    Automotive should be high, Health_and_Household near zero. If they are not
    far apart, the extractor in normalize.py is wrong and every later result
    is measuring the wrong thing.
    """
    raise NotImplementedError("stage 1")


# ------------------------------------------------------------------ stage 6 --


def open_call(audio_path: str | None) -> str:
    """Create a `calls` row, return its call_id."""
    raise NotImplementedError("stage 6")


def append_turn(call_id: str, turn: Turn) -> None:
    """Append one traced turn to `calls.turns`."""
    raise NotImplementedError("stage 6")


def save_cart(cart: Cart) -> None:
    raise NotImplementedError("stage 6")


def commit_order(
    call_id: str,
    parent_asin: str,
    quantity: int,
    transcript: Transcript,
    candidates: list[Candidate],
    confidence: float,
    was_confirmed: bool,
) -> str:
    """Write an order *with its trace*. Never call this without candidates.

    The signature is deliberately awkward: it is not possible to record an
    order here without also recording what produced it.
    """
    raise NotImplementedError("stage 6")

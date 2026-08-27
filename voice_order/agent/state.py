"""Stage 6 -- the cart, and the only code allowed to change it.

`OrderSession` is what the tools in `llm/tools.py` actually call. It owns
three things the model never gets to influence:

  * whether a product id is real
  * what anything costs
  * what the total is

Every method returns a plain dict, because that dict goes straight back to the
model as a tool result. They read like something a person could act on -- a
model told "no match, ask the caller for the brand" behaves better than one
told "KeyError".

The confidence score comes back with search results but the *decision* does
not: `policy.py` decides whether a match is good enough. A model asked "are
you confident?" will say yes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from voice_order.types import Candidate


class State(str, Enum):
    GREETING = "greeting"
    LISTENING = "listening"      # waiting for an item
    CONFIRMING = "confirming"    # read back, waiting for yes/no
    REVIEWING = "reviewing"      # reading the whole order back
    CLOSED = "closed"


@dataclass
class Line:
    """One item on the order. Price is copied from the catalog at add time."""

    product_id: str
    name: str
    unit_price: float | None
    quantity: int

    @property
    def subtotal(self) -> float | None:
        if self.unit_price is None:
            return None
        return round(self.unit_price * self.quantity, 2)


@dataclass
class OrderSession:
    """One call. Holds the cart and brokers every change to it."""

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    lines: list[Line] = field(default_factory=list)
    state: State = State.GREETING
    placed_order_ids: list[str] = field(default_factory=list)

    # Set by the agent so search results can be resolved back to products
    # without the model ever holding a catalog reference.
    _retriever: object | None = None
    _last_results: dict[str, Candidate] = field(default_factory=dict)
    # Cleared by every cart change, set by read_cart. Gates place_order.
    _total_read_since_change: bool = False

    # ------------------------------------------------------------- search --

    def search(self, query: str, top_k: int = 5) -> dict:
        """Look the caller's words up in the catalog.

        Returns candidates with a confidence score attached, so the model can
        tell a certain match from a guess and read the guess back instead of
        assuming.
        """
        from voice_order.agent import policy

        if self._retriever is None:
            return {"error": "the catalog is not loaded"}

        candidates = self._retriever.search_text(query, top_k=top_k, hydrate=True)
        if not candidates:
            return {
                "matches": [],
                "note": "nothing matched. Ask the caller for a brand or a part number.",
            }

        self._last_results = {c.parent_asin: c for c in candidates}
        confidence = policy.confidence(candidates)
        decision = policy.decide(candidates, confidence)

        return {
            "matches": [
                {
                    "product_id": c.parent_asin,
                    "name": (c.product.title[:90] if c.product else c.parent_asin),
                    "brand": (c.product.store if c.product else None),
                    "price": (c.product.price if c.product else None),
                }
                for c in candidates
            ],
            "confidence": round(confidence, 3),
            # Instruction rather than a number, because the number is only
            # meaningful next to thresholds the model cannot see.
            "guidance": policy.GUIDANCE[decision],
        }

    # --------------------------------------------------------------- cart --

    def _find(self, line_number: int) -> Line | None:
        if 1 <= line_number <= len(self.lines):
            return self.lines[line_number - 1]
        return None

    def add(self, product_id: str, quantity: int = 1) -> dict:
        """Put a product on the order, if it is a real product.

        The id must have come from a search in this call. That is what stops a
        model inventing a plausible-looking id, and it is checked here rather
        than trusted from the prompt.
        """
        candidate = self._last_results.get(product_id)
        if candidate is None:
            product = self._lookup(product_id)
            if product is None:
                return {
                    "error": (
                        f"no product {product_id!r} in the catalog. Use an id from "
                        "a search_products result."
                    )
                }
            name, price = product.title, product.price
        else:
            product = candidate.product
            name = product.title if product else product_id
            price = product.price if product else None

        for line in self.lines:
            if line.product_id == product_id:
                line.quantity = min(line.quantity + quantity, 99)
                self._total_read_since_change = False
                return {"ok": True, "note": "already on the order, quantity increased",
                        **self._snapshot()}

        self.lines.append(
            Line(product_id=product_id, name=name[:120], unit_price=price, quantity=quantity)
        )
        self._total_read_since_change = False
        self.state = State.LISTENING
        return {"ok": True, **self._snapshot()}

    def _lookup(self, product_id: str):
        from voice_order.db import repository

        return repository.get_products([product_id]).get(product_id)

    def change_quantity(self, line_number: int, quantity: int) -> dict:
        line = self._find(line_number)
        if line is None:
            return {"error": f"there is no line {line_number}. Call read_cart first."}
        line.quantity = quantity
        self._total_read_since_change = False
        return {"ok": True, **self._snapshot()}

    def remove(self, line_number: int) -> dict:
        line = self._find(line_number)
        if line is None:
            return {"error": f"there is no line {line_number}. Call read_cart first."}
        self.lines.remove(line)
        self._total_read_since_change = False
        return {"ok": True, **self._snapshot()}

    def read_cart(self) -> dict:
        """The order as it stands. The tool the model calls.

        Records that the total has been produced since the last change, which
        is what `place_order` requires. Internal callers use `_snapshot`
        instead -- `add` returning the cart is not the caller hearing it.
        """
        self._total_read_since_change = True
        return self._snapshot()

    def _snapshot(self) -> dict:
        """The cart as data, with the total computed here and nowhere else."""
        priced = [line for line in self.lines if line.unit_price is not None]
        total = round(sum(line.subtotal or 0.0 for line in priced), 2)
        unpriced = len(self.lines) - len(priced)

        out: dict = {
            "lines": [
                {
                    "line_number": i,
                    "product_id": line.product_id,
                    "name": line.name,
                    "quantity": line.quantity,
                    "unit_price": line.unit_price,
                    "subtotal": line.subtotal,
                }
                for i, line in enumerate(self.lines, start=1)
            ],
            "item_count": sum(line.quantity for line in self.lines),
            "total": total,
        }
        if unpriced:
            out["note"] = (
                f"{unpriced} item(s) have no price in the catalog and are not in "
                "the total. Tell the caller you will confirm those separately."
            )
        return out

    # -------------------------------------------------------------- close --

    def place_order(self) -> dict:
        if not self.lines:
            return {"error": "the order is empty -- nothing to place"}

        # The caller must have heard the total. A prompt can ask the model to
        # read it back; this makes it so. Without it a model that mistakes
        # "yes, go ahead" for agreement to a total it never said would place
        # an order nobody agreed to the price of -- which is the one mistake
        # here that costs somebody money.
        if not self._total_read_since_change:
            return {
                "error": (
                    "read_cart first, tell the caller the items and the total, "
                    "and wait for them to agree before placing the order."
                )
            }

        from voice_order.db import repository

        order_ids = []
        for line in self.lines:
            order_ids.append(
                repository.commit_order(
                    call_id=self.call_id,
                    parent_asin=line.product_id,
                    quantity=line.quantity,
                    query_text="",
                    nbest=[],
                    candidates=[],
                    confidence=None,
                    was_confirmed=True,
                )
            )
        self.placed_order_ids = order_ids
        self.state = State.CLOSED
        summary = self._snapshot()
        return {"ok": True, "order_ids": order_ids, "total": summary["total"]}

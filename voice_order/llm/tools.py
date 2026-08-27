"""Stage 6 -- the only things the model is allowed to do.

This is the boundary. The model proposes a call; `execute` validates it and
runs it against real code. Everything dangerous lives on this side of the
line:

  * an id that is not in the catalog cannot be added to a cart
  * a quantity outside 1-99 is refused
  * prices are read from the catalog, never accepted from the model
  * the total is arithmetic, not generation

Swapping the model cannot widen any of that, which is the point of putting it
here rather than in a prompt.

The descriptions matter as much as the code -- they are the only instructions
the model gets about when to call each one, so they are written for a reader
who knows nothing about the project.
"""

from __future__ import annotations

from typing import Any

MAX_QUANTITY = 99

# JSON-schema definitions sent to the model. Deliberately six, not seven --
# customers are deferred until after v1 works, so `set_customer` is absent.
SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Look up products in the shop's catalog by what the caller "
                "said. Call this whenever the caller names something they want, "
                "even if they are vague. Returns a short list of matches with "
                "an id, name, price and a confidence score. It does not add "
                "anything to the order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What the caller asked for, in their words. Include "
                            "the brand and any part number they gave, because "
                            "those are what make the match reliable."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": (
                "Add a product to the caller's order. Only ever use a "
                "product_id returned by search_products. If the search result "
                "was not confident, read the product name back to the caller "
                "and get a yes before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The id from a search_products result.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "How many, from 1 to 99. Defaults to 1.",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_quantity",
            "description": (
                "Change how many of a line the caller wants, for when they say "
                "something like 'actually make that three'. Use the line_number "
                "shown by read_cart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "line_number": {"type": "integer"},
                    "quantity": {"type": "integer"},
                },
                "required": ["line_number", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": (
                "Take a line off the order, for when the caller changes their "
                "mind. Use the line_number shown by read_cart."
            ),
            "parameters": {
                "type": "object",
                "properties": {"line_number": {"type": "integer"}},
                "required": ["line_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_cart",
            "description": (
                "Get everything currently on the order, with line numbers and "
                "the running total. Call this before telling the caller their "
                "total, and any time you need to know what is already on the "
                "order. Never work the total out yourself."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Finalise the order. Only call this after reading the full "
                "order and the total back to the caller and hearing them agree."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in SCHEMAS)


def _coerce_quantity(value: Any) -> int:
    """Models send "2", 2, and 2.0. Only the count matters."""
    if isinstance(value, bool):
        raise ValueError("quantity must be a number")
    if isinstance(value, (int, float)):
        quantity = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        quantity = int(value.strip())
    else:
        raise ValueError(f"quantity {value!r} is not a whole number")
    if not 1 <= quantity <= MAX_QUANTITY:
        raise ValueError(f"quantity must be between 1 and {MAX_QUANTITY}, got {quantity}")
    return quantity


def execute(name: str, arguments: dict, session) -> dict:
    """Run one tool call against the session. Never raises.

    Errors come back as `{"error": ...}` rather than exceptions on purpose: a
    model that asks for something impossible should be told so and given a
    chance to correct itself, exactly as it would be told a search found
    nothing. Crashing the call would be a worse answer to a recoverable
    mistake.
    """
    try:
        if name not in TOOL_NAMES:
            return {"error": f"no such tool {name!r}; available: {sorted(TOOL_NAMES)}"}

        if name == "search_products":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return {"error": "query was empty -- ask the caller what they need"}
            return session.search(query)

        if name == "add_to_cart":
            product_id = str(arguments.get("product_id") or "").strip()
            if not product_id:
                return {"error": "product_id is required; call search_products first"}
            quantity = _coerce_quantity(arguments.get("quantity", 1))
            return session.add(product_id, quantity)

        if name == "change_quantity":
            return session.change_quantity(
                int(arguments.get("line_number", 0)),
                _coerce_quantity(arguments.get("quantity", 1)),
            )

        if name == "remove_from_cart":
            return session.remove(int(arguments.get("line_number", 0)))

        if name == "read_cart":
            return session.read_cart()

        if name == "place_order":
            return session.place_order()

        return {"error": f"tool {name!r} is defined but not wired up"}

    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}

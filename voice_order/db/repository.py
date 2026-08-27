"""Reads and writes. No SQL escapes this module.

This is the seam. Everything above it -- retrieval, the agent, the eval
harness -- talks in `Product` / `Cart` / `Candidate`, never in rows. Adding a
Postgres backend later means writing one more implementation of these
functions, not touching anything that calls them.

Stage 1 for products; stage 6 for calls, carts and orders.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterable, Iterator, Sequence

from voice_order.db.session import connect
from voice_order.types import Product

# SQLite's default bound-parameter ceiling is well above this; 500 keeps the
# statement readable and is comfortably inside every version's limit.
_CHUNK = 500

_UPSERT_PRODUCT = """
INSERT INTO products
    (parent_asin, title, category, store, price, average_rating, features, details)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (parent_asin) DO UPDATE SET
    title = excluded.title,
    category = excluded.category,
    store = excluded.store,
    price = excluded.price,
    average_rating = excluded.average_rating,
    features = excluded.features,
    details = excluded.details
"""

# ------------------------------------------------------------------ stage 1 --


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row_to_product(row, part_numbers: list[str] | None = None) -> Product:
    return Product(
        parent_asin=row["parent_asin"],
        title=row["title"],
        category=row["category"],
        store=row["store"],
        price=row["price"],
        average_rating=row["average_rating"],
        features=json.loads(row["features"]),
        details=json.loads(row["details"]),
        part_numbers=part_numbers or [],
    )


def upsert_products(products: Iterable[Product], batch_size: int = 1000) -> int:
    """Bulk-insert normalised items and their part numbers. Returns rows written.

    Writes both `products` and `product_part_numbers` in one transaction per
    batch -- a product whose identifiers failed to land is worse than one that
    is missing entirely, because it looks fine and silently cannot be found.

    Uses ON CONFLICT DO UPDATE rather than INSERT OR REPLACE: REPLACE deletes
    the row first, which cascades and would silently drop part numbers that
    this call is not re-inserting.
    """
    from voice_order.catalog.normalize import part_number_rows

    written = 0
    batch: list[Product] = []

    def flush(items: list[Product]) -> int:
        if not items:
            return 0
        with connect() as conn:
            conn.executemany(
                _UPSERT_PRODUCT,
                [
                    (
                        p.parent_asin,
                        p.title,
                        p.category,
                        p.store,
                        p.price,
                        p.average_rating,
                        _dumps(p.features),
                        _dumps(p.details),
                    )
                    for p in items
                ],
            )
            conn.executemany(
                "DELETE FROM product_part_numbers WHERE parent_asin = ?",
                [(p.parent_asin,) for p in items],
            )
            rows: list[tuple[str, str, str]] = []
            for p in items:
                rows.extend(part_number_rows(p))
            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO product_part_numbers "
                    "(parent_asin, part_number, source) VALUES (?, ?, ?)",
                    rows,
                )
        return len(items)

    for product in products:
        batch.append(product)
        if len(batch) >= batch_size:
            written += flush(batch)
            batch = []
    written += flush(batch)
    return written


def iter_products(category: str | None = None) -> Iterator[Product]:
    """Stream every product with its part numbers. Builds the indexes (stage 2).

    Streams rather than returns a list: 100k hydrated products is enough
    memory to matter on a laptop, and the index builders only need one at a time.
    """
    sql = """
    SELECT p.*, (
        SELECT json_group_array(ppn.part_number)
        FROM product_part_numbers ppn
        WHERE ppn.parent_asin = p.parent_asin
    ) AS part_numbers
    FROM products p
    """
    params: tuple = ()
    if category:
        sql += " WHERE p.category = ?"
        params = (category,)
    sql += " ORDER BY p.parent_asin"

    with connect(readonly=True) as conn:
        for row in conn.execute(sql, params):
            pns = json.loads(row["part_numbers"] or "[]")
            yield _row_to_product(row, [p for p in pns if p])


def get_products(parent_asins: Sequence[str]) -> dict[str, Product]:
    """Hydrate retrieval hits back into full products.

    Retrieval returns ids and scores; this is what turns them into something
    the agent can read aloud. Batched because a fused candidate list is 20-50
    ids and 50 round trips is silly.
    """
    if not parent_asins:
        return {}

    out: dict[str, Product] = {}
    with connect(readonly=True) as conn:
        for start in range(0, len(parent_asins), _CHUNK):
            chunk = list(parent_asins[start : start + _CHUNK])
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""
                SELECT p.*, (
                    SELECT json_group_array(ppn.part_number)
                    FROM product_part_numbers ppn
                    WHERE ppn.parent_asin = p.parent_asin
                ) AS part_numbers
                FROM products p WHERE p.parent_asin IN ({marks})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                pns = json.loads(row["part_numbers"] or "[]")
                out[row["parent_asin"]] = _row_to_product(row, [p for p in pns if p])
    return out


def count_by_category() -> dict[str, int]:
    """Per-category row counts -- the stage 1 done-check."""
    with connect(readonly=True) as conn:
        return {
            r["category"]: r["n"]
            for r in conn.execute(
                "SELECT category, count(*) AS n FROM products "
                "GROUP BY category ORDER BY category"
            )
        }


def count_with_part_numbers() -> dict[str, int]:
    """Per-category count of items carrying at least one identifier.

    This is the number that says whether the category mix did its job:
    Automotive should be high, Health_and_Household near zero. If they are not
    far apart, the extractor in normalize.py is wrong and every later result
    is measuring the wrong thing.
    """
    with connect(readonly=True) as conn:
        return {
            r["category"]: r["n"]
            for r in conn.execute(
                """
                SELECT p.category, count(DISTINCT p.parent_asin) AS n
                FROM products p
                JOIN product_part_numbers ppn ON ppn.parent_asin = p.parent_asin
                GROUP BY p.category ORDER BY p.category
                """
            )
        }


def part_number_sources() -> dict[str, int]:
    """How many identifiers came from each source. Diagnoses the extractor.

    If `title` dominates `details`, the title scraper is doing the work and
    its false-positive rate is the thing to worry about.
    """
    with connect(readonly=True) as conn:
        return {
            r["source"]: r["n"]
            for r in conn.execute(
                "SELECT source, count(*) AS n FROM product_part_numbers "
                "GROUP BY source ORDER BY n DESC"
            )
        }


# ------------------------------------------------------------------ stage 6 --


def open_call(audio_path: str | None = None, asr_model: str | None = None) -> str:
    """Create a `calls` row, return its call_id."""
    call_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO calls (call_id, audio_path, asr_model) VALUES (?, ?, ?)",
            (call_id, audio_path, asr_model),
        )
    return call_id


def append_turn(call_id: str, turn: dict) -> None:
    """Append one traced turn to `calls.turns`.

    Read-modify-write rather than a turns table: a call has a handful of turns,
    they are only ever read back whole, and keeping them as one JSON blob means
    the trace travels with the call row.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT turns FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no call {call_id!r}")
        turns = json.loads(row["turns"] or "[]")
        turns.append(turn)
        conn.execute(
            "UPDATE calls SET turns = ? WHERE call_id = ?", (_dumps(turns), call_id)
        )


def close_call(call_id: str, total_latency_ms: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE calls SET ended_at = datetime('now'), total_latency_ms = ? "
            "WHERE call_id = ?",
            (total_latency_ms, call_id),
        )


def save_cart(call_id: str, lines: list[dict], status: str = "open") -> str:
    """Persist the in-progress cart so an interrupted call is recoverable."""
    with connect() as conn:
        row = conn.execute(
            "SELECT cart_id FROM carts WHERE call_id = ? AND status = 'open'",
            (call_id,),
        ).fetchone()
        cart_id = row["cart_id"] if row else str(uuid.uuid4())
        if row:
            conn.execute(
                "UPDATE carts SET lines = ?, status = ?, "
                "updated_at = datetime('now') WHERE cart_id = ?",
                (_dumps(lines), status, cart_id),
            )
        else:
            conn.execute(
                "INSERT INTO carts (cart_id, call_id, status, lines) VALUES (?, ?, ?, ?)",
                (cart_id, call_id, status, _dumps(lines)),
            )
    return cart_id


def commit_order(
    call_id: str,
    parent_asin: str,
    quantity: int,
    query_text: str = "",
    nbest: list | None = None,
    candidates: list | None = None,
    confidence: float | None = None,
    was_confirmed: bool = False,
) -> str:
    """Write an order *with its trace*.

    The trace arguments are plain data rather than objects so this stays the
    only module that knows about rows -- and so a trace can be written even
    when the order came from typed input with no audio behind it.

    Without these columns a wrong order teaches nothing: there is no way to
    tell afterwards whether the ASR or the search was at fault.
    """
    order_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO orders
                (order_id, call_id, parent_asin, quantity, query_text,
                 nbest, candidates, confidence, was_confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                call_id,
                parent_asin,
                quantity,
                query_text,
                _dumps(nbest or []),
                _dumps(candidates or []),
                confidence,
                1 if was_confirmed else 0,
            ),
        )
    return order_id


def orders_for_call(call_id: str) -> list[dict]:
    with connect(readonly=True) as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM orders WHERE call_id = ? ORDER BY created_at", (call_id,)
            )
        ]

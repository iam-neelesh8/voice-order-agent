"""Invent the shop data Amazon's catalog does not carry.

Three things a parts counter needs and the source data has none of:

  prices     42% of the catalog has no price at all, and an agent that says
             "your total is $0" has told the caller something false
  inventory  "do you have it?" is unanswerable without a quantity
  customers  the call flow starts with a name and a phone number

Everything here is fabricated, and everything fabricated is marked. Prices get
`price_source = 'estimated'`, so a query can always separate what Amazon said
from what this file made up. A dataset that cannot tell you which half is real
is worse than one that is half empty.

Deterministic from a seed, like the rest of the project: the same database
seeded twice is the same database.
"""

from __future__ import annotations

import random
import uuid

SEED = 20260827

# Sampled from the real distribution rather than drawn from a range. The
# observed prices are heavily skewed -- Automotive runs from $0.63 to $3,088
# with a median of $35 -- and a uniform range would produce a catalog where
# everything costs about the same, which is not what a shop looks like.
_SAMPLE_SIZE = 4000

# Roughly a tenth of a real parts shelf is out of stock at any moment, and a
# further slice is down to the last one or two. Both matter: "we're out" and
# "I've only got one left" are different conversations.
OUT_OF_STOCK_SHARE = 0.12
LOW_STOCK_SHARE = 0.18

# US-style numbers in the 555-01xx block, which is reserved for fiction.
_FIRST_NAMES = """James Mary Robert Patricia John Jennifer Michael Linda David
    Elizabeth William Barbara Richard Susan Joseph Jessica Thomas Sarah Charles
    Karen Christopher Nancy Daniel Lisa Matthew Betty Anthony Margaret""".split()
_LAST_NAMES = """Smith Johnson Williams Brown Jones Garcia Miller Davis
    Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor
    Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark""".split()
_TRADE_SUFFIX = ["Auto Repair", "Motors", "Garage", "Service Centre", "Fleet Services"]


def fill_prices(rng: random.Random) -> dict:
    """Give every unpriced product a plausible price for its category.

    Sampled from the prices that category actually has, so the shape of the
    distribution survives -- a cheap category stays cheap and an expensive one
    stays expensive, rather than everything converging on one invented mean.
    """
    from voice_order.db.session import connect

    filled = 0
    with connect() as conn:
        conn.execute(
            "UPDATE products SET price_source = 'catalog' "
            "WHERE price IS NOT NULL AND price_source IS NULL"
        )

        categories = [r["category"] for r in conn.execute(
            "SELECT DISTINCT category FROM products ORDER BY category"
        )]

        for category in categories:
            observed = [
                r["price"] for r in conn.execute(
                    "SELECT price FROM products WHERE category = ? AND price IS NOT NULL "
                    "ORDER BY random() LIMIT ?",
                    (category, _SAMPLE_SIZE),
                )
            ]
            if not observed:
                continue

            missing = [
                r["parent_asin"] for r in conn.execute(
                    "SELECT parent_asin FROM products WHERE category = ? AND price IS NULL",
                    (category,),
                )
            ]
            updates = []
            for asin in missing:
                # Jittered so the invented prices are not literal copies of
                # real ones sitting next to them in the same table.
                price = rng.choice(observed) * rng.uniform(0.85, 1.15)
                updates.append((round(max(price, 0.5), 2), asin))

            conn.executemany(
                "UPDATE products SET price = ?, price_source = 'estimated' "
                "WHERE parent_asin = ?",
                updates,
            )
            filled += len(updates)

    return {"filled": filled}


def fill_inventory(rng: random.Random) -> dict:
    """Put a quantity on every product.

    Wholly invented. Amazon's metadata has no stock information and there is no
    honest way to derive one, so this is a plausible shape rather than a guess
    at the truth: mostly in stock, a tenth out, a fifth down to the last few.
    """
    from voice_order.db.session import connect

    with connect() as conn:
        asins = [r["parent_asin"] for r in conn.execute("SELECT parent_asin FROM products")]

        rows = []
        out = low = 0
        for asin in asins:
            roll = rng.random()
            if roll < OUT_OF_STOCK_SHARE:
                on_hand = 0
                out += 1
            elif roll < OUT_OF_STOCK_SHARE + LOW_STOCK_SHARE:
                on_hand = rng.randint(1, 3)
                low += 1
            else:
                on_hand = rng.randint(4, 60)
            rows.append((asin, on_hand, rng.choice([2, 3, 5])))

        conn.executemany(
            "INSERT INTO inventory (parent_asin, on_hand, reorder_level) VALUES (?, ?, ?) "
            "ON CONFLICT (parent_asin) DO UPDATE SET on_hand = excluded.on_hand, "
            "reorder_level = excluded.reorder_level, updated_at = datetime('now')",
            rows,
        )

    return {"products": len(asins), "out_of_stock": out, "low_stock": low}


def fill_customers(rng: random.Random, count: int = 250) -> dict:
    """A book of existing customers, so a caller can be recognised.

    Phone numbers use the 555-01xx block, which is reserved for fiction and
    cannot reach a real person.
    """
    from voice_order.db.session import connect

    rows = []
    used: set[str] = set()
    while len(rows) < count:
        phone = f"555{rng.randint(100, 199):03d}{rng.randint(0, 9999):04d}"
        if phone in used:
            continue
        used.add(phone)

        trade = rng.random() < 0.3
        if trade:
            name = f"{rng.choice(_LAST_NAMES)} {rng.choice(_TRADE_SUFFIX)}"
        else:
            name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        rows.append((str(uuid.UUID(int=rng.getrandbits(128))), phone, name,
                     "trade" if trade else "retail"))

    with connect() as conn:
        conn.executemany(
            "INSERT INTO customers (customer_id, phone, name, kind) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (phone) DO NOTHING",
            rows,
        )
        total = conn.execute("SELECT count(*) n FROM customers").fetchone()["n"]
        trade_n = conn.execute(
            "SELECT count(*) n FROM customers WHERE kind = 'trade'"
        ).fetchone()["n"]

    return {"customers": total, "trade": trade_n}


def seed_all(seed: int = SEED) -> dict:
    """Everything, in one deterministic pass."""
    rng = random.Random(seed)
    return {
        "prices": fill_prices(rng),
        "inventory": fill_inventory(rng),
        "customers": fill_customers(rng),
    }


def report(stats: dict) -> None:
    from voice_order.db.session import connect

    print(f"prices     {stats['prices']['filled']:,} estimated")
    with connect(readonly=True) as conn:
        for row in conn.execute(
            "SELECT coalesce(price_source, 'none') src, count(*) n "
            "FROM products GROUP BY src ORDER BY n DESC"
        ):
            print(f"             {row['src']:<12}{row['n']:>8,}")

    inv = stats["inventory"]
    print(f"inventory  {inv['products']:,} products")
    print(f"             {'in stock':<12}{inv['products']-inv['out_of_stock']:>8,}")
    print(f"             {'out':<12}{inv['out_of_stock']:>8,}")
    print(f"             {'low (1-3)':<12}{inv['low_stock']:>8,}")

    cust = stats["customers"]
    print(f"customers  {cust['customers']:,}  ({cust['trade']} trade, "
          f"{cust['customers']-cust['trade']} retail)")
    print()
    print("Everything above is invented. Estimated prices carry")
    print("price_source='estimated'; inventory and customers have no real source at all.")

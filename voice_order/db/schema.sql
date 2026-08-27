-- SQLite. Stage 1 (products) and stage 6 (calls, carts, orders).
--
-- This database is storage and nothing else. No full-text index, no vectors:
-- retrieval runs in-process against files under data/index/, rebuilt from
-- these tables. That split is deliberate --
--
--   * the retrieval numbers must not depend on which database you ran,
--     and SQLite FTS5 bm25() and Postgres ts_rank are different rankers;
--   * stage 5 has to ablate three retrieval components independently, which
--     is a constructor flag in Python and a schema migration in SQL.
--
-- Everything here is plain SQL that Postgres also accepts, apart from the
-- PRAGMAs. Swapping backends means writing one more repository, not
-- rewriting the schema.

PRAGMA journal_mode = WAL;      -- concurrent calls read while one writes
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------- stage 1 --

CREATE TABLE IF NOT EXISTS products (
    parent_asin     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,
    store           TEXT,
    price           REAL,
    average_rating  REAL,
    features        TEXT NOT NULL DEFAULT '[]',   -- JSON array
    details         TEXT NOT NULL DEFAULT '{}',   -- JSON object
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);

-- Part numbers get their own table rather than a JSON column on products.
-- They are extracted once at load time (stage 1), and "how many Automotive
-- items actually carry an identifier" is a question worth being able to ask
-- in SQL -- it is the split that every later result is sliced by.
CREATE TABLE IF NOT EXISTS product_part_numbers (
    parent_asin  TEXT NOT NULL REFERENCES products (parent_asin) ON DELETE CASCADE,
    part_number  TEXT NOT NULL,          -- normalised form, see normalize.py
    source       TEXT NOT NULL,          -- title | details | features
    PRIMARY KEY (parent_asin, part_number)
);

CREATE INDEX IF NOT EXISTS ppn_number_idx ON product_part_numbers (part_number);

-- ------------------------------------------------- the shop around it --
--
-- None of this comes from Amazon. It is what a parts counter needs and the
-- catalog does not carry: who is calling, whether the thing is on the shelf,
-- and what an order is as opposed to a list of items.
--
-- Seeded by `voice-order seed`, which marks everything it invents so nothing
-- here is ever mistaken for real data.

CREATE TABLE IF NOT EXISTS customers (
    customer_id  TEXT PRIMARY KEY,
    -- The natural key. A caller gives a phone number, not an id, and it is
    -- how a returning customer is recognised.
    phone        TEXT NOT NULL UNIQUE,
    name         TEXT,
    kind         TEXT NOT NULL DEFAULT 'retail'
                 CHECK (kind IN ('retail', 'trade')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS inventory (
    parent_asin    TEXT PRIMARY KEY REFERENCES products (parent_asin) ON DELETE CASCADE,
    on_hand        INTEGER NOT NULL DEFAULT 0,
    reorder_level  INTEGER NOT NULL DEFAULT 2,
    location       TEXT NOT NULL DEFAULT 'MAIN',
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS inventory_on_hand_idx ON inventory (on_hand);

-- An order is a thing that happened, with a customer and a total. The old
-- `orders` table had one row per item and no header, so three items were
-- three unrelated rows and "this order came to $47.20" could not be said.
CREATE TABLE IF NOT EXISTS order_headers (
    order_id     TEXT PRIMARY KEY,
    call_id      TEXT REFERENCES calls (call_id),
    customer_id  TEXT REFERENCES customers (customer_id),
    status       TEXT NOT NULL DEFAULT 'placed'
                 CHECK (status IN ('placed', 'picking', 'shipped', 'cancelled')),
    subtotal     REAL NOT NULL DEFAULT 0,
    tax          REAL NOT NULL DEFAULT 0,
    total        REAL NOT NULL DEFAULT 0,
    -- How much of the total is real. The catalog cannot price 42% of itself,
    -- and an order that is half estimated must say so on its face.
    unpriced_lines INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS order_headers_customer_idx ON order_headers (customer_id);
CREATE INDEX IF NOT EXISTS order_headers_call_idx ON order_headers (call_id);

CREATE TABLE IF NOT EXISTS order_lines (
    order_id      TEXT NOT NULL REFERENCES order_headers (order_id) ON DELETE CASCADE,
    line_no       INTEGER NOT NULL,
    parent_asin   TEXT NOT NULL REFERENCES products (parent_asin),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    -- Copied at order time. A price that changes later must not silently
    -- rewrite what somebody already agreed to pay.
    unit_price    REAL,
    subtotal      REAL,

    -- The trace stays on the line, because it is per item: this is the
    -- utterance and the candidates that put THIS product on the order.
    query_text    TEXT,
    nbest         TEXT NOT NULL DEFAULT '[]',
    candidates    TEXT NOT NULL DEFAULT '[]',
    confidence    REAL,
    was_confirmed INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (order_id, line_no)
);

CREATE INDEX IF NOT EXISTS order_lines_asin_idx ON order_lines (parent_asin);

-- ---------------------------------------------------------------- stage 6 --

CREATE TABLE IF NOT EXISTS calls (
    call_id           TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at          TEXT,
    audio_path        TEXT,
    turns             TEXT NOT NULL DEFAULT '[]',   -- JSON: full turn-by-turn trace
    asr_model         TEXT,
    total_latency_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS carts (
    cart_id     TEXT PRIMARY KEY,
    call_id     TEXT NOT NULL REFERENCES calls (call_id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'committed', 'abandoned')),
    lines       TEXT NOT NULL DEFAULT '[]',         -- JSON
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS carts_call_idx ON carts (call_id);

CREATE TABLE IF NOT EXISTS orders (
    order_id       TEXT PRIMARY KEY,
    call_id        TEXT NOT NULL REFERENCES calls (call_id),
    parent_asin    TEXT NOT NULL REFERENCES products (parent_asin),
    quantity       INTEGER NOT NULL CHECK (quantity > 0),

    -- The trace. Without these four columns a wrong order teaches nothing:
    -- there is no way to tell whether ASR or retrieval was at fault.
    query_text     TEXT NOT NULL,                   -- what the caller was heard to say
    nbest          TEXT NOT NULL DEFAULT '[]',      -- JSON: every ASR hypothesis + score
    candidates     TEXT NOT NULL DEFAULT '[]',      -- JSON: retrieved candidates + component scores
    confidence     REAL,

    was_confirmed  INTEGER NOT NULL DEFAULT 0,      -- 0/1
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS orders_call_idx ON orders (call_id);
CREATE INDEX IF NOT EXISTS orders_asin_idx ON orders (parent_asin);

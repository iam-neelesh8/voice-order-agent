"""SQLite connection handling. Stage 1.

One file, no server, no extensions. `git clone && pip install -e . &&
voice-order catalog` has to work on a laptop with nothing else installed --
every install step between a reader and the stage 4 number costs a reader.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from voice_order import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

TABLES = ("products", "product_part_numbers", "calls", "carts", "orders")

# PRAGMAs are per-connection, not stored in the file, so schema.sql setting
# them is not enough -- every connection has to reapply them.
_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
)


@contextmanager
def connect(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a connection to the database at `config.database_path()`.

    Commits on clean exit, rolls back on exception. Rows come back as
    `sqlite3.Row` so callers index by column name.
    """
    path = config.database_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path, isolation_level="DEFERRED")

    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        conn.execute(pragma)

    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> Path:
    """Apply `schema.sql`. Idempotent -- everything in it is IF NOT EXISTS."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(sql)
    return config.database_path()


def healthcheck() -> dict[str, Any]:
    """Row counts per table, plus which on-disk indexes exist.

    The second half matters: the database can look perfectly healthy while the
    retrieval indexes are missing or stale, and that failure is otherwise
    silent until recall mysteriously drops.
    """
    db = config.database_path()
    out: dict[str, Any] = {
        "database": str(db),
        "exists": db.is_file(),
        "size_mb": round(db.stat().st_size / 1e6, 1) if db.is_file() else 0.0,
        "tables": {},
        "indexes": {},
    }

    if db.is_file():
        with connect(readonly=True) as conn:
            present = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            for table in TABLES:
                if table in present:
                    out["tables"][table] = conn.execute(
                        f"SELECT count(*) FROM {table}"  # noqa: S608 - fixed identifiers
                    ).fetchone()[0]
                else:
                    out["tables"][table] = None  # schema not applied yet

    idx = config.index_dir()
    for name in ("bm25", "embeddings.npy", "ids.json", "part_numbers.json"):
        target = idx / name
        out["indexes"][name] = target.exists()

    return out

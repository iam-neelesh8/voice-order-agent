"""SQLite connection handling. Stage 1.

One file, no server, no extensions. `git clone && pip install -e . &&
voice-order catalog` has to work on a laptop with nothing else installed --
every install step between a reader and the stage 4 number costs a reader.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def connect(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a connection to the database at `config.database_path()`.

    Sets `row_factory` to `sqlite3.Row` so callers get named columns, applies
    the PRAGMAs from schema.sql (they do not persist across connections), and
    commits on clean exit / rolls back on exception.
    """
    raise NotImplementedError("stage 1")


def init_schema() -> None:
    """Apply `schema.sql`. Idempotent -- everything in it is IF NOT EXISTS."""
    raise NotImplementedError("stage 1")


def healthcheck() -> dict[str, Any]:
    """Row counts per table, plus which on-disk indexes exist under data/index/.

    The second half matters: the database can look perfectly healthy while the
    retrieval indexes are missing or stale, and that failure is otherwise
    silent until recall mysteriously drops.
    """
    raise NotImplementedError("stage 1")

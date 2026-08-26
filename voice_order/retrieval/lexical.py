"""Stage 2 -- BM25 over titles, features and part numbers. In-process.

The fitment-rag study found lexical matching wins on this corpus *because*
product identifiers are exact tokens. That is the baseline everything else
has to beat, so it is built first and kept honest.

Why not the database's full-text search: SQLite FTS5 `bm25()` and Postgres
`ts_rank` are different ranking functions, so the same query would score
differently depending on which backend happened to be running. For a project
whose entire output is numbers, that is disqualifying. The index is ours,
lives at `data/index/bm25/`, and gives identical results on every machine.
"""

from __future__ import annotations

from pathlib import Path

from voice_order.types import Candidate, Product

INDEX_NAME = "bm25"


def tokenize(text: str) -> list[str]:
    """Lowercase, split, keep alphanumerics.

    Deliberately does *not* stem: stemming "41-993" or "P0420" is destructive,
    and identifiers are the tokens that matter most here. Separator handling
    is left to `part_number.py`, which is the module that understands them.
    """
    raise NotImplementedError("stage 2")


def document_text(product: Product) -> str:
    """The string that gets indexed for one product.

    Title, store, features and the raw identifier forms, concatenated. Field
    weighting is done by repetition rather than by a weighted index, because
    it keeps the index a plain bag of words that can be swapped out.
    """
    raise NotImplementedError("stage 2")


class LexicalIndex:
    """A built BM25 index, loaded once and queried many times.

    Instantiating this per query is the obvious performance mistake and the
    reason it is a class rather than a module-level function.
    """

    def __init__(self, index_dir: Path, ids: list[str]) -> None:
        raise NotImplementedError("stage 2")

    @classmethod
    def build(cls, index_dir: Path | None = None) -> "LexicalIndex":
        """Stream products out of the database, tokenize, build, save to disk."""
        raise NotImplementedError("stage 2")

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "LexicalIndex":
        """Load a previously built index. Raises if it is missing."""
        raise NotImplementedError("stage 2")

    def search(
        self, query: str, top_k: int = 50, category: str | None = None
    ) -> list[Candidate]:
        """Score the query against every document, return the top k.

        Candidates come back with ids and scores only; `fusion` hydrates them
        once, after fusing, rather than 3x per retriever.
        """
        raise NotImplementedError("stage 2")

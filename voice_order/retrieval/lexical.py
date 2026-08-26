"""Stage 2 -- BM25 over titles, stores and part numbers. In-process.

The fitment-rag study found lexical matching wins on this corpus *because*
product identifiers are exact tokens. That is the baseline everything else
has to beat, so it is built first and kept honest.

Why not the database's full-text search: SQLite FTS5 `bm25()` and Postgres
`ts_rank` are different ranking functions, so the same query would score
differently depending on which backend happened to be running. For a project
whose entire output is numbers, that is disqualifying. The index is ours,
lives at `data/index/bm25/`, and gives identical results on every machine.

Why not bm25s's own tokenizer: it splits on word characters, so `41-993`
becomes `41` and `993`. That is the one thing this project cannot afford --
it destroys the identifier before retrieval ever sees it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from voice_order import config
from voice_order.types import Candidate, Product

INDEX_NAME = "bm25"
_IDS_FILE = "doc_ids.json"

# Tokens may contain separators; they are split off at the edges only.
_TOKEN_RE = re.compile(r"[a-z0-9](?:[a-z0-9\-/_.]*[a-z0-9])?")

# Deliberately tiny. The fitment-rag questions are phrased as "How much does
# the X cost?", so the interrogative frame is pure noise -- but anything
# beyond that risks eating a real product word.
_STOPWORDS = frozenset(
    """a an the of for and or to in on at by with from is are was were be been
    what which who whom whose how much many does do did make makes made cost
    costs price rating rate customers give given this that these those it its
    you your i me my we our they them their there here""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split, and emit identifier variants alongside raw tokens.

    Deliberately does *not* stem: stemming `41-993` or `P0420` is destructive,
    and identifiers are the tokens that matter most here.

    A token that carries a separator *and* a digit is emitted twice -- once as
    written and once collapsed (`41-993` -> also `41993`). That makes the
    index reachable from either spelling without a second retriever, and it
    costs one extra posting per identifier.
    """
    out: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        token = match.group(0)
        if token in _STOPWORDS:
            continue
        out.append(token)
        if any(c.isdigit() for c in token) and not token.isalnum():
            collapsed = re.sub(r"[^a-z0-9]", "", token)
            if collapsed and collapsed != token:
                out.append(collapsed)
    return out


def document_text(product: Product) -> str:
    """The string that gets indexed for one product.

    Title, store and the extracted identifiers. Feature bullets are left out
    for the same reason they are not a part-number source: they are mostly
    marketing copy and compatibility cross-references, and they would dilute
    every title term through BM25's length normalisation.

    Field weighting is by repetition rather than a weighted index, which keeps
    this a plain bag of words that can be swapped out.
    """
    parts = [product.title, product.title]        # title counts twice
    if product.store:
        parts.append(product.store)
    parts.extend(product.part_numbers)
    return " ".join(parts)


class LexicalIndex:
    """A built BM25 index, loaded once and queried many times.

    Instantiating this per query is the obvious performance mistake and the
    reason it is a class rather than a module-level function.
    """

    def __init__(self, retriever, doc_ids: list[str], categories: list[str]) -> None:
        self._retriever = retriever
        self.doc_ids = doc_ids
        self.categories = categories

    @classmethod
    def build(cls, index_dir: Path | None = None) -> "LexicalIndex":
        """Stream products out of the database, tokenize, build, save to disk."""
        import bm25s

        from voice_order.db import repository

        cfg = config.load("retrieval")
        target = Path(index_dir or config.index_dir()) / INDEX_NAME
        target.parent.mkdir(parents=True, exist_ok=True)

        doc_ids: list[str] = []
        categories: list[str] = []
        corpus: list[list[str]] = []
        for product in repository.iter_products():
            doc_ids.append(product.parent_asin)
            categories.append(product.category)
            corpus.append(tokenize(document_text(product)))

        retriever = bm25s.BM25(
            k1=float(cfg.get("lexical.k1", 1.2)),
            b=float(cfg.get("lexical.b", 0.75)),
        )
        retriever.index(corpus, show_progress=False)
        retriever.save(str(target))
        (target / _IDS_FILE).write_text(
            json.dumps({"doc_ids": doc_ids, "categories": categories}), encoding="utf-8"
        )
        return cls(retriever, doc_ids, categories)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "LexicalIndex":
        """Load a previously built index. Raises if it is missing."""
        import bm25s

        target = Path(index_dir or config.index_dir()) / INDEX_NAME
        meta_path = target / _IDS_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no BM25 index at {target} -- run `voice-order index lexical`"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        retriever = bm25s.BM25.load(str(target), mmap=False)
        return cls(retriever, meta["doc_ids"], meta["categories"])

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(
        self, query: str, top_k: int = 50, category: str | None = None
    ) -> list[Candidate]:
        """Score the query against every document, return the top k.

        Candidates come back as ids and scores only -- `fusion` hydrates them
        once, after fusing, rather than three times per query.
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        # Over-fetch when filtering, since the filter is applied after ranking.
        fetch = top_k * 10 if category else top_k
        fetch = min(fetch, len(self.doc_ids))

        idx, scores = self._retriever.retrieve([tokens], k=fetch, show_progress=False)

        out: list[Candidate] = []
        for doc_i, score in zip(idx[0].tolist(), scores[0].tolist()):
            if score <= 0:
                continue
            if category and self.categories[doc_i] != category:
                continue
            out.append(
                Candidate(
                    parent_asin=self.doc_ids[doc_i],
                    score=float(score),
                    component_scores={"lexical": float(score)},
                )
            )
            if len(out) >= top_k:
                break
        return out

"""Stage 2 -- embeddings and vector search. One numpy array, brute force.

Handles the queries lexical cannot: "the ceramic brake pads", "something for
a 2012 Civic".

On why there is no ANN index here (open question #3, answered): 100k vectors
at 384 dimensions in float32 is ~150 MB. A full scan is a single matmul --
memory-bandwidth bound, tens of milliseconds, and the whole eval set can be
batched into a handful of chunked matmuls. pgvector, FAISS and HNSW are all
solving a problem that starts around a million vectors. The right move is to
report the measured latency and move on; add an ANN index when the catalog
grows, not before.

On the embedding backend: fastembed runs bge-small-en-v1.5 through ONNX
Runtime, so this pulls no PyTorch. sentence-transformers would drag in a
multi-hundred-megabyte torch wheel for a 33M-parameter model, and install
weight is a real cost for a project meant to be forked and run.

Vectors are stored L2-normalised so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from voice_order import config
from voice_order.types import Candidate

EMBEDDINGS_FILE = "embeddings.npy"   # float32, shape (n_products, dim)
IDS_FILE = "ids.json"                # row i of the array is ids[i]

_MODEL_CACHE: dict[str, object] = {}


def _model(name: str):
    """One ONNX session per process. Reloading it per call dominates runtime."""
    if name not in _MODEL_CACHE:
        from fastembed import TextEmbedding

        _MODEL_CACHE[name] = TextEmbedding(model_name=name)
    return _MODEL_CACHE[name]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Encode with the model named in configs/retrieval.yaml, L2-normalised.

    CPU is fine for querying. Embedding the full catalog is the one step worth
    borrowing a GPU for, and it takes about a minute on a T4.
    """
    cfg = config.load("retrieval")
    name = str(cfg.get("dense.model", "BAAI/bge-small-en-v1.5"))
    vectors = np.asarray(
        list(_model(name).embed(list(texts), batch_size=batch_size)), dtype=np.float32
    )
    return _normalise(vectors)


def document_text(product) -> str:
    """What gets embedded.

    Title and store only. Identifiers are deliberately left out: a dense model
    maps `41-993` to a fuzzy neighbourhood of other digit strings, which is
    the opposite of what an exact identifier needs. Part numbers are the
    lexical and part-number retrievers' job; this one covers the description
    words they miss.
    """
    return f"{product.title} {product.store}" if product.store else product.title


class DenseIndex:
    """Embeddings held in memory as one array. Loaded once, queried many times."""

    def __init__(self, vectors: np.ndarray, ids: list[str], categories: list[str]) -> None:
        self.vectors = vectors
        self.ids = ids
        self.categories = np.asarray(categories)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, index_dir: Path | None = None, batch_size: int | None = None) -> "DenseIndex":
        """Embed every product and write embeddings.npy + ids.json."""
        from voice_order.db import repository

        cfg = config.load("retrieval")
        target = Path(index_dir or config.index_dir())
        target.mkdir(parents=True, exist_ok=True)
        batch = int(batch_size or cfg.get("dense.batch_size", 64))

        ids: list[str] = []
        categories: list[str] = []
        texts: list[str] = []
        for product in repository.iter_products():
            ids.append(product.parent_asin)
            categories.append(product.category)
            texts.append(document_text(product))

        chunks = []
        step = max(batch * 64, 2048)
        for start in range(0, len(texts), step):
            chunks.append(embed_texts(texts[start : start + step], batch_size=batch))
            done = min(start + step, len(texts))
            print(f"    embedded {done:,}/{len(texts):,}", flush=True)
        vectors = np.vstack(chunks) if chunks else np.zeros((0, 384), dtype=np.float32)

        np.save(target / EMBEDDINGS_FILE, vectors)
        (target / IDS_FILE).write_text(
            json.dumps({"ids": ids, "categories": categories}), encoding="utf-8"
        )
        return cls(vectors, ids, categories)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "DenseIndex":
        """Memory-map the array from disk. Raises if it is missing."""
        target = Path(index_dir or config.index_dir())
        vec_path, ids_path = target / EMBEDDINGS_FILE, target / IDS_FILE
        if not vec_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                f"no dense index at {target} -- run `voice-order index dense`"
            )
        meta = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path, mmap_mode="r")
        return cls(vectors, meta["ids"], meta["categories"])

    def _rank(self, query_vec: np.ndarray, top_k: int, category: str | None) -> list[Candidate]:
        scores = np.asarray(self.vectors @ query_vec, dtype=np.float32)
        if category:
            scores = np.where(self.categories == category, scores, -np.inf)

        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Candidate(
                parent_asin=self.ids[i],
                score=float(scores[i]),
                component_scores={"dense": float(scores[i])},
            )
            for i in top
            if np.isfinite(scores[i])
        ]

    def search(
        self, query: str, top_k: int = 50, category: str | None = None
    ) -> list[Candidate]:
        return self._rank(embed_texts([query])[0], top_k, category)

    def search_batch(
        self, queries: Sequence[str], top_k: int = 50, chunk: int = 256
    ) -> list[list[Candidate]]:
        """Score many queries in one matmul. The eval harness path.

        Chunk the query side -- 10k queries x 100k products is a 4 GB score
        matrix if materialised whole, which is the one way to make this slow.
        """
        out: list[list[Candidate]] = []
        for start in range(0, len(queries), chunk):
            block = embed_texts(list(queries[start : start + chunk]))
            for row in block:
                out.append(self._rank(row, top_k, None))
        return out

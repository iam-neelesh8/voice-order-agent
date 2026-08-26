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

Vectors are stored L2-normalised so cosine similarity is a plain dot product.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from voice_order.types import Candidate

EMBEDDINGS_FILE = "embeddings.npy"   # float32, shape (n_products, dim)
IDS_FILE = "ids.json"                # row i of the array is ids[i]


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> "list[list[float]]":
    """Encode with the model named in configs/retrieval.yaml.

    CPU is fine for querying. Embedding the full catalog is the one step worth
    borrowing a GPU for, and it takes about a minute on a T4.
    """
    raise NotImplementedError("stage 2")


class DenseIndex:
    """Embeddings held in memory as one array. Loaded once, queried many times."""

    def __init__(self, vectors, ids: list[str]) -> None:
        raise NotImplementedError("stage 2")

    @classmethod
    def build(cls, index_dir: Path | None = None) -> "DenseIndex":
        """Embed every product and write embeddings.npy + ids.json."""
        raise NotImplementedError("stage 2")

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "DenseIndex":
        """Memory-map the array from disk. Raises if it is missing."""
        raise NotImplementedError("stage 2")

    def search(
        self, query: str, top_k: int = 50, category: str | None = None
    ) -> list[Candidate]:
        raise NotImplementedError("stage 2")

    def search_batch(self, queries: Sequence[str], top_k: int = 50) -> list[list[Candidate]]:
        """Score many queries in one matmul. The eval harness path.

        Chunk the query side -- 10k queries x 100k products is a 4 GB score
        matrix if materialised whole, which is the one way to make this slow.
        """
        raise NotImplementedError("stage 2")

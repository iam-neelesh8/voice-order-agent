"""Run the heavy embedding step somewhere else, bring the result back.

Embedding 100k catalog titles is ~3 hours on a laptop CPU and about a minute
on a rented GPU. Rather than make that a manual science project, it is two
commands and a file:

    voice-order export-embed-input      -> data/exports/embed_input.jsonl.gz
    (run notebooks/embed_catalog_gpu.py on Kaggle / Colab / anywhere)
    voice-order import-embeddings out/  -> data/index/embeddings.npy

The exported file carries the exact texts the local pipeline would have
embedded, so nothing about normalisation has to be reimplemented on the
remote box and the two cannot drift.

The import is verified, not trusted. Two things get checked before the index
is installed, because both failures are silent and would show up much later
as an unexplained drop in recall:

  1. The ids must match the current catalog exactly, in order. Re-running
     `voice-order catalog` after an export changes the row set, and an index
     built against the old one would map vectors to the wrong products.
  2. A sample of vectors is re-embedded locally and compared. Document and
     query vectors must come from the same model; if the remote box used a
     different backend or a non-quantised variant, cosine similarities are
     miscalibrated and every dense result is subtly wrong.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from voice_order import config
from voice_order.retrieval.dense import EMBEDDINGS_FILE, IDS_FILE, document_text

EXPORT_NAME = "embed_input.jsonl.gz"

# How many vectors to re-embed locally when verifying an import. Enough to
# catch a wrong model; small enough to finish in seconds on CPU.
_VERIFY_SAMPLE = 24
_VERIFY_MIN_COSINE = 0.99


def export_dir() -> Path:
    """Where exports land. Redirectable, so tests cannot clobber a real one.

    VOICE_ORDER_DB and VOICE_ORDER_INDEX_DIR were already overridable; this
    was not, and a test fixture duly overwrote a real 6 MB catalog export with
    its own 24-product one. Nothing failed -- the file was just quietly wrong.
    """
    import os

    return Path(os.environ.get("VOICE_ORDER_EXPORT_DIR") or config.data_dir() / "exports")


def catalog_fingerprint(ids: list[str]) -> str:
    """Stable hash of the row set an index was built against."""
    digest = hashlib.sha256()
    for asin in ids:
        digest.update(asin.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def export_embed_input(dest: Path | None = None) -> tuple[Path, int, str]:
    """Write every product's id and embedding text, gzipped.

    Returns (path, count, fingerprint). ~100k rows lands around 3 MB
    compressed -- small enough to attach to a notebook or upload as a dataset.
    """
    from voice_order.db import repository

    path = Path(dest or export_dir() / EXPORT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config.load("retrieval")
    ids: list[str] = []

    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        header = {
            "_meta": True,
            "model": str(cfg.get("dense.model", "BAAI/bge-small-en-v1.5")),
            "dim": int(cfg.get("dense.dim", 384)),
            "normalise": bool(cfg.get("dense.normalise", True)),
        }
        fh.write(json.dumps(header) + "\n")
        for product in repository.iter_products():
            ids.append(product.parent_asin)
            fh.write(
                json.dumps(
                    {"id": product.parent_asin, "text": document_text(product)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    fingerprint = catalog_fingerprint(ids)
    (path.parent / "embed_input.fingerprint").write_text(fingerprint, encoding="utf-8")
    return path, len(ids), fingerprint


def _expected_ids() -> list[str]:
    from voice_order.db import repository

    return [p.parent_asin for p in repository.iter_products()]


def import_embeddings(
    source: Path, index_dir: Path | None = None, skip_verify: bool = False
) -> dict:
    """Install embeddings produced elsewhere, after checking they belong here.

    `source` is a directory holding embeddings.npy and ids.json, or the .npy
    itself if ids.json sits beside it.
    """
    source = Path(source)
    vec_path = source / EMBEDDINGS_FILE if source.is_dir() else source
    ids_path = vec_path.parent / IDS_FILE

    if not vec_path.is_file():
        raise FileNotFoundError(f"no {EMBEDDINGS_FILE} at {vec_path}")
    if not ids_path.is_file():
        raise FileNotFoundError(f"no {IDS_FILE} beside {vec_path.name}")

    vectors = np.load(vec_path)
    meta = json.loads(ids_path.read_text(encoding="utf-8"))
    ids = meta["ids"] if isinstance(meta, dict) else list(meta)

    report: dict = {
        "vectors": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "ids": len(ids),
    }

    if vectors.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {vectors.shape}")
    if vectors.shape[0] != len(ids):
        raise ValueError(
            f"{vectors.shape[0]:,} vectors but {len(ids):,} ids -- these do not belong together"
        )

    # --- check 1: does this index describe the catalog we actually have? ---
    expected = _expected_ids()
    report["catalog_rows"] = len(expected)
    if ids != expected:
        if len(ids) != len(expected):
            raise ValueError(
                f"index covers {len(ids):,} products but the catalog holds "
                f"{len(expected):,}. Re-export and re-embed: the catalog changed."
            )
        first = next(
            i for i, (a, b) in enumerate(zip(ids, expected, strict=True)) if a != b
        )
        raise ValueError(
            f"index ids diverge from the catalog at row {first} "
            f"({ids[first]} vs {expected[first]}). Re-export and re-embed."
        )
    report["fingerprint"] = catalog_fingerprint(ids)

    # --- check 2: did the remote box use the same model we query with? ---
    if not skip_verify:
        from voice_order.db import repository
        from voice_order.retrieval.dense import embed_texts

        step = max(1, len(ids) // _VERIFY_SAMPLE)
        sample_ids = ids[::step][:_VERIFY_SAMPLE]
        products = repository.get_products(sample_ids)
        texts = [document_text(products[i]) for i in sample_ids]
        local = embed_texts(texts)

        rows = [ids.index(i) for i in sample_ids]
        remote = np.asarray(vectors[rows], dtype=np.float32)
        remote /= np.maximum(np.linalg.norm(remote, axis=1, keepdims=True), 1e-12)

        cosines = np.sum(local * remote, axis=1)
        report["verify_min_cosine"] = float(cosines.min())
        report["verify_mean_cosine"] = float(cosines.mean())

        if cosines.min() < _VERIFY_MIN_COSINE:
            raise ValueError(
                f"imported vectors do not match locally computed ones "
                f"(min cosine {cosines.min():.4f} < {_VERIFY_MIN_COSINE}). "
                "The remote box used a different model or variant. Document and "
                "query vectors must come from the same model or every dense "
                "score is miscalibrated. Re-run the notebook unmodified, or "
                "pass --skip-verify if you know what you are doing."
            )

    # Categories are filled in locally rather than shipped to the remote box
    # and back. Check 1 already proved the id order matches the catalog row
    # for row, so this is exact -- and it keeps the export to id + text, which
    # is the only thing the GPU side actually needs.
    from voice_order.db import repository

    lookup = {p.parent_asin: p.category for p in repository.iter_products()}
    categories = [lookup[i] for i in ids]

    target = Path(index_dir or config.index_dir())
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / EMBEDDINGS_FILE, vectors.astype(np.float32))
    (target / IDS_FILE).write_text(
        json.dumps({"ids": ids, "categories": categories}), encoding="utf-8"
    )
    report["installed"] = str(target)
    return report

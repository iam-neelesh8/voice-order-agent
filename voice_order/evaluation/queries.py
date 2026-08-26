"""Stage 3 -- generate order queries with known answers.

Evaluation needs to know which product *should* have been found, so queries
are generated from the catalog rather than collected. Deterministic from a
seed, one product per query, answer known.

Dev and test are fixed and separate. Configuration is chosen on dev; test is
run once at the end of a stage and not looked at while deciding anything.

The loader for the fitment-rag lookup set lands here in stage 2, because
stage 2 needs something to measure against before the order generator exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from voice_order import config

LOOKUP_SET = "amazon_automotive_10k_n2000.jsonl"


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    question: str
    relevant_doc_ids: list[str]
    category: str
    # Stage 3 metadata -- lets results be sliced by what makes a query hard.
    has_part_number: bool = False
    has_brand: bool = False
    has_disfluency: bool = False
    quantity: int = 1
    kind: str = "lookup"
    extra: dict = field(default_factory=dict)


def evalset_dir() -> Path:
    return config.data_dir() / "evalsets"


def load_lookup_queries(path: Path | None = None) -> list[EvalQuery]:
    """Load the fitment-rag lookup set -- ground truth already verified.

    Phrased as catalog questions ("How much does the X cost?") rather than
    orders. Open question #2 is whether they transfer at all; the gap against
    the generated order set in stage 3 is the answer.

    Every gold id in this file was checked to be present in our catalog slice
    before it was adopted -- fitment-rag used the first 10k Automotive items,
    which is a subset of our first 40k. Without that overlap the recall would
    be zero by construction rather than by measurement.
    """
    path = Path(path or evalset_dir() / LOOKUP_SET)
    if not path.is_file():
        raise FileNotFoundError(
            f"no lookup evalset at {path}. It comes from the fitment-rag repo; "
            "see data/README.md."
        )

    out: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(
            EvalQuery(
                query_id=row["query_id"],
                question=row["question"],
                relevant_doc_ids=list(row["relevant_doc_ids"]),
                category="Automotive",   # this set is Automotive-only
                extra={
                    "source_field": row.get("source_field"),
                    "n_terms": row.get("n_terms"),
                },
            )
        )
    return out


# The split seed is separate from the generation seed and must never change.
# It decides which products belong to dev and which to test; changing it would
# silently leak test products into dev and invalidate every earlier number.
SPLIT_SEED = 20260826


def _split_products(split: str) -> list:
    """Products belonging to one split. Dev and test are disjoint by product.

    Splitting by product rather than by query matters: the same catalog row
    phrased two ways is still the same answer, and having it on both sides
    would let a choice made on dev quietly tune for a test item.
    """
    import random

    from voice_order.db import repository

    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev or test, got {split!r}")

    products = list(repository.iter_products())
    random.Random(SPLIT_SEED).shuffle(products)
    half = len(products) // 2
    return products[:half] if split == "dev" else products[half:]


def generate_order_queries(n: int, seed: int, split: str) -> list[EvalQuery]:
    """Template order phrasings and fill them from catalog fields.

    Covers: quantity, brand + identifier, partial names, disfluencies. The
    templates are the experiment -- they decide what "hard" means here.
    """
    from voice_order.evaluation import generate

    rows = generate.generate_for_products(_split_products(split), n, seed, split)
    return [
        EvalQuery(
            query_id=r["query_id"],
            question=r["question"],
            relevant_doc_ids=r["relevant_doc_ids"],
            category=r["category"],
            has_part_number=r["has_part_number"],
            has_brand=r["has_brand"],
            has_disfluency=r["has_disfluency"],
            quantity=r["quantity"],
            kind=r["kind"],
        )
        for r in rows
    ]


def order_set_path(split: str) -> Path:
    return evalset_dir() / f"orders_{split}.jsonl"


def write_split(queries: list[EvalQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in queries:
            fh.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "question": q.question,
                        "relevant_doc_ids": q.relevant_doc_ids,
                        "category": q.category,
                        "kind": q.kind,
                        "has_part_number": q.has_part_number,
                        "has_brand": q.has_brand,
                        "has_disfluency": q.has_disfluency,
                        "quantity": q.quantity,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_order_queries(split: str = "dev", path: Path | None = None) -> list[EvalQuery]:
    """Load a generated order split. Regenerate with `voice-order gen-queries`."""
    path = Path(path or order_set_path(split))
    if not path.is_file():
        raise FileNotFoundError(
            f"no order evalset at {path} -- run `voice-order gen-queries --split {split}`"
        )
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(
            EvalQuery(
                query_id=r["query_id"],
                question=r["question"],
                relevant_doc_ids=r["relevant_doc_ids"],
                category=r["category"],
                has_part_number=r.get("has_part_number", False),
                has_brand=r.get("has_brand", False),
                has_disfluency=r.get("has_disfluency", False),
                quantity=r.get("quantity", 1),
                kind=r.get("kind", "lookup"),
            )
        )
    return out

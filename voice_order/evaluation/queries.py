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
    has_disfluency: bool = False
    quantity: int = 1
    extra: dict = field(default_factory=dict)


def evalset_dir() -> Path:
    return config.DATA_DIR / "evalsets"


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


def generate_order_queries(n: int, seed: int, split: str) -> list[EvalQuery]:
    """Template order phrasings and fill them from catalog fields.

    Covers: quantity, brand + identifier, partial names, disfluencies. The
    templates are the experiment -- they decide what "hard" means here.
    """
    raise NotImplementedError("stage 3")


def write_split(queries: list[EvalQuery], path: Path) -> None:
    raise NotImplementedError("stage 3")

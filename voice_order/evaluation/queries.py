"""Stage 3 — generate order queries with known answers.

Evaluation needs to know which product *should* have been found, so queries
are generated from the catalog rather than collected. Deterministic from a
seed, one product per query, answer known.

Dev and test are fixed and separate. Configuration is chosen on dev; test is
run once at the end of a stage and not looked at while deciding anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    question: str
    relevant_doc_ids: list[str]
    category: str
    # Stage 3 metadata — lets results be sliced by what makes a query hard.
    has_part_number: bool = False
    has_disfluency: bool = False
    quantity: int = 1


def generate_order_queries(n: int, seed: int, split: str) -> list[EvalQuery]:
    """Template order phrasings and fill them from catalog fields.

    Covers: quantity, brand + identifier, partial names, disfluencies. The
    templates are the experiment — they decide what "hard" means here.
    """
    raise NotImplementedError("stage 3")


def load_lookup_queries(path: Path) -> list[EvalQuery]:
    """Load the fitment-rag lookup set — ground truth already verified.

    Phrased as catalog questions rather than orders. Open question #2 is
    whether they transfer at all; the gap against the generated order set is
    the answer.
    """
    raise NotImplementedError("stage 3")


def write_split(queries: list[EvalQuery], path: Path) -> None:
    raise NotImplementedError("stage 3")

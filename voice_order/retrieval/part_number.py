"""Stage 5 -- the retriever this project exists for.

Speech destroys identifiers. Stage 4 measured it: 84% word error rate on
queries carrying a part number against 15% without, and a six-times-larger
model recovering only 8% of the loss. The digits arrive correct and spelled as
English, so the fix is orthographic -- `spoken_digits` does that part.

This module is the other half: an index from normalised identifier straight to
product id, so a recovered `41993` becomes a retrieval hit rather than a token
BM25 has to compete over.

Why a separate retriever rather than better tokenisation in BM25: an
identifier is not evidence, it is an answer. When a caller says a part number
correctly there is exactly one right product, and a scoring function that
weighs it against title words will sometimes rank something else first. BM25
gets `brand_id` to 0.919 on typed text and this should get it higher, because
an exact identifier match is not a similarity judgement at all.

Kept behind `retrieval.part_number.enabled` so stages 2 and 4 measure a clean
baseline without it, and so the stage 5 ablation can turn it off.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from voice_order import config
from voice_order.retrieval.spoken_digits import candidates
from voice_order.types import Candidate

INDEX_FILE = "part_numbers.json"

# An identifier that maps to this many products is not identifying anything.
# Measured on the catalog: a handful of short numerics like "1000" collide
# across hundreds of items, and matching all of them is worse than matching
# none -- it buries the real answer under noise it cannot be ranked out of.
MAX_PRODUCTS_PER_IDENTIFIER = 25


class PartNumberIndex:
    """Normalised identifier -> product ids. A dictionary, not a ranker."""

    def __init__(self, exact: dict[str, list[str]]) -> None:
        self.exact = exact

    def __len__(self) -> int:
        return len(self.exact)

    @classmethod
    def build(cls, index_dir: Path | None = None) -> "PartNumberIndex":
        """Read `product_part_numbers` into a dict and save it.

        Identifiers shared by more than `MAX_PRODUCTS_PER_IDENTIFIER` products
        are dropped at build time rather than filtered at query time, so the
        cost is paid once and the index stays small.
        """
        from voice_order.db.session import connect

        target = Path(index_dir or config.index_dir())
        target.mkdir(parents=True, exist_ok=True)

        exact: dict[str, list[str]] = defaultdict(list)
        with connect(readonly=True) as conn:
            for row in conn.execute(
                "SELECT part_number, parent_asin FROM product_part_numbers"
            ):
                exact[row["part_number"]].append(row["parent_asin"])

        dropped = [k for k, v in exact.items() if len(v) > MAX_PRODUCTS_PER_IDENTIFIER]
        for key in dropped:
            del exact[key]

        payload = {"exact": exact, "dropped_ambiguous": len(dropped)}
        (target / INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")
        return cls(dict(exact))

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "PartNumberIndex":
        target = Path(index_dir or config.index_dir()) / INDEX_FILE
        if not target.is_file():
            raise FileNotFoundError(
                f"no part-number index at {target} -- run "
                "`voice-order index part-number`"
            )
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(payload["exact"])

    def search(
        self, query: str, top_k: int = 50, category: str | None = None
    ) -> list[Candidate]:
        """Pull every identifier out of the utterance and look each one up.

        `category` is accepted so every retriever presents the same interface
        to `fusion`; it is not used, because an identifier match is already
        specific enough that narrowing it by category cannot help and could
        only discard the right answer.

        Scores are 1.0 for an exact hit, divided by how many products share
        that identifier. A code matching one product is worth more than one
        matching twelve, and that is a property of the match rather than a
        tuned weight.
        """
        found: list[Candidate] = []
        seen: set[str] = set()

        for token in candidates(query):
            asins = self.exact.get(token)
            if not asins:
                continue
            score = 1.0 / len(asins)
            for asin in asins:
                if asin in seen:
                    continue
                seen.add(asin)
                found.append(
                    Candidate(
                        parent_asin=asin,
                        score=score,
                        component_scores={"part_number": score, "matched": token},
                    )
                )

        found.sort(key=lambda c: c.score, reverse=True)
        return found[:top_k]

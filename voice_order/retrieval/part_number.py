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


# Fuzzy matching is only offered for identifiers this long. Below it, a single
# edit reaches too many real neighbours to be useful -- "1000" is one edit from
# "1001", "1010", "2000" and dozens more, all real products, so a fuzzy hit is
# noise. Long codes have sparse neighbourhoods and a fuzzy hit is usually the
# digit Whisper got wrong.
_MIN_FUZZY_LENGTH = 6


def _deletions(token: str) -> set[str]:
    """Every string one character shorter. The SymSpell trick.

    Two identifiers are within edit distance 1 (one substitution, insertion or
    deletion) iff they share a deletion. So indexing deletions lets us find
    edit-1 neighbours with dictionary lookups instead of scanning 160k codes.
    """
    return {token[:i] + token[i + 1:] for i in range(len(token))}


class PartNumberIndex:
    """Normalised identifier -> product ids. A dictionary, not a ranker.

    Two lookups live here. Exact is the answer when the caller and Whisper both
    got the code right. Fuzzy is the fallback for when Whisper got one digit
    wrong -- it finds real catalog codes one edit away, so the agent has
    something to read back instead of nothing. Fuzzy always scores below exact,
    because a near-match is a guess and an exact match is not.
    """

    def __init__(
        self, exact: dict[str, list[str]], deletions: dict[str, list[str]] | None = None
    ) -> None:
        self.exact = exact
        # deletion-string -> the real identifiers that produce it
        self.deletions = deletions or {}

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

        # Deletion index for edit-distance-1 fuzzy fallback. Built only for
        # long-enough codes; short ones are excluded because their neighbours
        # are dense and every fuzzy hit would be noise.
        deletions: dict[str, list[str]] = defaultdict(list)
        for token in exact:
            if len(token) >= _MIN_FUZZY_LENGTH:
                for d in _deletions(token):
                    deletions[d].append(token)

        payload = {
            "exact": exact,
            "deletions": deletions,
            "dropped_ambiguous": len(dropped),
        }
        (target / INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")
        return cls(dict(exact), dict(deletions))

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "PartNumberIndex":
        target = Path(index_dir or config.index_dir()) / INDEX_FILE
        if not target.is_file():
            raise FileNotFoundError(
                f"no part-number index at {target} -- run "
                "`voice-order index part-number`"
            )
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(payload["exact"], payload.get("deletions"))

    def _fuzzy_neighbours(self, token: str) -> set[str]:
        """Real catalog identifiers within edit distance 1 of `token`.

        Found through the deletion index: two strings are edit-1 apart iff they
        share a deletion, so we compare `token` and its deletions against the
        index rather than against 160k codes.
        """
        if len(token) < _MIN_FUZZY_LENGTH:
            return set()
        neighbours: set[str] = set()
        # token itself as a deletion key catches insertions in the catalog code;
        # its deletions catch substitutions and deletions.
        for key in {token} | _deletions(token):
            neighbours.update(self.deletions.get(key, ()))
        neighbours.discard(token)
        return neighbours

    def search(
        self, query: str, top_k: int = 50, category: str | None = None, fuzzy: bool = True
    ) -> list[Candidate]:
        """Pull every identifier out of the utterance and look each one up.

        Exact first. Fuzzy only fills the gaps -- an exact hit is never
        displaced by a near one, because an exact match is the answer and a
        fuzzy match is a guess to read back.

        `category` is accepted so every retriever presents the same interface
        to `fusion`; it is not used, because an identifier match is already
        specific enough that narrowing it by category could only discard the
        right answer.

        Exact scores are 1.0 divided by how many products share the identifier
        (a code matching one product is worth more than one matching twelve).
        Fuzzy scores are held below any exact score by design.
        """
        found: list[Candidate] = []
        seen: set[str] = set()

        tokens = candidates(query)

        # Pass 1 -- exact.
        for token in tokens:
            for asin in self.exact.get(token, ()):
                if asin in seen:
                    continue
                seen.add(asin)
                score = 1.0 / len(self.exact[token])
                found.append(
                    Candidate(
                        parent_asin=asin,
                        score=score,
                        component_scores={"part_number": score, "matched": token},
                    )
                )

        # Pass 2 -- fuzzy, only for tokens that found no exact product. A code
        # that matched exactly does not need its typos guessed at.
        if fuzzy:
            for token in tokens:
                if token in self.exact:
                    continue
                for neighbour in self._fuzzy_neighbours(token):
                    asins = self.exact.get(neighbour, ())
                    for asin in asins:
                        if asin in seen:
                            continue
                        seen.add(asin)
                        # Capped below the weakest exact score, and further
                        # divided by how many products the neighbour matches.
                        score = 0.4 / len(asins)
                        found.append(
                            Candidate(
                                parent_asin=asin,
                                score=score,
                                component_scores={
                                    "part_number_fuzzy": score,
                                    "matched": neighbour,
                                    "heard": token,
                                },
                            )
                        )

        found.sort(key=lambda c: c.score, reverse=True)
        return found[:top_k]

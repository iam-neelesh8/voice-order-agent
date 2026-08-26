"""Stages 2 and 5 -- the retriever the rest of the project talks to.

Two axes of fusion:
  1. across retrievers  (lexical, dense, part-number)   -- stage 2
  2. across n-best ASR hypotheses                        -- stage 5

Axis 2 is what makes the agent robust to a mangled transcript: the correct
product need only surface for *one* hypothesis.

`Retriever` is the ablation surface. Because every retriever is an in-process
object rather than a database query, turning one off for a stage 5 run is a
constructor argument and a rerun -- not a schema change.
"""

from __future__ import annotations

from voice_order import config
from voice_order.types import Candidate, Transcript

ALL_RETRIEVERS = ("lexical", "dense", "part_number")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Candidate]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Fuse ranked lists by reciprocal rank.

    Rank-based rather than score-based on purpose: BM25 scores, cosine
    similarities and part-number tier scores are not on comparable scales, and
    normalising them introduces a tuning knob that would have to be fitted --
    on dev, and then defended.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}

    for name, ranked in ranked_lists.items():
        weight = float(weights.get(name, 1.0))
        for rank, candidate in enumerate(ranked, start=1):
            asin = candidate.parent_asin
            fused[asin] = fused.get(asin, 0.0) + weight / (k + rank)
            components.setdefault(asin, {}).update(candidate.component_scores)

    order = sorted(fused, key=lambda a: fused[a], reverse=True)
    return [
        Candidate(parent_asin=a, score=fused[a], component_scores=components.get(a, {}))
        for a in order
    ]


def _parse_retrievers(spec: str | list[str] | None) -> list[str]:
    if spec is None:
        return ["lexical"]
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in spec if s not in ALL_RETRIEVERS]
    if unknown:
        raise ValueError(f"unknown retriever(s) {unknown}; choose from {ALL_RETRIEVERS}")
    return list(spec)


class Retriever:
    """Holds the loaded indexes. Built once per process.

    Which retrievers are loaded is the ablation surface: `load(retrievers=...)`
    selects them by name, so stage 5 can answer "which of the three actually
    helped" rather than only "the bundle helped". Every retriever must accept
    the same `(query, top_k, category)` signature -- see `_search_all`.
    """

    def __init__(self, indexes: dict[str, object], use_nbest: bool = False) -> None:
        self.indexes = indexes
        self.use_nbest = use_nbest
        self._cfg = config.load("retrieval")

    @classmethod
    def load(
        cls, retrievers: str | list[str] | None = None, use_nbest: bool = False
    ) -> "Retriever":
        """Load whichever indexes are requested. Raises if one is missing."""
        names = _parse_retrievers(retrievers)
        indexes: dict[str, object] = {}

        if "lexical" in names:
            from voice_order.retrieval.lexical import LexicalIndex

            indexes["lexical"] = LexicalIndex.load()
        if "dense" in names:
            from voice_order.retrieval.dense import DenseIndex

            indexes["dense"] = DenseIndex.load()
        if "part_number" in names:
            from voice_order.retrieval.part_number import PartNumberIndex

            indexes["part_number"] = PartNumberIndex.load()

        return cls(indexes, use_nbest=use_nbest)

    def _search_all(
        self, query: str, top_k: int, category: str | None
    ) -> dict[str, list[Candidate]]:
        per_retriever: dict[str, list[Candidate]] = {}
        for name, index in self.indexes.items():
            fetch = int(self._cfg.get(f"{name}.top_k", 50))
            per_retriever[name] = index.search(query, top_k=max(fetch, top_k), category=category)
        return per_retriever

    def search_text(
        self, query: str, top_k: int = 20, category: str | None = None, hydrate: bool = False
    ) -> list[Candidate]:
        """One string in, fused candidates out. The stage 2/3 eval entry point.

        Hydration is off by default: the eval only compares ids, and looking up
        20 product rows per query across 2,000 queries is pure overhead. The
        agent turns it on because it has to read the product back to a caller.
        """
        per_retriever = self._search_all(query, top_k, category)

        # One retriever needs no fusion, and skipping RRF keeps the baseline
        # exactly what BM25 ranked rather than a re-ranked copy of it.
        if len(per_retriever) == 1:
            fused = next(iter(per_retriever.values()))[:top_k]
        else:
            fused = reciprocal_rank_fusion(
                per_retriever,
                k=int(self._cfg.get("fusion.rrf_k", 60)),
                weights=dict(self._cfg.get("fusion.weights", {})),
            )[:top_k]

        return self.hydrate(fused) if hydrate else fused

    @staticmethod
    def hydrate(candidates: list[Candidate]) -> list[Candidate]:
        """Attach full products in one batched lookup."""
        from voice_order.db import repository

        products = repository.get_products([c.parent_asin for c in candidates])
        return [
            Candidate(
                parent_asin=c.parent_asin,
                score=c.score,
                component_scores=c.component_scores,
                product=products.get(c.parent_asin),
                matched_hypothesis=c.matched_hypothesis,
            )
            for c in candidates
        ]

    def search_transcript(self, transcript: Transcript, top_k: int = 20) -> list[Candidate]:
        """Retrieve over every hypothesis and fuse, weighted by ASR score.

        The stage 4/5 eval entry point. With `use_nbest=False` this must reduce
        *exactly* to `search_text(transcript.best)` -- that equivalence is what
        makes the stage 5 ablation trustworthy, and it is worth a test.
        """
        raise NotImplementedError("stage 5")

    def confidence(self, candidates: list[Candidate]) -> float:
        """Calibrated 0-1 score for the top candidate.

        Drives the stage 6 commit / confirm / re-ask decision, so it has to
        reflect the *margin* over the runner-up and not just the absolute
        score. Two plausible part numbers scoring 0.9 each is the dangerous
        case, and an absolute threshold sails straight past it.
        """
        raise NotImplementedError("stage 5")


def build_all_indexes(which: str = "all") -> dict[str, int]:
    """Build the requested indexes from the database.

    Returns item counts per index. This is `voice-order index`, and it has to
    be rerun whenever the catalog changes -- the indexes are derived data, and
    nothing checks that they are in sync.
    """
    built: dict[str, int] = {}

    if which in ("all", "lexical"):
        from voice_order.retrieval.lexical import LexicalIndex

        print("  building lexical (BM25) ...", flush=True)
        built["lexical"] = len(LexicalIndex.build())

    if which in ("all", "dense"):
        from voice_order.retrieval.dense import DenseIndex

        print("  building dense (embeddings) ...", flush=True)
        built["dense"] = len(DenseIndex.build())

    if which in ("all", "part-number", "part_number"):
        from voice_order.retrieval.part_number import PartNumberIndex

        print("  building part-number ...", flush=True)
        built["part_number"] = len(PartNumberIndex.build())

    return built

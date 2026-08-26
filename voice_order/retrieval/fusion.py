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

from voice_order.types import Candidate, Transcript


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
    raise NotImplementedError("stage 2")


class Retriever:
    """Holds the three indexes. Built once per process.

    Ablation flags exist so stage 5 can answer "which of the three actually
    helped" instead of only "the bundle helped".
    """

    def __init__(
        self,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_part_number: bool = False,   # stage 5; off for the stage 2/4 baseline
        use_nbest: bool = False,         # stage 5; off for the stage 2/4 baseline
    ) -> None:
        raise NotImplementedError("stage 2")

    @classmethod
    def load(cls, **flags) -> "Retriever":
        """Load whichever indexes the flags require. Raises if one is missing."""
        raise NotImplementedError("stage 2")

    def search_text(
        self, query: str, top_k: int = 20, category: str | None = None
    ) -> list[Candidate]:
        """One string in, fused and hydrated candidates out.

        The stage 2/3 eval entry point.
        """
        raise NotImplementedError("stage 2")

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


def build_all_indexes() -> dict[str, int]:
    """Build lexical, dense and part-number indexes from the database.

    Returns item counts per index. This is `voice-order index all`, and it is
    the step that has to be rerun whenever the catalog changes -- the indexes
    are derived data, and nothing checks that they are in sync.
    """
    raise NotImplementedError("stage 2")

"""Stage 2 -- reciprocal rank fusion and the metrics.

RRF is rank-based on purpose: BM25 scores, cosine similarities and
part-number tier scores are not on comparable scales. These tests pin that
property, because the tempting "just normalise the scores" refactor would
quietly introduce a knob that has to be fitted and defended.
"""

from __future__ import annotations

from voice_order.evaluation import metrics
from voice_order.retrieval.fusion import reciprocal_rank_fusion
from voice_order.types import Candidate


def cand(asin: str, score: float = 1.0, **components) -> Candidate:
    return Candidate(parent_asin=asin, score=score, component_scores=components)


# --------------------------------------------------------------------- rrf --


def test_a_document_ranked_first_by_both_wins():
    fused = reciprocal_rank_fusion(
        {
            "lexical": [cand("A"), cand("B"), cand("C")],
            "dense": [cand("A"), cand("C"), cand("B")],
        }
    )
    assert fused[0].parent_asin == "A"


def test_raw_score_magnitude_does_not_leak_through():
    """BM25 scores in the tens must not outvote cosines below one."""
    huge = reciprocal_rank_fusion(
        {"lexical": [cand("A", 900.0), cand("B", 800.0)], "dense": [cand("B", 0.9)]}
    )
    tiny = reciprocal_rank_fusion(
        {"lexical": [cand("A", 0.9), cand("B", 0.8)], "dense": [cand("B", 0.09)]}
    )
    assert [c.parent_asin for c in huge] == [c.parent_asin for c in tiny]


def test_agreement_across_retrievers_beats_a_single_top_hit():
    """B is second in both lists; A is first in one and absent from the other."""
    fused = reciprocal_rank_fusion(
        {
            "lexical": [cand("A"), cand("B")],
            "dense": [cand("C"), cand("B")],
        }
    )
    assert fused[0].parent_asin == "B"


def test_weights_shift_the_outcome():
    lists = {"lexical": [cand("A"), cand("B")], "part_number": [cand("B"), cand("A")]}
    assert reciprocal_rank_fusion(lists)[0].parent_asin == "A"  # tie broken by order
    weighted = reciprocal_rank_fusion(lists, weights={"part_number": 5.0})
    assert weighted[0].parent_asin == "B"


def test_component_scores_survive_fusion():
    """The trace is the point -- a fused candidate must say what fired."""
    fused = reciprocal_rank_fusion(
        {
            "lexical": [cand("A", lexical=12.5)],
            "part_number": [cand("A", part_number=1.0)],
        }
    )
    assert fused[0].component_scores == {"lexical": 12.5, "part_number": 1.0}


def test_empty_lists_are_harmless():
    assert reciprocal_rank_fusion({"lexical": [], "dense": []}) == []


# ----------------------------------------------------------------- metrics --


def test_recall_at_k_respects_the_cutoff():
    ranked = [cand(x) for x in "ABCDE"]
    assert metrics.recall_at_k(ranked, ["C"], k=1) == 0.0
    assert metrics.recall_at_k(ranked, ["C"], k=3) == 1.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    ranked = [cand(x) for x in "ABC"]
    assert metrics.mrr(ranked, ["A"]) == 1.0
    assert metrics.mrr(ranked, ["C"]) == 1 / 3
    assert metrics.mrr(ranked, ["Z"]) == 0.0


def test_aggregate_groups_and_always_adds_an_overall_row():
    rows = [
        {"recall@1": 1.0, "mrr": 1.0, "category": "Automotive"},
        {"recall@1": 0.0, "mrr": 0.0, "category": "Automotive"},
        {"recall@1": 1.0, "mrr": 1.0, "category": "Electronics"},
    ]
    agg = metrics.aggregate(rows, by="category")
    assert agg["Automotive"]["recall@1"] == 0.5
    assert agg["Automotive"]["n"] == 2
    assert agg["Electronics"]["recall@1"] == 1.0
    assert agg["ALL"]["n"] == 3


def test_aggregate_can_slice_by_identifier_presence():
    """The stage 5 story lives in this split."""
    rows = [
        {"recall@1": 1.0, "mrr": 1.0, "has_part_number": True},
        {"recall@1": 0.0, "mrr": 0.0, "has_part_number": False},
    ]
    agg = metrics.aggregate(rows, by="has_part_number")
    assert agg["True"]["recall@1"] == 1.0
    assert agg["False"]["recall@1"] == 0.0

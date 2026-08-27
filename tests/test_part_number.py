"""Stage 5 -- the part-number retriever and n-best fusion.

The ablation on the dev set, phone condition, small.en, recall@1 on queries
that carry an identifier:

    lexical only                0.226
    + part-number               0.328
    + part-number + n-best      0.352
    n-best only                 0.183   <- worse than the baseline

That last row is the interesting one and these tests pin why. n-best is not
independently useful: four wrong transcripts outvote one right one under
rank-based fusion. It only pays off next to a retriever that can turn a single
correct hypothesis into a definitive hit, which is what exact identifier
matching is and BM25 is not.
"""

from __future__ import annotations

import pytest

from voice_order.retrieval.part_number import PartNumberIndex
from voice_order.types import Hypothesis, Transcript


@pytest.fixture
def index():
    return PartNumberIndex(
        {
            "41993": ["B-plug"],
            "P0420": ["B-sensor"],
            "3397118933": ["B-wiper"],
            "1000": [f"B-common-{i}" for i in range(12)],
        }
    )


def test_a_written_identifier_is_found(index):
    hits = index.search("I need an AC Delco 41-993 spark plug")
    assert [c.parent_asin for c in hits] == ["B-plug"]


def test_a_spoken_identifier_reaches_the_same_product(index):
    """The whole point of stage 5: this is what Whisper actually writes."""
    hits = index.search("I need an AC Delco forty one nine ninety three")
    assert "B-plug" in [c.parent_asin for c in hits]


def test_a_digit_by_digit_reading_works(index):
    hits = index.search("part number four one nine nine three please")
    assert "B-plug" in [c.parent_asin for c in hits]


def test_a_split_alphanumeric_code_is_rejoined(index):
    """ASR routinely writes P0420 as "P 0420"."""
    assert "B-sensor" in [c.parent_asin for c in index.search("the P 0420 sensor")]


def test_a_shared_identifier_scores_lower_than_a_unique_one(index):
    """A code matching twelve products is worth less than one matching one.

    That is a property of the match, not a tuned weight.
    """
    unique = index.search("41-993")[0].score
    shared = index.search("1000")[0].score
    assert unique > shared


def test_the_match_is_recorded_on_the_candidate(index):
    """Which token produced the hit, for tracing a wrong order later."""
    hit = index.search("an AC Delco 41-993")[0]
    assert hit.component_scores["matched"] == "41993"


def test_a_query_with_no_identifier_returns_nothing(index):
    assert index.search("do you have any brake pads") == []


def test_an_unknown_identifier_returns_nothing(index):
    assert index.search("the 55-555 widget") == []


def test_category_is_accepted_and_ignored(index):
    """Fusion passes it to every retriever; narrowing an exact identifier
    match could only discard the right answer."""
    with_filter = index.search("41-993", category="Automotive")
    without = index.search("41-993")
    assert [c.parent_asin for c in with_filter] == [c.parent_asin for c in without]


# ------------------------------------------------------------- n-best --


def transcript(*texts_and_scores) -> Transcript:
    return Transcript(
        audio_path=None,
        hypotheses=[
            Hypothesis(text=t, score=s, rank=i)
            for i, (t, s) in enumerate(texts_and_scores)
        ],
        duration_s=3.0,
        latency_ms=100.0,
    )


class StubIndex:
    """Records what it was asked, returns nothing."""

    def __init__(self):
        self.queries = []

    def search(self, query, top_k=50, category=None):
        self.queries.append(query)
        return []


def retriever(use_nbest: bool):
    from voice_order.retrieval.fusion import Retriever

    stub = StubIndex()
    return Retriever({"lexical": stub}, use_nbest=use_nbest), stub


def test_without_nbest_only_the_best_hypothesis_is_searched():
    """The equivalence the ablation depends on.

    With n-best off this must reduce exactly to searching the 1-best -- the
    same code path, not a reimplementation -- or the comparison between the
    two rows means nothing.
    """
    r, stub = retriever(use_nbest=False)
    r.search_transcript(transcript(("first guess", -0.2), ("second guess", -0.9)))
    assert stub.queries == ["first guess"]


def test_with_nbest_every_hypothesis_is_searched():
    r, stub = retriever(use_nbest=True)
    r.search_transcript(transcript(("first", -0.2), ("second", -0.9), ("third", -1.4)))
    assert stub.queries == ["first", "second", "third"]


def test_a_single_hypothesis_needs_no_fusion():
    r, stub = retriever(use_nbest=True)
    r.search_transcript(transcript(("only one", -0.2)))
    assert stub.queries == ["only one"]


def test_hypothesis_weights_favour_what_the_asr_preferred():
    from voice_order.retrieval.fusion import _hypothesis_weights

    weights = _hypothesis_weights([-0.2, -1.5, -3.0])
    assert weights[0] > weights[1] > weights[2]


def test_a_disliked_hypothesis_keeps_a_floor():
    """It must still be able to win when it is the only one carrying a
    usable identifier -- which is the entire reason to look at it."""
    weights = __import__(
        "voice_order.retrieval.fusion", fromlist=["_hypothesis_weights"]
    )._hypothesis_weights([-0.1, -20.0])
    assert weights[1] >= 0.05


def test_weighting_can_be_disabled_for_the_ablation():
    from voice_order.retrieval.fusion import _hypothesis_weights

    assert _hypothesis_weights([-0.2, -1.5, -3.0], enabled=False) == [1.0, 1.0, 1.0]

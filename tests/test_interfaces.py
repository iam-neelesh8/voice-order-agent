"""Contracts between modules that no single module's tests would catch.

Every bug pinned here was found by reading, not by a failing test, because
each one lives in the gap between two components that are individually fine.
"""

from __future__ import annotations

import inspect

import pytest

from voice_order.evaluation import metrics
from voice_order.retrieval import dense, fusion, lexical, part_number

RETRIEVERS = (
    ("lexical", lexical.LexicalIndex),
    ("dense", dense.DenseIndex),
    ("part_number", part_number.PartNumberIndex),
)


@pytest.mark.parametrize("name,cls", RETRIEVERS)
def test_every_retriever_accepts_the_same_search_signature(name, cls):
    """`fusion._search_all` calls all of them identically.

    part_number.search was missing `category` entirely. Nothing failed,
    because the retriever is off by default -- it would have raised TypeError
    the moment stage 5 switched it on, a long way from the cause.
    """
    params = inspect.signature(cls.search).parameters
    assert "query" in params
    assert "top_k" in params
    assert "category" in params, (
        f"{name}.search must accept `category`; fusion passes it to every retriever"
    )


@pytest.mark.parametrize("name,cls", RETRIEVERS)
def test_every_retriever_exposes_build_and_load(name, cls):
    """`build_all_indexes` and `Retriever.load` assume both exist."""
    assert hasattr(cls, "build") and hasattr(cls, "load")


def test_fusion_knows_exactly_the_retrievers_that_exist():
    """A name in ALL_RETRIEVERS with no loader would fail only at runtime."""
    source = inspect.getsource(fusion.Retriever.load)
    for name in fusion.ALL_RETRIEVERS:
        assert f'"{name}" in names' in source, f"{name} has no branch in Retriever.load"


def test_unknown_retriever_names_are_rejected_early():
    with pytest.raises(ValueError, match="unknown retriever"):
        fusion._parse_retrievers("lexical,telepathy")


# ------------------------------------------------------------------ tables --


def test_format_table_survives_a_single_group():
    """The old separator logic indexed [-2] and raised here."""
    agg = {"ALL": {"recall@1": 1.0, "mrr": 1.0, "n": 1}}
    out = metrics.format_table(agg, "one row")
    assert "ALL" in out


def test_format_table_puts_a_rule_above_the_total():
    rows = [{"recall@1": 1.0, "mrr": 1.0, "category": c} for c in ("Auto", "Elec")]
    out = metrics.format_table(metrics.aggregate(rows, by="category")).splitlines()
    assert out[-1].startswith("ALL")
    assert set(out[-2]) == {"-"}, "the total row should be separated from the groups"


def test_word_error_rate_reaches_the_reported_tables():
    """WER was computed per query and then dropped from every breakdown."""
    rows = [
        {"recall@1": 1.0, "mrr": 1.0, "wer": 0.2, "kind": "id_only"},
        {"recall@1": 0.0, "mrr": 0.0, "wer": 0.6, "kind": "noun_only"},
    ]
    agg = metrics.aggregate(rows, by="kind")
    assert agg["id_only"]["wer"] == pytest.approx(0.2)
    assert agg["ALL"]["wer"] == pytest.approx(0.4)


# ------------------------------------------------------------------ catalog --


def test_part_number_rows_takes_only_a_product():
    """It used to take a `raw` dict it never read; callers passed `{}`."""
    from voice_order.catalog.normalize import part_number_rows

    params = list(inspect.signature(part_number_rows).parameters)
    assert params == ["product"]


def test_catalog_loader_binds_its_loop_variables():
    """The inner generator closed over `path` and `name` (ruff B023).

    It worked only because `upsert_products` drained it inside the same
    iteration -- a property of the call order, not of the code.
    """
    from voice_order.catalog import load

    source = inspect.getsource(load.build_catalog)
    assert "def products(source_path=path, category=name" in source

"""Stage 3 -- the order query generator.

These queries are ground truth: they get committed, spoken by TTS in stage 4,
and every later number is measured against them. A generator that drifts
between runs would silently invalidate comparisons across stages, so
determinism is tested first and hardest.
"""

from __future__ import annotations

import pytest

from voice_order.evaluation import generate
from voice_order.types import Product


def product(
    asin="B001",
    title="AC Delco 41-993 Professional Iridium Spark Plug",
    store="ACDelco",
    part_numbers=("41993",),
    category="Automotive",
) -> Product:
    return Product(
        parent_asin=asin,
        title=title,
        category=category,
        store=store,
        part_numbers=list(part_numbers),
    )


# ------------------------------------------------------------ extraction --


def test_the_head_noun_comes_from_the_end_of_the_phrase():
    parts = generate.extract_parts(product())
    assert parts.noun == "Spark Plug"
    assert parts.modifier == "Iridium"


def test_the_identifier_keeps_its_written_spelling():
    """The catalog stores `41993`; a caller reads `41-993` off the box.

    Stage 4's TTS voices the two very differently, so the query has to carry
    the spelling, not the normalised form.
    """
    assert generate.extract_parts(product()).part_number == "41-993"


def test_packaging_copy_is_cut_before_the_noun_is_taken():
    """A comma with no space before it still ends the product name."""
    parts = generate.extract_parts(
        product(title="Wagner 0503000 Heat Gun, 2 Temp Settings 750F", part_numbers=("0503000",))
    )
    assert parts.noun == "Heat Gun"


def test_a_trailing_packaging_word_is_still_the_head_noun():
    """Stripping "Kit" would turn "Radio Install Kit" into "Radio Install"."""
    parts = generate.extract_parts(
        product(title="American International FMK549 Radio Install Kit",
                store="American International", part_numbers=("FMK549",))
    )
    assert parts.noun == "Install Kit"


def test_a_product_without_an_identifier_cannot_reach_the_id_rungs():
    parts = generate.extract_parts(product(title="Lavender Hand Soap", part_numbers=()))
    kinds = generate.eligible_kinds(parts)
    assert "id_only" not in kinds and "brand_id" not in kinds
    assert "noun_only" in kinds


# ----------------------------------------------------------- determinism --


def test_the_same_seed_produces_the_same_file():
    products = [product(asin=f"B{i:03d}") for i in range(60)]
    a = generate.generate_for_products(products, 30, seed=7, split="dev")
    b = generate.generate_for_products(products, 30, seed=7, split="dev")
    assert a == b


def test_a_different_seed_produces_different_queries():
    products = [product(asin=f"B{i:03d}") for i in range(60)]
    a = generate.generate_for_products(products, 30, seed=7, split="dev")
    b = generate.generate_for_products(products, 30, seed=8, split="dev")
    assert [r["question"] for r in a] != [r["question"] for r in b]


def test_a_product_is_used_at_most_once():
    """Otherwise one catalog row is gold for two queries and recall inflates."""
    products = [product(asin=f"B{i:03d}") for i in range(60)]
    rows = generate.generate_for_products(products, 40, seed=1, split="dev")
    gold = [d for r in rows for d in r["relevant_doc_ids"]]
    assert len(gold) == len(set(gold))


# ------------------------------------------------------------- rendering --


@pytest.mark.parametrize("kind", generate.KINDS)
def test_metadata_matches_the_text(kind):
    """has_part_number must mean the identifier is really in the utterance."""
    parts = generate.extract_parts(product())
    if kind not in generate.eligible_kinds(parts):
        pytest.skip(f"{kind} not eligible for this fixture")
    import random

    text, _ = generate.render(kind, parts, random.Random(3))
    expected = kind in ("id_only", "brand_id", "brand_id_noun")
    assert ("41-993" in text) == expected


def test_disfluency_never_produces_a_doubled_article():
    """The stutter used to collide with the quantity word: "a the ... the X"."""
    import random

    parts = generate.extract_parts(product())
    for seed in range(60):
        text, _ = generate.render("noun_only", parts, random.Random(seed), disfluent=True)
        assert " a the " not in text
        assert " the ... the the " not in text


def test_a_leading_pronoun_keeps_its_capital():
    """"Uh, i need" reads as a typo and would be voiced oddly by TTS."""
    import random

    parts = generate.extract_parts(product())
    for seed in range(60):
        text, _ = generate.render("noun_only", parts, random.Random(seed), disfluent=True)
        assert " i need " not in text
        assert " i want " not in text
        assert not text.startswith("i ")


def test_plural_quantities_pluralise_the_noun():
    import random

    parts = generate.extract_parts(product())
    seen_plural = False
    for seed in range(40):
        text, qty = generate.render("noun_only", parts, random.Random(seed))
        if qty > 1:
            seen_plural = True
            assert "Spark Plugs" in text
    assert seen_plural, "fixture never produced a plural -- widen the seed range"

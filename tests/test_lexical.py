"""Stage 2 -- the tokenizer.

bm25s ships its own tokenizer that splits on word characters, turning
`41-993` into `41` and `993`. That is the one thing this project cannot
afford, so the custom tokenizer is load-bearing and worth pinning.
"""

from __future__ import annotations

from voice_order.retrieval.lexical import tokenize


def test_identifiers_survive_tokenization():
    """The failure that motivates a custom tokenizer at all."""
    assert "41-993" in tokenize("AC Delco 41-993 spark plug")


def test_identifiers_are_also_emitted_collapsed():
    """So a query spelled either way reaches the same document."""
    tokens = tokenize("AC Delco 41-993")
    assert "41-993" in tokens
    assert "41993" in tokens


def test_plain_words_are_not_duplicated():
    assert tokenize("spark plug").count("spark") == 1


def test_hyphenated_words_without_digits_are_not_collapsed():
    """`heavy-duty` is a word, not an identifier -- one posting is enough."""
    assert tokenize("heavy-duty") == ["heavy-duty"]


def test_question_framing_is_dropped():
    """The lookup set is phrased "How much does the X cost?"."""
    assert tokenize("How much does the fel pro plenum cost?") == ["fel", "pro", "plenum"]


def test_case_is_folded():
    assert tokenize("Bosch") == tokenize("BOSCH") == ["bosch"]


def test_empty_input_is_safe():
    assert tokenize("") == []
    assert tokenize("   ?  ") == []

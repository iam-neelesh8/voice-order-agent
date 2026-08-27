"""Stage 5 -- spoken numbers back into figures.

Stage 4 measured the problem this solves: 84% word error rate on queries
carrying a part number against 15% without, and a six-times-larger model
recovering only 8% of the damage. Whisper hears the digits correctly and
writes them as English, so the failure is orthographic and a better acoustic
model cannot fix it.

These tests are the specification for that rewriting. Getting them wrong makes
stage 5 look like it does not work, for reasons unrelated to retrieval.
"""

from __future__ import annotations

import pytest

from voice_order.retrieval.spoken_digits import candidates, spoken_to_digits

# ------------------------------------------------------------- rewriting --


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("one two three", "1 2 3"),
        ("forty one", "41"),
        ("twenty three", "23"),
        ("ninety three", "93"),
        ("nineteen", "19"),
        ("fifteen", "15"),
    ],
)
def test_spoken_numbers_become_figures(spoken, expected):
    assert spoken_to_digits(spoken) == expected


@pytest.mark.parametrize("spoken,expected", [("oh", "0"), ("o", "0"), ("nought", "0")])
def test_zero_has_several_spoken_forms(spoken, expected):
    """People say "oh" for zero when reading a code aloud."""
    assert spoken_to_digits(spoken) == expected


def test_double_and_triple_are_repetition():
    assert spoken_to_digits("double seven") == "77"
    assert spoken_to_digits("triple eight") == "888"


@pytest.mark.parametrize(
    "spoken,expected",
    [("to", "2"), ("too", "2"), ("for", "4"), ("ate", "8"), ("won", "1")],
)
def test_homophones_are_read_as_numbers(spoken, expected):
    """ASR writes what it hears: "for" and "four" are the same sound."""
    assert spoken_to_digits(spoken) == expected


def test_spoken_separators_become_hyphens():
    assert spoken_to_digits("forty one dash nine") == "41 - 9"


def test_words_that_are_not_numbers_survive():
    assert spoken_to_digits("ac delco spark plug") == "ac delco spark plug"


def test_a_tens_word_alone_keeps_its_zero():
    """"twenty" is 20, but "twenty three" is 23, not 203."""
    assert spoken_to_digits("twenty") == "20"
    assert spoken_to_digits("twenty three") == "23"


# ------------------------------------------------------------ candidates --


def test_an_identifier_already_written_as_digits_is_found():
    assert "41993" in candidates("I need an AC Delco 41-993 spark plug")


def test_the_spoken_form_reaches_the_same_string():
    """The whole point: this is what Whisper actually writes.

    "forty one nine ninety three" is how a person reads 41-993 aloud, and
    the digits are correct -- only the spelling is wrong.
    """
    assert "41993" in candidates("I need an AC Delco forty one nine ninety three")


def test_digit_by_digit_reading_is_recovered():
    """"four one nine nine three" -- how people read codes over a bad line."""
    assert "41993" in candidates("part number four one nine nine three")


def test_a_split_alphanumeric_code_is_rejoined():
    """ASR routinely writes "P0420" as "P 0420"."""
    assert "P0420" in candidates("the P 0420 sensor")


def test_alphanumeric_identifiers_survive_intact():
    assert "AV1200" in candidates("a Black and Decker AV1200 vacuum")


def test_separators_are_dropped_from_the_lookup_form():
    """`41-993` and `41993` must reach the same index entry."""
    assert "41993" in candidates("the 41-993")
    assert "41993" in candidates("the 41 993")


def test_short_numbers_are_not_candidates():
    """"two" and "12" are quantities, not part numbers."""
    assert candidates("I need two of them") == []


def test_a_plain_sentence_yields_nothing():
    assert candidates("do you have any brake pads") == []


def test_over_generation_is_deliberate():
    """Ambiguous readings all come back; the index decides which exists.

    "forty one ninety three" is 4193 read as pairs, or part of 41993 read as
    "forty one, nine ninety three". Nothing in the text distinguishes them.
    """
    got = candidates("forty one ninety three")
    assert "4193" in got
    assert len(got) >= 1


def test_candidates_are_normalised_for_lookup():
    """Uppercase, alphanumeric only -- ready to hit the index directly."""
    for value in candidates("the ac-1200 and the P 0420"):
        assert value.isalnum()
        assert value == value.upper()


def test_a_real_mangled_transcript_still_yields_the_identifier():
    """From the stage 4 transcripts: what small.en actually produced."""
    assert "E3340" in candidates("I need a single case out E3340 skin cover")


def test_quantity_words_do_not_contaminate_the_identifier():
    """"two AC Delco 41-993" must not become "241993"."""
    got = candidates("I need two AC Delco 41-993 spark plugs")
    assert "41993" in got
    assert "241993" not in got

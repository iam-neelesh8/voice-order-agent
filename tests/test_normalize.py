"""Stage 1 -- the part-number extractor.

These are the tests worth having early. `extract_part_numbers` runs once, at
load time, and everything downstream inherits its mistakes: a missed
identifier is a product that can never be found, and a spurious one is a
product that matches queries it has nothing to do with. Both failures are
silent and show up as an unexplained recall number in stage 2.
"""

from __future__ import annotations

import pytest

from voice_order.catalog.normalize import (
    extract_part_numbers,
    normalize_item,
    normalize_part_number,
)


def ids(title="", details=None, features=None) -> set[str]:
    return {pn for pn, _ in extract_part_numbers(title, details or {}, features or [])}


def sources(title="", details=None, features=None) -> dict[str, str]:
    return dict(extract_part_numbers(title, details or {}, features or []))


# --------------------------------------------------------------- normalise --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("41-993", "41993"),
        ("41 993", "41993"),
        ("41.993", "41993"),
        ("41/993", "41993"),
        ("ac-41993", "AC41993"),
        ("P0420", "P0420"),
    ],
)
def test_separators_are_noise_letters_are_signal(raw, expected):
    assert normalize_part_number(raw) == expected


def test_letters_are_not_collapsed_away():
    """`AC41993` and `41993` are different parts and must stay different."""
    assert normalize_part_number("AC-41993") != normalize_part_number("41-993")


# ----------------------------------------------------------------- extract --


def test_finds_a_hyphenated_part_number_in_a_title():
    assert "41993" in ids("AC Delco 41-993 Professional Iridium Spark Plug")


def test_finds_a_long_numeric_part_number():
    assert "3397118933" in ids("Bosch 3397118933 Aerotwin Wiper Blade Set")


def test_finds_an_alphanumeric_code():
    assert "P0420" in ids("P0420 Catalytic Converter Diagnostic Tool")


@pytest.mark.parametrize(
    "title",
    [
        "Brake Pads for 2012 Honda Civic",
        "Fits 1998 Ford F-150",
    ],
)
def test_model_years_are_not_part_numbers(title):
    """A fitment year in a title is not an identifier.

    This deliberately also rejects genuine 4-digit part numbers in the
    1900-2099 range -- see the note in normalize.py for why that is the
    better trade on this corpus.
    """
    assert not {t for t in ids(title) if t.isdigit() and len(t) == 4}


@pytest.mark.parametrize(
    "title",
    [
        "12V LED Bulb",
        "10mm Socket Wrench",
        "3 Pack Microfiber Towels",
        "16.9oz Bottle",
        "5000mAh Power Bank",
    ],
)
def test_quantities_and_measures_are_rejected(title):
    assert ids(title) == set()


def test_short_tokens_are_rejected():
    assert ids("Set of 4 Wheel Nuts M12") == set()


def test_plain_names_yield_nothing():
    """Health_and_Household is in the catalog slice precisely for this case."""
    assert ids("Organic Lavender Hand Soap, Gentle Daily Cleanser") == set()


# ------------------------------------------------------------------ source --


def test_declared_detail_fields_win_over_the_title():
    got = sources(
        "AC Delco 41-993 Spark Plug",
        {"Manufacturer Part Number": "41-993"},
    )
    assert got["41993"] == "details"


def test_detail_keys_are_matched_loosely():
    """Punctuation and casing in the key must not matter."""
    assert "12345678" in ids("Widget", {"OEM Part Number:": "12345678"})
    assert "12345678" in ids("Widget", {"manufacturer part number": "12345678"})


def test_a_bare_number_before_a_unit_word_is_a_measure():
    """"5000mAh" is one token; "5000 IU" is two and needs a lookahead."""
    assert ids("Vitamin D 5000 IU Softgels") == set()
    assert ids("Bandages 200 Count Box") == set()
    # ...but a bare number that is not a measure is still an identifier
    assert "4348" in ids("Auto Meter 4348 Ultra-Lite Oil Temperature Gauge")


def test_feature_bullets_are_not_a_source():
    """Measured and removed, not forgotten.

    On a 500-item sample, features produced more identifiers than details and
    title combined while adding under two points of Automotive coverage. Most
    of what they contribute is compatibility cross-references, which index a
    product under *another* product's part number -- the one false-positive
    class that actively produces wrong orders.
    """
    assert ids("Spark Plug", features=["Replaces OEM 3397118933"]) == set()


def test_item_model_number_is_not_a_spoken_identifier():
    """An Amazon-internal SKU is on nearly every listing and nobody says it.

    Including it was measured to take Health_and_Household from 22% to 47%
    identifier coverage while adding under two points to Automotive -- it
    flattens the exact distinction the category mix exists to expose.
    """
    assert ids("Selenium Tablets, 100 Count", {"Item model number": "4791949"}) == set()


def test_manufacturer_and_oem_part_numbers_are_spoken_identifiers():
    assert "APLN008" in ids("Lug Nuts", {"Manufacturer Part Number": "APL-N008"})
    assert "F4ZZ2B293A" in ids("Caliper", {"OEM Part Number": "F4ZZ2B293A"})


@pytest.mark.parametrize(
    "title",
    [
        "Microfiber Cleaning Cloths 15x15 Inch",
        "Drawstring Velvet Bags 40x48",
        "Contour Flip Pillow 10-in-1 Rest Positions",
        "Cotton Compression Socks 20-30mmHg",
        "Vitamin D 5000 IU Softgels",
    ],
)
def test_specs_and_dimensions_are_not_identifiers(title):
    """The Health_and_Household noise floor: dosages, dimensions, ratings."""
    assert ids(title) == set()


def test_real_health_identifiers_still_survive():
    """Health is the easy case, not the empty case.

    It genuinely contains hardware -- AC adaptors, first aid cabinets -- with
    real part numbers. Rejecting those would be overcorrecting.
    """
    assert "A10166" in ids("Imak A10166 Mouse Wrist Cushion, Gray")
    assert "62003" in ids("Rapid Care First Aid 62003 3 Shelf First Aid Cabinet")


# ------------------------------------------------------------- normalize_item --


def test_records_without_an_id_or_title_are_dropped():
    assert normalize_item({"title": "No asin here"}, "Automotive") is None
    assert normalize_item({"parent_asin": "B01", "title": "  "}, "Automotive") is None


def test_prices_arrive_in_several_shapes():
    def price(value):
        return normalize_item(
            {"parent_asin": "B01", "title": "Thing", "price": value}, "Automotive"
        ).price

    assert price("$12.99") == 12.99
    assert price("1,299.00") == 1299.00
    assert price(12.5) == 12.5
    assert price("None") is None
    assert price(None) is None


def test_out_of_range_ratings_are_dropped():
    def rating(value):
        return normalize_item(
            {"parent_asin": "B01", "title": "Thing", "average_rating": value},
            "Automotive",
        ).average_rating

    assert rating(4.5) == 4.5
    assert rating("4.5") == 4.5
    assert rating(9.9) is None
    assert rating("n/a") is None


def test_category_is_the_slice_not_the_breadcrumb():
    """Per-category reporting is about which file the item came from."""
    product = normalize_item(
        {
            "parent_asin": "B01",
            "title": "Thing",
            "categories": ["Automotive", "Replacement Parts"],
        },
        "Automotive",
    )
    assert product.category == "Automotive"
    # nothing is lost -- the item's own breadcrumb is folded into details
    assert product.details["categories"] == ["Automotive", "Replacement Parts"]

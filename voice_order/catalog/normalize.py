"""Stage 1 -- raw metadata record -> `Product`.

The interesting part is `extract_part_numbers`. Doing it at load time rather
than query time means retrieval can index identifiers directly, which is what
makes the Automotive-vs-Health_and_Household split measurable.

Note the asymmetry with stage 5's query-side extractor: this one is *strict*.
A false positive here indexes a product under a garbage identifier and creates
spurious matches for every future query. On the query side a false candidate
costs one dict lookup. So the catalog over-rejects and the query over-accepts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from voice_order.types import Product

# Detail keys whose value is an identifier a caller would plausibly *say* --
# the kind printed on a box or quoted in a parts catalog. That is the test,
# not "is this field an identifier": the point of the index is to be reachable
# from speech.
#
# "Item model number" is deliberately absent. It is an Amazon-internal SKU
# present on nearly every listing, and including it was measured to take
# Health_and_Household from 22% identifier coverage to 47% while adding under
# two points to Automotive -- it flattens the exact distinction the category
# mix exists to expose. Nobody phones in an order for hand soap by SKU.
_DETAIL_KEYS = frozenset(
    {
        "part number",
        "manufacturer part number",
        "mpn",
        "oem part number",
        "model number",
        "model",
        "model name",
        "style number",
        "catalog number",
    }
)

# A token has to survive all of these to be treated as an identifier.
_MIN_LEN = 4
_MAX_LEN = 20

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# "12v", "10mm", "3pack", "16.9oz" -- quantities and measures, not identifiers.
_UNIT_RE = re.compile(
    r"^\d+(?:\.\d+)?\s?(?:"
    r"mm|cm|m|in|inch|inches|ft|feet|yd|yard|"
    r"v|volt|volts|w|watt|watts|a|amp|amps|ah|mah|wh|"
    r"oz|ounce|lb|lbs|pound|kg|g|gram|ml|l|liter|litre|qt|quart|gal|gallon|"
    r"pc|pcs|piece|pieces|pack|packs|pk|ct|count|set|sets|pair|pairs|"
    r"psi|rpm|hp|nm|hz|khz|mhz|ghz|kb|mb|gb|tb|k|x|"
    # dosages and pack copy -- the Health_and_Household noise floor
    r"mcg|mg|iu|ply|tablet|tablets|capsule|capsules|softgel|softgels|"
    r"serving|servings|sheet|sheets|roll|rolls|wipe|wipes|load|loads|"
    r"floz|tsp|tbsp|cal|kcal|mmhg|sqft|sq"
    r")$",
    re.IGNORECASE,
)

# "15x15", "40x48x2" -- dimensions, not identifiers.
_DIMENSION_RE = re.compile(r"^\d+(?:\.\d+)?(?:X\d+(?:\.\d+)?){1,2}$", re.IGNORECASE)

# "10-in-1", "3in1" -- marketing copy.
_NIN1_RE = re.compile(r"^\d+IN\d+$", re.IGNORECASE)

# "20-30mmHg" compression, "15-20mmHg" -- ratings, not identifiers.
_RANGE_UNIT_RE = re.compile(r"^\d+\d*(?:MMHG|MM|CM|IN|V|W|MG|MCG)$", re.IGNORECASE)

# "18-by-7-1/3-Foot", "24x36inch" -- spelled-out dimensions. These survive the
# other filters because they mix digits and letters like a real part number.
_SPELLED_DIMENSION_RE = re.compile(
    r"^\d.*(?:BY\d|FOOT|FEET|INCH|INCHES|YARD|METER|METRE)$", re.IGNORECASE
)

# Candidate tokens: start and end alphanumeric, may contain - / _ . inside.
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-/_.]*[A-Za-z0-9])?")

# The same units again, as standalone words. "5000mAh" is caught by _UNIT_RE,
# but "5000 IU" tokenizes into two tokens and the number sails through on its
# own. A bare number followed by a unit word is a measure, not a part number.
_UNIT_WORDS = frozenset(
    """mm cm in inch inches ft feet yd yard v volt volts w watt watts a amp amps
    ah mah wh oz ounce ounces lb lbs pound pounds kg g gram grams ml l liter
    liters litre qt quart gal gallon pc pcs piece pieces pack packs pk ct count
    set sets pair pairs psi rpm hp nm hz khz mhz ghz kb mb gb tb mcg mg iu ply
    tablet tablets capsule capsules softgel softgels serving servings sheet
    sheets roll rolls wipe wipes load loads floz tsp tbsp cal kcal mmhg
    sqft sq""".split()
)


def normalize_part_number(token: str) -> str:
    """Collapse an identifier to its comparison form.

    Uppercases and drops everything that is not alphanumeric, so `41-993`,
    `41 993` and `41.993` all become `41993`. The alphanumeric core is
    preserved, so `AC41993` stays distinct from `41993` -- separators are
    noise, letters are signal.
    """
    return re.sub(r"[^A-Za-z0-9]", "", token).upper()


def _is_identifier_shaped(normalized: str) -> bool:
    """Strict test, used on tokens scraped from free text."""
    if not (_MIN_LEN <= len(normalized) <= _MAX_LEN):
        return False
    if not any(c.isdigit() for c in normalized):
        return False
    if _YEAR_RE.match(normalized):
        # "Fits 2012 Honda Civic" -- a fitment year, not a part number.
        # This does reject genuine 4-digit part numbers in the 1900-2099
        # range. Accepting them would pull a model year into the identifier
        # index for a large share of the Automotive catalog, which is the
        # worse trade.
        return False
    if _UNIT_RE.match(normalized):
        return False
    if _DIMENSION_RE.match(normalized):
        return False
    if _NIN1_RE.match(normalized):
        return False
    if _RANGE_UNIT_RE.match(normalized):
        return False
    if _SPELLED_DIMENSION_RE.match(normalized):
        return False
    return True


def _tokens(text: str) -> Iterable[str]:
    for match in _TOKEN_RE.finditer(text or ""):
        yield match.group(0)


def _normalize_detail_key(key: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(key).lower()).strip()


def extract_part_numbers(
    title: str, details: dict[str, Any], features: list[str] | None = None
) -> list[tuple[str, str]]:
    """Pull identifier-shaped tokens out of an item.

    Returns `(normalized_part_number, source)` pairs, where source is
    `details` | `title`, deduplicated with the more trustworthy source
    winning. `features` is accepted and ignored -- see the note below.

    The source is stored so a later result can be sliced by where the
    identifier came from: if all the wins come from `details`, the title
    scraper is not earning its false positives.
    """
    found: dict[str, str] = {}

    # 1. Declared identifier fields. Highest trust: the key already asserts
    #    that the value is a part number, so only sanity limits apply.
    for key, value in (details or {}).items():
        if _normalize_detail_key(key) not in _DETAIL_KEYS:
            continue
        if not isinstance(value, str):
            continue
        for token in _tokens(value):
            norm = normalize_part_number(token)
            if _MIN_LEN <= len(norm) <= _MAX_LEN and any(c.isdigit() for c in norm):
                found.setdefault(norm, "details")

    # 2. Title tokens. Medium trust, full strictness, plus a one-token
    #    lookahead so "5000 IU" and "100 Count" are read as measures.
    title_tokens = list(_tokens(title or ""))
    for i, token in enumerate(title_tokens):
        norm = normalize_part_number(token)
        if not _is_identifier_shaped(norm):
            continue
        if norm.isdigit() and i + 1 < len(title_tokens):
            if title_tokens[i + 1].lower() in _UNIT_WORDS:
                continue
        found.setdefault(norm, "title")

    # Feature bullets are deliberately NOT a source. Measured on a 500-item
    # sample they produced more identifiers than details and title combined
    # while adding under two points of Automotive coverage, and most of what
    # they contribute is compatibility cross-references -- which index a
    # product under *another* product's part number. That is the one
    # false-positive class that actively produces wrong orders.
    #
    # KNOWN LIMITATION: titles carry cross-references too ("Compatible with
    # BRRC105, BRRC107, ..."), and this does not suppress them. It should show
    # up in stage 2 as a specific error class before it is worth the parsing.

    return sorted(found.items())


def _parse_price(value: Any) -> float | None:
    """Amazon prices arrive as "$12.99", "12.99", "", "None", or a number."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.replace(",", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else None


def _parse_rating(value: Any) -> float | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if 0.0 <= rating <= 5.0 else None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def normalize_item(raw: dict, category: str) -> Product | None:
    """Map one raw record to a `Product`, or None if required fields are missing.

    `category` is the catalog slice the record came from, not the item's own
    breadcrumb -- per-category reporting is about which file it was drawn from.
    The item's own `categories` list is folded into `details` so nothing is
    lost, since the schema has no column for it.
    """
    parent_asin = raw.get("parent_asin")
    title = raw.get("title")
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        return None
    if not isinstance(title, str) or not title.strip():
        return None

    details = raw.get("details")
    if not isinstance(details, dict):
        details = {}
    else:
        details = dict(details)

    item_categories = _as_list(raw.get("categories"))
    if item_categories:
        details.setdefault("categories", item_categories)

    features = _as_list(raw.get("features"))
    store = raw.get("store")

    return Product(
        parent_asin=parent_asin.strip(),
        title=title.strip(),
        category=category,
        store=store.strip() if isinstance(store, str) and store.strip() else None,
        price=_parse_price(raw.get("price")),
        average_rating=_parse_rating(raw.get("average_rating")),
        features=features,
        details=details,
        part_numbers=[pn for pn, _ in extract_part_numbers(title, details, features)],
    )


def part_number_rows(raw: dict, product: Product) -> list[tuple[str, str, str]]:
    """`(parent_asin, part_number, source)` rows for `product_part_numbers`."""
    pairs = extract_part_numbers(product.title, product.details, product.features)
    return [(product.parent_asin, pn, source) for pn, source in pairs]


def dumps(value: Any) -> str:
    """Compact JSON for the TEXT columns."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

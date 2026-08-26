"""Stage 3 -- turn catalog rows into things a caller would actually say.

Stage 2 established why this module exists. The fitment-rag lookup set gives a
real first number, but only 6% of its queries contain an identifier at all --
so it measures name-based retrieval and cannot test the premise the project is
built on. This generator puts part numbers in front of the retriever.

The organising idea is a SPECIFICITY LADDER. Real callers give wildly
different amounts of information, from a bare part number to "uh, the ceramic
ones", and retrieval difficulty is dominated by which rung they are on. Every
generated query is tagged with its rung, so results are reported per rung
rather than as one average that hides the whole effect.

The bottom rungs are deliberately ambiguous. "I need spark plugs" matches
hundreds of products and only one is marked gold, so recall there is low by
construction. That is not a flaw in the metric -- it is the signal that the
agent must ask a clarifying question instead of guessing, and stage 6 uses
exactly this to set its confirmation thresholds.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

from voice_order.types import Product

# ---------------------------------------------------------------------------
# Rungs of the ladder, most informative first.
# ---------------------------------------------------------------------------

KINDS = (
    "brand_id_noun",   # "two AC Delco 41-993 spark plugs"      -- everything
    "brand_id",        # "an AC Delco 41-993"                   -- no product word
    "id_only",         # "a 41-993"                             -- identifier alone
    "brand_noun",      # "AC Delco spark plugs"                 -- no identifier
    "noun_modifier",   # "the iridium spark plugs"              -- description only
    "noun_only",       # "spark plugs"                          -- the floor
)

_OPENERS = (
    "I need {q}{body}",
    "I'd like {q}{body}",
    "Can I get {q}{body}",
    "Can you get me {q}{body}",
    "I'm looking for {q}{body}",
    "Do you have {q}{body}",
    "Give me {q}{body}",
    "I want to order {q}{body}",
)

# Spoken quantities. The written digit forms matter for stage 4: TTS says
# "2" and "two" identically, but ASR does not always write them back the same.
_QUANTITIES = {
    1: ("a", "one", "a single"),
    2: ("two", "a pair of", "a couple of", "2"),
    3: ("three", "3"),
    4: ("four", "4", "a set of four"),
}

_DISFLUENCY_PREFIX = ("Uh, ", "Um, ", "So, ", "Yeah, so ", "Hi, uh, ")
_DISFLUENCY_SUFFIX = (", I think", ", if you have it", " please", ", whatever you've got")

# Title tails that are packaging copy rather than product identity.
_TITLE_CUT = re.compile(
    # Commas and parens need no leading space -- "Heat Gun, 2 Temp Settings"
    # must cut at the comma, or the head noun becomes "Temp Settings".
    r"\s*[,(\[|]\s*|\s+[-–—]\s+|\s+(?:for|fits|with|compatible|replaces|by)\s+",
    re.IGNORECASE,
)

_WORD = re.compile(r"^[A-Za-z][A-Za-z\-]*$")

# Words that are never the head noun of a product.
_NOT_A_NOUN = frozenset(
    """new pack set kit pcs pc piece pieces premium professional heavy duty
        original genuine oem universal replacement standard deluxe ultra
        super high low black white red blue green silver gold grey gray
        chrome clear large small medium mini compact adjustable""".split()
)


@dataclass(frozen=True)
class Parts:
    """The pieces of a product a caller might name."""

    brand: str | None
    part_number: str | None      # as spelled in the title, e.g. "41-993"
    noun: str | None             # "spark plug"
    modifier: str | None         # "iridium"


def _title_head(title: str) -> list[str]:
    """The part of a title before packaging copy takes over."""
    return _TITLE_CUT.split(title, maxsplit=1)[0].split()


def spoken_part_number(product: Product) -> str | None:
    """Recover the identifier as *written*, not as normalised.

    The catalog stores `41993`; the title says `41-993`. A caller reads the
    hyphenated form off a box, and stage 4's TTS will voice the two very
    differently, so the query has to carry the original spelling wherever it
    can be recovered.
    """
    if not product.part_numbers:
        return None

    normalised = set(product.part_numbers)
    for token in _title_head(product.title):
        stripped = token.strip(",.;:()[]")
        collapsed = re.sub(r"[^A-Za-z0-9]", "", stripped).upper()
        if collapsed in normalised and len(collapsed) >= 4:
            return stripped

    # Came from a details field with no title spelling to recover -- use the
    # longest normalised form, which is the most discriminative one.
    return max(product.part_numbers, key=len)


def extract_parts(product: Product) -> Parts:
    """Pull brand, identifier, head noun and modifier out of one product."""
    brand = product.store.strip() if product.store else None
    part_number = spoken_part_number(product)

    words = _title_head(product.title)

    brand_words = {w.lower() for w in (brand or "").split()}
    normalised = set(product.part_numbers)

    def is_identifier(word: str) -> bool:
        return re.sub(r"[^A-Za-z0-9]", "", word).upper() in normalised

    plain = [
        w.strip(",.;:()[]")
        for w in words
        if _WORD.match(w.strip(",.;:()[]"))
        and w.lower() not in brand_words
        and not is_identifier(w)
    ]

    noun = modifier = None
    if plain:
        # The head noun sits at the END of the descriptive phrase: "...
        # Iridium Spark Plug". Take it verbatim -- filtering packaging words
        # here would turn "Radio Install Kit" into "Radio Install".
        noun_words = plain[-2:]
        noun = " ".join(noun_words)
        # The modifier is the nearest preceding word that carries meaning,
        # so "Standard Brake Rotor" reaches past "Standard" for "C-Tek".
        for word in reversed(plain[: -len(noun_words)]):
            if word.lower() not in _NOT_A_NOUN and len(word) > 2:
                modifier = word
                break

    return Parts(brand=brand, part_number=part_number, noun=noun, modifier=modifier)


def eligible_kinds(parts: Parts) -> list[str]:
    """Which rungs this product can actually populate."""
    kinds: list[str] = []
    if parts.part_number:
        kinds.append("id_only")
        if parts.brand:
            kinds.append("brand_id")
            if parts.noun:
                kinds.append("brand_id_noun")
    if parts.noun:
        kinds.append("noun_only")
        if parts.brand:
            kinds.append("brand_noun")
        if parts.modifier:
            kinds.append("noun_modifier")
    return kinds


def _body(kind: str, parts: Parts, plural: bool) -> str:
    noun = parts.noun or ""
    if plural and noun and not noun.lower().endswith("s"):
        noun = noun + "s"

    if kind == "brand_id_noun":
        return f"{parts.brand} {parts.part_number} {noun}"
    if kind == "brand_id":
        return f"{parts.brand} {parts.part_number}"
    if kind == "id_only":
        return f"{parts.part_number}"
    if kind == "brand_noun":
        return f"{parts.brand} {noun}"
    if kind == "noun_modifier":
        return f"{parts.modifier} {noun}"
    if kind == "noun_only":
        return noun
    raise ValueError(f"unknown kind {kind!r}")


def _apply_disfluency(text: str, rng: random.Random, noun: str | None) -> str:
    """Make it sound like a phone call rather than a search box.

    Three kinds, because they break ASR differently: a filler prefix adds a
    token the retriever must ignore, a stutter duplicates a real word, and a
    trailing hedge puts noise after the useful content.
    """
    choice = rng.random()
    if choice < 0.4:
        prefix = rng.choice(_DISFLUENCY_PREFIX)
        # "Uh, i want to order" -- do not lowercase a bare "I".
        head = text if text[:1] == "I" else text[0].lower() + text[1:]
        return prefix + head
    if choice < 0.7 and noun:
        # Repeat the first word of the noun phrase rather than inserting an
        # article, which would collide with the quantity word already there
        # and produce "a the ... the Bed Extender".
        first = noun.split()[0]
        return text.replace(noun, f"{first}... {noun}", 1)
    return text + rng.choice(_DISFLUENCY_SUFFIX)


def render(
    kind: str, parts: Parts, rng: random.Random, disfluent: bool = False
) -> tuple[str, int]:
    """Build one utterance. Returns (text, quantity)."""
    quantity = 1
    if kind in ("brand_id_noun", "brand_noun", "noun_only", "noun_modifier"):
        quantity = rng.choices([1, 2, 3, 4], weights=[55, 25, 10, 10])[0]

    q_word = rng.choice(_QUANTITIES[quantity])
    body = _body(kind, parts, plural=quantity > 1)

    # "a" before a vowel sounds wrong and TTS will voice it that way.
    if q_word == "a" and body[:1].lower() in "aeiou":
        q_word = "an"

    text = rng.choice(_OPENERS).format(q=f"{q_word} " if q_word else "", body=body)
    text = re.sub(r"\s+", " ", text).strip()

    if disfluent:
        text = _apply_disfluency(text, rng, parts.noun)
    return text, quantity


def generate_for_products(
    products: Sequence[Product],
    n: int,
    seed: int,
    split: str,
    disfluency_rate: float = 0.30,
) -> list[dict]:
    """Generate `n` queries, spread as evenly as possible across the ladder.

    Deterministic: the same products, n and seed always produce the same file.
    One product yields at most one query, so no gold document appears twice
    and recall cannot be inflated by duplicates.
    """
    rng = random.Random(seed)

    by_kind: dict[str, list[tuple[Product, Parts]]] = {k: [] for k in KINDS}
    for product in products:
        parts = extract_parts(product)
        for kind in eligible_kinds(parts):
            by_kind[kind].append((product, parts))

    for pool in by_kind.values():
        rng.shuffle(pool)

    per_kind = max(1, n // len(KINDS))
    used: set[str] = set()
    rows: list[dict] = []

    # Rarest rung first, so a product that only fits one rung is not consumed
    # by a rung with plenty of other candidates.
    for kind in sorted(KINDS, key=lambda k: len(by_kind[k])):
        taken = 0
        for product, parts in by_kind[kind]:
            if taken >= per_kind or len(rows) >= n:
                break
            if product.parent_asin in used:
                continue
            used.add(product.parent_asin)
            disfluent = rng.random() < disfluency_rate
            text, quantity = render(kind, parts, rng, disfluent=disfluent)
            rows.append(
                {
                    "query_id": f"{split}-{len(rows):05d}",
                    "question": text,
                    "relevant_doc_ids": [product.parent_asin],
                    "category": product.category,
                    "kind": kind,
                    "has_part_number": kind in ("id_only", "brand_id", "brand_id_noun"),
                    "has_brand": kind in ("brand_id", "brand_id_noun", "brand_noun"),
                    "has_disfluency": disfluent,
                    "quantity": quantity,
                }
            )
            taken += 1

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["query_id"] = f"{split}-{i:05d}"
    return rows

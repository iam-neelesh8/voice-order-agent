"""Stage 5 -- turn spoken numbers back into figures.

The single highest-leverage function in the project. Stage 4 measured an 84%
word error rate on queries containing a part number against 15% without, and
showed that a six-times-larger model recovers 8% of the damage. Whisper does
not mishear identifiers so much as *rewrite* them: it hears the digits
correctly and writes them as English.

    "41-993"   ->  "forty one ninety three"
                   "41 99 3"
                   "four one dash nine nine three"
                   "41-99 3"

All four are the same number, and none of them matches `41993` in an index.
That is why a better acoustic model does not help: the acoustics were fine.
The failure is orthographic.

Deliberately over-generates. Every reading of an ambiguous phrase is returned,
because a false candidate costs one dictionary lookup while a missed one costs
the caller's order. That is the opposite of the catalog-side extractor in
`catalog/normalize.py`, which is strict -- indexing a product under a wrong
identifier creates spurious matches forever, whereas a wrong query candidate
is discarded within the millisecond.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "oh": 0, "o": 0, "nought": 0,
    "one": 1, "won": 1,
    "two": 2, "to": 2, "too": 2,
    "three": 3, "tree": 3,
    "four": 4, "for": 4, "fore": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8, "ate": 8,
    "nine": 9,
}

TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# "double seven" -> 77, "triple eight" -> 888. Common when reading a code out.
MULTIPLIERS = {"double": 2, "triple": 3, "treble": 3}

# Spoken separators. A caller says these; an index does not want them.
SEPARATORS = {"dash", "hyphen", "slash", "stroke", "point", "dot", "space"}

# Scale words. Not handled as arithmetic on purpose -- see `_scale_note`.
SCALES = {"hundred": 100, "thousand": 1000}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _digits_for(token: str) -> str | None:
    """One token -> the digits it spells, or None if it is not a number."""
    if token.isdigit():
        return token
    if token in UNITS:
        return str(UNITS[token])
    if token in TEENS:
        return str(TEENS[token])
    if token in TENS:
        return str(TENS[token])
    return None


def spoken_to_digits(text: str) -> str:
    """Rewrite spoken numbers in a sentence as figures, in place.

    "I need an AC Delco forty one ninety three"
        -> "i need an ac delco 41 93"

    Note that this alone does not solve the problem: "forty one ninety three"
    is 41-993 to a person reading a part number aloud, but 41 and 93 as plain
    arithmetic. Which is why `candidates` below returns several readings
    rather than one, and this function is only the first pass.
    """
    out: list[str] = []
    tokens = _tokens(text)
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # "double seven" -> "77"
        if token in MULTIPLIERS and i + 1 < len(tokens):
            digits = _digits_for(tokens[i + 1])
            if digits is not None:
                out.append(digits * MULTIPLIERS[token])
                i += 2
                continue

        # "twenty three" -> "23"; "twenty" alone stays "20"
        if token in TENS and i + 1 < len(tokens) and tokens[i + 1] in UNITS:
            unit = UNITS[tokens[i + 1]]
            if unit != 0:
                out.append(str(TENS[token] + unit))
                i += 2
                continue

        if token in SEPARATORS:
            out.append("-")
            i += 1
            continue

        digits = _digits_for(token)
        out.append(digits if digits is not None else token)
        i += 1

    return " ".join(out)


def _join_runs(tokens: list[str]) -> list[str]:
    """Glue adjacent numeric tokens into one, dropping spoken separators."""
    out: list[str] = []
    run: list[str] = []

    for token in tokens + ["\x00"]:
        if token.isdigit() or token == "-":
            run.append(token)
            continue
        if run:
            joined = "".join(t for t in run if t != "-")
            if joined:
                out.append(joined)
            run = []
        if token != "\x00":
            out.append(token)
    return out


def candidates(text: str, min_length: int = 4, max_length: int = 20) -> list[str]:
    """Every string in an utterance that could be the identifier.

    Returns normalised forms -- uppercase, alphanumeric only -- ready to look
    up directly.

    Several readings are produced for the same phrase on purpose. "forty one
    ninety three" is genuinely ambiguous: 4193 as spoken pairs, or 41993 if
    the speaker meant "forty one, nine ninety three". A person reading a part
    number off a box produces both patterns and it is not recoverable from the
    text, so both are offered and the index decides which exists.
    """
    found: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
        if min_length <= len(cleaned) <= max_length and cleaned not in found:
            found.append(cleaned)

    # 1. Anything already written as an identifier: "41-993", "P0420", "AV1200"
    for token in _tokens(text):
        if any(c.isdigit() for c in token):
            add(token)

    # 2. The digitised reading, with runs glued: "forty one ninety three"
    #    -> tokens 41, 93 -> "4193"
    digitised = _tokens(spoken_to_digits(text))
    for token in _join_runs(digitised):
        if any(c.isdigit() for c in token):
            add(token)

    # 3. Every maximal run of adjacent numeric tokens, concatenated. This is
    #    what catches "forty one nine ninety three" -> 41 9 93 -> "41993".
    run: list[str] = []
    for token in digitised + ["\x00"]:
        if token.isdigit():
            run.append(token)
            continue
        if len(run) > 1:
            add("".join(run))
            # Also the digit-by-digit reading: a caller saying "four one nine
            # nine three" produces single digits that must concatenate.
            for start in range(len(run)):
                for stop in range(start + 2, len(run) + 1):
                    add("".join(run[start:stop]))
        run = []

    # 4. The quantity glued to the front of the identifier. ASR writes "a
    #    CAT6A" as "1C-AT6A" and "one W-21411" as "1W-21411", so the leading
    #    digit is the article, not part of the code. Measured on the dev
    #    transcripts, this pattern was half of all near-misses.
    #
    #    Both readings are kept rather than choosing: "110WCB" is a real code
    #    beginning with 1 as often as it is "1" plus "10WCB", and the index is
    #    a better judge of which exists than a rule here would be.
    for value in list(found):
        if len(value) > min_length and value[0].isdigit():
            add(value[1:])

    # 5. A letter prefix attached to the following number: "P 0420" -> "P0420",
    #    which is how ASR often splits an alphanumeric code.
    tokens = _tokens(text)
    for i, token in enumerate(tokens[:-1]):
        if token.isalpha() and len(token) <= 3:
            nxt = _digits_for(tokens[i + 1])
            if nxt is not None:
                add(token + nxt)

    return found

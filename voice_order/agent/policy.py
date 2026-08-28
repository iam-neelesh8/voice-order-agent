"""Stage 6 -- commit, confirm, clarify, or re-ask.

This is the mechanism that turns a low-confidence retrieval into either a
correct order or a cheap re-ask, instead of a silently wrong one. It is the
difference between a demo and something you would let take real orders.

It is deliberately not the model's job. A model asked "are you confident?"
will say yes, fluently, and be wrong at exactly the rate it was already wrong.
A threshold can be read, tested, and moved.

The model is told what to *do* rather than what the number was -- a score is
meaningless to it without the thresholds, and handing it both would just be
inviting it to overrule them.
"""

from __future__ import annotations

import math

from voice_order.types import Candidate

# What the model is told for each decision. These are instructions, not
# explanations: they are the model's entire view of the confidence system.
GUIDANCE = {
    "commit": (
        "Strong match. Pick the best of the matches, add it with add_to_cart, "
        "and say what you added in one sentence so the caller can correct you."
    ),
    "confirm": (
        "Good match but not certain. Pick the best of the matches, add it, and "
        "say what you added and that they can correct it. Do NOT stop to ask "
        "first -- the whole order and total are read back at the end, which is "
        "where a wrong item gets caught."
    ),
    "clarify": (
        "Two or more products fit equally well and you cannot tell which. Ask "
        "one short question that separates them -- brand, or a digit of the "
        "part number. Only when they are genuinely tied."
    ),
    "reask": (
        "No usable match. Say you did not catch it and ask for the brand or the "
        "part number. Do not guess."
    ),
}


def confidence(candidates: list[Candidate]) -> float:
    """A 0-1 score for the top candidate.

    Two things decide it, and the second is the one that matters:

      strength -- how well the best candidate scored at all
      margin   -- how far clear of the runner-up it is

    A lone strong hit and a strong hit with an equally strong twin are very
    different situations, and only the margin separates them. Two plausible
    part numbers scoring alike is precisely the case that produces confidently
    wrong orders, so a thin margin caps the score no matter how high the raw
    number was.

    CALIBRATION: `_STRENGTH_SCALE` is a starting point, not a fitted value.
    BM25 scores have no fixed range, so it should be set from the dev set --
    run the stage 3 eval, take the score distribution of correct versus
    incorrect top hits, and pick the scale that separates them. Until that is
    done, treat the absolute number as ordinal and only trust the ordering.
    """
    if not candidates:
        return 0.0

    top = candidates[0].score
    if top <= 0:
        return 0.0

    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    margin = max(0.0, (top - runner_up) / top)

    # Saturating, so a very high score cannot buy its way past a thin margin.
    strength = 1.0 - math.exp(-top / _STRENGTH_SCALE)

    return round(strength * (0.35 + 0.65 * margin), 4)


# BM25 score at which `strength` reaches ~63%.
#
# Set so that an unambiguous match actually clears the commit threshold. At
# 12.0 it did not: a 24-vs-3 winner scored 0.795 against a 0.85 threshold, so
# every match fell through to "confirm" and the confident path was dead code.
# The agent would have read every single item back -- correct, but tedious
# enough that nobody would use it.
#
# Still provisional. See the calibration note in `confidence`.
_STRENGTH_SCALE = 6.0


def decide(candidates: list[Candidate], score: float | None = None) -> str:
    """-> "commit" | "confirm" | "clarify" | "reask"."""
    from voice_order import config

    if not candidates:
        return "reask"

    cfg = config.load("agent")
    commit_at = float(cfg.get("confidence.commit_threshold", 0.85))
    confirm_at = float(cfg.get("confidence.confirm_threshold", 0.45))
    min_margin = float(cfg.get("confidence.margin_requires_confirm", 0.10))

    score = confidence(candidates) if score is None else score

    top = candidates[0].score
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    margin = (top - runner_up) / top if top > 0 else 0.0

    if score < confirm_at:
        # Several near-identical candidates is a different problem from no
        # candidates, and needs a different question.
        return "clarify" if len(candidates) > 1 and margin < min_margin else "reask"

    # A thin margin forces a read-back even at a high absolute score. This is
    # the branch that catches the dangerous case.
    if score >= commit_at and margin >= min_margin:
        return "commit"

    return "confirm"


def readback(candidate: Candidate, quantity: int = 1) -> str:
    """Speakable confirmation text, for when the model is not driving.

    Never reads the full catalog title -- they run to 200 characters of
    keyword stuffing and are unspeakable. Brand, identifier, and a short noun
    phrase is what a person would say.
    """
    product = candidate.product
    if product is None:
        return f"{quantity} of item {candidate.parent_asin}"

    parts: list[str] = []
    if product.store:
        parts.append(product.store)
    if product.part_numbers:
        parts.append(max(product.part_numbers, key=len))

    words = [w for w in product.title.split() if w.isalpha()]
    if words:
        parts.append(" ".join(words[-2:]))

    name = " ".join(parts) if parts else product.title[:60]
    return f"{quantity} {name}" if quantity > 1 else name

"""The eval harness. One function per stage number.

Nothing downstream starts until the upstream number exists — otherwise there
is no way to tell which stage caused a regression.
"""

from __future__ import annotations

from typing import Any


def eval_typed_retrieval(split: str = "dev", query_set: str = "lookup") -> dict[str, Any]:
    """Stage 2/3 — recall@k and MRR on typed queries. No audio anywhere.

    The baseline every later stage is measured against.
    """
    raise NotImplementedError("stage 2")


def eval_spoken_retrieval(
    split: str = "dev", condition: str = "phone", nbest: bool = False
) -> dict[str, Any]:
    """Stage 4/5 — the same metrics, through ASR.

    `condition` is clean | phone | phone_snr20 | ... The drop from
    `eval_typed_retrieval` is the headline problem; n-best fusion and the
    part-number matcher are how much of it comes back.
    """
    raise NotImplementedError("stage 4")


def eval_end_to_end(split: str = "dev") -> dict[str, Any]:
    """Stage 6 — order accuracy, turns per order, latency per turn."""
    raise NotImplementedError("stage 6")


def eval_human(condition: str = "phone") -> dict[str, Any]:
    """Stage 8 — the ~100 human recordings.

    Never used for tuning. If synthetic and human disagree, the humans are
    right, and the synthetic pipeline is what needs fixing.
    """
    raise NotImplementedError("stage 8")

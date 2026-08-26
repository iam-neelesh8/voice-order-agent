"""Metrics, shared by every stage that reports a number.

Reported per category, never as one average -- Automotive and
Health_and_Household are in the slice precisely because they differ, and an
average hides the only interesting result.
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from voice_order.types import Candidate

DEFAULT_KS = (1, 5, 20)


def _ranked_ids(candidates: Sequence[Candidate]) -> list[str]:
    return [c.parent_asin for c in candidates]


def recall_at_k(candidates: Sequence[Candidate], relevant: Iterable[str], k: int) -> float:
    """Share of relevant documents that appear in the top k.

    With one gold document per query -- which is how both evalsets are built --
    this is hit-or-miss, and the mean over queries is the hit rate.
    """
    gold = set(relevant)
    if not gold:
        return 0.0
    top = _ranked_ids(candidates)[:k]
    return len(gold.intersection(top)) / len(gold)


def mrr(candidates: Sequence[Candidate], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant hit, 0 if it never appears."""
    gold = set(relevant)
    for rank, doc_id in enumerate(_ranked_ids(candidates), start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def score_query(
    candidates: Sequence[Candidate], relevant: Iterable[str], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    gold = list(relevant)
    out = {f"recall@{k}": recall_at_k(candidates, gold, k) for k in ks}
    out["mrr"] = mrr(candidates, gold)
    return out


def aggregate(per_query: list[dict], by: str = "category") -> dict[str, dict]:
    """Group per-query results and summarise.

    `by` also accepts "has_part_number" -- that split is the stage 5 story,
    and it is the reason every query carries its own metadata rather than the
    harness inferring it later.
    """
    if not per_query:
        raise ValueError(
            "aggregate() got no results to summarise -- the eval matched zero "
            "queries. Check the split, the limit, and that the evalset is built."
        )

    groups: dict[str, list[dict]] = {}
    for row in per_query:
        key = str(row.get(by, "all"))
        groups.setdefault(key, []).append(row)
    groups["ALL"] = list(per_query)

    metric_names = [
        k for k in per_query[0] if k.startswith("recall@") or k in ("mrr", "wer")
    ]
    out: dict[str, dict] = {}
    for key, rows in groups.items():
        summary = {m: mean(r[m] for r in rows) for m in metric_names}
        summary["n"] = len(rows)
        out[key] = summary
    return out


def format_table(agg: dict[str, dict], title: str = "") -> str:
    """Fixed-width report. ASCII only -- this has to render on a Windows console."""
    metric_names = [m for m in next(iter(agg.values())) if m != "n"]
    width = max(len(k) for k in agg) + 2

    lines = []
    if title:
        lines.append(title)
    header = f"{'group':<{width}}{'n':>7}" + "".join(f"{m:>12}" for m in metric_names)
    lines.append(header)
    lines.append("-" * len(header))

    # "ALL" sorts last and gets a rule above it. The order is computed once
    # rather than re-sorting the whole table on every row, and this is safe
    # when "ALL" is the only key -- the previous version indexed [-2] and
    # would have raised on a single-group table.
    order = sorted(agg, key=lambda k: (k == "ALL", k))
    for key in order:
        if key == "ALL" and len(order) > 1:
            lines.append("-" * len(header))
        row = agg[key]
        line = f"{key:<{width}}{row['n']:>7,}"
        line += "".join(f"{row[m]:>11.3f} " for m in metric_names)
        lines.append(line.rstrip())
    return "\n".join(lines)

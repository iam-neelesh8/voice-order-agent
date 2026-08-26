"""The eval harness. One function per stage number.

Nothing downstream starts until the upstream number exists -- otherwise there
is no way to tell which stage caused a regression.
"""

from __future__ import annotations

import time
from typing import Any

from voice_order.evaluation import metrics, queries


def eval_typed_retrieval(
    split: str = "dev",
    query_set: str = "lookup",
    limit: int | None = None,
    category: str | None = None,
    retrievers: str = "lexical",
) -> dict[str, Any]:
    """Stage 2/3 -- recall@k and MRR on typed queries. No audio anywhere.

    The baseline every later stage is measured against. `retrievers` selects
    which components are switched on, so the same harness reports the
    lexical-only baseline and the fused result without a second code path.
    """
    if query_set == "lookup":
        rows = queries.load_lookup_queries()
    elif query_set == "orders":
        rows = queries.load_order_queries(split)
    else:
        raise ValueError(f"unknown query_set {query_set!r}; use lookup or orders")

    if limit:
        rows = rows[:limit]

    from voice_order.retrieval.fusion import Retriever

    retriever = Retriever.load(retrievers=retrievers)

    per_query: list[dict] = []
    latencies: list[float] = []

    for q in rows:
        t0 = time.perf_counter()
        candidates = retriever.search_text(q.question, top_k=20, category=category)
        latencies.append((time.perf_counter() - t0) * 1000)

        row = metrics.score_query(candidates, q.relevant_doc_ids)
        row["category"] = q.category
        row["query_id"] = q.query_id
        row["kind"] = q.kind
        row["has_part_number"] = q.has_part_number
        row["has_disfluency"] = q.has_disfluency
        per_query.append(row)

    latencies.sort()
    return {
        "query_set": query_set,
        "split": split,
        "retrievers": retrievers,
        "n_queries": len(rows),
        "aggregate": metrics.aggregate(per_query, by="category"),
        "by_kind": metrics.aggregate(per_query, by="kind"),
        "by_identifier": metrics.aggregate(per_query, by="has_part_number"),
        "by_disfluency": metrics.aggregate(per_query, by="has_disfluency"),
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "mean": round(sum(latencies) / len(latencies), 2),
        },
        "per_query": per_query,
    }


def eval_spoken_retrieval(
    split: str = "dev",
    condition: str = "phone",
    nbest: bool = False,
    model: str | None = None,
    retrievers: str = "lexical",
    limit: int | None = None,
) -> dict[str, Any]:
    """Stage 4/5 -- the same metrics, through ASR.

    `condition` is clean | phone | phone_snr20 | ... The drop from
    `eval_typed_retrieval` is the headline problem; n-best fusion and the
    part-number matcher are how much of it comes back.

    Retrieval runs over cached transcripts, never over audio, so an ablation
    re-runs in seconds instead of re-transcribing hours of speech.
    """
    from voice_order.asr import batch
    from voice_order.asr.transcribe import word_error_rate
    from voice_order.retrieval.fusion import Retriever

    rows = queries.load_order_queries(split)
    if limit:
        rows = rows[:limit]

    transcripts = batch.load_transcripts(split, condition, model)
    retriever = Retriever.load(retrievers=retrievers, use_nbest=nbest)

    per_query: list[dict] = []
    missing = 0

    for q in rows:
        transcript = transcripts.get(q.query_id)
        if transcript is None:
            missing += 1
            continue

        if nbest:
            candidates = retriever.search_transcript(transcript, top_k=20)
        else:
            candidates = retriever.search_text(transcript.best, top_k=20)

        row = metrics.score_query(candidates, q.relevant_doc_ids)
        row["wer"] = word_error_rate(q.question, transcript.best)
        row["category"] = q.category
        row["kind"] = q.kind
        row["has_part_number"] = q.has_part_number
        row["has_disfluency"] = q.has_disfluency
        per_query.append(row)

    if not per_query:
        raise RuntimeError(
            f"no transcripts matched {split}/{condition} -- transcribe it first"
        )

    return {
        "query_set": f"orders ({condition})",
        "split": split,
        "condition": condition,
        "retrievers": retrievers + (" +nbest" if nbest else ""),
        "n_queries": len(per_query),
        "missing_transcripts": missing,
        "wer": round(sum(r["wer"] for r in per_query) / len(per_query), 4),
        "aggregate": metrics.aggregate(per_query, by="category"),
        "by_kind": metrics.aggregate(per_query, by="kind"),
        "by_identifier": metrics.aggregate(per_query, by="has_part_number"),
        "by_disfluency": metrics.aggregate(per_query, by="has_disfluency"),
        "latency_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0},
        "per_query": per_query,
    }


def eval_end_to_end(split: str = "dev") -> dict[str, Any]:
    """Stage 6 -- order accuracy, turns per order, latency per turn."""
    raise NotImplementedError("stage 6")


def eval_human(condition: str = "phone") -> dict[str, Any]:
    """Stage 8 -- the ~100 human recordings.

    Never used for tuning. If synthetic and human disagree, the humans are
    right, and the synthetic pipeline is what needs fixing.
    """
    raise NotImplementedError("stage 8")


def report(result: dict[str, Any]) -> None:
    """Print an eval result. ASCII only, for the Windows console."""
    print()
    print(
        f"{result['query_set']} queries"
        f"  |  retrievers: {result['retrievers']}"
        f"  |  n = {result['n_queries']:,}"
    )
    print()
    if "wer" in result:
        print(f"word error rate (1-best): {result['wer']:.3f}")
        if result.get("missing_transcripts"):
            print(f"  ! {result['missing_transcripts']:,} queries had no transcript")
        print()
    print(metrics.format_table(result["aggregate"], "by category"))
    for key, title in (
        ("by_kind", "by specificity rung -- how much the caller gave us"),
        ("by_identifier", "by whether the query carries an identifier"),
        ("by_disfluency", "by whether the query is disfluent"),
    ):
        if result.get(key):
            print()
            print(metrics.format_table(result[key], title))
    lat = result["latency_ms"]
    if lat.get("mean"):
        print()
        print(
            f"latency per query   p50 {lat['p50']} ms"
            f"   p95 {lat['p95']} ms   mean {lat['mean']} ms"
        )

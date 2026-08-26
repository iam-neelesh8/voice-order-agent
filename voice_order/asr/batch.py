"""Stage 4 -- transcribe a whole condition, once, and cache the result.

Transcription is the slowest thing in the project and the only part that
wants a GPU. It is also the part that gets re-run least: the transcripts for
a given (split, condition, model) never change, while stage 5 will re-run
retrieval over them dozens of times while ablating.

So it is a batch job with a file for an interface:

    manifest.jsonl  ->  transcribe  ->  <split>_<condition>_<model>.jsonl

Nothing here touches the database, the indexes or the catalog. That is what
lets the same job run on a laptop, on Kaggle, or on a rented box, and come
back as one file.

Checkpointed by line: a run that dies halfway is resumed by re-invoking it,
which matters when a free-tier GPU session can be reclaimed mid-job.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from voice_order import config
from voice_order.types import Hypothesis, Transcript


def transcripts_dir() -> Path:
    return config.DATA_DIR / "transcripts"


def transcripts_path(split: str, condition: str, model: str) -> Path:
    safe = model.replace("/", "_").replace(".", "-")
    return transcripts_dir() / f"{split}_{condition}_{safe}.jsonl"


def _row(query_id: str, condition: str, model: str, transcript: Transcript) -> dict:
    return {
        "query_id": query_id,
        "condition": condition,
        "model": model,
        "hypotheses": [
            {"text": h.text, "score": h.score, "rank": h.rank} for h in transcript.hypotheses
        ],
        "duration_s": round(transcript.duration_s, 3),
        "latency_ms": round(transcript.latency_ms, 1),
    }


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["query_id"])
        except (json.JSONDecodeError, KeyError):
            continue          # a torn final line from a killed run
    return done


def transcribe_manifest(
    split: str = "dev",
    condition: str = "phone",
    model: str | None = None,
    n_best: int | None = None,
    limit: int | None = None,
    audio_root: Path | None = None,
    out_path: Path | None = None,
    progress_every: int = 50,
) -> tuple[Path, dict]:
    """Transcribe every clip of one condition. Resumable, appends as it goes."""
    from voice_order.asr.transcribe import Transcriber
    from voice_order.evaluation import audio as audio_mod

    transcriber = Transcriber(model=model, n_best=n_best)
    model_name = transcriber.model_name

    rows = audio_mod.load_manifest(split, condition)
    if limit:
        rows = rows[:limit]

    path = Path(out_path or transcripts_path(split, condition, model_name))
    path.parent.mkdir(parents=True, exist_ok=True)

    done = _done_ids(path)
    todo = [r for r in rows if r["query_id"] not in done]
    root = Path(audio_root or config.DATA_DIR)

    stats = {
        "condition": condition,
        "model": model_name,
        "total": len(rows),
        "resumed": len(done),
        "transcribed": 0,
        "missing_audio": 0,
        "empty": 0,
    }
    if todo:
        transcriber._loaded()          # pay the model load before timing anything

    started = time.perf_counter()
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(todo, start=1):
            clip = root / row["path"]
            if not clip.is_file():
                stats["missing_audio"] += 1
                continue
            transcript = transcriber.transcribe_file(clip)
            if not transcript.hypotheses:
                stats["empty"] += 1
            fh.write(json.dumps(_row(row["query_id"], condition, model_name, transcript)) + "\n")
            fh.flush()                 # so a killed run resumes from real progress
            stats["transcribed"] += 1
            if progress_every and i % progress_every == 0:
                rate = i / (time.perf_counter() - started)
                remaining = (len(todo) - i) / rate / 60 if rate else 0
                print(f"    {i:,}/{len(todo):,}  ({rate:.1f} clips/s, ~{remaining:.0f} min left)",
                      flush=True)

    stats["elapsed_s"] = round(time.perf_counter() - started, 1)
    stats["path"] = str(path)
    return path, stats


def load_transcripts(
    split: str, condition: str, model: str | None = None, path: Path | None = None
) -> dict[str, Transcript]:
    """Read a cached transcript file back into `Transcript` objects."""
    if path is None:
        if model is None:
            model = str(config.load("asr").get("model.name", "small.en"))
        path = transcripts_path(split, condition, model)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no transcripts at {path} -- run "
            f"`voice-order transcribe --split {split} --condition {condition}`"
        )

    out: dict[str, Transcript] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["query_id"]] = Transcript(
            audio_path=None,
            hypotheses=[
                Hypothesis(text=h["text"], score=h["score"], rank=h["rank"])
                for h in row["hypotheses"]
            ],
            duration_s=row.get("duration_s", 0.0),
            latency_ms=row.get("latency_ms", 0.0),
        )
    return out


def available(split: str) -> list[dict]:
    """What has already been transcribed, so the CLI can say what is missing."""
    out = []
    for path in sorted(transcripts_dir().glob(f"{split}_*.jsonl")):
        stem = path.stem[len(split) + 1 :]
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"file": path.name, "key": stem, "clips": n})
    return out

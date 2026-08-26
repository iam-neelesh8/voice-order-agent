"""Run transcription somewhere else, bring the transcripts back.

Measured on this laptop: ~80 minutes per condition for 1-best with `small.en`,
so the full five-condition matrix is about 6.75 hours, and the n-best sweep
stage 5 needs is roughly 34. On a T4 the same matrix is well under an hour.
Batching does not rescue the CPU path -- the clips are ~3 seconds, so there is
nothing to batch within one.

Same shape as the embedding round trip:

    voice-order export-asr-input        -> data/exports/asr_dev.zip
    (run notebooks/transcribe_gpu.py on Kaggle / Colab)
    voice-order import-transcripts out/ -> data/transcripts/*.jsonl

The bundle carries audio and the manifest and nothing else. No database, no
indexes, no catalog -- which is what lets the job run anywhere.

Import is checked, not trusted: transcripts whose query_ids do not belong to
the manifest are refused, because a mismatched file would silently evaluate
one query's retrieval against another query's speech.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from voice_order import config
from voice_order.asr import batch
from voice_order.evaluation import audio as audio_mod


def export_dir() -> Path:
    """Redirectable via VOICE_ORDER_EXPORT_DIR -- see retrieval/portable.py."""
    from voice_order.retrieval.portable import export_dir as _dir

    return _dir()


def export_asr_input(
    split: str = "dev", conditions: list[str] | None = None, dest: Path | None = None
) -> tuple[Path, dict]:
    """Bundle the manifest and its audio into one zip for a GPU box."""
    rows = audio_mod.load_manifest(split)
    if conditions:
        rows = [r for r in rows if r["condition"] in conditions]
    if not rows:
        raise RuntimeError(f"no clips for {split} -- run `voice-order gen-audio` first")

    path = Path(dest or export_dir() / f"asr_{split}.zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict = {"clips": 0, "missing": 0, "conditions": sorted({r["condition"] for r in rows})}

    # Audio is FLAC, already compressed, so storing beats deflating it: the
    # zip is a container here, not a compressor.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        )
        for row in rows:
            clip = config.DATA_DIR / row["path"]
            if not clip.is_file():
                stats["missing"] += 1
                continue
            zf.write(clip, row["path"])
            stats["clips"] += 1

    stats["path"] = str(path)
    stats["mb"] = round(path.stat().st_size / 1e6, 1)
    return path, stats


def import_transcripts(source: Path, split: str = "dev") -> list[dict]:
    """Install transcript files produced elsewhere, after checking they fit.

    `source` is a directory of *.jsonl (or a single .jsonl). Each file is
    keyed by (split, condition, model), so importing a second model's run
    sits alongside the first rather than overwriting it -- which is what makes
    the model-size comparison possible at all.
    """
    source = Path(source)
    files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f"no .jsonl transcripts in {source}")

    known = {r["query_id"] for r in audio_mod.load_manifest(split)}
    installed: list[dict] = []

    for file in files:
        rows = [
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            continue

        conditions = {r.get("condition") for r in rows}
        models = {r.get("model") for r in rows}
        if len(conditions) != 1 or len(models) != 1:
            raise ValueError(
                f"{file.name} mixes conditions {sorted(conditions)} or models "
                f"{sorted(models)}; one file must cover exactly one of each"
            )
        condition, model = conditions.pop(), models.pop()

        unknown = [r["query_id"] for r in rows if r["query_id"] not in known]
        if unknown:
            raise ValueError(
                f"{file.name} has {len(unknown):,} query_ids that are not in the "
                f"{split} manifest (first: {unknown[0]}). These transcripts were "
                "made from a different audio set -- evaluating them here would "
                "score one query's retrieval against another query's speech."
            )

        empty = sum(1 for r in rows if not r.get("hypotheses"))
        target = batch.transcripts_path(split, condition, model)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        installed.append(
            {
                "file": target.name,
                "condition": condition,
                "model": model,
                "clips": len(rows),
                "empty": empty,
                "coverage": round(len(rows) / max(len(known), 1), 3),
            }
        )

    return installed

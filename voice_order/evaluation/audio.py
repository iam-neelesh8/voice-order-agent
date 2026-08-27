"""Stage 4 -- build the spoken test set.

The queries already have known answers, so speaking them with TTS gives
perfect ground truth for free. Then degrade, because clean TTS audio is
nothing like a phone call and skipping that produces numbers that will not
survive contact with a real caller.

Output is keyed by `query_id`, so audio stays joined to ground truth no matter
how the files are shuffled, copied to a GPU box, or partially regenerated.

The manifest is the contract with the ASR step. It lists every clip and its
condition and nothing else -- no database, no indexes, no catalog. That is
what lets transcription run anywhere and come back as a single JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_order import config
from voice_order.evaluation import degrade

MANIFEST_NAME = "manifest.jsonl"

# Standard ASR input rate. The clean condition is stored here rather than at
# Piper's native 22 kHz -- nothing downstream reads above 8 kHz of bandwidth.
CLEAN_RATE = 16000


def audio_root() -> Path:
    return config.data_dir() / "audio" / "synthetic"


def split_dir(split: str) -> Path:
    return audio_root() / split


def clip_path(split: str, query_id: str, condition: str, fmt: str = "flac") -> Path:
    return split_dir(split) / condition / f"{query_id}.{fmt}"


def _conditions() -> list[str]:
    return list(config.load("asr").get("conditions", ["clean", "phone"]))


def _parse_condition(condition: str) -> tuple[bool, float | None]:
    """"phone_snr10" -> (degraded, 10.0). "clean" -> (False, None)."""
    if condition == "clean":
        return False, None
    if condition == "phone":
        return True, None
    if condition.startswith("phone_snr"):
        return True, float(condition.removeprefix("phone_snr"))
    raise ValueError(f"unknown condition {condition!r}")


BABBLE_POOL_SIZE = 24


def _build_babble_pool(rows, speakers, size: int = BABBLE_POOL_SIZE) -> list:
    """Synthesise a fixed set of clips to use as background speech.

    Always the first `size` queries, always the same voices, regardless of
    what is already cached on disk. That is the whole point: the pool must not
    depend on build order, or the noise stops being reproducible.
    """
    pool = []
    for i in range(min(size, len(rows))):
        audio, rate = speakers[i % len(speakers)].synthesize(rows[i].question)
        if len(audio) == 0:
            continue
        phone, _ = degrade.telephone(audio, rate)
        pool.append(phone)
    return pool


def build_spoken_set(
    split: str = "dev", limit: int | None = None, force: bool = False
) -> dict:
    """Synthesise every query, then render each condition. Resumable.

    Voices rotate deterministically by query index, so a given query always
    gets the same speaker across reruns -- otherwise a rebuild would quietly
    change what the ASR was measured on.
    """
    from voice_order.evaluation.queries import load_order_queries
    from voice_order.tts.speak import speaker_pool

    cfg = config.load("asr")
    fmt = str(cfg.get("synthesis.format", "flac"))
    seed = int(cfg.get("degradation.seed", 0))
    noise_kinds = list(cfg.get("degradation.noise.kinds", ["babble"]))
    conditions = _conditions()

    rows = load_order_queries(split)
    if limit:
        rows = rows[:limit]

    speakers = speaker_pool()
    for condition in conditions:
        (split_dir(split) / condition).mkdir(parents=True, exist_ok=True)

    # Babble needs other people's speech, and it must be the SAME speech on
    # every run. Filling the pool from whatever happened to be synthesised
    # would make it depend on what was already cached on disk: a resumed build
    # would skip the early clips, leave the pool short or empty, and quietly
    # mix different noise than a fresh build produced. Nothing would fail --
    # the test set would just stop being the reproducible artefact this module
    # claims it is, and stage 4 numbers from before and after a rebuild would
    # not be comparable.
    #
    # Built lazily, so a run with no noise conditions never pays for it.
    babble_pool: list[np.ndarray] | None = None
    needs_babble = "babble" in noise_kinds and any(
        _parse_condition(c)[1] is not None for c in conditions
    )

    manifest: list[dict] = []
    stats = {"synthesised": 0, "skipped": 0, "clips": 0}

    for i, query in enumerate(rows):
        speaker = speakers[i % len(speakers)]
        rng = np.random.default_rng(seed + i)

        targets = {c: clip_path(split, query.query_id, c, fmt) for c in conditions}
        if not force and all(p.is_file() for p in targets.values()):
            stats["skipped"] += 1
            for condition, path in targets.items():
                manifest.append(_manifest_row(query, condition, path, speaker.name))
            continue

        audio, rate = speaker.synthesize(query.question)
        if len(audio) == 0:
            continue
        stats["synthesised"] += 1

        phone, phone_rate = degrade.telephone(audio, rate)
        if needs_babble and babble_pool is None:
            babble_pool = _build_babble_pool(rows, speakers)

        for condition in conditions:
            degraded, snr = _parse_condition(condition)
            if not degraded:
                # Down to 16 kHz: every ASR front-end resamples there anyway,
                # and Piper's 22 kHz output is a third more bytes for
                # information no model in this pipeline will ever look at.
                data, out_rate = degrade.resample(audio, rate, CLEAN_RATE), CLEAN_RATE
                # Piper peaks at full scale and the resampler overshoots it --
                # measured up to 1.0096, which a 16-bit file clamps. Only a
                # few samples per clip, but this is the condition that is
                # supposed to be the *undistorted* ceiling, so it should not
                # carry clipping of its own. Scaled only when it would clip,
                # to leave the other 99% of clips at their natural level.
                peak = float(np.max(np.abs(data))) if len(data) else 0.0
                if peak > 1.0:
                    data = data * (0.999 / peak)
            elif snr is None:
                data, out_rate = phone, phone_rate
            else:
                kind = noise_kinds[int(rng.integers(0, len(noise_kinds)))]
                noise = degrade.make_noise(
                    kind, len(phone), phone_rate, rng, babble_pool or []
                )
                data, out_rate = degrade.mix_at_snr(phone, noise, snr, rng), phone_rate

            sf.write(targets[condition], data, out_rate)
            manifest.append(_manifest_row(query, condition, targets[condition], speaker.name))
            stats["clips"] += 1

    write_manifest(split, manifest)
    stats["manifest"] = len(manifest)
    stats["conditions"] = conditions
    return stats


def audio_fingerprint(split: str, rows: list[dict] | None = None) -> str:
    """Hash of the clip bytes themselves, in manifest order.

    Exists because `import-transcripts` could not catch the one mistake most
    likely to happen: uploading a stale audio bundle. The query_ids match --
    the queries never changed, only the audio did -- so the id check passes
    and transcripts of the wrong recordings install silently.

    Hashing the bytes rather than paths or sizes: a regenerated clip has the
    same name and a very similar size, so anything cheaper would miss exactly
    the case this is for.
    """
    import hashlib

    rows = rows if rows is not None else load_manifest(split)
    digest = hashlib.sha256()
    for row in rows:
        clip = config.data_dir() / row["path"]
        digest.update(row["query_id"].encode("utf-8"))
        digest.update(b"|")
        if clip.is_file():
            digest.update(clip.read_bytes())
    return digest.hexdigest()[:16]


def _manifest_row(query, condition: str, path: Path, voice: str) -> dict:
    return {
        "query_id": query.query_id,
        "condition": condition,
        # Relative so the manifest survives being copied to another machine.
        "path": str(path.relative_to(config.data_dir())).replace("\\", "/"),
        "voice": voice,
        "reference": query.question,
    }


def write_manifest(split: str, rows: list[dict]) -> Path:
    path = split_dir(split) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_manifest(split: str, condition: str | None = None) -> list[dict]:
    path = split_dir(split) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no manifest at {path} -- run `voice-order gen-audio --split {split}`"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if not r.get("_meta")]
    if condition is None:
        return rows
    return [r for r in rows if r["condition"] == condition]


def describe(split: str) -> dict:
    """Size and duration per condition -- what a GPU trip would have to carry."""
    out: dict[str, dict] = {}
    for row in load_manifest(split):
        path = config.data_dir() / row["path"]
        entry = out.setdefault(row["condition"], {"clips": 0, "mb": 0.0, "seconds": 0.0})
        entry["clips"] += 1
        if path.is_file():
            entry["mb"] += path.stat().st_size / 1e6
            info = sf.info(path)
            entry["seconds"] += info.duration
    for entry in out.values():
        entry["mb"] = round(entry["mb"], 1)
        entry["seconds"] = round(entry["seconds"], 1)
    return out

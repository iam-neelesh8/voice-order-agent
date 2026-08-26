"""Stage 4 — build the spoken test set.

The queries already have known answers, so speaking them with TTS gives
perfect ground truth for free. Then degrade, because clean TTS audio is
nothing like a phone call and skipping this produces numbers that will not
survive contact with a real caller.
"""

from __future__ import annotations

from pathlib import Path


def synthesize_split(split: str, out_dir: Path) -> dict[str, Path]:
    """Speak every query in a split.

    Voices rotate across 4+ Piper models, or this measures one speaker rather
    than the ASR. Keyed by `query_id` so audio stays joined to ground truth.
    """
    raise NotImplementedError("stage 4")


def apply_phone_codec(src: Path, dest: Path) -> Path:
    """Narrowband: 8 kHz mono mu-law, via ffmpeg.

    The clean version is kept — the clean-vs-phone gap is itself a result.
    """
    raise NotImplementedError("stage 4")


def mix_noise(src: Path, noise_path: Path, snr_db: float, dest: Path) -> Path:
    """Mix a MUSAN clip at a target SNR."""
    raise NotImplementedError("stage 4")


def build_spoken_set(split: str) -> dict[str, list[Path]]:
    """Whole pipeline: synthesize -> phone codec -> noise at each SNR.

    Returns `query_id -> [clean, phone, phone_snr20, ...]`. Regenerates from a
    seed in under an hour, which is why none of it is committed.
    """
    raise NotImplementedError("stage 4")

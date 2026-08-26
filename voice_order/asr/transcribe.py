"""Stage 4 -- faster-whisper, returning n-best rather than one transcript.

One transcript is not enough. The whole robustness argument depends on
retrieval seeing several hypotheses, so this wrapper never collapses to a
single string.

A note on how n-best is obtained. faster-whisper exposes beam search but not
the beam's runners-up, so the alternatives are produced by decoding the same
audio at several temperatures and keeping the distinct results, ranked by the
model's own average log-probability. That is a weaker n-best than a true
lattice, and it is stated plainly rather than papered over: if stage 5 shows
n-best fusion helping, a real lattice would help at least as much.
"""

from __future__ import annotations

import time
from pathlib import Path

from voice_order import config
from voice_order.types import Hypothesis, Transcript

_MODEL_CACHE: dict[str, object] = {}


class Transcriber:
    """Loads the model once. Instantiating this per call is the obvious
    performance mistake and the reason it is a class."""

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        cfg = config.load("asr")
        self.model_name = model or str(cfg.get("model.name", "small.en"))
        self.device = device or str(cfg.get("model.device", "cpu"))
        self.compute_type = compute_type or str(cfg.get("model.compute_type", "int8"))
        self.beam_size = int(cfg.get("decode.beam_size", 5))
        self.n_best = int(cfg.get("decode.n_best", 5))
        self.vad_filter = bool(cfg.get("decode.vad_filter", True))
        self._model = None

    def _loaded(self):
        key = f"{self.model_name}:{self.device}:{self.compute_type}"
        if key not in _MODEL_CACHE:
            from faster_whisper import WhisperModel

            _MODEL_CACHE[key] = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        self._model = _MODEL_CACHE[key]
        return self._model

    def _decode(self, path: Path, temperature: float) -> tuple[str, float]:
        segments, _ = self._loaded().transcribe(
            str(path),
            beam_size=self.beam_size if temperature == 0.0 else 1,
            temperature=temperature,
            vad_filter=self.vad_filter,
            language="en",
            condition_on_previous_text=False,
        )
        parts, logprobs = [], []
        for segment in segments:
            parts.append(segment.text)
            logprobs.append(segment.avg_logprob)
        text = " ".join(p.strip() for p in parts).strip()
        score = sum(logprobs) / len(logprobs) if logprobs else -10.0
        return text, float(score)

    def transcribe_file(self, path: Path) -> Transcript:
        """One clip in, an n-best list out.

        The greedy/beam pass at temperature 0 is always hypothesis 1. Higher
        temperatures explore alternatives; duplicates are dropped, so a clip
        the model is confident about legitimately returns fewer than n_best.
        """
        path = Path(path)
        started = time.perf_counter()

        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8][: max(1, self.n_best)]
        seen: dict[str, float] = {}
        for temperature in temperatures:
            text, score = self._decode(path, temperature)
            if not text:
                continue
            # Keep the best score seen for a given string.
            if text not in seen or score > seen[text]:
                seen[text] = score

        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        hypotheses = [
            Hypothesis(text=text, score=score, rank=i)
            for i, (text, score) in enumerate(ranked[: self.n_best])
        ]

        import soundfile as sf

        try:
            duration = sf.info(path).duration
        except Exception:
            duration = 0.0

        return Transcript(
            audio_path=str(path),
            hypotheses=hypotheses,
            duration_s=float(duration),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def transcribe_stream(self, audio_chunks) -> Transcript:
        """For the live loop (stage 7). Batch path is stage 4."""
        raise NotImplementedError("stage 7")


# ---------------------------------------------------------------------------


def normalise_for_wer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    WER is otherwise dominated by the ASR's punctuation and casing choices,
    which retrieval never sees and nobody cares about.
    """
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER. Reported clean vs phone-degraded -- the gap is a result."""
    import jiwer

    ref, hyp = normalise_for_wer(reference), normalise_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))

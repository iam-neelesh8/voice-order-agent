"""Stages 4 and 7 -- Piper.

Stage 4 uses it to *build the test set*: several voices, so the eval measures
the ASR rather than one speaker. Stage 7 uses it to answer the caller.

Voices are ONNX files pulled from the rhasspy/piper-voices repo on first use
and cached. They are ~60 MB each, which is why they are downloaded on demand
rather than vendored, and why the cache lives outside the repo.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from voice_order import config

VOICE_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Where each voice lives inside that repo. Keeping this explicit beats
# deriving it from the name -- the layout is not perfectly regular.
VOICE_PATHS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-ryan-medium": "en/en_US/ryan/medium",
    "en_US-joe-medium": "en/en_US/joe/medium",
    "en_US-kusal-medium": "en/en_US/kusal/medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium",
    "en_GB-cori-medium": "en/en_GB/cori/medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium",
}

_VOICE_CACHE: dict[str, object] = {}


def voices_dir() -> Path:
    """Deliberately NOT under `config.data_dir()`.

    Voice models are a 60 MB-per-voice download cache, not generated data.
    Redirecting them with the rest of a run would make an isolated test
    re-download five voices, which is the opposite of useful.
    """
    return config.DATA_DIR / "voices"


def download_voice(name: str) -> Path:
    """Fetch a voice's .onnx and .onnx.json if not already cached."""
    if name not in VOICE_PATHS:
        raise ValueError(f"unknown voice {name!r}; known: {sorted(VOICE_PATHS)}")

    target = voices_dir()
    target.mkdir(parents=True, exist_ok=True)
    model = target / f"{name}.onnx"

    for suffix in (".onnx", ".onnx.json"):
        path = target / f"{name}{suffix}"
        if path.is_file() and path.stat().st_size > 0:
            continue
        url = f"{VOICE_REPO}/{VOICE_PATHS[name]}/{name}{suffix}"
        tmp = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)

    return model


class Speaker:
    """One loaded Piper voice. Loading is expensive; keep the instance."""

    def __init__(self, voice: str | None = None, deterministic: bool | None = None) -> None:
        cfg = config.load("asr")
        self.name = voice or cfg.get("synthesis.voices", ["en_US-lessac-medium"])[0]
        if deterministic is None:
            deterministic = bool(cfg.get("synthesis.deterministic", True))
        self.deterministic = deterministic
        self._voice = None

    def _synthesis_config(self):
        """Piper is stochastic by default and there is no seed to set.

        VITS samples from a stochastic duration predictor, so the same text
        twice gives different audio -- measured here at 99,840 vs 97,792
        samples for one sentence. Seeding numpy does not help; the sampling
        happens inside ONNX Runtime.

        Zeroing both noise scales removes the sampling entirely and makes
        synthesis reproducible. The cost is flatter prosody. That is the right
        trade for a test set: five voices already supply the speaker variety,
        and without this the audio is not a regenerable artefact at all --
        every rebuild would silently produce a different test set, and stage 4
        numbers taken before and after one would not be comparable.

        Flatter prosody is also slightly *easier* for ASR, so the degradation
        measured on this set is a lower bound on what a real caller costs.
        """
        if not self.deterministic:
            return None
        from piper import SynthesisConfig

        return SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

    def _loaded(self):
        if self._voice is None:
            if self.name not in _VOICE_CACHE:
                from piper import PiperVoice

                _VOICE_CACHE[self.name] = PiperVoice.load(str(download_voice(self.name)))
            self._voice = _VOICE_CACHE[self.name]
        return self._voice

    @property
    def sample_rate(self) -> int:
        return int(self._loaded().config.sample_rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Text -> float32 waveform in [-1, 1], plus its sample rate.

        Returns an array rather than writing a file: stage 4 pipes the audio
        straight into degradation, and a temp wav per clip per condition would
        be thousands of pointless writes.
        """
        voice = self._loaded()
        chunks = [
            np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            for chunk in voice.synthesize(text, syn_config=self._synthesis_config())
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio, self.sample_rate

    def say(self, text: str) -> None:
        """Synthesize and play. Stage 7."""
        raise NotImplementedError("stage 7")


def speaker_pool(names: list[str] | None = None) -> list[Speaker]:
    """One Speaker per configured voice.

    Four or more, or the result measures how well the ASR handles a single
    speaker rather than how well it handles speech.
    """
    cfg = config.load("asr")
    names = names or list(cfg.get("synthesis.voices", []))
    if len(names) < 2:
        raise ValueError(
            "at least two voices are needed, or the eval measures one speaker "
            "rather than the ASR"
        )
    return [Speaker(n) for n in names]

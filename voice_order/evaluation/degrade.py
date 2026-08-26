"""Stage 4 -- make clean TTS audio sound like it came down a phone line.

This step is not optional. Clean 22 kHz studio audio is nothing like a call,
and skipping it produces numbers that will not survive contact with a real
caller.

Done in numpy and scipy rather than by shelling out to ffmpeg. Three reasons,
and they all matter for something meant to be forked and run:

  * ffmpeg is a system dependency that is absent on most Windows machines and
    cannot be pip-installed. It was absent here.
  * `audioop`, the stdlib module that used to do mu-law, was removed in
    Python 3.13.
  * ffmpeg output can differ across builds and versions. This is deterministic
    on every platform, which is what makes a regenerable test set actually
    regenerable.

Determinism took a second fix to be real. Piper is stochastic by default --
VITS samples a duration predictor, so the same sentence twice gave different
audio and there is no seed to set. See `Speaker._synthesis_config`.

What a phone line does, in order:

  1. band-limits to roughly 300-3400 Hz -- the G.712 passband
  2. samples at 8 kHz
  3. companding to 8-bit mu-law (G.711), which is lossy and adds quantisation
     noise that is loudest in quiet passages
  4. adds whatever is going on in the room

Steps 1-3 are what destroys sibilants and makes spoken digits collide, which
is the whole reason this project exists.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

PHONE_RATE = 8000
PASSBAND = (300.0, 3400.0)
_MU = 255.0


# ---------------------------------------------------------------- G.711 --


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """Float in [-1, 1] -> 8-bit mu-law codes, 0..255."""
    x = np.clip(x, -1.0, 1.0)
    compressed = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    return np.clip(np.round((compressed + 1.0) * 127.5), 0, 255).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """8-bit mu-law codes -> float in [-1, 1]."""
    compressed = codes.astype(np.float32) / 127.5 - 1.0
    return (
        np.sign(compressed) * (np.power(1.0 + _MU, np.abs(compressed)) - 1.0) / _MU
    ).astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample. Public because the clean condition needs it too."""
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    gcd = np.gcd(src_rate, dst_rate)
    return signal.resample_poly(audio, dst_rate // gcd, src_rate // gcd).astype(np.float32)


def telephone(audio: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """Band-limit, downsample to 8 kHz, and push through G.711 mu-law.

    Returns 8 kHz audio -- the rate a phone line actually carries. ASR
    front-ends resample internally, and storing the true rate keeps the test
    set honest about how much information is really there.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Band-limit before decimating, or aliasing folds high frequencies back in
    # as artefacts a real line would never produce.
    nyquist = rate / 2.0
    high = min(PASSBAND[1], nyquist * 0.99)
    sos = signal.butter(
        4, [PASSBAND[0] / nyquist, high / nyquist], btype="band", output="sos"
    )
    filtered = signal.sosfilt(sos, audio).astype(np.float32)

    # Piper emits at full scale and the band-pass overshoots, so mu-law would
    # clip on almost every clip. That would add hard-clipping distortion on
    # top of the codec artefacts this is trying to isolate. Normalise into
    # headroom first -- SNR mixing is RMS-relative, so absolute level is free.
    peak = float(np.max(np.abs(filtered)))
    if peak > 0:
        filtered = filtered * (0.95 / peak)

    narrow = resample(filtered, rate, PHONE_RATE)
    return mulaw_decode(mulaw_encode(narrow)), PHONE_RATE


# ---------------------------------------------------------------- noise --


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def mix_at_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add noise scaled to hit a target signal-to-noise ratio.

    The noise is tiled or randomly cropped to length, so a clip longer than
    the noise source does not fall silent halfway through.
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > len(speech):
        start = int(rng.integers(0, len(noise) - len(speech) + 1))
        noise = noise[start : start + len(speech)]

    scale = _rms(speech) / (_rms(noise) * (10.0 ** (snr_db / 20.0)))
    mixed = speech + scale * noise

    # Keep headroom rather than clipping, which would add its own distortion
    # on top of the effect being measured.
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise -- line hiss and room tone.

    Closer to real ambient noise than white noise, which is unnaturally harsh
    in the high band a phone throws away anyway.
    """
    white = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum /= np.sqrt(freqs)
    out = np.fft.irfft(spectrum, n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-12)


def mains_hum(n: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    """50/60 Hz hum and harmonics -- bad wiring on a landline.

    Mostly below the phone passband, so it survives only as its harmonics,
    which is exactly what makes it a realistic nuisance rather than an
    obvious one.
    """
    t = np.arange(n, dtype=np.float32) / rate
    base = float(rng.choice([50.0, 60.0]))
    out = np.zeros(n, dtype=np.float32)
    for harmonic in (1, 2, 3, 4, 5):
        out += (1.0 / harmonic) * np.sin(
            2 * np.pi * base * harmonic * t + float(rng.random()) * 2 * np.pi
        )
    return out / (np.max(np.abs(out)) + 1e-12)


def babble(
    clips: list[np.ndarray], n: int, rng: np.random.Generator, voices: int = 6
) -> np.ndarray:
    """Background speech, built by overlapping other clips at random offsets.

    Real recorded babble (MUSAN, DEMAND) would be better and can drop in
    behind this same function. It is not used here because MUSAN is an 11 GB
    single tarball with no subset, which is a poor trade for a repo whose
    whole point is that it clones and runs. Built from the test set's own
    speech, babble costs nothing, needs no download, and is reproducible from
    a seed -- and it is the hardest noise type for ASR, because the
    interference has speech statistics.
    """
    if not clips:
        return pink_noise(n, rng)

    out = np.zeros(n, dtype=np.float32)
    for _ in range(voices):
        clip = clips[int(rng.integers(0, len(clips)))]
        if len(clip) < n:
            clip = np.tile(clip, int(np.ceil(n / len(clip))))
        start = int(rng.integers(0, max(1, len(clip) - n + 1)))
        out += clip[start : start + n]
    return out / (np.max(np.abs(out)) + 1e-12)


NOISE_KINDS = ("babble", "pink", "hum")


def make_noise(
    kind: str, n: int, rate: int, rng: np.random.Generator, clips: list[np.ndarray] | None = None
) -> np.ndarray:
    if kind == "pink":
        return pink_noise(n, rng)
    if kind == "hum":
        return mains_hum(n, rate, rng)
    if kind == "babble":
        return babble(clips or [], n, rng)
    raise ValueError(f"unknown noise kind {kind!r}; choose from {NOISE_KINDS}")

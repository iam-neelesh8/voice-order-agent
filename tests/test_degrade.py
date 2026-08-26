"""Stage 4 -- the phone line simulation.

This is the step that creates the problem the whole project exists to solve,
so it has to actually do what it claims. A degradation that quietly does
nothing would produce a stage 4 "result" showing no drop at all, and the
wrong conclusion would look like good news.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from voice_order.evaluation import degrade


def tone(freq: float, rate: int = 22050, seconds: float = 0.5, amp: float = 0.8) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_energy(x: np.ndarray, rate: int, low: float, high: float) -> float:
    freqs, power = signal.welch(x, fs=rate, nperseg=min(1024, len(x)))
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(power[mask]))


# ------------------------------------------------------------------ G.711 --


def test_mulaw_round_trips_approximately():
    """8-bit companding is lossy, but it must not be destructive."""
    x = np.linspace(-1, 1, 4096, dtype=np.float32)
    back = degrade.mulaw_decode(degrade.mulaw_encode(x))
    assert np.max(np.abs(back - x)) < 0.05


def test_mulaw_preserves_small_signals_better_than_linear():
    """That is the entire point of companding -- fine steps near zero."""
    quiet = np.full(512, 0.01, dtype=np.float32)
    back = degrade.mulaw_decode(degrade.mulaw_encode(quiet))
    assert abs(float(np.mean(back)) - 0.01) < 0.002


def test_mulaw_codes_are_bytes():
    codes = degrade.mulaw_encode(np.linspace(-1, 1, 256, dtype=np.float32))
    assert codes.dtype == np.uint8
    assert codes.min() >= 0 and codes.max() <= 255


# -------------------------------------------------------------- telephone --


def test_output_is_eight_kilohertz():
    out, rate = degrade.telephone(tone(1000), 22050)
    assert rate == degrade.PHONE_RATE
    assert len(out) == pytest.approx(0.5 * 8000, rel=0.02)


def test_energy_above_the_passband_is_removed():
    """A 6 kHz tone cannot survive a 300-3400 Hz line."""
    high = tone(6000)
    out, rate = degrade.telephone(high, 22050)
    kept = band_energy(out, rate, 300, 3400)
    assert kept < 1e-3, "high-frequency tone survived the band-pass"


def test_energy_below_the_passband_is_removed():
    """A 100 Hz rumble is below the line's passband."""
    out, rate = degrade.telephone(tone(100), 22050)
    assert band_energy(out, rate, 300, 3400) < 1e-3


def test_speech_band_survives():
    """The filter must pass what it is supposed to pass."""
    out, rate = degrade.telephone(tone(1000), 22050)
    assert band_energy(out, rate, 800, 1200) > 1e-4


def test_degradation_actually_changes_a_broadband_signal():
    """The guard against a no-op pipeline silently reporting 'no drop'.

    Tested on broadband noise rather than a tone: a 1 kHz tone sits inside
    the passband and legitimately survives almost untouched, so it would pass
    this test even if the filter were removed. Speech is broadband, and so is
    this.
    """
    rng = np.random.default_rng(0)
    clean = (0.5 * rng.standard_normal(22050)).astype(np.float32)

    out, rate = degrade.telephone(clean, 22050)
    reference = degrade.resample(clean, 22050, rate)

    n = min(len(out), len(reference))
    assert np.corrcoef(out[:n], reference[:n])[0, 1] < 0.95


def test_mulaw_quantisation_leaves_a_measurable_trace():
    """Even inside the passband, 8-bit companding is not free.

    Pinned because a pipeline that band-limited but skipped the codec would
    pass every other test here while under-stating the damage.
    """
    clean = tone(1000)
    out, rate = degrade.telephone(clean, 22050)

    # Compare against the same path without the codec.
    nyquist = 22050 / 2
    sos = signal.butter(4, [300 / nyquist, 3400 / nyquist], btype="band", output="sos")
    filtered = signal.sosfilt(sos, clean).astype(np.float32)
    filtered *= 0.95 / max(float(np.max(np.abs(filtered))), 1e-9)
    reference = degrade.resample(filtered, 22050, rate)

    n = min(len(out), len(reference))
    error = degrade._rms(out[:n] - reference[:n]) / degrade._rms(reference[:n])
    assert error > 1e-4, "the mu-law codec appears to be a no-op"


def test_output_leaves_headroom():
    """Piper emits at full scale; clipping would add its own distortion."""
    out, _ = degrade.telephone(tone(1000, amp=1.0), 22050)
    assert float(np.max(np.abs(out))) <= 1.0


# ------------------------------------------------------------------ noise --


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0, 0.0])
def test_mixing_hits_the_requested_snr(snr_db):
    rng = np.random.default_rng(0)
    speech = tone(1000, rate=8000, seconds=2.0)
    noise = degrade.pink_noise(len(speech), rng)

    mixed = degrade.mix_at_snr(speech, noise, snr_db, rng)
    assert len(mixed) == len(speech)

    # Recover the achieved ratio from the components rather than from `mixed`,
    # which has been peak-normalised and no longer decomposes cleanly.
    scale = degrade._rms(speech) / (degrade._rms(noise) * (10.0 ** (snr_db / 20.0)))
    achieved = 20 * np.log10(degrade._rms(speech) / degrade._rms(scale * noise))
    assert achieved == pytest.approx(snr_db, abs=0.5)


def test_lower_snr_means_more_noise():
    rng = np.random.default_rng(1)
    speech = tone(1000, rate=8000, seconds=1.0)
    noise = degrade.pink_noise(len(speech), rng)

    quiet = degrade.mix_at_snr(speech, noise, 20.0, np.random.default_rng(2))
    loud = degrade.mix_at_snr(speech, noise, 0.0, np.random.default_rng(2))
    assert degrade._rms(loud - speech) > degrade._rms(quiet - speech)


def test_mixing_never_clips():
    rng = np.random.default_rng(3)
    speech = tone(1000, rate=8000, amp=0.99)
    noise = degrade.pink_noise(len(speech), rng)
    mixed = degrade.mix_at_snr(speech, noise, 0.0, rng)
    assert float(np.max(np.abs(mixed))) <= 1.0


def test_noise_is_tiled_when_shorter_than_the_clip():
    rng = np.random.default_rng(4)
    speech = tone(1000, rate=8000, seconds=2.0)
    short = degrade.pink_noise(1000, rng)
    mixed = degrade.mix_at_snr(speech, short, 10.0, rng)
    assert len(mixed) == len(speech)
    # The tail must not be silent -- a clip that goes quiet halfway is not a
    # noisy clip, it is two different conditions in one file.
    assert degrade._rms(mixed[-4000:] - speech[-4000:]) > 1e-4


@pytest.mark.parametrize("kind", degrade.NOISE_KINDS)
def test_every_noise_kind_produces_usable_audio(kind):
    rng = np.random.default_rng(5)
    clips = [tone(440, rate=8000, seconds=1.0)]
    noise = degrade.make_noise(kind, 8000, 8000, rng, clips)
    assert len(noise) == 8000
    assert np.all(np.isfinite(noise))
    assert 0.0 < float(np.max(np.abs(noise))) <= 1.0


def test_an_unknown_noise_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown noise kind"):
        degrade.make_noise("traffic", 100, 8000, np.random.default_rng(0))


def test_babble_falls_back_when_no_clips_are_available():
    """The first clips of a run have nothing to babble with yet."""
    noise = degrade.babble([], 4000, np.random.default_rng(0))
    assert len(noise) == 4000
    assert np.all(np.isfinite(noise))


def test_the_pipeline_is_deterministic_from_a_seed():
    """A regenerable test set has to actually regenerate identically."""
    speech, _ = degrade.telephone(tone(1000), 22050)
    a = degrade.mix_at_snr(speech, degrade.pink_noise(len(speech), np.random.default_rng(7)),
                           10.0, np.random.default_rng(7))
    b = degrade.mix_at_snr(speech, degrade.pink_noise(len(speech), np.random.default_rng(7)),
                           10.0, np.random.default_rng(7))
    assert np.array_equal(a, b)

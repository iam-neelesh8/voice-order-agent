"""Stage 4 -- the spoken set has to regenerate identically.

`degrade.py` and `configs/asr.yaml` both promise a test set that rebuilds from
a seed. That promise is load-bearing: stage 4 and stage 5 numbers are only
comparable if the audio underneath them is the same audio.

The interesting case is a *partial* rebuild, which is the normal case -- runs
get interrupted, and the builder skips clips already on disk.
"""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("piper", reason="needs the speech extra")
pytest.importorskip("soundfile", reason="needs the speech extra")


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@pytest.fixture
def tiny_split(tmp_path, monkeypatch):
    """Three queries in an isolated data root."""
    monkeypatch.setenv("VOICE_ORDER_DATA_DIR", str(tmp_path))

    evalsets = tmp_path / "evalsets"
    evalsets.mkdir(parents=True)
    rows = [
        {
            "query_id": f"dev-{i:05d}",
            "question": q,
            "relevant_doc_ids": [f"B{i:04d}"],
            "category": "Automotive",
            "kind": "brand_id_noun",
            "has_part_number": True,
            "has_brand": True,
            "has_disfluency": False,
            "quantity": 1,
        }
        for i, q in enumerate(
            [
                "I need two AC Delco 41-993 spark plugs",
                "Can you get me a Bosch 3397 wiper blade",
                "Give me one Denso 234-4587 oxygen sensor",
            ]
        )
    ]
    (evalsets / "orders_dev.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_a_resumed_build_reproduces_the_original_audio(tiny_split):
    """The bug this pins: babble was drawn from whatever happened to be
    synthesised, so a resumed build -- which skips cached clips -- left the
    pool short or empty and mixed different noise. Nothing failed; the test
    set simply stopped being reproducible, and numbers from before and after a
    rebuild were no longer comparable.
    """
    from voice_order.evaluation import audio

    audio.build_spoken_set("dev", limit=3)

    noisy = sorted((audio.split_dir("dev") / "phone_snr10").glob("*.flac"))
    assert len(noisy) == 3, "fixture did not produce the noisy condition"
    before = {p.name: digest(p) for p in noisy}

    # Force exactly one query to be re-rendered; the other two stay cached.
    # That is what a resumed run looks like.
    victim = noisy[0].name
    for condition in ("clean", "phone", "phone_snr20", "phone_snr10", "phone_snr5"):
        (audio.split_dir("dev") / condition / victim).unlink(missing_ok=True)

    stats = audio.build_spoken_set("dev", limit=3)
    assert stats["synthesised"] == 1 and stats["skipped"] == 2, (
        "the fixture must exercise a partial rebuild, not a full one"
    )

    after = {
        p.name: digest(p)
        for p in sorted((audio.split_dir("dev") / "phone_snr10").glob("*.flac"))
    }
    assert after[victim] == before[victim], (
        "a resumed build produced different noise than the original run"
    )


def test_two_fresh_builds_agree(tiny_split):
    """The easier guarantee, and the one that would have passed anyway."""
    from voice_order.evaluation import audio

    audio.build_spoken_set("dev", limit=3)
    first = {
        p.name: digest(p)
        for p in sorted((audio.split_dir("dev") / "phone_snr5").glob("*.flac"))
    }

    audio.build_spoken_set("dev", limit=3, force=True)
    second = {
        p.name: digest(p)
        for p in sorted((audio.split_dir("dev") / "phone_snr5").glob("*.flac"))
    }

    assert first == second


def test_the_clean_condition_does_not_clip(tiny_split):
    """`clean` is the ceiling the other conditions are compared against.

    Piper emits at full scale and the 22k->16k resampler overshoots it
    (measured 1.0096), which a 16-bit file silently clamps. A handful of
    samples per clip, but clipping in the undegraded reference makes the
    clean-vs-phone gap slightly smaller than it should be -- it understates
    the very thing stage 4 exists to measure.
    """
    import numpy as np
    import soundfile as sf

    from voice_order.evaluation import audio

    audio.build_spoken_set("dev", limit=3)

    for path in sorted((audio.split_dir("dev") / "clean").glob("*.flac")):
        data, _ = sf.read(path, dtype="float32")
        assert float(np.max(np.abs(data))) < 1.0, f"{path.name} is clipped"

"""The transcription round trip, and the mistake it has to catch.

Uploading a stale audio bundle is the easiest thing to get wrong here: the
audio gets regenerated locally, the old zip is still sitting in exports, and
an hour of GPU time goes into transcribing recordings that no longer exist.

The query_id check cannot catch it. The queries never change -- only the audio
does -- so the ids match perfectly and the transcripts install clean. Only a
fingerprint of the audio itself closes that.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("piper", reason="needs the speech extra")
pytest.importorskip("soundfile", reason="needs the speech extra")


@pytest.fixture
def tiny_audio(tmp_path, monkeypatch):
    """Two queries, spoken and degraded, in an isolated data root."""
    monkeypatch.setenv("VOICE_ORDER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOICE_ORDER_EXPORT_DIR", str(tmp_path / "exports"))

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
            ["I need an AC Delco 41-993 spark plug", "Give me a Bosch 3330 oil filter"]
        )
    ]
    (evalsets / "orders_dev.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    from voice_order.evaluation import audio

    audio.build_spoken_set("dev", limit=2)
    return tmp_path


def fake_transcripts(tmp_path, fingerprint, condition="phone", model="small.en"):
    """What the notebook hands back, with a fingerprint we control."""
    from voice_order.evaluation import audio

    out = tmp_path / "transcripts"
    out.mkdir(exist_ok=True)
    rows = [
        {
            "query_id": r["query_id"],
            "condition": condition,
            "model": model,
            "hypotheses": [{"text": r["reference"], "score": -0.2, "rank": 0}],
            "audio_fingerprint": fingerprint,
            "duration_s": 0.0,
            "latency_ms": 0.0,
        }
        for r in audio.load_manifest("dev", condition)
    ]
    path = out / f"{condition}__{model}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return out


def test_the_bundle_carries_a_fingerprint_of_its_audio(tiny_audio):
    import zipfile

    from voice_order.asr import portable

    path, stats = portable.export_asr_input("dev")

    with zipfile.ZipFile(path) as zf:
        header = json.loads(zf.open("manifest.jsonl").readline().decode())

    assert header["_meta"] is True
    assert header["audio_fingerprint"] == stats["fingerprint"]
    assert len(stats["fingerprint"]) == 16


def test_matching_transcripts_import(tiny_audio, tmp_path):
    from voice_order.asr import portable
    from voice_order.evaluation import audio

    good = fake_transcripts(tmp_path, audio.audio_fingerprint("dev"))
    installed = portable.import_transcripts(good, "dev")

    assert len(installed) == 1
    assert installed[0]["clips"] == 2


def test_transcripts_from_regenerated_audio_are_refused(tiny_audio, tmp_path):
    """The whole point. Ids match; the recordings do not.

    Simulated by stamping a fingerprint that is not the local one -- which is
    exactly what a stale bundle produces.
    """
    from voice_order.asr import portable

    stale = fake_transcripts(tmp_path, "0000badf1e0000ff")

    with pytest.raises(ValueError, match="different audio"):
        portable.import_transcripts(stale, "dev")


def test_the_error_names_both_fingerprints(tiny_audio, tmp_path):
    """So the fix is obvious without going digging."""
    from voice_order.asr import portable
    from voice_order.evaluation import audio

    stale = fake_transcripts(tmp_path, "0000badf1e0000ff")
    local = audio.audio_fingerprint("dev")

    with pytest.raises(ValueError) as excinfo:
        portable.import_transcripts(stale, "dev")

    message = str(excinfo.value)
    assert "0000badf1e0000ff" in message
    assert local in message
    assert "export-asr-input" in message


def test_transcripts_without_a_fingerprint_still_import(tiny_audio, tmp_path):
    """Older transcript files predate the stamp and must not break.

    The id check still applies to them; only the audio check is skipped.
    """
    from voice_order.asr import portable
    from voice_order.evaluation import audio

    out = tmp_path / "legacy"
    out.mkdir()
    rows = [
        {
            "query_id": r["query_id"],
            "condition": "phone",
            "model": "small.en",
            "hypotheses": [{"text": r["reference"], "score": -0.2, "rank": 0}],
        }
        for r in audio.load_manifest("dev", "phone")
    ]
    (out / "phone__small.en.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    assert portable.import_transcripts(out, "dev")[0]["clips"] == 2

# =============================================================================
# voice-order-agent -- transcribe the spoken test set on a GPU
#
# Paste into ONE cell of a Kaggle or Colab notebook with a GPU runtime, after
# uploading data/exports/asr_dev.zip.
#
#   Kaggle : Add Data -> Upload -> asr_dev.zip. Settings -> Accelerator: GPU.
#   Colab  : run the cell; it will prompt you to upload.
#
# Measured on the laptop this was built on: ~80 min per condition for 1-best
# with small.en, ~6.75 h for the five-condition matrix, ~34 h with the n-best
# sweep. On a T4 the whole matrix is well under an hour.
#
# MODELS is the experiment. Running more than one size answers the question
# that actually matters: does a bigger model fix phone audio, or does the
# robustness have to live in retrieval? If large-v3 recovers most of the loss,
# the answer is "use a bigger model". If it does not, that is the justification
# for everything in stage 5.
#
# Download transcripts.zip at the end, unzip, and run:
#   voice-order import-transcripts transcripts
# =============================================================================

import glob
import json
import os
import time
import zipfile

os.system("pip install -q faster-whisper")

from faster_whisper import WhisperModel

# --- what to run -------------------------------------------------------------

MODELS = ["small.en", "large-v3"]        # add "medium.en" for a third point
CONDITIONS = None                         # None = every condition in the bundle
N_BEST = 5                                # 1 is ~5x faster and enough for stage 4

# --- locate and unpack the bundle -------------------------------------------

candidates = (
    glob.glob("/kaggle/input/**/asr_*.zip", recursive=True)
    + glob.glob("./asr_*.zip")
    + glob.glob("/content/asr_*.zip")
)
if not candidates:
    try:
        from google.colab import files

        files.upload()
        candidates = glob.glob("asr_*.zip")
    except ImportError:
        pass
if not candidates:
    raise SystemExit("asr_*.zip not found -- upload it first.")

with zipfile.ZipFile(candidates[0]) as zf:
    zf.extractall("audio")
print("unpacked", candidates[0])

raw = [
    json.loads(line)
    for line in open("audio/manifest.jsonl", encoding="utf-8")
    if line.strip()
]
# The first line carries a fingerprint of the audio this bundle was built
# from. It is stamped into every transcript so the import can prove the
# transcripts and the local audio are the same recordings.
meta = next((r for r in raw if r.get("_meta")), {})
AUDIO_FINGERPRINT = meta.get("audio_fingerprint")
manifest = [r for r in raw if not r.get("_meta")]
print("audio fingerprint:", AUDIO_FINGERPRINT)
if CONDITIONS:
    manifest = [r for r in manifest if r["condition"] in CONDITIONS]

conditions = sorted({r["condition"] for r in manifest})
print(f"{len(manifest):,} clips across {conditions}")

os.makedirs("transcripts", exist_ok=True)

# --- transcribe --------------------------------------------------------------
# Temperature sweep, because faster-whisper exposes beam search but not the
# beam's runners-up. Weaker than a true lattice; keep it identical to the
# local implementation so the two are comparable.
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8][:N_BEST]


def transcribe_one(model, path):
    seen = {}
    for temp in TEMPERATURES:
        segments, _ = model.transcribe(
            path,
            beam_size=5 if temp == 0.0 else 1,
            temperature=temp,
            vad_filter=True,
            language="en",
            condition_on_previous_text=False,
        )
        parts, logprobs = [], []
        for seg in segments:
            parts.append(seg.text)
            logprobs.append(seg.avg_logprob)
        text = " ".join(p.strip() for p in parts).strip()
        if not text:
            continue
        score = sum(logprobs) / len(logprobs) if logprobs else -10.0
        if text not in seen or score > seen[text]:
            seen[text] = score
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:N_BEST]
    return [{"text": t, "score": s, "rank": i} for i, (t, s) in enumerate(ranked)]


for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    for condition in conditions:
        rows = [r for r in manifest if r["condition"] == condition]
        out_path = f"transcripts/{condition}__{model_name.replace('.', '-')}.jsonl"

        # Resume: free-tier sessions get reclaimed, and re-doing an hour of
        # transcription because of it is avoidable.
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path, encoding="utf-8"):
                if line.strip():
                    done.add(json.loads(line)["query_id"])

        todo = [r for r in rows if r["query_id"] not in done]
        started = time.time()
        with open(out_path, "a", encoding="utf-8") as fh:
            for i, row in enumerate(todo, 1):
                hyps = transcribe_one(model, f"audio/{row['path']}")
                fh.write(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "condition": condition,
                            "model": model_name,
                            "hypotheses": hyps,
                            "audio_fingerprint": AUDIO_FINGERPRINT,
                            "duration_s": 0.0,
                            "latency_ms": 0.0,
                        }
                    )
                    + "\n"
                )
                if i % 200 == 0:
                    rate = i / (time.time() - started)
                    print(f"  {condition}: {i:,}/{len(todo):,}  "
                          f"({rate:.1f}/s, ~{(len(todo)-i)/rate/60:.0f} min left)", flush=True)
        print(f"  {condition}: {len(rows):,} clips in {time.time()-started:.0f}s", flush=True)

    del model

os.system("zip -qr transcripts.zip transcripts")
print(f"\nwrote transcripts.zip ({os.path.getsize('transcripts.zip')/1e6:.1f} MB)")
print("download it, unzip, then:  voice-order import-transcripts transcripts")

try:
    from google.colab import files

    files.download("transcripts.zip")
except ImportError:
    print("(Kaggle: find it under Output on the right-hand panel)")

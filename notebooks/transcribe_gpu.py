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

import ctranslate2
from faster_whisper import WhisperModel

# --- fail loudly if this is not actually on a GPU ----------------------------
# faster-whisper raises on device="cuda" without CUDA, but the message is deep
# in ctranslate2 and easy to miss. Better to say it plainly up front.

if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit(
        "No CUDA device. On CPU this run takes hours rather than under one.\n"
        "Fix: Settings -> Accelerator -> GPU T4 x2, then "
        "Run -> Restart & Clear Cell Outputs."
    )
print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}", flush=True)

# --- what to run -------------------------------------------------------------

MODELS = ["small.en", "large-v3"]        # add "medium.en" for a third point
CONDITIONS = None                         # None = every condition in the bundle
N_BEST = 5                                # 1 is ~5x faster and enough for stage 4

# --- locate and unpack the bundle -------------------------------------------

# Kaggle DECOMPRESSES uploads: a .zip arrives already extracted, as a folder.
# So look for the manifest first -- if it is there, the bundle is unpacked and
# there is no zip to open. Searching only for the zip finds nothing and falls
# into a recursive walk of 10,000 audio files, which looks exactly like a hang.

AUDIO_ROOT = None

manifest_hits: list[str] = []
for pattern in (
    "/kaggle/input/*/manifest.jsonl",
    "/kaggle/input/*/*/manifest.jsonl",
    "./audio/manifest.jsonl",
):
    manifest_hits += glob.glob(pattern)

if manifest_hits:
    AUDIO_ROOT = os.path.dirname(manifest_hits[0])
    print(f"bundle already extracted at {AUDIO_ROOT}", flush=True)
else:
    zips: list[str] = []
    for pattern in ("/kaggle/input/*/asr_*.zip", "/kaggle/input/*/*/asr_*.zip",
                    "./asr_*.zip", "/content/asr_*.zip"):
        zips += glob.glob(pattern)

    if not zips:
        try:
            from google.colab import files

            files.upload()
            zips = glob.glob("asr_*.zip")
        except ImportError:
            pass

    if not zips:
        print("nothing found. What IS attached:", flush=True)
        for root, _dirs, fs in os.walk("/kaggle/input"):
            if root.count("/") - 2 <= 2:
                print(f"   {root} -> {fs[:6]}{' ...' if len(fs) > 6 else ''}", flush=True)
        raise SystemExit(
            "No asr_*.zip and no manifest.jsonl. Upload data/exports/asr_dev.zip; "
            "note that Kaggle extracts it, so it appears as a folder."
        )

    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall("audio")
    AUDIO_ROOT = "audio"
    print("unpacked", zips[0], flush=True)

raw = [
    json.loads(line)
    for line in open(f"{AUDIO_ROOT}/manifest.jsonl", encoding="utf-8")
    if line.strip()
]
meta = next((r for r in raw if r.get("_meta")), {})
AUDIO_FINGERPRINT = meta.get("audio_fingerprint")
print("audio fingerprint:", AUDIO_FINGERPRINT, flush=True)
manifest = [r for r in raw if not r.get("_meta")]
print("audio fingerprint:", AUDIO_FINGERPRINT, flush=True)
if CONDITIONS:
    manifest = [r for r in manifest if r["condition"] in CONDITIONS]

conditions = sorted({r["condition"] for r in manifest})
print(f"{len(manifest):,} clips across {conditions}", flush=True)

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
                hyps = transcribe_one(model, f"{AUDIO_ROOT}/{row['path']}")
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

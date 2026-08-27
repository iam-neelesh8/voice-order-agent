# =============================================================================
# voice-order-agent -- transcribe the spoken test set on a GPU
#
# Paste into ONE cell of a Kaggle or Colab notebook with a GPU runtime, after
# uploading data/exports/asr_dev.zip.
#
#   Kaggle : Add Data -> Upload -> asr_dev.zip. Settings -> Accelerator: GPU.
#   Colab  : run the cell; it will prompt you to upload.
#
# MEASURED ON A KAGGLE T4, 1,998 clips per condition:
#
#   small.en, N_BEST=5   ~55 min per condition   ~4.6 h for all five
#   large-v3, N_BEST=5   roughly 3-4x that       ~15 h for all five
#
# Both models across all five conditions is about 20 hours, and a Kaggle
# session is capped at 12. Start with the defaults below -- clean and phone
# at 1-best is the stage 4 headline number and takes about 20 minutes.
# Widen only once you have that.
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

# Defaults chosen to finish. Each of these multiplies the runtime:
#   every extra model      ~1x (large-v3 is ~3-4x small.en on its own)
#   every extra condition  ~1x
#   every extra n-best     ~1x  (N_BEST=5 is five decodes per clip)
MODELS = ["large-v3"]                     # small.en is already done -- see data/transcripts_gpu/
CONDITIONS = ["clean", "phone"]           # None = all five; the drop lives in these two
N_BEST = 5                                # matches the small.en run, so the two are comparable

# --- locate and unpack the bundle -------------------------------------------

# Kaggle both decompresses uploads and nests them unpredictably: a .zip
# arrives already extracted, and the dataset may sit at /kaggle/input/<slug>/
# or /kaggle/input/datasets/<user>/<slug>/. Searching by filename with an
# early exit is immune to all of that; a glob written for one layout is not.


def find_file(names, roots=("/kaggle/input", "/kaggle/working", ".", "/content")):
    wanted = set(names)
    skip = {"clean", "phone", "phone_snr5", "phone_snr10", "phone_snr20"}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for filename in filenames:
                if filename in wanted:
                    return os.path.join(dirpath, filename)
    return None


print("looking for the audio bundle ...", flush=True)

# The manifest first: if it is there the bundle is already extracted and there
# is no archive to open.
manifest_path = find_file(("manifest.jsonl",))

if manifest_path:
    AUDIO_ROOT = os.path.dirname(manifest_path)
    print(f"bundle already extracted at {AUDIO_ROOT}", flush=True)
else:
    archive = find_file(("asr_dev.zip", "asr_test.zip"))
    if archive is None:
        print("nothing found. What IS attached:", flush=True)
        for dirpath, _d, filenames in os.walk("/kaggle/input"):
            if filenames:
                print(f"   {dirpath} -> {filenames[:6]}"
                      f"{' ...' if len(filenames) > 6 else ''}", flush=True)
        try:
            from google.colab import files

            files.upload()
            archive = find_file(("asr_dev.zip",))
        except ImportError:
            pass
    if archive is None:
        raise SystemExit(
            "No manifest.jsonl and no asr_*.zip -- see the listing above. "
            "Add Data -> Upload data/exports/asr_dev.zip."
        )
    with zipfile.ZipFile(archive) as zf:
        zf.extractall("audio")
    AUDIO_ROOT = "audio"
    print("unpacked", archive, flush=True)

raw = [
    json.loads(line)
    for line in open(f"{AUDIO_ROOT}/manifest.jsonl", encoding="utf-8")
    if line.strip()
]
meta = next((r for r in raw if r.get("_meta")), {})
AUDIO_FINGERPRINT = meta.get("audio_fingerprint")
print("audio fingerprint:", AUDIO_FINGERPRINT, flush=True)
manifest = [r for r in raw if not r.get("_meta")]
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

        # Re-zip after every condition. A run that gets stopped or reclaimed
        # part way still leaves a downloadable archive -- zipping only at the
        # very end means an interrupted four-hour run hands back nothing.
        os.system("zip -qr transcripts.zip transcripts")
        print(f"  transcripts.zip updated "
              f"({os.path.getsize('transcripts.zip')/1e6:.0f} MB) -- safe to "
              f"download and stop here", flush=True)

    del model

os.system("zip -qr transcripts.zip transcripts")
size_mb = os.path.getsize("transcripts.zip") / 1e6
print(f"\nwrote transcripts.zip ({size_mb:.2f} MB)", flush=True)
for name in sorted(os.listdir("transcripts")):
    clips = sum(1 for _ in open(f"transcripts/{name}", encoding="utf-8"))
    print(f"   {name}  {clips:,} clips", flush=True)

# How to actually get the file. On Kaggle this renders a clickable link and the
# file also appears in the Output panel. The Colab helper throws a JavaScript
# error on Kaggle, so it goes last and its failure is swallowed -- a finished
# run used to end with that error on screen, which reads like a failure.
try:
    from IPython.display import FileLink, display

    print("\nclick to download:", flush=True)
    display(FileLink("transcripts.zip"))
except Exception:
    pass

print(
    "\nOr: right-hand panel -> Output -> /kaggle/working -> transcripts.zip"
    "\nThen locally:  voice-order import-transcripts <unzipped folder>",
    flush=True,
)

try:                                    # Colab only; errors on Kaggle
    from google.colab import files

    files.download("transcripts.zip")
except Exception:
    pass

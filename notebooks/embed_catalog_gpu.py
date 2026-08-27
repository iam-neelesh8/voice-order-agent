# =============================================================================
# voice-order-agent -- embed the catalog on a GPU
#
# Paste this into ONE cell of a Kaggle or Colab notebook with a GPU runtime,
# after uploading data/exports/embed_input.jsonl.gz.
#
#   Kaggle : Add Data -> Upload -> the .gz file. Settings -> Accelerator: GPU.
#            It lands under /kaggle/input/<your-dataset-name>/
#   Colab  : run the cell; it will prompt you to upload the file.
#
# Runtime is about a minute on a T4. Download catalog_embeddings.zip at the
# end, unzip it locally, and run:
#
#   voice-order import-embeddings <unzipped folder>
#
# The import refuses the file if it does not match your catalog or was made
# with a different model, so a bad run cannot silently poison retrieval.
#
# DO NOT change the model name. Document and query vectors must come from the
# same model; if they do not, every cosine score is miscalibrated and the
# import will reject the result.
# =============================================================================

import glob
import gzip
import json
import os
import time

_T0 = time.time()


def log(message):
    """Timestamped and flushed. Every step, so a hang has an address."""
    print(f"[{time.time() - _T0:6.1f}s] {message}", flush=True)

# fastembed-gpu keeps the exact ONNX weights the local CPU path uses, so the
# vectors are interchangeable. Installing plain `fastembed` here would still
# work but would run on CPU and defeat the point.
# Kaggle images ship CPU onnxruntime. Installing the GPU build alongside it
# leaves the CPU one winning and the whole job silently runs ~180x slower, so
# the CPU package is removed first.
os.system("pip uninstall -y -q onnxruntime fastembed >/dev/null 2>&1")
os.system("pip install -q fastembed-gpu onnxruntime-gpu")

import numpy as np
import onnxruntime
from fastembed import TextEmbedding

# --- fail loudly if this is not actually on a GPU ----------------------------

providers = onnxruntime.get_available_providers()
log(f"onnxruntime providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "No CUDAExecutionProvider. This would run on CPU and take about three "
        "hours instead of one minute.\n"
        "Fix: Settings -> Accelerator -> GPU T4 x2, then Run -> Restart & Clear "
        "Cell Outputs, and run this cell again."
    )

# --- locate the export -------------------------------------------------------

log("looking for the upload ...")

# Kaggle DECOMPRESSES uploads. A .gz arrives as a plain .jsonl with no
# extension to tell you, so both names have to be searched -- looking only for
# the name you uploaded finds nothing and falls into a recursive walk of every
# attached dataset, which looks exactly like a hang.
NAMES = ("embed_input.jsonl.gz", "embed_input.jsonl")

candidates: list[str] = []
for name in NAMES:
    for pattern in (
        f"/kaggle/input/*/{name}",
        f"/kaggle/input/*/*/{name}",
        f"./{name}",
        f"/content/{name}",
    ):
        candidates += glob.glob(pattern)

if not candidates:
    log("  not in the usual places; listing what IS attached:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count("/") - 2 <= 2:
            log(f"    {root}  ->  {files[:6]}{' ...' if len(files) > 6 else ''}")
    try:                                     # Colab: prompt for the upload
        from google.colab import files as colab_files

        colab_files.upload()
        candidates = [f for n in NAMES for f in glob.glob(n)]
    except ImportError:
        pass

if not candidates:
    raise SystemExit(
        "embed_input.jsonl(.gz) not found -- see the listing above for what is "
        "actually attached. On Kaggle: Add Data -> Upload the file, and note "
        "that Kaggle decompresses it, so it appears without the .gz."
    )

source = candidates[0]
log(f"reading {source}")

# --- read ids and texts ------------------------------------------------------

def open_maybe_gzip(path):
    """Kaggle strips the .gz, so sniff the magic bytes instead of trusting it."""
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == bytes([0x1F, 0x8B])  # gzip magic
    log(f"  {'gzipped' if gzipped else 'plain text'}")
    if gzipped:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


meta, ids, texts = None, [], []
with open_maybe_gzip(source) as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("_meta"):
            meta = row
            continue
        ids.append(row["id"])
        texts.append(row["text"])

model_name = (meta or {}).get("model", "BAAI/bge-small-en-v1.5")
log(f"{len(texts):,} texts read  |  model {model_name}")

# --- embed -------------------------------------------------------------------

log("constructing TextEmbedding -- downloads ~130 MB on a cold cache, then")
log("  builds a CUDA session. This is the step with no progress bar.")
model = TextEmbedding(model_name=model_name, providers=["CUDAExecutionProvider"])
log("model constructed")

# What the session actually chose, not what was merely available. An
# onnxruntime-gpu that cannot bind to the CUDA runtime falls back silently.
try:
    active = model.model.model.get_providers()
    log(f"session providers: {active}")
    if "CUDAExecutionProvider" not in active:
        raise SystemExit(
            "The session fell back to CPU despite CUDA being available. "
            "Restart & Clear Cell Outputs and run again."
        )
except AttributeError:
    log("(could not read session providers on this fastembed version)")

# Eight texts, not 256. If something is wrong this returns in a second rather
# than looking like another hang.
log("tiny smoke test: 8 texts ...")
list(model.embed(["spark plug"] * 8, batch_size=8))
log("smoke test ok")

# Time a small batch before committing to the whole catalog. A GPU does
# hundreds of texts a second; CPU does about ten. Better to find out now than
# forty minutes in with nothing on screen.
log("warm-up: 256 texts ...")
warm = texts[:256]
t0 = time.time()
list(model.embed(warm, batch_size=256))
rate = len(warm) / max(time.time() - t0, 1e-6)
log(f"warm-up: {rate:,.0f} texts/s")
if rate < 100:
    raise SystemExit(
        f"Only {rate:,.0f} texts/s -- that is CPU speed, and the full run would "
        f"take about {len(texts)/rate/60:,.0f} minutes.\n"
        "Check Settings -> Accelerator is GPU, then Restart & Clear Cell Outputs."
    )

# Chunked so progress is visible. One blocking call over 100k texts prints
# nothing for the entire run, which is indistinguishable from being hung.
t0 = time.time()
chunks = []
STEP = 5000
for start in range(0, len(texts), STEP):
    chunks.append(
        np.asarray(
            list(model.embed(texts[start : start + STEP], batch_size=256)),
            dtype=np.float32,
        )
    )
    done = min(start + STEP, len(texts))
    speed = done / (time.time() - t0)
    log(f"  {done:,}/{len(texts):,}  ({speed:,.0f}/s, "
        f"~{(len(texts)-done)/speed:,.0f}s left)")

vectors = np.vstack(chunks)
elapsed = time.time() - t0
log(f"embedded in {elapsed:.1f}s ({len(texts)/elapsed:,.0f} texts/s) shape {vectors.shape}")

# L2-normalise so cosine similarity is a plain dot product, matching what the
# local pipeline stores.
vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

# --- write the artefact ------------------------------------------------------

os.makedirs("catalog_embeddings", exist_ok=True)
np.save("catalog_embeddings/embeddings.npy", vectors)
with open("catalog_embeddings/ids.json", "w", encoding="utf-8") as fh:
    json.dump({"ids": ids, "model": model_name}, fh)

os.system("zip -qr catalog_embeddings.zip catalog_embeddings")
size_mb = os.path.getsize("catalog_embeddings.zip") / 1e6
print(f"\nwrote catalog_embeddings.zip  ({size_mb:.1f} MB)")
print("download it, unzip, then run:  voice-order import-embeddings catalog_embeddings")

try:                                          # Colab: start the download
    from google.colab import files

    files.download("catalog_embeddings.zip")
except ImportError:
    print("(Kaggle: find it under Output on the right-hand panel)")

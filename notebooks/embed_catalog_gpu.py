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
print("onnxruntime providers:", providers, flush=True)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "No CUDAExecutionProvider. This would run on CPU and take about three "
        "hours instead of one minute.\n"
        "Fix: Settings -> Accelerator -> GPU T4 x2, then Run -> Restart & Clear "
        "Cell Outputs, and run this cell again."
    )

# --- locate the export -------------------------------------------------------

candidates = (
    glob.glob("/kaggle/input/**/embed_input.jsonl.gz", recursive=True)
    + glob.glob("./embed_input.jsonl.gz")
    + glob.glob("/content/embed_input.jsonl.gz")
)
if not candidates:
    try:                                     # Colab: prompt for the upload
        from google.colab import files

        files.upload()
        candidates = glob.glob("embed_input.jsonl.gz")
    except ImportError:
        pass
if not candidates:
    raise SystemExit(
        "embed_input.jsonl.gz not found. Upload it first "
        "(Kaggle: Add Data -> Upload; Colab: run this cell and pick the file)."
    )

source = candidates[0]
print("reading", source, flush=True)

# --- read ids and texts ------------------------------------------------------

meta, ids, texts = None, [], []
with gzip.open(source, "rt", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("_meta"):
            meta = row
            continue
        ids.append(row["id"])
        texts.append(row["text"])

model_name = (meta or {}).get("model", "BAAI/bge-small-en-v1.5")
print(f"{len(texts):,} texts  |  model {model_name}", flush=True)

# --- embed -------------------------------------------------------------------

model = TextEmbedding(model_name=model_name, providers=["CUDAExecutionProvider"])

# Time a small batch before committing to the whole catalog. A GPU does
# hundreds of texts a second; CPU does about ten. Better to find out now than
# forty minutes in with nothing on screen.
warm = texts[:256]
t0 = time.time()
list(model.embed(warm, batch_size=256))
rate = len(warm) / max(time.time() - t0, 1e-6)
print(f"warm-up: {rate:,.0f} texts/s", flush=True)
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
    print(f"  {done:,}/{len(texts):,}  ({speed:,.0f}/s, "
          f"~{(len(texts)-done)/speed:,.0f}s left)", flush=True)

vectors = np.vstack(chunks)
elapsed = time.time() - t0
print(f"embedded in {elapsed:.1f}s  ({len(texts)/elapsed:,.0f} texts/s)  shape {vectors.shape}",
      flush=True)

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

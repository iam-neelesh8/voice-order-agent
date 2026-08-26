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
os.system("pip install -q fastembed-gpu onnxruntime-gpu")

import numpy as np
from fastembed import TextEmbedding

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
print("reading", source)

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
print(f"{len(texts):,} texts  |  model {model_name}")

# --- embed -------------------------------------------------------------------

model = TextEmbedding(model_name=model_name, providers=["CUDAExecutionProvider"])

t0 = time.time()
vectors = np.asarray(list(model.embed(texts, batch_size=256)), dtype=np.float32)
elapsed = time.time() - t0
print(f"embedded in {elapsed:.1f}s  ({len(texts)/elapsed:,.0f} texts/s)  shape {vectors.shape}")

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

# Running the GPU steps on Kaggle

Two steps in this project want a GPU. Everything else is CPU work and already
runs on your laptop.

| step | your laptop, measured | Kaggle T4 |
|---|---|---|
| Embed 100k catalog titles | ~3 hours | ~1 minute |
| Transcribe 5 conditions, 1-best | ~6.75 hours | under an hour |
| Transcribe 5 conditions, n-best | ~34 hours | ~2 hours |

Kaggle gives roughly 30 GPU-hours a week free. Both jobs together use about
two, so one session covers it with room to spare.

---

## Before you start

Generate the two upload bundles:

```bash
voice-order export-embed-input     # -> data/exports/embed_input.jsonl.gz   6.2 MB
voice-order export-asr-input       # -> data/exports/asr_dev.zip          481.5 MB
```

Then, once on kaggle.com:

1. **Create** → **New Notebook**
2. Right panel → **Session options** → **Accelerator** → **GPU T4 x2**
3. Right panel → **Session options** → **Internet** → **On**
   *(required — both notebooks pip-install and download model weights)*

**Check the GPU is really attached.** Kaggle ships a CPU build of
`onnxruntime`, and installing the GPU one alongside it leaves the CPU build
winning — silently, at roughly 1/180th the speed. Both notebooks now refuse to
start if that happens, but if you want to check by hand:

```python
import onnxruntime, subprocess
print(onnxruntime.get_available_providers())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:400])
```

`CUDAExecutionProvider` must be in that list.

**If a cell sits with no output**, it is not necessarily working. Stop it,
**Run → Restart & Clear Cell Outputs**, and re-run — restarting matters because
the wrong `onnxruntime` stays loaded in memory otherwise.

---

## Job 1 — embed the catalog (~5 minutes total)

**Upload the data**

Right panel → **Input** → **Upload** → **New Dataset**. Drag in
`data/exports/embed_input.jsonl.gz`, name it `voice-order-embed`, **Create**.
It lands at `/kaggle/input/voice-order-embed/`.

**Run it**

Open `notebooks/embed_catalog_gpu.py` from this repo, copy the whole file into
one notebook cell, and run. The notebook finds the upload itself. You should
see something like:

```
100,000 texts  |  model BAAI/bge-small-en-v1.5
embedded in 58.3s  (1,715 texts/s)  shape (100000, 384)
wrote catalog_embeddings.zip  (154.2 MB)
```

**Bring it back**

Right panel → **Output** → download `catalog_embeddings.zip`. Unzip it, then:

```bash
voice-order import-embeddings catalog_embeddings
```

Expect `min cosine 1.00000`. Anything below 0.99 and the import refuses the
file — see *If the import rejects your file* below.

**Then dense retrieval is live:**

```bash
voice-order eval typed --query-set orders --split dev --retrievers lexical,dense
```

Compare it against the lexical-only number (recall@1 **0.576**) to find out
whether dense earns its place at all.

---

## Job 2 — transcribe the spoken set (~1 hour)

**Upload the data**

Same as above, but with `data/exports/asr_dev.zip`, named `voice-order-asr`.
It is ~456 MB, so give the upload a few minutes.

The bundle carries a fingerprint of the audio it was built from, and the
import checks it. If you regenerate the audio locally after exporting, run
`voice-order export-asr-input` again -- otherwise you would spend an hour of
GPU transcribing recordings that no longer exist on your machine, and the
`query_id` check alone cannot detect that.

**Run it**

Copy `notebooks/transcribe_gpu.py` into a cell. Check the settings at the top:

```python
MODELS = ["small.en", "large-v3"]   # the comparison worth the GPU hour
CONDITIONS = None                    # None = all five
N_BEST = 5                           # 1 is ~5x faster; enough for stage 4 alone
```

**Leave `MODELS` with both entries.** That comparison is the point of the run:

- If `large-v3` recovers most of the loss, the answer is *use a bigger model*.
- If it does not, that is the justification for everything in stage 5.

If you are short on time, set `N_BEST = 1` first — it gives the stage 4 drop
in a fifth of the time. Stage 5 is what actually consumes the alternatives.

**Bring it back**

Download `transcripts.zip` from **Output**, unzip, then:

```bash
voice-order import-transcripts transcripts
```

**Then the stage 4 numbers:**

```bash
voice-order eval spoken --split dev --condition clean
voice-order eval spoken --split dev --condition phone
voice-order eval spoken --split dev --condition phone_snr10
```

The drop from **0.576** is the headline result of the project.

---

## If the session dies partway

Both notebooks resume. Re-run the same cell and they skip what is already
done — embeddings by chunk, transcripts by clip. Kaggle reclaims free-tier
sessions without warning, which is why this matters.

Kaggle keeps `/kaggle/working` between runs of the same notebook, so the
partial output is still there.

## If the import rejects your file

That is the import working. Both are checked because both failure modes are
silent — nothing crashes, the answers are just wrong.

| message | cause | fix |
|---|---|---|
| `index ids diverge from the catalog` | the catalog was reloaded after exporting | re-export, re-run the notebook |
| `do not match locally computed` | the notebook used a different model | re-run it unmodified |
| `query_ids that are not in the manifest` | transcripts from a different evalset | re-export the audio bundle |
| `transcribed from different audio` | the audio was regenerated after you exported the zip | `voice-order export-asr-input`, upload the new zip |
| `N vectors but M ids` | truncated download | download again |

## Notes

- **Do not edit the model name** in either notebook. Document and query
  vectors must come from the same model or every cosine score is
  miscalibrated — the import will catch it, but only after you have spent the
  GPU time.
- `fastembed-gpu` is pinned deliberately: it uses the exact ONNX weights the
  local CPU path uses, which is why the verification returns cosine 1.00000
  rather than merely "close".
- Kaggle's free GPU quota resets weekly. Check **Settings → Accelerator** for
  what you have left.

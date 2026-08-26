# voice-order-agent

A voice agent that takes product orders over the phone — speech in, order in
a database out. Open-source ASR, hybrid retrieval, and an eval harness that
measures whether it actually works.

```
caller ──► ASR ──► agent loop ──► retrieval ──► SQLite
  ▲    (faster-whisper)  │      (BM25 + dense)     │
  └────── TTS ◄──────────┘                      orders,
        (Piper)                            carts, transcripts
```

Everything runs locally on open-source models and stdlib SQLite. No speech
APIs, no LLM API keys, no database server, no GPU required.

## Why this problem and not a generic chatbot

The interesting failure is not dialogue. It is **identifiers**.

> "I need an AC Delco 41-993"

Speech recognition mangles alphanumerics — `forty one dash nine ninety three`,
`41-99 3`, `4199 3`. A companion study
([fitment-rag](https://github.com/iam-neelesh8/fitment-rag)) measured typed
retrieval on this same corpus and found lexical matching wins *because*
product identifiers are exact tokens. Speech destroys exactly those tokens.

So the agent never trusts a single transcript: retrieval runs over n-best ASR
hypotheses, part numbers are digit-normalised and phonetically matched, and
anything ambiguous gets a confirmation turn before it reaches the cart.

## Status

Stage 0 of 8 — scaffold. Nothing is implemented yet; every module raises
`NotImplementedError` naming the stage it belongs to.

| doc | what it covers |
|---|---|
| [docs/voice-order-agent.drawio](docs/voice-order-agent.drawio) | **start here** — 4-page draw.io diagram: build order, system, one turn, data pipeline |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the same thing in text + mermaid, for reading on GitHub |
| [data/README.md](data/README.md) | the four datasets, what is committed and what is not |
| [DATA_LICENSES.md](DATA_LICENSES.md) | attribution and terms |

## Getting started

```bash
pip install -e ".[retrieval,speech,dev]"
voice-order db init           # stage 1 — creates data/voice_order.db
voice-order catalog           # stage 1 — ~100k items into SQLite
voice-order index all         # stage 2 — build BM25 + dense indexes on disk
voice-order eval typed        # stage 2 — the first number
```

No config needed to start: the database is one file at `data/voice_order.db`
and the indexes are files under `data/index/`. Delete both and rerun to reset.
`voice-order --help` lists one command per stage.

## Running the heavy steps on a GPU

Two steps want a GPU. Both are the same round trip: export a file, run a
notebook, import the result with checks.

| step | this laptop, measured | a T4 |
|---|---|---|
| embed 100k catalog titles | ~3 hours | ~1 minute |
| transcribe 5 conditions, 1-best | ~6.75 hours | under an hour |
| transcribe 5 conditions, n-best | ~34 hours | ~2 hours |

```bash
# embeddings
voice-order export-embed-input          # -> data/exports/embed_input.jsonl.gz (6 MB)
# run notebooks/embed_catalog_gpu.py, download + unzip
voice-order import-embeddings catalog_embeddings

# transcripts
voice-order export-asr-input            # -> data/exports/asr_dev.zip
# run notebooks/transcribe_gpu.py, download + unzip
voice-order import-transcripts transcripts
```

Both imports are **verified, not trusted** -- see below.

Embedding the 100k catalog is ~3 hours on a laptop CPU and about a minute on
a T4. It is a two-command round trip, not a science project:

```bash
voice-order export-embed-input          # -> data/exports/embed_input.jsonl.gz (6 MB)
# run notebooks/embed_catalog_gpu.py on Kaggle or Colab with a GPU runtime,
# upload that file, download catalog_embeddings.zip, unzip
voice-order import-embeddings catalog_embeddings
```

`import-embeddings` refuses the file unless the ids match the catalog exactly
row for row, and a sample re-embedded locally matches what came back -- document
and query vectors must come from the same model or every cosine score is
miscalibrated.

`import-transcripts` refuses any file whose query_ids are not in the manifest,
because evaluating those would score one query's retrieval against another
query's speech.

Every one of those failures is silent: nothing crashes, the answers are just
wrong. That is why they are checked rather than assumed.

## Storage and retrieval, and why they are separate

The database is **storage only** — products, calls, carts, orders. Retrieval
runs in-process against index files rebuilt from it. Two reasons:

- **The numbers must not depend on the backend.** SQLite FTS5 `bm25()` and
  Postgres `ts_rank` are different ranking functions. For a project whose
  whole output is recall@k, a score that changes with the database is
  disqualifying. The BM25 index is ours, so it is identical everywhere.
- **Stage 5 has to ablate three components independently.** In-process that
  is a constructor flag; in SQL it is a schema migration per variant.

That leaves nothing that needs a server at this scale. `db/repository.py` is
the only module touching SQL, so a Postgres backend is one more implementation
of that interface if you want one — it is not needed to run this.

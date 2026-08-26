# Data

Four datasets, different purposes, different lifetimes. Only the first is a
real dependency; the rest are generated.

| # | dataset | purpose | in git? |
|---|---|---|---|
| 1 | Amazon Reviews 2023 item metadata | the catalog | no — streamed |
| 2 | Order queries with known answers | evaluation ground truth | yes |
| 3 | Synthesized + degraded audio | the spoken test set | no — regenerable |
| 4 | Human recordings | held-out reality check | yes, it is small |

## Layout

```
data/
  README.md          this file
  catalog/           downloaded metadata shards      (gitignored)
  evalsets/          queries + ground truth          (committed)
  audio/
    synthetic/       TTS output, clean and degraded  (gitignored)
    human/           ~100 real recordings            (committed)
  noise/             MUSAN subset                    (gitignored)
```

---

## 1. Catalog — Amazon Reviews 2023

McAuley Lab, UC San Diego. Public, released for research.
`huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`

Item **metadata** only (`meta_<Category>.jsonl`). The reviews themselves are
not used — this is a product lookup problem, not a sentiment problem.

The full dataset is ~48M items across 33 categories. Do not load it. Take a
bounded slice, set in `configs/catalog.yaml`:

| category | items | why |
|---|---|---|
| Automotive | 40k | part numbers — the hard case |
| Tools_and_Home_Improvement | 20k | model numbers, sizes |
| Electronics | 20k | model numbers, high name collision |
| Office_Products | 10k | mundane, high-frequency reorders |
| Health_and_Household | 10k | plain names, no identifiers — the easy case |

~100k items total. The mix is deliberate: it spans catalogs where identifiers
carry the signal and catalogs where they do not, so results can be reported
per category rather than as one average that hides the difference.

Fields used per item: `parent_asin` (the id), `title`, `price`,
`average_rating`, `store`, `categories`, `details`, `features`.

Streamed as raw JSONL over HTTP rather than via `datasets`, which dropped
script-based loaders in 4.x. fitment-rag already has this loader.

**License:** research use. Attribution and terms go in `DATA_LICENSES.md`.
Nothing here is company data.

## 2. Order queries — generated, with known answers

The evaluation depends on knowing which product *should* have been found.
That means generating queries from the catalog rather than collecting them.

**Lookup queries** — reused from fitment-rag
(`evalsets/amazon_automotive_10k_n2000.jsonl`, 2,000 items). Each has a
`question`, `relevant_doc_ids`, and a `query_id`. Phrased as catalog
questions, not orders, but the ground truth is already built and verified —
the cheapest way to get a first number.

**Order queries** — new, and the ones that matter. Templated into how people
actually order, then filled from catalog fields:

```
"I need two AC Delco 41-993 spark plugs"
"Can you get me a Bosch oil filter for a 2012 Civic"
"Uh, the ... the brake pads, the ceramic ones"
```

Quantities, brand + identifier combinations, partial names, disfluencies.
Same rule as fitment-rag: generated deterministically from a seed, one
product per query, answer known.

**Split discipline** — dev and test are fixed and separate. Every
configuration choice is made on dev. Every published number comes from test,
which is not looked at while deciding anything.

## 3. Spoken test set — synthesized, then degraded

The queries already have known answers, so speaking them with TTS gives
perfect ground truth for free.

**Synthesis** — Piper, CPU, faster than realtime. 4+ voice models, not one,
or the result measures how well the ASR handles a single speaker. Output
keyed by `query_id` so audio stays joined to ground truth.

**Degradation** — not optional. Clean TTS audio is nothing like a phone call.
Phone lines are narrowband:

```
ffmpeg -i clean.wav -ar 8000 -ac 1 -c:a pcm_mulaw -f wav phone.wav
```

Then mix background noise from **MUSAN** at several SNRs. MUSAN is ~11 GB; a
subset is enough. Keep the clean version too — the clean-vs-phone gap is
itself a result worth reporting.

Size: ~2,000 clips at ~3 s is roughly 190 MB at 16 kHz, ~50 MB after the
phone codec. Gitignored — regenerates from a seed in under an hour.

## 4. Human recordings — the reality check

~100 utterances, read by real people, ideally some over an actual phone call,
with a spread of accents.

Small enough to commit. Never used for tuning. This is the only thing that
says whether the synthetic 2,000 can be trusted, and if the two disagree, the
humans are right.

---

## Open questions

- How many categories before per-category results stop being meaningful?
- ~~Do the fitment-rag lookup queries transfer at all, or is order phrasing
  different enough that only the new set counts?~~
  **Partly answered in stage 2, and the answer is sharper than expected.**
  All 2,000 gold ids are present in our slice, so the set is usable and gives
  a real first number (recall@1 0.744, recall@20 0.931, MRR 0.811 on BM25).
  But only **6% of its queries contain an identifier at all** -- they are
  phrased as attribute questions ("How much does the fel pro plenum cost?").
  So this set measures name-based retrieval and cannot test the premise the
  project is built on. Worse, the 6% that *do* carry an identifier score
  lower (recall@1 0.579 vs 0.755), because they are short ambiguous numerics
  like `1080` and `3344` that collide across the catalog.
  The stage 3 order generator is therefore not optional -- it is the only
  thing that will put part numbers in front of the retriever.
- ~~Is pgvector fast enough at 100k, or does FAISS stay a separate index?~~
  Answered before building: neither is needed. 100k x 384 float32 is ~150 MB
  and a brute-force scan is one matmul. The measured latency is the result;
  an ANN index becomes worth it around a million vectors.
- How many human recordings are actually needed to detect a gap between
  synthetic and real? 100 is a guess, not a calculation.

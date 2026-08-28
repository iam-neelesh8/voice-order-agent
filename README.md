# voice-order-agent

A phone-to-purchase agent for a parts shop. A caller speaks, the agent finds
the product, takes the order, and reads back the total — by voice.

```
you speak ──► ASR ──► search ──► agent (LLM) ──► TTS ──► you hear
           (Whisper)  (BM25 +     picks, cart,   (Piper)
                    part numbers)  confirms
```

Three stages, each built the cheapest good way — open source first, an API
key only where it clearly wins. Every technology choice is **measured, not
assumed**, and the reasoning is kept as a learning journal:
**[docs/LEARNINGS.md](docs/LEARNINGS.md)**.

## Demo

<!--
  To make the video play INLINE on GitHub (recommended):
  open this README on github.com, click the edit (pencil) icon, drag Demo.mp4
  into the editor, and GitHub replaces this block with an embedded player
  hosted on its CDN. That keeps the 92 MB file out of the repo entirely.

  Paste the GitHub-generated video URL here once you have it:
-->

https://github.com/iam-neelesh8/voice-order-agent/assets/PLACEHOLDER/Demo.mp4

A caller speaks an order; the agent finds the part, builds the cart, reads back
the total, and places it — end to end, by voice.

## Try it

```bash
pip install -e ".[retrieval,speech]"
python -m voice_order serve
```

Open **http://localhost:8000**, press the mic, and speak an order — or type
one. Pick the model from the dropdown (local Ollama, or a fast API). The right
panel shows every layer: what the model asked for, what the catalog returned,
and the cart building.

First run needs a one-time data build (a few minutes):

```bash
python -m voice_order db init       # create the database
python -m voice_order catalog       # ~100k products from Amazon Reviews 2023
python -m voice_order seed           # prices, inventory, customers
python -m voice_order index all      # build the search indexes
```

## The hard part is identifiers, not dialogue

> "I need an AC Delco 41-993"

Speech recognition mangles part numbers — `forty one dash nine ninety three`,
`41-99 3`, `4199 3`. Typed, the agent finds the right product **58%** of the
time; after speech, **~30%**. That gap is almost entirely part numbers: word
error rate is **84%** on queries with an identifier against **15%** without.

So the agent never trusts one transcript. Part numbers are digit-normalised,
matched exactly then fuzzily, and Whisper is biased toward writing digits as
digits. A six-times-larger model recovered 8% of the loss; a free biasing
prompt recovered more. The measurements are the point — see the journal.

## The design: the model proposes, the code disposes

The LLM holds the conversation but can only *ask* for six functions; code
validates and runs them. It cannot invent a product, a price, or a total. A
model that hallucinates a part number is refused in code, not asked nicely in
a prompt. That means swapping the model — local for API, one for another —
can never widen what the agent is allowed to do, and a wrong order is always
traceable to the step that caused it.

| doc | what it covers |
|---|---|
| [docs/LEARNINGS.md](docs/LEARNINGS.md) | how BM25, embeddings, local LLMs and Whisper actually behave here, with the number behind every claim |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the system, one turn end to end, and the build order |
| [data/README.md](data/README.md) | the catalog, the test set, and the results |

## Switching the model

Local Ollama by default. To use a hosted model, put the key in a `.env` file
and switch:

```bash
python -m voice_order model gemini     # or in the demo's dropdown, live
```

Any OpenAI-compatible endpoint works — Ollama, Gemini, Groq, OpenAI — because
the tools and rules live outside the model. See `configs/agent.yaml`.

## Data

Catalog is item metadata from Amazon Reviews 2023 (McAuley Lab, UC San Diego),
research use, streamed at build time. Prices, inventory and customers are
generated and marked as such. Nothing here is company data. See
[DATA_LICENSES.md](DATA_LICENSES.md).

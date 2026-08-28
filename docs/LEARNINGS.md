# Learning journal

The goal is a phone-to-purchase application: a caller speaks, and an order is
placed. Three stages, each to be built the cheapest good way — open source
first, an API key only where it clearly wins:

    voice -> text      (ASR)
    text  -> product   (retrieval + an agent)
    text  -> voice     (TTS)

This file is the running notebook. Every experiment, every number, every thing
that turned out differently from the textbook. It is written so that someone
who has never seen the project can learn *how these pieces actually behave*,
not just that they exist.

Rule for this file: **no claim without a number, and every number came from a
run.** Guesses are labelled as guesses.

---

## The one finding that shapes everything

Typed, the system finds the right product **58%** of the time. Spoken, **27%**.

That 31-point gap is the whole problem, and it lives in one place: **speech
destroys part numbers, and nothing else.**

| query contains... | word error rate |
|---|---|
| a part number (`41-993`) | **84%** |
| no part number (`brake pads`) | 15% |

Whisper transcribes ordinary words almost perfectly and mangles identifiers
nearly six times as often. Everything below follows from that.

---

## Stage 1 — voice to text (ASR)

### What we use and why

**faster-whisper** (`small.en` / `large-v3`). Open source, runs on CPU or GPU.
`faster-whisper` is a re-implementation of OpenAI's Whisper on CTranslate2; it
is several times faster than the original for the same weights.

### How Whisper actually fails — the thing to understand

Whisper does **not mishear** identifiers. It hears the digits correctly and
**writes them as English**:

    41-993  ->  "forty one ninety three"
                "four one nine nine three"
                "41 99 3"

The sound was fine. The spelling is wrong. This is why it is worth
understanding: it means the fix is not a better microphone or a bigger model —
it is text processing after the fact.

### Experiment: does a bigger model fix it?  (measured)

| model | params | identifier WER | recall@1 |
|---|---|---|---|
| small.en | 39M | 84% | 0.212 |
| large-v3 | 1550M | 54% | 0.241 |

**No.** Six times the model, 4.4 GPU-hours, and it recovered 2.9 points of a
36-point drop — 8%. It transcribes better, but recall barely moves, because a
part number is either exactly right or useless. `41-993` heard as `41-99 3` is
as unfindable as `forty one ninety three`. There is no partial credit in an
exact-match lookup.

**Lesson: throwing model size at a problem is the expensive reflex. Measure
what the errors actually are first.** Here they were orthographic, and no
acoustic model fixes orthography.

### Speed, measured

| | per clip | 100k-clip job |
|---|---|---|
| small.en, CPU (14 cores) | ~2.5 s | ~80 min/condition |
| small.en | ~0.3 s | ~5 min |

Whisper is ~20% of a voice turn. It is not the latency problem (the LLM is).

### Experiment: initial_prompt biasing  (free, CPU, ~8 min)

Whisper generates text token by token, each conditioned on what came before.
`initial_prompt` is fed in as fake *previous context*, so the model's built-in
language model is primed. Prime it with hyphenated digit codes and it keeps
writing that pattern instead of spelling numbers out.

Prompt used:

    "Ordering auto parts by part number. Examples: 41-993, P0420, AV1200,
     3397118933, F4ZZ-2B293-A, H9004S, 8542043983."

Same 80 phone clips, small.en, identical decoding, prompt vs no prompt:

    no prompt         36.2%
    + initial_prompt  47.5%   (+11.2)

**Bigger than predicted, and the reason is instructive.** I expected a small
gain, because the spoken-digit normaliser already turns "forty one ninety
three" into 41993. But the prompt fixes a *different, worse* failure that the
normaliser cannot touch: Whisper writing a code as a spelled-out cardinal
number.

    want 532135
      no prompt:  "five hundred thirty two thousand one hundred thirty five"
      prompted:   "532135"

"Five hundred thirty two thousand..." is arithmetic prose, not a digit string
-- nothing downstream recovers it. The prompt stops Whisper doing this at the
source by teaching it "these are codes, not quantities". It also keeps letters
attached (H9004S, W21411, I31115G4) instead of splitting them off.

**Lessons:**
- A prompt can bias *style* (how the model writes) for free, without touching
  the acoustics. Cheapest ASR win in the project so far.
- It has a hard ceiling: ~224 tokens, so it cannot carry a 160k-part-number
  vocabulary. It is a style hint, not a lookup. Getting a *wrong* digit
  (failure mode 2) is untouched by it -- that still needs vocabulary-constrained
  ASR (paid) or fuzzy matching (our own code).
- I predicted "modest" and was wrong by running it. The spelled-out-cardinal
  failure was invisible until the transcripts were read.

Now on by default -- `configs/asr.yaml: decode.initial_prompt`.

### Still worth trying

`hotwords` (a newer faster-whisper feature) boosts specific words with the
same short-context limit. And large-v3-turbo -- large-v3 accuracy at a speed
that runs locally -- is the next thing to measure.

---

## Stage 2 — text to product (the interesting stage)

Three ways to turn words into a product. We measured all three. Only two earn
their place.

### BM25 — keyword matching

**What it is.** A scoring formula from the 1990s that ranks documents by how
many query words they contain, weighted so that rare words count more (a word
in every product tells you nothing) and long documents do not win just by being
long. No neural network, no training. `bm25s` runs it in-process in
milliseconds over 100k products.

**Why it wins here.** A part number is a rare, exact token. BM25 rewards exactly
that. Measured recall@1 by how much the caller gives:

    brand + part number + name     0.970
    brand + part number            0.919
    name alone                     0.117

The identifier is doing the work.

### Dense / embeddings — meaning matching

**What it is.** A small neural model (`bge-small`, 33M params, via `fastembed`
on ONNX — no PyTorch) turns each product title into a 384-number vector, so
that things with *similar meaning* sit close together. "the ceramic ones" lands
near ceramic brake pads even with no shared words. Search is one big matmul over
100k vectors — ~150 MB, tens of milliseconds, no fancy vector database needed.

**Experiment: does adding it help?  (measured)**

| retriever | recall@1 |
|---|---|
| BM25 | **0.576** |
| dense | 0.282 |
| BM25 + dense fused | 0.492  ← *worse than BM25 alone* |

**No, it hurts.** Dense scores **0.006** on bare part numbers — near zero. An
embedding maps `41-993` into a fog of similar-looking digit strings; it has no
concept of "this exact token". And because the fusion is rank-based, a
retriever that is near-random on identifiers drags the good one down.

**Lesson: embeddings are for meaning, not for identity.** When the answer is an
exact code, similarity is the wrong tool. The GPU hour was still worth it — it
turned "everyone says add embeddings" into a measured *no*.

### Part-number index — exact lookup

**What it is.** Not a ranker. A dictionary: normalised identifier -> product.
An identifier is not *evidence* for a product, it *is* the product.

**Experiment: the stage 5 ablation** (phone audio, recall@1 on identifier
queries):

    BM25 only                       0.226
    + part-number index             0.328   (+0.102)
    + n-best fusion                 0.352   (+0.024)

The part-number index is the single largest win in the project — twice what
large-v3 bought, for microseconds instead of GPU-hours.

### The spoken-digit recovery — the highest-leverage code in the project

Before the exact lookup can work, `"forty one ninety three"` has to become
`41993`. A hand-written normaliser does this: word digits, "double seven",
"oh" for zero, homophones ("for"/"four"), and the quantity glued onto the code
("a CAT6A" -> "1CAT6A").

Share of identifiers recoverable from the ASR output:

    raw ASR                    24%
    + spoken-digit recovery    36%   (+12, more than large-v3's +5)
    + n-best (all hypotheses)  46%

Microseconds of Python, worth more than 4.4 GPU-hours.

### Fuzzy matching — where a recovery proxy lied

For the wrong-digit case (Whisper heard `VAG1809`, the real code is
`VAGG1809`), a deletion index finds real catalog codes one edit away. A quick
proxy — "is the gold code within edit distance 1 of anything we produced" —
said **+11 points**. That proxy was misleading.

End-to-end, measured on 200 phone clips:

    identifier queries, top-1     41% -> 42%   (+1)
    identifier queries, top-5     51% -> 53%   (+2)

**Lesson: recoverable is not the same as findable.** Fuzzy floods the
candidate list with plausible neighbours (`41994` -> `1994`, `41984`, ...), so
the right product gets *into* the top-5 but rarely ranks first among the noise.
The proxy counted "is it reachable"; ranking it first when the set is full of
near-misses is a different, harder thing.

But the top-5 gain is exactly the setup the LLM reranker needs: fuzzy puts the
right product in the shortlist, and a reranker that reads all five products'
details picks it out. Fuzzy alone is +2; fuzzy feeding a reranker is where it
converts. Kept, scored strictly below any exact match, off by default via a
flag for the ablation.

### n-best fusion — the counter-intuitive one

Whisper can return its top 5 guesses, not just 1. Retrieving over all of them:

    BM25 + n-best (no part-number index)    0.180   ← WORSE than 1-best (0.212)
    part-number index + n-best              0.352   ← better than without (0.328)

**On its own, n-best hurts.** Four wrong transcripts outvote one right one. It
only pays off next to a retriever that can turn a *single* correct guess into a
definitive hit. **Lesson: a technique can be useless alone and valuable in
combination — you cannot judge it in isolation.**

---

## Stage 3 — text to voice (TTS)

**Piper.** Open source, runs on CPU at ~23x realtime, no GPU. Used both to
speak replies and to build the synthetic test set.

**Learning that cost real time:** Piper is *non-deterministic* by default —
the same sentence twice produces different audio, because its duration
predictor samples randomly. Seeding numpy does nothing (the randomness is
inside ONNX). Setting `noise_scale=0` makes it reproducible, at the cost of
flatter prosody. A test set that will not regenerate identically is not a test
set, so this mattered.

---

## The LLM — how the agent actually works

### Local vs API, measured on a 16 GB laptop, no GPU

| | latency per turn | tool-calling |
|---|---|---|
| qwen2.5:1.5b | ~4 s | **fails** — no tool calls, hallucinates a product and a price |
| qwen2.5:3b | ~15 s | 3/7 — never searches, invents ids |
| qwen2.5:7b | ~24 s | **7/7** — correct |
| API (est.) | ~2-5 s | best |

**Lesson on local LLMs: tool-calling has a size floor.** All three emit valid
JSON; only 7b *sequences* the calls correctly (search before adding, read the
total before placing). Below ~7B the model does not understand *when* to call
what. And the LLM is ~80% of voice-turn latency — the single strongest reason
to consider an API key, and it changes nothing about retrieval quality.

### The design that makes the model safe: it proposes, code disposes

The model can only ask for six functions; code validates and runs them. A model
that invents `product_id: "ACDelco spark plug"` is refused — checked in code,
not asked of the prompt. Measured: the 1.5b model hallucinated a Bosch part at
$2.50, and the cart stayed empty because nothing reached it without a real
search result.

**This is the core architectural lesson.** Prices, the cart, and the
confidence decision live *outside* the model. Swapping the model — local for
API, 3b for 7b — cannot widen what the agent is allowed to do. When an order
comes out wrong, you can point at the step that did it.

---

## Running tally against the goal

| stage | best cheap choice so far | open? | state |
|---|---|---|---|
| voice -> text | small.en + initial_prompt + digit recovery | yes | prompt-biasing measured +11.2 |
| text -> product | BM25 + part-number index (no embeddings) | yes | 0.226 -> 0.352 on identifiers |
| text -> voice | Piper | yes | works |
| the agent | qwen2.5:7b local, API-swappable | yes | 7/7, but 24s/turn on CPU |

Recall on identifier queries: **0.226 -> 0.352**, against a typed ceiling of
**0.779**. About a third of the speech gap closed with pure text processing and
zero extra model cost.

## Open experiments, in priority order

1. ~~Whisper `initial_prompt` biasing~~ — **done, +11.2 points, now default.**
2. **large-v3-turbo** — large-v3 accuracy at a speed that runs on the laptop.
3. **General fuzzy matching** — ~8 points of identifier headroom remain after
   the digit recovery. Needs a deletion index; an afternoon.
3. **API LLM for latency** — 24s -> ~4s per turn. Fixes the demo, not the
   numbers.
4. **The real voice loop** — mic -> Whisper -> agent -> Piper, end to end.

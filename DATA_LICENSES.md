# Data licenses and attribution

Nothing in this repository is company data. Every dataset below is public.

## Amazon Reviews 2023 — item metadata

McAuley Lab, UC San Diego.
<https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023>

Released for **research use**. Only the item metadata (`meta_<Category>.jsonl`)
is used here; the review text is not. A bounded ~100k-item slice is streamed
at build time and is not redistributed in this repository.

> Hou et al. *Bridging Language and Items for Retrieval and Recommendation.*
> arXiv:2403.03952, 2024.

## MUSAN — background noise

<https://www.openslr.org/17/> — CC BY 4.0. A subset is downloaded for audio
degradation and is not redistributed here.

## Piper — TTS voices

<https://github.com/rhasspy/piper> — MIT. Voice models carry their own
licenses; check each before redistributing generated audio.

## faster-whisper / Whisper

<https://github.com/SYSTRAN/faster-whisper> — MIT, over OpenAI Whisper (MIT).

## Human recordings — `data/audio/human/`

Recorded by consenting speakers for this project. No personal data beyond the
voice itself; utterances are read from a fixed script and contain no real
customer or order information.

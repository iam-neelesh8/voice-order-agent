# Transcripts behind the stage 4 result

Committed on purpose, which is an exception to the rule in `data/README.md`
that regenerable data stays out of git.

These are regenerable, but only from an hour of GPU time, and they are the
evidence for the headline claim -- that speech costs 36 points of recall and
the phone line costs one. Someone checking that number should not have to
rent a GPU to see the data it came from.

    clean__small-en.jsonl    1,998 clips, no degradation
    phone__small-en.jsonl    1,998 clips, G.711 8 kHz mu-law

Both from `small.en` at 1-best, produced by `notebooks/transcribe_gpu.py` on a
Kaggle T4, against audio with fingerprint `a8342d02045241a4`.

To use them:

    voice-order import-transcripts data/transcripts_gpu
    voice-order eval spoken --split dev --condition phone

The import checks that fingerprint against your local audio and refuses if it
does not match, so these cannot be scored against a different test set.

A partial `phone_snr10` run of 23 clips was deleted rather than kept. At 1.2%
coverage it would evaluate cleanly and mean nothing, which is worse than not
being there.

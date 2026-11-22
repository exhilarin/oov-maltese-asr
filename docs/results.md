# Results and error analysis

All numbers: FLEURS `mt_mt` test (926 utterances, 3.55 h), one shared normalization (`src/common.py`), Tesla T4.

## Main table

| system | WER % | CER % | RTF |
|---|---|---|---|
| nvidia/canary-1b-v2 | **22.15** | 7.10 | 0.041 |
| nvidia/parakeet-tdt-0.6b-v3 | 24.77 | **6.45** | 0.010 |
| wav2vec2-large-xlsr-53-maltese-64h | 25.41 | 7.38 | 0.007 |
| openai/whisper-small, zero-shot | 111.93 | 54.01 | 0.193 |
| **ours: whisper-small fine-tuned** | 34.24 | 9.35 | 0.012 |

## Reading the table

* **Zero-shot Whisper-small does not know Maltese.** WER above 100% means it inserts many words that are not in
  the audio (hallucination); it is also slow (RTF 0.19) because it generates long outputs.
* **Ten hours of fine-tuning change this completely:** 111.9 → 34.2 WER and 54.0 → 9.4 CER, a 78-point drop.
* **The large multilingual models stay ahead.** Canary (22.2) and Parakeet (24.8) are trained on much more
  European speech, and the wav2vec2 baseline on 64 h of Maltese, versus our 10 h.
* **CER is much lower than WER for every model** (e.g. ours 9.4 vs 34.2): most wrong words are wrong by one or two letters.

## Checkpoint selection (dev only)

| step | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|
| dev WER % | 44.6 | 36.1 | **34.6** | 34.7 |
| dev CER % | 12.3 | 9.7 | 9.4 | 9.4 |

Step 750 was chosen on dev and decoded on test once. Dev and test WER agree (34.6 vs 34.2).

## Error analysis

Word error on reference words with Maltese-specific spelling (ċ ġ ħ ż, word-final apostrophe, clitic hyphen) vs plain words:

| system | Maltese-spelling words % | plain words % |
|---|---|---|
| canary-1b-v2 | 23.0 | 18.2 |
| parakeet-tdt-0.6b-v3 | 24.9 | 22.0 |
| wav2vec2 Maltese-64h | 19.6 | 20.3 |
| whisper-small, zero-shot | 99.9 | 85.3 |
| ours | 33.8 | 28.6 |

* For our model and both NVIDIA models, Maltese-spelling words are 3–5 points harder.
* The wav2vec2 model, the only one trained only on Maltese, shows no gap: Maltese-specific training data removes the spelling penalty.
* Frequent substitutions are near misses: lost hyphens (*l-ebda → lebda*), the silent *għ* (*jistgħu → jistaw*),
  *ħ → h*, dropped apostrophes (*ma' → ma*). Full lists in `results/test_errors.md`.

## Limitations

* FLEURS has no speaker IDs, so speaker-disjoint test cannot be verified.
* One seed, one run per model (no confidence intervals).
* Read speech only; spontaneous Maltese would be harder.

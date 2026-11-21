# Out of Vocabulary · Maltese ASR (COE025 Project 3)

Automatic speech recognition for **Maltese (mt)**, a low-resource Semitic language written in Latin script
(ċ, ġ, ħ, ż, word-final apostrophes such as *ta'*, clitic hyphens such as *il-*).

Group **Out of Vocabulary**: İlyas Güney, Eren Taşkın, Nuri Tayfun Akbeniz.

## Data

FLEURS `mt_mt` (google/fleurs, CC-BY-4.0), official splits. Stats from `python src/split_stats.py`:

| split | utts | sentences | hours | male | female |
|---|---|---|---|---|---|
| train | 2895 | 1462 | 9.91 | 1646 | 1249 |
| dev | 404 | 150 | 1.49 | 140 | 264 |
| test | 926 | 348 | 3.55 | 352 | 574 |

Sentence overlap between all split pairs = **0**. FLEURS does not release speaker IDs, so speaker
disjointness cannot be checked; this is a stated limitation. Dev is used for checkpoint selection, test is decoded once per model.

## Method

- One normalization for every system (`src/common.py`): NFC, lowercase, punctuation removed, Maltese
  letters, in-word hyphens and word-final apostrophes kept. WER and CER with `jiwer`.
- Each run stores the model id, Hub revision (commit sha), real-time factor and GPU (`results/preds/*.json`).

| baseline | type |
|---|---|
| carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h | CTC, Maltese-specific |
| openai/whisper-small | multilingual seq2seq, zero-shot |
| nvidia/canary-1b-v2 | multilingual AED (25 EU languages) |
| nvidia/parakeet-tdt-0.6b-v3 | multilingual TDT (25 EU languages) |

Qwen3-ASR was considered but does not list Maltese among its supported languages
([model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)), so it is not evaluated.

**Our contribution:** fine-tune whisper-small on FLEURS mt train (`src/finetune_whisper.py`, seed 42, 1000 steps,
best checkpoint by dev WER) and compare against the zero-shot model and the baselines.

## Run

Everything runs on a free Colab (or Kaggle) T4: open `notebooks/run_all.ipynb` and run all cells. Locally:

```
pip install -r requirements.txt
python src/transcribe.py openai/whisper-small --split test
python src/score.py --split test
```

## Results (FLEURS mt_mt test, 926 utterances, Tesla T4)

| system | WER % | CER % | RTF |
|---|---|---|---|
| nvidia/canary-1b-v2 | **22.15** | 7.10 | 0.041 |
| nvidia/parakeet-tdt-0.6b-v3 | 24.77 | **6.45** | 0.010 |
| wav2vec2-large-xlsr-53-maltese-64h | 25.41 | 7.38 | 0.007 |
| openai/whisper-small (zero-shot) | 111.93 | 54.01 | 0.193 |
| **ours: whisper-small fine-tuned on FLEURS mt train** | 34.24 | 9.35 | 0.012 |

Fine-tuning (1000 steps, best checkpoint = step 750 by dev WER 34.60; dev WER at steps 250/500/750/1000:
44.6 / 36.1 / 34.6 / 34.7) cuts Whisper-small's WER by 78 points, but the three models that saw far more Maltese
or EU speech stay ahead. Full tables: `results/test_scores.md`, error analysis: `results/test_errors.md`,
training log: `results/finetune_trainer_state.json`.

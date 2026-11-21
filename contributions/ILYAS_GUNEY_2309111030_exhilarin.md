# İlyas Güney (2309111030, @exhilarin)

**Role:** code developer.

## What I did

* **Pipeline.** Shared normalization and data loading (`src/common.py`), split statistics (`src/split_stats.py`),
  one transcription script for Hugging Face and NeMo models with revision/RTF logging (`src/transcribe.py`),
  scoring (`src/score.py`) and error analysis (`src/error_analysis.py`).
* **Our model.** Whisper-small fine-tuning on FLEURS mt train with dev-based checkpoint selection
  (`src/finetune_whisper.py`), and the Colab notebook that runs everything on a T4.
* **Experiments.** Ran all baselines and the fine-tuning on Colab, fixed the >30 s clip issue for Whisper
  and the GPU memory limit, and collected the results in `results/`.
* **Slides.** Built the deck from the result files.

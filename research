# Research notes: why Maltese, which data, which models

## 1. The problem

**Task:** automatic speech recognition (ASR) for **Maltese**: 16 kHz read speech in, Maltese text out.
We measure word error rate (WER) and character error rate (CER) after one shared text normalization.

## 2. Why Maltese

* **Low-resource.** About half a million speakers, an official EU language, but open speech data
  adds up to only tens of hours (FLEURS ~15 h; Common Voice and the MASRI corpora are small too).
  The course gives a bonus for low-resource languages.
* **Mixed lexicon.** Maltese grammar and core vocabulary come from Siculo-Arabic, while a large part
  of the vocabulary comes from Italian/Sicilian and English. One sentence can contain words of
  all three origins, so a model cannot lean on a single "parent" language.
* **Own spelling.** It is the only Semitic language written in the Latin alphabet, with extra
  letters **ċ ġ ħ ż** and the digraphs **għ** and **ie**. The article is joined with a hyphen
  (*il-logħob*, *ir-rugby*) and many words end in an apostrophe (*ta'*, *jista'*). A model that
  writes *hafna* for *ħafna* or *ta* for *ta'* is wrong in WER even if it "heard" the word.
* **Silent letters.** *għ* and *h* are mostly silent, so the same sound can have several spellings;
  this is a known difficulty for Maltese spelling and for ASR.

## 3. Data decision

| source | status | why |
|---|---|---|
| FLEURS `mt_mt` (Conneau et al., 2022) | **used** | open (CC-BY-4.0), official train/dev/test, no sentence overlap |
| MASRI corpora (Hernández Mena et al., 2020) | dropped | licence form would not be answered before the deadline |
| Common Voice `mt` (Ardila et al., 2020) | dropped | now distributed through a separate data portal with its own account |

FLEURS releases no speaker IDs, so a speaker-disjoint test cannot be verified; we report this as a limitation.

## 4. Model choices

* **wav2vec2 XLSR-53, Maltese 64 h** (Baevski et al., 2020; Conneau et al., 2021): a self-supervised
  multilingual encoder fine-tuned with CTC on about 64 hours of Maltese (MASRI + Common Voice).
  The strongest "language-specific" baseline.
* **Whisper-small** (Radford et al., 2023): multilingual encoder-decoder trained on weakly labelled
  web audio; Maltese is in its language list but with very little training audio. Used zero-shot
  and as the base of our own fine-tuned model.
* **NVIDIA Parakeet-TDT 0.6B v3 and Canary-1B v2** (2025): multilingual models for 25 European
  languages including Maltese, trained on NVIDIA's Granary data. Modern, large baselines.
* **Qwen3-ASR**: considered, but Maltese is not in its supported-language list, so it is not evaluated.
* **Our method:** fine-tune Whisper-small on the 10 h FLEURS training set (seed 42, 2,000 steps,
  best checkpoint by dev WER). Question: how far do ten hours of in-language data move a general model?

## 5. References

* Ardila, R. et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus. LREC 2020.
* Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS 2020.
* Conneau, A. et al. (2021). Unsupervised Cross-lingual Representation Learning for Speech Recognition (XLSR). Interspeech 2021.
* Conneau, A. et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. IEEE SLT 2022.
* Hernández Mena, C. D. et al. (2020). MASRI-HEADSET: A Maltese Corpus for Speech Recognition. LREC 2020.
* Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). ICML 2023.
* Model cards: carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h, nvidia/parakeet-tdt-0.6b-v3,
  nvidia/canary-1b-v2, openai/whisper-small, Qwen/Qwen3-ASR-1.7B (Hugging Face).

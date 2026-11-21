"""Shared data loading + text normalization. Every model is scored through these two functions."""
import re
import unicodedata

FLEURS = "hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet/mt_mt/{}/*.parquet"  # parquet branch; no loading script
SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def normalize(text: str) -> str:
    # lowercase + NFC; keep Maltese letters (ċ ġ ħ ż), digits, and hyphens/apostrophes inside words (il-, ta')
    t = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = re.sub(r"(?<!\w)'", " ", t)  # Maltese keeps word-final apostrophes (ta', jista')
    t = re.sub(r"(?<!\w)-|-(?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_split(split: str, limit: int | None = None):
    from datasets import Audio, load_dataset
    ds = load_dataset("parquet", data_files={split: FLEURS.format(SPLITS[split])}, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    # decode=False + soundfile: datasets>=4 would otherwise need torchcodec, which Colab/Kaggle may lack
    return ds.cast_column("audio", Audio(decode=False))


def wav(x):
    """16 kHz float32 waveform of one FLEURS row (FLEURS audio is already 16 kHz mono wav)."""
    import io
    import soundfile as sf
    a, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32")
    assert sr == 16000, sr
    return a


if __name__ == "__main__":
    assert normalize("Il-Logħob Olimpiku, ta’ Malta!") == "il-logħob olimpiku ta' malta"
    assert normalize(" - Ċirku   “ŻEJT” ") == "ċirku żejt"
    print("ok")

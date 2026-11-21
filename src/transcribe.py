"""Transcribe a FLEURS mt split with one model; writes results/preds/<name>_<split>.csv and a meta json (revision, RTF, GPU).
usage: python src/transcribe.py MODEL [--split test] [--limit N]
MODEL: HF id (wav2vec2 / whisper), a local fine-tuned checkpoint dir, or nvidia/canary-* / nvidia/parakeet-* (NeMo)."""
import argparse, csv, json, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(__file__))
from common import load_split, wav

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--split", default="test", choices=["dev", "test"])
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

import torch
ds = load_split(a.split, a.limit)
audios = [wav(x) for x in ds]
audio_sec = sum(len(x) for x in audios) / 16000

revision = "local"
if not os.path.isdir(a.model):
    from huggingface_hub import HfApi
    revision = HfApi().model_info(a.model).sha

t0 = time.time()
if a.model.startswith("nvidia/"):
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
    m = ASRModel.from_pretrained(a.model).eval()
    tmp = tempfile.mkdtemp()
    paths = []
    for i, x in enumerate(audios):
        paths.append(f"{tmp}/{i}.wav"); sf.write(paths[-1], x, 16000)
    t0 = time.time()  # time decoding only, not model download / wav writing
    kw = {"source_lang": "mt", "target_lang": "mt"} if "canary" in a.model else {}
    out = m.transcribe(paths, batch_size=a.batch, **kw)
    hyps = [o.text if hasattr(o, "text") else o for o in out]
else:
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model=a.model, device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # FLEURS has clips > 30 s; Whisper sees 30 s windows, so long clips are chunked
    kw = {"generate_kwargs": {"language": "maltese", "task": "transcribe"}, "chunk_length_s": 30} \
        if "whisper" in pipe.model.config.model_type else {}
    t0 = time.time()
    hyps = [o["text"] for o in pipe([{"raw": x, "sampling_rate": 16000} for x in audios], batch_size=a.batch, **kw)]
elapsed = time.time() - t0

name = os.path.basename(a.model.rstrip("/"))
os.makedirs("results/preds", exist_ok=True)
with open(f"results/preds/{name}_{a.split}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "ref", "hyp"])
    for x, h in zip(ds, hyps):
        w.writerow([x["id"], x["raw_transcription"], h])
meta = {"model": a.model, "revision": revision, "split": a.split, "utts": len(hyps), "audio_sec": round(audio_sec, 1),
        "decode_sec": round(elapsed, 1), "rtf": round(elapsed / audio_sec, 4),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
json.dump(meta, open(f"results/preds/{name}_{a.split}.json", "w"), indent=1)
print(meta)

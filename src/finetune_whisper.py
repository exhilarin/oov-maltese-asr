"""Fine-tune openai/whisper-small on FLEURS mt_mt train; pick the best checkpoint by dev WER (test is never touched here).
Kaggle T4 16 GB, fp16. usage: python src/finetune_whisper.py [--steps 2000] [--out whisper-small-mt]"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import load_split, normalize, wav
import jiwer, torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="openai/whisper-small")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default="whisper-small-mt")
a = ap.parse_args()
set_seed(42)

proc = WhisperProcessor.from_pretrained(a.base, language="maltese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(a.base)
model.generation_config.language, model.generation_config.task = "maltese", "transcribe"
model.generation_config.forced_decoder_ids = None


def prep(x):
    return {"input_features": proc.feature_extractor(wav(x), sampling_rate=16000).input_features[0],
            "labels": proc.tokenizer(x["raw_transcription"]).input_ids}  # train on cased+punctuated text, score normalized


train, dev = (d.map(prep, remove_columns=d.column_names) for d in (load_split("train"), load_split("dev")))


def collate(b):
    feats = proc.feature_extractor.pad([{"input_features": x["input_features"]} for x in b], return_tensors="pt")
    lab = proc.tokenizer.pad([{"input_ids": x["labels"]} for x in b], return_tensors="pt")
    labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
    if (labels[:, 0] == model.config.decoder_start_token_id).all():
        labels = labels[:, 1:]  # model prepends start token itself
    feats["labels"] = labels
    return feats


def metrics(p):
    ids = p.label_ids.copy(); ids[ids == -100] = proc.tokenizer.pad_token_id
    hyp = [normalize(t) for t in proc.batch_decode(p.predictions, skip_special_tokens=True)]
    ref = [normalize(t) for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return {"wer": 100 * jiwer.wer(ref, hyp), "cer": 100 * jiwer.cer(ref, hyp)}


args = Seq2SeqTrainingArguments(
    output_dir=a.out + "-ckpt", per_device_train_batch_size=16, per_device_eval_batch_size=16,
    learning_rate=1e-5, warmup_steps=200, max_steps=a.steps, fp16=torch.cuda.is_available(),
    gradient_checkpointing=True, eval_strategy="steps", eval_steps=500, save_steps=500, save_total_limit=2,
    predict_with_generate=True, generation_max_length=225, logging_steps=50, report_to="none",
    load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False, seed=42)
tr = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev, data_collator=collate,
                    compute_metrics=metrics, processing_class=proc.feature_extractor)
tr.train()
tr.save_model(a.out); proc.save_pretrained(a.out)
print("best dev:", tr.state.best_metric, tr.state.best_model_checkpoint)

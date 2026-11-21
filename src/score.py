"""Score every results/preds/*_<split>.csv with the same normalization; WER + CER (jiwer).
Baselines and our own models go to separate tables. usage: python src/score.py [--split test]"""
import argparse, csv, glob, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import normalize
import jiwer

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); a = ap.parse_args()
OURS = ("whisper-small-mt", "gmm-hmm")  # our own systems; everything else is a baseline
tables = {"Baselines": [], "Ours": []}
for p in sorted(glob.glob(f"results/preds/*_{a.split}.csv")):
    rows = list(csv.DictReader(open(p)))
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]
    meta = json.load(open(p[:-4] + ".json")) if os.path.exists(p[:-4] + ".json") else {}
    name = os.path.basename(p)[: -len(f"_{a.split}.csv")]
    line = f"| {meta.get('model', name)} | {meta.get('revision', '-')[:10]} | {100*jiwer.wer(refs, hyps):.2f} | " \
           f"{100*jiwer.cer(refs, hyps):.2f} | {meta.get('rtf', '-')} | {meta.get('hardware', '-')} | {len(rows)} |"
    tables["Ours" if name.startswith(OURS) else "Baselines"].append(line)

head = "| model | revision | WER % | CER % | RTF | hardware | utts |\n|---|---|---|---|---|---|---|"
md = "\n\n".join(f"### {k} (FLEURS mt_mt {a.split})\n\n{head}\n" + "\n".join(v) for k, v in tables.items() if v)
print(md)
open(f"results/{a.split}_scores.md", "w").write(md + "\n")

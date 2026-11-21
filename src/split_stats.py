"""Per-split utterances / sentences / hours / gender and cross-split sentence overlap (FLEURS mt_mt).
FLEURS releases no speaker IDs, so speaker-disjointness cannot be verified; we use the official
splits (sentence-disjoint by construction) and report the overlap count to prove it is 0."""
import csv
import sys

rows, sents = [], {}
for s in ("train", "dev", "test"):
    r = list(csv.reader(open(f"data/fleurs/{s}.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
    sents[s] = {x[0] for x in r}
    hours = sum(int(x[5]) for x in r) / 16000 / 3600
    rows.append((s, len(r), len(sents[s]), f"{hours:.2f}", sum(x[6] == "MALE" for x in r), sum(x[6] == "FEMALE" for x in r)))

out = ["| split | utts | sentences | hours | male | female |", "|---|---|---|---|---|---|"]
out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
out += ["", "Sentence overlap: train∩dev = {}, train∩test = {}, dev∩test = {}".format(
    len(sents["train"] & sents["dev"]), len(sents["train"] & sents["test"]), len(sents["dev"] & sents["test"]))]
print("\n".join(out))
assert not (sents["train"] & sents["test"]), "test sentences leak into train"
open("results/split_stats.md", "w").write("\n".join(out) + "\n")

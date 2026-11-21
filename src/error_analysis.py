"""Where do models fail on Maltese? For every prediction file: WER on reference words that carry
Maltese-specific spelling (ċ ġ ħ ż, word-final apostrophe, clitic hyphen like il-) vs plain words,
plus the most frequent word substitutions. Writes results/<split>_errors.json and .md."""
import argparse, csv, glob, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import normalize
import jiwer

SPECIAL = re.compile(r"[ċġħż]|'$|\w-\w")


def analyse(rows):
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]
    out = jiwer.process_words(refs, hyps)
    err, tot, subs = Counter(), Counter(), Counter()
    for ref, hyp, align in zip(out.references, out.hypotheses, out.alignments):
        for a in align:
            for k in range(a.ref_end_idx - a.ref_start_idx):
                w = ref[a.ref_start_idx + k]
                g = "special" if SPECIAL.search(w) else "plain"
                tot[g] += 1
                if a.type != "equal":
                    err[g] += 1
                if a.type == "substitute":
                    subs[(w, hyp[a.hyp_start_idx + k])] += 1
    return {"word_error_special": round(100 * err["special"] / max(tot["special"], 1), 2),
            "word_error_plain": round(100 * err["plain"] / max(tot["plain"], 1), 2),
            "n_special": tot["special"], "n_plain": tot["plain"],
            "top_subs": [[r, h, n] for (r, h), n in subs.most_common(8)]}


if __name__ == "__main__":
    assert SPECIAL.search("ħafna") and SPECIAL.search("ta'") and SPECIAL.search("il-logħob") and not SPECIAL.search("kemm")
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); a = ap.parse_args()
    res = {}
    for p in sorted(glob.glob(f"results/preds/*_{a.split}.csv")):
        res[os.path.basename(p)[: -len(f"_{a.split}.csv")]] = analyse(list(csv.DictReader(open(p))))
    json.dump(res, open(f"results/{a.split}_errors.json", "w"), ensure_ascii=False, indent=1)
    md = ["| model | error on Maltese-spelling words % | error on plain words % |", "|---|---|---|"]
    md += [f"| {m} | {r['word_error_special']} | {r['word_error_plain']} |" for m, r in res.items()]
    for m, r in res.items():
        md += ["", f"**{m}** top substitutions (ref → hyp): " + ", ".join(f"{x} → {y} ({n})" for x, y, n in r["top_subs"])]
    open(f"results/{a.split}_errors.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

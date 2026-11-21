#!/usr/bin/env python3
"""Evaluate a GRECO re-ranking output of the 12 CoNLL-2014 shared-task systems (authors' public example) against the paper (Table 2,
multi-system: Spearman rho 0.415; re-rank P/R/F0.5 67.39/30.71/54.40). F0.5 is approximated from the official per-sentence M2Scorer
counts in data/conll-official/reports (# correct/# proposed/# gold of the chosen annotator of the system whose hypothesis was picked);
the M2Scorer itself (python2) is not installed. Public data only.
  python -I g_greco_conll_eval.py --greco_dir <clone> --rerank <out file>"""
import argparse, os, re, json
ap = argparse.ArgumentParser(); ap.add_argument("--greco_dir", required=True); ap.add_argument("--rerank", required=True); a = ap.parse_args()
G = a.greco_dir; T = f"{G}/data/conll-official/texts"; R = f"{G}/data/conll-official/reports"
names = sorted(os.listdir(T)); hyp = {s: [x.strip() for x in open(f"{T}/{s}").read().split("\n")] for s in names}


def counts(path):
    txt = open(path).read(); out = {}
    for blk in re.split(r"\n(?=>> Annotator:)", txt):
        pass
    # per sentence: blocks of annotators, then '>> Chosen Annotator for line i : k'
    cur = {}
    for line in txt.split("\n"):
        m = re.match(r">> Annotator: (\d+)", line)
        if m: ann = int(m.group(1)); cur[ann] = {}
        for k in ["correct", "proposed", "gold"]:
            mm = re.match(rf"# {k}\s*: (\d+)", line)
            if mm: cur[ann][k] = int(mm.group(1))
        m = re.match(r">> Chosen Annotator for line (\d+) : (\d+)", line)
        if m: out[int(m.group(1)) - 1] = cur[int(m.group(2))]; cur = {}
    return out


C = {s: counts(f"{R}/{s}.m2-report") for s in names}
out = [x.strip() for x in open(a.rerank).read().split("\n")]
n = len(hyp["INPUT"]) if hyp["INPUT"][-1] else len(hyp["INPUT"]) - 1
n = 1312; tp = fp_ = gold = 0; unmatched = 0
for i in range(n):
    cand = [s for s in names if hyp[s][i] == out[i]]
    if not cand: unmatched += 1; continue
    c = C[cand[0]][i]; tp += c["correct"]; fp_ += c["proposed"] - c["correct"]; gold += c["gold"]
P = tp / (tp + fp_) if tp + fp_ else 1.0; Rr = tp / gold; F = 1.25 * P * Rr / (0.25 * P + Rr)
chk = {s: None for s in names}
for s in names:   # sanity: summing a system's own counts reproduces its official F0.5
    t = sum(C[s][i]["correct"] for i in range(n)); p = sum(C[s][i]["proposed"] for i in range(n)); g = sum(C[s][i]["gold"] for i in range(n))
    pp = t / p if p else 1.0; rr = t / g; chk[s] = round(1.25 * pp * rr / (0.25 * pp + rr), 4)
print(json.dumps(dict(n=n, unmatched=unmatched, P=round(100 * P, 2), R=round(100 * Rr, 2), F05=round(100 * F, 2),
                      paper_table2_multi=dict(P=67.39, R=30.71, F05=54.40), per_system_F05_from_counts=chk)))

#!/usr/bin/env python3
"""C5 side analysis: oracle upper bounds for B1 (SGECF overall P / R / F0.5) if false feedback were withheld perfectly.
Recall is unchanged (only FP removed): R = TP / n_ref. Usage: python3 -I c5_oracle.py [labels_subdir, default labels; labels_v4 = markup-free reference V] > 06_results/oracle_upper_bounds[_v4].json"""
import csv, json, os, sys
LAB = sys.argv[1] if len(sys.argv) > 1 else "labels"
E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
f05 = lambda p, r: 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
out = {}
rows = {ds: list(csv.DictReader(open(f"{E}/05_runs/2026-11-10_B1_{ds}/{LAB}/hyp_edit_labels_{ds}.csv"))) for ds in ["dev", "train"]}
nref = {ds: sum(1 for _ in csv.DictReader(open(f"{E}/05_runs/2026-11-10_B1_{ds}/{LAB}/ref_edit_labels_{ds}.csv"))) for ds in ["dev", "train"]}
for name, dss in [("dev", ["dev"]), ("train", ["train"]), ("train+dev", ["train", "dev"])]:
    h = sum((rows[d] for d in dss), []); n = sum(nref[d] for d in dss)
    tp = sum(1 for x in h if x["false_feedback"] == "0"); fp = len(h) - tp
    R = tp / n
    res = {}
    for sc, removed in [("as_is", 0),
                        ("remove_asr_induced_primary", sum(int(x["asr_induced"]) for x in h)),
                        ("remove_asr_induced_literal", sum(int(x["asr_induced_lit"]) for x in h)),
                        ("remove_asr_and_dd_induced_primary", sum(int(x["asr_induced"]) + int(x["dd_induced"]) for x in h)),
                        ("remove_all_fp", fp)]:
        P = tp / (tp + fp - removed)
        res[sc] = dict(fp_removed=removed, edits_shown=tp + fp - removed, P=round(P, 4), R=round(R, 4), F05=round(f05(P, R), 4))
    out[name] = dict(n_hyp=len(h), tp=tp, fp=fp, n_ref=n, scenarios=res)
print(json.dumps(out, indent=1))

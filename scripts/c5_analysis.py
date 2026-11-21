#!/usr/bin/env python3
"""C5 analyses on the pooled OOF scores (05_runs/2026-11-10_C2_cv/oof_scores.csv), eswa-c env. Numbers only.
PLANNED (03_configs/2026-11-10_C_plan.yaml): per test part; label-definition sensitivity of the cause composition of the
withheld edits (primary vs literal, labels v4).  UNPLANNED (marked): AUROC per FP cause (asr/dd/gec-induced FP vs TP).
Usage: python -I c5_analysis.py > 05_runs/2026-11-10_C5_ablation/c5_analysis.json"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
o = pd.read_csv(f"{E}/05_runs/2026-11-10_C2_cv/oof_scores.csv")
L = pd.concat([pd.read_csv(f"{E}/05_runs/2026-11-10_B1_{ds}/labels_v4/hyp_edit_labels_{ds}.csv").assign(ds=ds) for ds in ["train", "dev"]])
L = L[["ds", "utt_id", "edit_idx", "asr_induced", "dd_induced", "gec_induced", "asr_induced_lit", "dd_induced_lit", "gec_induced_lit"]]
o = o.drop(columns=["asr_induced", "dd_induced", "gec_induced", "asr_induced_lit"]).merge(L, on=["ds", "utt_id", "edit_idx"], how="left", validate="1:1")
assert o["asr_induced"].notna().all()
U = pd.concat([pd.read_csv(f"{E}/05_runs/2026-11-10_C1_features/utt_counts_{ds}.csv") for ds in ["train", "dev"]])
U["part"] = U["utt_id"].str.extract(r"-P(\d)")[0].radd("P")
M = ["asr_conf", "asr_lr", "gec_lr", "type_prior", "risk_lr", "risk_lgbm", "random"]
SEED = 20261110
def kept_mask(s, cov):
    tie = np.random.RandomState(SEED).permutation(len(s)); order = np.lexsort((tie, s))
    m = np.zeros(len(s), bool); m[order[:int(np.ceil(cov * len(s)))]] = True; return m
out = {"per_part": {}, "withheld_composition": {}, "UNPLANNED_auroc_per_cause": {}}
for part, g in o.groupby("part"):
    nref = int(U.loc[U["part"] == part, "n_ref_edits"].sum()); y = g["false_feedback"].values
    out["per_part"][part] = {"n_edits": len(g), "fp_rate": round(float(y.mean()), 4), "n_ref": nref}
    for m in M:
        s = g[f"s_{m}"].values; k = kept_mask(s, 0.5)
        P = 1 - y[k].mean()
        prec = [(1 - y[kept_mask(s, c / 100)].mean()) for c in range(1, 101)]
        out["per_part"][part][m] = {"AUROC": round(float(roc_auc_score(y, s)), 4), "P@0.5": round(float(P), 4), "AURC": round(float(np.mean([1 - p for p in prec])), 4)}
for m in M:
    s = o[f"s_{m}"].values; res = {}
    for cov in [0.5, 0.7]:
        w = ~kept_mask(s, cov); r = {"n_withheld": int(w.sum()), "fp_share_of_withheld": round(float(o.loc[w, "false_feedback"].mean()), 4)}
        for defn, suf in [("primary", ""), ("literal", "_lit")]:
            for cause in ["asr", "dd", "gec"]:
                col = f"{cause}_induced{suf}"; tot = int(o[col].sum())
                r[f"{defn}_{cause}_fp_withheld_share"] = round(float(o.loc[w, col].sum() / tot), 4) if tot else None
        r["tp_withheld_share"] = round(float(((o["false_feedback"] == 0) & w).sum() / (o["false_feedback"] == 0).sum()), 4)
        res[f"cov{cov}"] = r
    out["withheld_composition"][m] = res
    tp = o["false_feedback"] == 0
    out["UNPLANNED_auroc_per_cause"][m] = {c: round(float(roc_auc_score(o.loc[tp | (o[f"{c}_induced"] == 1), "false_feedback"], o.loc[tp | (o[f"{c}_induced"] == 1), f"s_{m}"])), 4) for c in ["asr", "dd", "gec"]}
print(json.dumps(out, indent=1))

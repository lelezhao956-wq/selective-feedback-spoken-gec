#!/usr/bin/env python3
"""ASR-confidence diagnostics for manuscript section 6.4 (B1, train+dev = the C2 cross-validation set; no model is (re)trained).
Recomputes from existing files the numbers first noted in the lab notebook at C4 (2026-11-10 09:43; not archived then):
  score  s = 1 - asr_conf_win_min  (minimum official-baseline Whisper word confidence in the edit window; = the planned asr_conf
         baseline of c_model.py; fixed score, no fitting, so it needs no CV folds; median fill as c_model.py (0 missing here))
  (1) AUROC of s for the window ASR-error label asr_err_window (labels v4, primary; literal and asr_induced as sensitivity)
  (2) FP rate (false_feedback) of edits with / without a window ASR error
  (3) FP rate by quartile of asr_conf_win_min: pandas.qcut(asr_conf_win_min, 4) on the train+dev edits, ascending confidence
      (Q1 = lowest confidence), right-closed intervals; ties at a cut point fall in the lower quartile.
Inputs: 05_runs/2026-11-10_C1_features/features_{train,dev}.csv, 05_runs/2026-11-10_B1_{train,dev}/labels_v4/hyp_edit_labels_*.csv.
Output: 06_results/asr_conf_diagnostics.json. Numbers only.  python -I asr_conf_diagnostics.py  (env eswa-c)"""
import hashlib, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)")); R = f"{E}/05_runs"; F = f"{R}/2026-11-10_C1_features"
d = pd.concat([pd.read_csv(f"{F}/features_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
L = pd.concat([pd.read_csv(f"{R}/2026-11-10_B1_{ds}/labels_v4/hyp_edit_labels_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
m = d[["utt_id", "edit_idx", "false_feedback", "asr_conf_win_min"]].merge(
    L[["utt_id", "edit_idx", "false_feedback", "asr_err_window", "asr_err_window_lit", "asr_induced", "gec_induced"]],
    on=["utt_id", "edit_idx"], how="left", validate="1:1", suffixes=("", "_lab"))
assert len(m) == len(d) and (m["false_feedback"] == m["false_feedback_lab"]).all()
n_missing = int(m["asr_conf_win_min"].isna().sum())
conf = m["asr_conf_win_min"].fillna(m["asr_conf_win_min"].median()).values; s = 1 - conf; y = m["false_feedback"].values
r = dict(status="recomputed 2026-11-11 from existing features/labels (no retraining); first noted in 09_lab_notebook.md (C4, 2026-11-10)",
         system="B1", data="train+dev (C2 cross-validation set)", n_edits=int(len(m)), n_conf_missing=n_missing,
         score="s = 1 - asr_conf_win_min (official-baseline Whisper word confidence, minimum over the edit window)")
auc = lambda lab: float(roc_auc_score(m[lab].values, s))
tp = y == 0
r["auroc_window_asr_error"] = dict(primary_asr_err_window=auc("asr_err_window"), literal_asr_err_window_lit=auc("asr_err_window_lit"),
                                   asr_induced_vs_all=auc("asr_induced"),
                                   gec_induced_fp_vs_tp=float(roc_auc_score(m["gec_induced"].values[tp | (m["gec_induced"].values == 1)],
                                                                            s[tp | (m["gec_induced"].values == 1)])),
                                   false_feedback_vs_tp=float(roc_auc_score(y, s)))
a = m["asr_err_window"].values
r["fp_rate_by_window_asr_error"] = {"with_asr_error": dict(n=int(a.sum()), fp_rate=float(y[a == 1].mean())),
                                    "without_asr_error": dict(n=int((a == 0).sum()), fp_rate=float(y[a == 0].mean())),
                                    "label": "asr_err_window (labels v4, primary)"}
q, bins = pd.qcut(m["asr_conf_win_min"], 4, labels=False, retbins=True)
r["fp_rate_by_confidence_quartile"] = dict(
    rule="pandas.qcut(asr_conf_win_min, 4) over the train+dev edits; Q1 = lowest confidence ... Q4 = highest; right-closed; ties at a cut point go to the lower quartile",
    cut_points=[float(b) for b in bins],
    quartiles={f"Q{k + 1}": dict(conf_range=[float(bins[k]), float(bins[k + 1])], n=int((q == k).sum()), fp_rate=float(y[q == k].mean()))
               for k in range(4)})
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
r["inputs_sha256"] = {os.path.relpath(p, E): sha(p) for p in [f"{F}/features_train.csv", f"{F}/features_dev.csv"] +
                      [f"{R}/2026-11-10_B1_{ds}/labels_v4/hyp_edit_labels_{ds}.csv" for ds in ["train", "dev"]]}
r["script_sha256"] = sha(os.path.abspath(__file__))
json.dump(r, open(f"{E}/06_results/asr_conf_diagnostics.json", "w"), indent=1)
print(json.dumps({k: r[k] for k in ["auroc_window_asr_error", "fp_rate_by_window_asr_error"]}))
print(json.dumps({k: (v["conf_range"], v["n"], round(v["fp_rate"], 3)) for k, v in r["fp_rate_by_confidence_quartile"]["quartiles"].items()}))

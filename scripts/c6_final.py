#!/usr/bin/env python3
"""C6: fit the final models on all B1 train+dev edits and freeze them (eswa-c env). No eval data is read.
Chosen risk model (plan: best OOF AUROC of lr / lgbm): LightGBM, all feature groups. Hyperparameters tuned on all train+dev
with the plan's inner scheme (GroupKFold 3, criterion pooled AUROC). Baselines for C7 refitted the same way: gec_lr, asr_lr;
type_prior table from all train+dev; asr_conf needs no fitting (fill value for missing = train+dev median).
Usage: python -I c6_final.py --out 05_runs/2026-11-10_C6_final"""
import argparse, hashlib, json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C
import lightgbm, sklearn

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d, nref = C.load()
allc = sum(C.GROUPS.values(), [])
res = {}
p_lgb, auc_lgb = C.tune("lgbm", allc, d)
_, m_lgb = C.fit_predict("lgbm", allc, p_lgb, d, d.head(1))
m_lgb.booster_.save_model(f"{a.out}/risk_lgbm.txt")
cats = {c: sorted(d[c].astype(str).unique()) for c in allc if c in C.CATS}
res["risk_lgbm"] = dict(params=p_lgb, inner_auc=round(auc_lgb, 4), categories=cats)
for name, cols in [("gec_lr", C.GROUPS["GEC"]), ("asr_lr", C.GROUPS["ASR"])]:
    p, auc = C.tune("lr", cols, d)
    _, m = C.fit_predict("lr", cols, p, d, d.head(1))
    pickle.dump(m, open(f"{a.out}/{name}.pkl", "wb"))
    res[name] = dict(params=p, inner_auc=round(auc, 4))
mu = d["false_feedback"].mean(); g = d.groupby("type")["false_feedback"].agg(["sum", "count"])
tp_table = ((g["sum"] + mu) / (g["count"] + 1)).to_dict()
json.dump(dict(table=tp_table, fallback=mu), open(f"{a.out}/type_prior.json", "w"), indent=1)
asr_fill = float(d["asr_conf_win_min"].median())
# in-sample sanity + operating thresholds from the OOF scores of the CV (for reporting only; C7 metrics are coverage-based)
oof = pd.read_csv(f"{C.EXP}/05_runs/2026-11-10_C2_cv/oof_scores.csv")
thr = {f"cov{c}": float(np.quantile(oof["s_risk_lgbm"], c)) for c in [0.3, 0.5, 0.7, 0.9]}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
S = os.path.dirname(os.path.abspath(__file__))
scripts = {f: sha(f"{S}/{f}") for f in ["c1_bart_score.py", "c1_features.py", "c_model.py", "c6_final.py", "b3_edit_labels.py",
                                         "stm_to_ctm.py", "b_norm_verbatim.sh", "b_pipeline2.sh"]}
info = dict(models=res, asr_conf_fill=asr_fill, oof_risk_thresholds_for_coverage=thr, n_edits=len(d), n_groups=int(d["group"].nunique()),
            feature_columns=allc, versions=dict(lightgbm=lightgbm.__version__, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
            artifact_sha256={f: sha(f"{a.out}/{f}") for f in ["risk_lgbm.txt", "gec_lr.pkl", "asr_lr.pkl", "type_prior.json"]},
            script_sha256=scripts)
json.dump(info, open(f"{a.out}/final_fit.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ["models", "asr_conf_fill", "oof_risk_thresholds_for_coverage", "n_edits", "versions"]}, default=str)[:1500])

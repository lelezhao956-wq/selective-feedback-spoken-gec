#!/usr/bin/env python3
"""C6 addendum (2026-11-10, before any eval result): uses ONLY train+dev.
1. final risk_lr (all feature groups, LR, tuned with the plan's inner scheme on all train+dev) for the pre-specified
   eval comparison 'risk_lr vs gec_lr' (same model class: gain from features, not from the model type);
2. operating thresholds on the pooled OOF scores (05_runs/2026-11-10_C2_cv/oof_scores.csv) for risk_lgbm (primary), risk_lr, gec_lr:
   (a) coverage 0.5: the 0.5 quantile of the OOF risk scores (show an edit iff risk <= threshold);
   (b) FFR <= 30%: the HIGHEST risk threshold t (i.e. the largest coverage) such that the false-feedback rate of the OOF edits
       with risk <= t is <= 0.30;
3. OOF calibration: Brier score and ECE (10 equal-width bins on [0, 1], weighted by bin size) of the probability-valued scores.
Usage (eswa-c env): python -I c6b_addendum.py --out 05_runs/2026-11-10_C6_final"""
import argparse, hashlib, json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C


def ece(y, p, bins=10):
    e = np.clip(np.floor(p * bins).astype(int), 0, bins - 1); out = 0.0
    for b in range(bins):
        m = e == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def thresholds(y, s):
    tie = np.random.RandomState(C.SEED).permutation(len(s)); order = np.lexsort((tie, s))
    ss, yy = s[order], y[order]
    ffr = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    t_a = float(np.quantile(s, 0.5))
    ok = np.where(ffr <= 0.30)[0]
    # highest threshold with cumulative FFR <= 0.30; ties at the threshold value are all shown at application time,
    # so take the largest index whose score is strictly below the next score
    i = int(ok.max()) if len(ok) else -1
    t_b = float(ss[i]) if i >= 0 else float("nan")
    shown = s <= t_b
    return dict(cov05=dict(threshold=t_a, oof_coverage=float((s <= t_a).mean()), oof_ffr=float(y[s <= t_a].mean())),
                ffr30=dict(threshold=t_b, oof_coverage=float(shown.mean()), oof_ffr=float(y[shown].mean()) if shown.any() else None))


ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
d, nref = C.load()
allc = sum(C.GROUPS.values(), [])
p, auc = C.tune("lr", allc, d)
_, m = C.fit_predict("lr", allc, p, d, d.head(1))
pickle.dump(m, open(f"{a.out}/risk_lr.pkl", "wb"))
oof = pd.read_csv(f"{C.EXP}/05_runs/2026-11-10_C2_cv/oof_scores.csv")
y = oof["false_feedback"].values
res = dict(risk_lr=dict(params=p, inner_auc=round(auc, 4),
                        artifact_sha256=hashlib.sha256(open(f"{a.out}/risk_lr.pkl", "rb").read()).hexdigest()),
           operating_points={}, oof_calibration={})
for n in ["risk_lgbm", "risk_lr", "gec_lr", "asr_lr", "type_prior"]:
    s = oof[f"s_{n}"].values
    if n in ["risk_lgbm", "risk_lr", "gec_lr"]:
        res["operating_points"][n] = thresholds(y, s)
    res["oof_calibration"][n] = dict(brier=float(np.mean((s - y) ** 2)), ece10=ece(y, s), mean_pred=float(s.mean()), fp_rate=float(y.mean()))
res["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
json.dump(res, open(f"{a.out}/addendum_fit.json", "w"), indent=1)
print(json.dumps(res, indent=1))

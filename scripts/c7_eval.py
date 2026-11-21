#!/usr/bin/env python3
"""C7: score the B1 EVAL edits with the frozen artifacts (03_configs/final.yaml + 2026-11-10_eval_plan_supplement.yaml) and
compute every pre-specified eval result ONCE. No fitting, no tuning, no threshold choice on eval. eswa-c env.
Usage: python -I c7_eval.py --features <dir with features_eval.csv, utt_counts_eval.csv> --labels <B1_eval labels_v4 dir> --out <json>
"""
import argparse, hashlib, json, os, pickle, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
FD = f"{C.EXP}/05_runs/2026-11-10_C6_final"
fit = json.load(open(f"{FD}/final_fit.json")); add = json.load(open(f"{FD}/addendum_fit.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
art_ok = {f: sha(f"{FD}/{f}") == h for f, h in fit["artifact_sha256"].items()}
art_ok["risk_lr.pkl"] = sha(f"{FD}/risk_lr.pkl") == add["risk_lr"]["artifact_sha256"]
assert all(art_ok.values()), art_ok

d = pd.read_csv(f"{a.features}/features_eval.csv")
u = pd.read_csv(f"{a.features}/utt_counts_eval.csv")
nref_g = u.groupby("group")["n_ref_edits"].sum(); N = int(nref_g.sum())
L = pd.read_csv(f"{a.labels}/hyp_edit_labels_eval.csv")[["utt_id", "edit_idx", "asr_induced", "dd_induced", "gec_induced",
                                                         "asr_induced_lit", "dd_induced_lit", "gec_induced_lit"]]
d = d.drop(columns=["asr_induced", "dd_induced", "gec_induced", "asr_induced_lit"]).merge(L, on=["utt_id", "edit_idx"], how="left", validate="1:1")
assert d["asr_induced"].notna().all()
y = d["false_feedback"].values
allc = fit["feature_columns"]

# ---- scores with frozen artifacts
S = {}
X = d[allc].copy()
for c, lev in fit["models"]["risk_lgbm"]["categories"].items():
    X[c] = X[c].astype(str).astype(pd.CategoricalDtype(lev))
S["risk_lgbm"] = lgb.Booster(model_file=f"{FD}/risk_lgbm.txt").predict(X)
for n, cols in [("risk_lr", allc), ("gec_lr", C.GROUPS["GEC"]), ("asr_lr", C.GROUPS["ASR"])]:
    S[n] = pickle.load(open(f"{FD}/{n}.pkl", "rb")).predict_proba(d[cols])[:, 1]
tp = json.load(open(f"{FD}/type_prior.json"))
S["type_prior"] = d["type"].map(tp["table"]).fillna(tp["fallback"]).values
S["asr_conf"] = 1 - d["asr_conf_win_min"].fillna(fit["asr_conf_fill"]).values
S["random"] = np.random.RandomState(C.SEED).rand(len(d))
names = list(S)

# ---- point metrics, bootstrap, paired tests
point = {n: C.metrics(y, S[n], N) for n in names}
groups = np.array(sorted(set(nref_g.index) | set(d["group"])))
gi = {g: d.index[d["group"] == g].values for g in groups}
OPS = {n: add["operating_points"][n] for n in ["risk_lgbm", "risk_lr", "gec_lr"]}


def op_metrics(yy, s, nr, t):
    m = s <= t; k = int(m.sum()); tpk = int(((1 - yy) * m).sum())
    P = tpk / k if k else float("nan"); R = tpk / nr if nr else float("nan")
    F = 1.25 * P * R / (0.25 * P + R) if k and (P + R) > 0 else 0.0
    return dict(coverage=k / len(yy), P=P, FFR=1 - P if k else float("nan"), F05=F, R=R, n_shown=k)


opp = {n: {op: op_metrics(y, S[n], N, v["threshold"]) for op, v in OPS[n].items()} for n in OPS}
keys = ["AURC", "P@0.3", "P@0.5", "P@0.7", "P@0.9", "FFR@0.3", "FFR@0.5", "FFR@0.7", "FFR@0.9", "F05@0.5", "maxF05", "AUROC", "AUPRC"]
B = {n: {k: [] for k in keys} for n in names}
BO = {n: {op: {k: [] for k in ["coverage", "P", "FFR", "F05"]} for op in OPS[n]} for n in OPS}
rng = np.random.RandomState(C.SEED)
for b in range(a.n_boot):
    gs = rng.choice(groups, size=len(groups), replace=True)
    idx = np.concatenate([gi[g] for g in gs]); nr = int(sum(nref_g.get(g, 0) for g in gs)); yb = y[idx]
    for n in names:
        m = C.metrics(yb, S[n][idx], nr)
        for k in keys:
            B[n][k].append(m[k])
    for n in OPS:
        for op, v in OPS[n].items():
            m = op_metrics(yb, S[n][idx], nr, v["threshold"])
            for k in BO[n][op]:
                BO[n][op][k].append(m[k])
pc = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
ci = {n: {k: pc(B[n][k]) for k in keys} for n in names}
op_ci = {n: {op: {k: pc(v) for k, v in BO[n][op].items()} for op in BO[n]} for n in BO}
PRESPEC = [("risk_lgbm", "asr_conf", "planned (C plan)"), ("risk_lgbm", "gec_lr", "pre-specified before eval (added after CV)"),
           ("risk_lr", "gec_lr", "pre-specified before eval (added after CV)")]
SECOND = [("risk_lgbm", "asr_lr", "secondary"), ("risk_lr", "asr_conf", "secondary")]
tests = {}
for cand, ref, status in PRESPEC + SECOND:
    t = dict(status=status)
    for k in keys:
        diff = np.array(B[cand][k]) - np.array(B[ref][k])
        t[k] = dict(diff=point[cand][k] - point[ref][k], ci=pc(diff), p=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
    tests[f"{cand}_vs_{ref}"] = t


# ---- calibration
def ece(yy, p, bins=10):
    e = np.clip(np.floor(p * bins).astype(int), 0, bins - 1)
    return float(sum((e == b).mean() * abs(yy[e == b].mean() - p[e == b].mean()) for b in range(bins) if (e == b).any()))


calib = {n: dict(brier=float(np.mean((S[n] - y) ** 2)), ece10=ece(y, S[n]), mean_pred=float(S[n].mean()), fp_rate=float(y.mean()))
         for n in ["risk_lgbm", "risk_lr", "gec_lr", "asr_lr", "type_prior"]}

# ---- per FP cause (primary v4; literal for sensitivity)
tpm = y == 0
cause = {}
for n in names:
    r = {"auroc_primary": {}, "auroc_literal": {}}
    for c in ["asr", "dd", "gec"]:
        for defn, suf in [("primary", ""), ("literal", "_lit")]:
            m = tpm | (d[f"{c}_induced{suf}"].values == 1)
            yy = y[m]
            r[f"auroc_{defn}"][c] = float(roc_auc_score(yy, S[n][m])) if 0 < yy.sum() < len(yy) else None
    thr = {"cov0.5": float(np.quantile(S[n], 0.5))}
    if n in OPS:
        thr.update({f"op_{k}": v["threshold"] for k, v in OPS[n].items()})
    for tn, t in thr.items():
        w = S[n] > t if tn.startswith("op_") else None
        if w is None:
            tie = np.random.RandomState(C.SEED).permutation(len(y)); order = np.lexsort((tie, S[n]))
            w = np.ones(len(y), bool); w[order[:int(np.ceil(0.5 * len(y)))]] = False
        rr = dict(n_withheld=int(w.sum()), fp_share_of_withheld=float(y[w].mean()) if w.any() else None,
                  tp_withheld_share=float((w & tpm).sum() / tpm.sum()))
        for c in ["asr", "dd", "gec"]:
            for defn, suf in [("primary", ""), ("literal", "_lit")]:
                col = d[f"{c}_induced{suf}"].values == 1
                rr[f"{defn}_{c}_fp_withheld_share"] = float((w & col).sum() / col.sum()) if col.sum() else None
        r[f"withheld_{tn}"] = rr
    cause[n] = r

fp_by_cause = {defn: {c: int(d[f"{c}_induced{suf}"].sum()) for c in ["asr", "dd", "gec"]} for defn, suf in [("primary", ""), ("literal", "_lit")]}
out = dict(n_edits=len(d), n_groups_with_edits=int(d["group"].nunique()), n_groups=len(groups), n_ref=N, fp_rate=float(y.mean()),
           fp_by_cause=fp_by_cause, artifacts_verified=art_ok, n_boot=a.n_boot, seed=C.SEED,
           point=point, ci95=ci, paired_tests=tests, operating_points=dict(thresholds=OPS, eval=opp, ci95=op_ci),
           calibration_eval=calib, calibration_oof=add["oof_calibration"], per_cause=cause,
           script_sha256=sha(os.path.abspath(__file__)))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(dict(n_edits=len(d), fp_rate=round(float(y.mean()), 4), n_ref=N, artifacts_verified=all(art_ok.values()))))

#!/usr/bin/env python3
"""GRECO comparison, modelling (eswa-c env; plan 03_configs/2026-11-10_GRECO_plan_supplement.yaml; added after the C7 eval).
  python -I g_greco_model.py --system B1|W --out 06_results/G_<system>_eval.json
Uses the system's existing train+dev / eval feature files (unchanged) joined 1:1 with the per-edit GRECO scores
(05_runs/2026-11-10_G_greco/greco_<system>_<ds>.csv). Models:
  greco_zs   : zero-shot ranking, risk = -greco_single (no fitting)
  greco_lr   : LogisticRegression on [greco_single, greco_loo, greco_sent] (c_model.lr_pipe, C grid, nested grouped CV as C2)
  risk_greco : LightGBM on the frozen feature set + the 3 GRECO scores (c_model LGB grid, nested grouped CV as C2)
  risk_lgbm  : the system's existing final model (frozen artifacts; reference, not refitted)
CV: the C2 outer folds (c_model.folds on the same train+dev rows). Final fit: inner-CV tuning on all train+dev (c_model.tune), then fit;
eval scored ONCE (output file must not exist). Endpoints as C7 (c_model.metrics), 1000 submission-grouped bootstrap, seed 20261110.
Pre-specified comparisons: risk_lgbm vs greco_lr, risk_greco vs greco_lr (primary GRECO-only = greco_lr); greco_zs reported alongside.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C   # noqa: E402

R = f"{C.EXP}/05_runs"
SYS = {"B1": dict(td=f"{R}/2026-11-10_C1_features", ev=f"{R}/2026-11-10_C7_B1_eval/features", art=f"{R}/2026-11-10_C6_final",
                  oof=f"{R}/2026-11-10_C2_cv/oof_scores.csv", evlab=f"{R}/2026-11-10_C7_B1_eval/labels_v4"),
       "W": dict(td=f"{R}/2026-11-10_D_W_model/features", ev=f"{R}/2026-11-10_D_W_eval/features",
                 art=f"{R}/2026-11-10_D_W_model/05_runs/2026-11-10_C6_final", oof=f"{R}/2026-11-10_D_W_model/05_runs/2026-11-10_C2_cv/oof_scores.csv",
                 evlab=f"{R}/2026-11-10_D_W_eval/labels_v4"),
       # Q: added after the eval (GRECO supplement amendment 2026-11-11); features copied 1:1 into D_Q_model/features (cmp-verified),
       # utt_counts read from the per-set feature dirs
       "Q": dict(td=f"{R}/2026-11-10_D_Q_model/features", ev=f"{R}/2026-11-10_D_Q_eval/features",
                 art=f"{R}/2026-11-10_D_Q_model/05_runs/2026-11-10_C6_final", oof=f"{R}/2026-11-10_D_Q_model/05_runs/2026-11-10_C2_cv/oof_scores.csv",
                 evlab=f"{R}/2026-11-10_D_Q_eval/labels_v4", uc={ds: f"{R}/2026-11-10_D_Q_{ds}/features" for ds in ["train", "dev"]})}
GF = ["greco_single", "greco_loo", "greco_sent"]
ap = argparse.ArgumentParser(); ap.add_argument("--system", required=True, choices=list(SYS)); ap.add_argument("--out", required=True)
ap.add_argument("--n_boot", type=int, default=1000)
ap.add_argument("--gd", default="", help="score dir (default 05_runs/2026-11-10_G_greco); only for the synthetic-score smoke test")
a = ap.parse_args(); cfg = SYS[a.system]
assert not os.path.exists(a.out), f"{a.out} exists (eval scored once only)"
GD = a.gd or f"{R}/2026-11-10_G_greco"


def join(df, ds):
    g = pd.read_csv(f"{GD}/greco_{a.system}_{ds}.csv")
    m = df.merge(g, on=["utt_id", "edit_idx"], how="left", validate="1:1")
    assert len(m) == len(df) and m[GF].notna().all().all(), f"GRECO scores missing for {ds}"
    return m


td = pd.concat([join(pd.read_csv(f"{cfg['td']}/features_{ds}.csv"), ds) for ds in ["train", "dev"]], ignore_index=True)
utd = pd.concat([pd.read_csv(f"{cfg.get('uc', {}).get(ds, cfg['td'])}/utt_counts_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
ev = join(pd.read_csv(f"{cfg['ev']}/features_eval.csv"), "eval"); uev = pd.read_csv(f"{cfg['ev']}/utt_counts_eval.csv")
fit = json.load(open(f"{cfg['art']}/final_fit.json")); ALLC = fit["feature_columns"]
oof_ref = pd.read_csv(cfg["oof"])
for c in ["utt_id", "edit_idx", "false_feedback"]:
    assert (oof_ref[c].values == td[c].values).all(), f"OOF rows not aligned ({c})"
y = td["false_feedback"].values; yE = ev["false_feedback"].values
N_TD = int(utd["n_ref_edits"].sum()); N_EV = int(uev["n_ref_edits"].sum())
fl = C.folds(td)
fold_of = np.zeros(len(td), int)
for k, (_, b) in enumerate(fl):
    fold_of[b] = k
assert (fold_of == oof_ref["fold"].values).all(), "outer folds differ from C2"
SPEC = {"greco_lr": ("lr", GF), "risk_greco": ("lgbm", ALLC + GF)}
OOF = {"greco_zs": -td["greco_single"].values, "risk_lgbm": oof_ref["s_risk_lgbm"].values}
chosen = {}
for n, (kind, cols) in SPEC.items():
    OOF[n], chosen[n] = C.cv_scores(td, kind, cols, fl); print(n, "OOF AUROC", round(C.roc_auc_score(y, OOF[n]), 4), flush=True)
oof_point = {n: C.metrics(y, s, N_TD) for n, s in OOF.items()}
# final fits on all train+dev, eval once
SE_ = {"greco_zs": -ev["greco_single"].values}
X = ev[ALLC].copy()
for c, lev in fit["models"]["risk_lgbm"]["categories"].items():
    X[c] = X[c].astype(str).astype(pd.CategoricalDtype(lev))
SE_["risk_lgbm"] = lgb.Booster(model_file=f"{cfg['art']}/risk_lgbm.txt").predict(X)
final = {}
for n, (kind, cols) in SPEC.items():
    p, inner = C.tune(kind, cols, td); SE_[n] = C.fit_predict(kind, cols, p, td, ev)[0]; final[n] = dict(params=p, inner_auc=inner)
point = {n: C.metrics(yE, s, N_EV) for n, s in SE_.items()}
groups = np.array(sorted(set(uev["group"]) | set(ev["group"]))); nref_g = uev.groupby("group")["n_ref_edits"].sum()
gi = {g: ev.index[ev["group"] == g].values for g in groups}
keys = ["AURC", "P@0.3", "P@0.5", "P@0.7", "P@0.9", "FFR@0.3", "FFR@0.5", "FFR@0.7", "FFR@0.9", "F05@0.5", "maxF05", "AUROC", "AUPRC"]
B = {n: {k: [] for k in keys} for n in SE_}; rng = np.random.RandomState(C.SEED)
for b in range(a.n_boot):
    gs = rng.choice(groups, size=len(groups), replace=True); idx = np.concatenate([gi[g] for g in gs])
    nr = int(sum(nref_g.get(g, 0) for g in gs))
    for n in SE_:
        m = C.metrics(yE[idx], SE_[n][idx], nr)
        for k in keys:
            B[n][k].append(m[k])
pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
tests = {}
for cand, ref, st in [("risk_lgbm", "greco_lr", "pre-specified (GRECO supplement)"), ("risk_greco", "greco_lr", "pre-specified (GRECO supplement)"),
                      ("risk_lgbm", "greco_zs", "secondary"), ("risk_greco", "risk_lgbm", "secondary")]:
    t = dict(status=st)
    for k in keys:
        d = np.array(B[cand][k]) - np.array(B[ref][k])
        t[k] = dict(diff=point[cand][k] - point[ref][k], ci=pc(d), p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    tests[f"{cand}_vs_{ref}"] = t
# ---- operating points (rule of c6b_addendum.thresholds on the OOF scores: cov05 = OOF median; ffr30 = highest OOF threshold with
#      cumulative FFR <= 0.30) applied to eval, with grouped-bootstrap CIs (same resamples as above, re-drawn with the same seed)
def thresholds(yy, s):
    tie = np.random.RandomState(C.SEED).permutation(len(s)); order = np.lexsort((tie, s)); ss, ys = s[order], yy[order]
    ffr = np.cumsum(ys) / np.arange(1, len(ys) + 1); ok = np.where(ffr <= 0.30)[0]; i = int(ok.max()) if len(ok) else -1
    t_b = float(ss[i]) if i >= 0 else float("nan"); t_a = float(np.quantile(s, 0.5)); shown = s <= t_b
    return dict(cov05=dict(threshold=t_a, oof_coverage=float((s <= t_a).mean()), oof_ffr=float(yy[s <= t_a].mean())),
                ffr30=dict(threshold=t_b, oof_coverage=float(shown.mean()), oof_ffr=float(yy[shown].mean()) if shown.any() else None))


def op_metrics(yy, s, nr, t):
    m = s <= t; k = int(m.sum()); tpk = int(((1 - yy) * m).sum())
    P = tpk / k if k else float("nan"); Rr = tpk / nr if nr else float("nan")
    F = 1.25 * P * Rr / (0.25 * P + Rr) if k and (P + Rr) > 0 else 0.0
    return dict(coverage=k / len(yy), P=P, FFR=1 - P if k else float("nan"), F05=F, R=Rr, n_shown=k)


OPS = {n: thresholds(y, OOF[n]) for n in SE_}
opp = {n: {op: op_metrics(yE, SE_[n], N_EV, v["threshold"]) for op, v in OPS[n].items()} for n in SE_}
BO = {n: {op: {k: [] for k in ["coverage", "P", "FFR", "F05"]} for op in OPS[n]} for n in SE_}
rng = np.random.RandomState(C.SEED)
for b in range(a.n_boot):
    gs = rng.choice(groups, size=len(groups), replace=True); idx = np.concatenate([gi[g] for g in gs])
    nr = int(sum(nref_g.get(g, 0) for g in gs))
    for n in SE_:
        for op, v in OPS[n].items():
            m = op_metrics(yE[idx], SE_[n][idx], nr, v["threshold"])
            for k in BO[n][op]:
                BO[n][op][k].append(m[k])
pcn = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))] if np.isfinite(x).any() else None
op_ci = {n: {op: {k: pcn(np.array(v, float)) for k, v in BO[n][op].items()} for op in BO[n]} for n in BO}
# ---- per FP cause (primary v4 labels of the eval set; literal for sensitivity), AUROC of FP-of-cause vs all TP
LE = pd.read_csv(f"{cfg['evlab']}/hyp_edit_labels_eval.csv")[["utt_id", "edit_idx", "false_feedback", "asr_induced", "dd_induced",
                                                              "gec_induced", "asr_induced_lit", "dd_induced_lit", "gec_induced_lit"]]
LE = ev[["utt_id", "edit_idx", "false_feedback"]].merge(LE, on=["utt_id", "edit_idx"], how="left", validate="1:1", suffixes=("", "_lab"))
assert LE["asr_induced"].notna().all() and (LE["false_feedback"].values == LE["false_feedback_lab"].values).all()
tpm = yE == 0
per_cause = {n: {f"auroc_{defn}": {c: (float(C.roc_auc_score(yE[tpm | (LE[f"{c}_induced{suf}"].values == 1)],
                                                              SE_[n][tpm | (LE[f"{c}_induced{suf}"].values == 1)]))
                                         if LE[f"{c}_induced{suf}"].sum() else None) for c in ["asr", "dd", "gec"]}
                 for defn, suf in [("primary", ""), ("literal", "_lit")]} for n in SE_}
fp_by_cause = {defn: {c: int(LE[f"{c}_induced{suf}"].sum()) for c in ["asr", "dd", "gec"]} for defn, suf in [("primary", ""), ("literal", "_lit")]}
diag_g = {ds: json.load(open(f"{GD}/greco_{a.system}_{ds}.diag.json")) for ds in ["train", "dev", "eval"]}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(status="added after the C7 eval (GRECO supplement); eval scored once", system=a.system, n_edits=len(ev), n_ref=N_EV,
               oof_point=oof_point, cv_chosen=chosen, final=final, point=point, ci95={n: {k: pc(v) for k, v in B[n].items()} for n in B},
               paired_tests=tests, operating_points=dict(thresholds=OPS, eval=opp, ci95=op_ci), per_cause=per_cause, fp_by_cause=fp_by_cause,
               greco_scoring={ds: {k: diag_g[ds].get(k) for k in ["n_edits", "n_utts", "seconds", "s_per_edit", "s_per_utt", "s_per_pair",
                                                                 "pairs_truncated", "nonfinite_rows", "device", "commit", "checkpoint_sha256"]}
                              for ds in diag_g},
               script_sha256=sha(os.path.abspath(__file__))), open(a.out, "w"), indent=1)
print(json.dumps({n: {k: round(point[n][k], 4) for k in ["AURC", "P@0.5", "AUROC"]} for n in point}))

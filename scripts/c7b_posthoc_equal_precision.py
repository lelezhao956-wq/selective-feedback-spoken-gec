#!/usr/bin/env python3
"""POST HOC (unplanned; requested 2026-11-10 after the eval checkpoint). Uses only the existing eval predictions:
the per-edit eval scores are the deterministic outputs of the frozen C6 artifacts on the existing eval features
(no fitting, no tuning, no pipeline rerun); they are regenerated exactly as in c7_eval.py and checked against
06_results/eval_results.json (all point metrics must match to 1e-12), then saved as derived data.
Question: at equal precision (0.65 and 0.67), how much coverage does each of risk_lgbm, risk_lr and gec_lr keep?
Coverage at precision p = largest k/n such that the precision of the k lowest-risk edits is >= p (same ranking and
tie-break as c_model.curve); 0 if no k reaches p. 1000 submission-grouped bootstrap resamples, seed 20261110, paired.
Usage (eswa-c env): python -I c7b_posthoc_equal_precision.py
"""
import hashlib, json, os, pickle, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C

R = f"{C.EXP}/05_runs/2026-11-10_C7_B1_eval"; FD = f"{C.EXP}/05_runs/2026-11-10_C6_final"
EV = json.load(open(f"{C.EXP}/06_results/eval_results.json"))
fit = json.load(open(f"{FD}/final_fit.json")); add = json.load(open(f"{FD}/addendum_fit.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert all(sha(f"{FD}/{f}") == h for f, h in fit["artifact_sha256"].items())
assert sha(f"{FD}/risk_lr.pkl") == add["risk_lr"]["artifact_sha256"]

d = pd.read_csv(f"{R}/features/features_eval.csv")
u = pd.read_csv(f"{R}/features/utt_counts_eval.csv")
nref_g = u.groupby("group")["n_ref_edits"].sum(); N = int(nref_g.sum())
y = d["false_feedback"].values; allc = fit["feature_columns"]
S = {}
X = d[allc].copy()
for c, lev in fit["models"]["risk_lgbm"]["categories"].items():
    X[c] = X[c].astype(str).astype(pd.CategoricalDtype(lev))
S["risk_lgbm"] = lgb.Booster(model_file=f"{FD}/risk_lgbm.txt").predict(X)
for n, cols in [("risk_lr", allc), ("gec_lr", C.GROUPS["GEC"])]:
    S[n] = pickle.load(open(f"{FD}/{n}.pkl", "rb")).predict_proba(d[cols])[:, 1]
names = list(S)
# check: identical to the single eval run
for n in names:
    m = C.metrics(y, S[n], N)
    bad = {k: (m[k], v) for k, v in EV["point"][n].items() if abs(m[k] - v) > 1e-12}
    assert not bad, (n, bad)
assert len(d) == EV["n_edits"] and N == EV["n_ref"]
sc = d[["utt_id", "edit_idx", "group", "false_feedback"]].copy()
for n in names:
    sc[f"s_{n}"] = S[n]
sc.to_csv(f"{R}/eval_scores_frozen.csv", index=False)

PS = [0.65, 0.67]


def cov_at_p(yy, s, p):
    _, P, _, _ = C.curve(yy, s, 1)
    ok = np.where(P >= p - 1e-12)[0]
    return (int(ok.max()) + 1) / len(yy) if len(ok) else 0.0


point = {n: {str(p): cov_at_p(y, S[n], p) for p in PS} for n in names}
groups = np.array(sorted(set(nref_g.index) | set(d["group"])))
gi = {g: d.index[d["group"] == g].values for g in groups}
B = {n: {str(p): [] for p in PS} for n in names}
rng = np.random.RandomState(C.SEED)
for b in range(1000):
    gs = rng.choice(groups, size=len(groups), replace=True)
    idx = np.concatenate([gi[g] for g in gs])
    for n in names:
        for p in PS:
            B[n][str(p)].append(cov_at_p(y[idx], S[n][idx], p))
pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
ci = {n: {p: pc(v) for p, v in B[n].items()} for n in names}
zero = {n: {p: float(np.mean(np.array(v) == 0)) for p, v in B[n].items()} for n in names}
tests = {}
for a_, b_ in [("risk_lgbm", "gec_lr"), ("risk_lr", "gec_lr"), ("risk_lgbm", "risk_lr")]:
    tests[f"{a_}_vs_{b_}"] = {}
    for p in map(str, PS):
        diff = np.array(B[a_][p]) - np.array(B[b_][p])
        tests[f"{a_}_vs_{b_}"][p] = dict(diff=point[a_][p] - point[b_][p], ci=pc(diff),
                                          p=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
out = dict(status="POST HOC (unplanned, requested after the eval checkpoint; not pre-specified)",
           definition="coverage at precision p = max k/n with precision of the k lowest-risk edits >= p (c_model.curve ranking/tie-break)",
           n_edits=len(d), n_groups=len(groups), n_boot=1000, seed=C.SEED, eval_scores_check="all point metrics identical to eval_results.json",
           coverage=point, ci95=ci, share_resamples_coverage_zero=zero, paired=tests,
           scores_file=os.path.relpath(f"{R}/eval_scores_frozen.csv", C.EXP), script_sha256=sha(os.path.abspath(__file__)))
json.dump(out, open(f"{C.EXP}/06_results/eval_posthoc_equal_precision.json", "w"), indent=1)
print(json.dumps(dict(coverage=point, ci95=ci, paired=tests), indent=0))

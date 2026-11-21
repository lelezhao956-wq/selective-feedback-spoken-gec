#!/usr/bin/env python3
"""I (POST HOC, planned=False, added 2026-11-19 during revision): Lu-inspired proxy baseline on B1.

Lu, Banno & Gales (2022, BEA, Sec. 4, Eq. 18, p. 55; weights in Fig. 4 caption, p. 58):
    log P = alpha log P_ASR + beta log P_DD + gamma log P_GEC,
each P the lowest token probability of the module (ASR always over the sentence), weights chosen by F0.5.
This is NOT a reproduction: the DD model's own posteriors were never saved (official DD writes labels only,
outputs/gec/dd/pre-gec.tsv), and dd_del_conf_min is the ASR confidence of deleted words, so the DD term is dropped
(beta = 0). Proxy:
    log P = alpha * log(max(utt_asr_conf_min, 1e-6)) + (1 - alpha) * gec_lp_c_min,
    gec_lp_c_min of U edits (empty correction; Lu does not state a rule) := 0, i.e. P_GEC = 1.0 (as Qian Sec. 3.4.2).
Two variants: 'tuned' -- alpha in {0.0,...,1.0} and threshold t in {0.00,...,1.00} chosen jointly on B1 dev by
SGECF-style F0.5 (keep exp(logP) >= t); 'lu_ratio' -- alpha = 0.5 (Lu's alpha:gamma = 0.3:0.3), threshold on dev.
Test labels are used only for evaluation. Metrics: c_model.metrics/curve; bootstrap: c7_eval.py:50-91 replicated
(same as H_editconf). Numbers only are written.  Usage (eswa-c): python -I i_lu_proxy.py --out <dir>
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

S1 = "~/Desktop/11.12-eswa-ASR语音投稿资料/Supplementary_Material_S1/scripts"
os.environ.setdefault("EXP_DIR", "~/Desktop/11.9剑桥")
sys.path.insert(0, S1)
import c_model as C

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
R = f"{C.EXP}/05_runs"
EV = json.load(open(f"{C.EXP}/06_results/eval_results.json"))
P = {"dev": f"{R}/2026-11-10_C1_features/features_dev.csv", "test": f"{R}/2026-11-10_C7_B1_eval/features/features_eval.csv"}
UC = {"dev": f"{R}/2026-11-10_C1_features/utt_counts_dev.csv", "test": f"{R}/2026-11-10_C7_B1_eval/features/utt_counts_eval.csv"}
D = {k: pd.read_csv(v) for k, v in P.items()}
ALPHAS = [round(i / 10, 1) for i in range(11)]


def logs(d):
    la = np.log(np.clip(d["utt_asr_conf_min"].astype(float).values, 1e-6, 1.0))
    lg = d["gec_lp_c_min"].astype(float).values.copy()
    U = (d["op"] == "U").values
    assert (np.isnan(lg) == U).all() and not np.isnan(la).any()
    lg[U] = 0.0
    return la, lg


def conf(d, alpha):
    la, lg = logs(d)
    return np.exp(alpha * la + (1 - alpha) * lg)


def f05(c, y, nref, t):
    keep = c >= t - 1e-9; n = int(keep.sum()); tp = int(((1 - y) * keep).sum())
    p = tp / n if n else 0.0; r = tp / nref
    return (1.25 * p * r / (0.25 * p + r) if p + r > 0 else 0.0), p, r, n / len(y)


ydev = D["dev"]["false_feedback"].values; Ndev = int(pd.read_csv(UC["dev"])["n_ref_edits"].sum())
grid = []
for al in ALPHAS:
    c = conf(D["dev"], al)
    for k in range(101):
        F, p, r, cov = f05(c, ydev, Ndev, k / 100); grid.append((al, k / 100, F, p, r, cov))
best = max(grid, key=lambda g: (g[2], -g[1], g[0]))   # max dev F0.5; ties -> lower threshold, then lower alpha
lu_ratio = max([g for g in grid if g[0] == 0.5], key=lambda g: (g[2], -g[1]))
SEL = {"lu_proxy_tuned": dict(alpha=best[0], threshold=best[1], dev_F05=best[2], dev_P=best[3], dev_R=best[4], dev_cov=best[5]),
       "lu_proxy_ratio": dict(alpha=0.5, threshold=lu_ratio[1], dev_F05=lu_ratio[2], dev_P=lu_ratio[3], dev_R=lu_ratio[4], dev_cov=lu_ratio[5])}
SEL["dev_F05_no_filter"] = f05(np.ones(len(ydev)), ydev, Ndev, 0.0)[0]

# ---- test (labels read for evaluation only; alpha and t fixed above) ----
d = D["test"]
fr = pd.read_csv(f"{R}/2026-11-10_C7_B1_eval/eval_scores_frozen.csv")
m = d[["utt_id", "edit_idx"]].merge(fr, on=["utt_id", "edit_idx"], how="left", validate="1:1")
y = d["false_feedback"].values
assert (m["false_feedback"].values == y).all()
u = pd.read_csv(UC["test"]); nref_g = u.groupby("group")["n_ref_edits"].sum(); N = int(nref_g.sum())
S = {n: m[f"s_{n}"].values for n in ["risk_lgbm", "gec_lr"]}
CONF = {n: conf(d, SEL[n]["alpha"]) for n in ["lu_proxy_tuned", "lu_proxy_ratio"]}
for n, c in CONF.items():
    S[n] = 1.0 - c
names = list(S)
point = {n: C.metrics(y, S[n], N) for n in names}
sc = dict(n_edits=len(d), n_ref=N, risk_lgbm_AUROC=point["risk_lgbm"]["AUROC"], risk_lgbm_P05=point["risk_lgbm"]["P@0.5"])
sc["auroc_p05_ok"] = round(sc["risk_lgbm_AUROC"], 3) == 0.834 and round(sc["risk_lgbm_P05"], 3) == 0.616
sc["n_ok"] = len(d) == EV["n_edits"] and N == EV["n_ref"]
assert sc["auroc_p05_ok"] and sc["n_ok"], sc


def op(yy, c, nr, t):
    keep = c >= t - 1e-9; k = int(keep.sum()); tp = int(((1 - yy) * keep).sum())
    p = tp / k if k else float("nan"); r = tp / nr
    return dict(coverage=k / len(yy), n_shown=k, P=p, R=r, F05=(1.25 * p * r / (0.25 * p + r) if k and p + r > 0 else 0.0))


def at_k(yy, s, nr, k):
    cov, Pc, Rc, Fc = C.curve(yy, s, nr)
    return float(Pc[k - 1]) if k > 0 else float("nan")


OP = {}
for n in CONF:
    r = op(y, CONF[n], N, SEL[n]["threshold"]); r["risk_lgbm_P_same_n"] = at_k(y, S["risk_lgbm"], N, r["n_shown"]); OP[n] = r

groups = np.array(sorted(set(nref_g.index) | set(d["group"])))
gi = {g: d.index[d["group"] == g].values for g in groups}
keys = ["AURC", "P@0.3", "P@0.5", "P@0.7", "AUROC"]
B = {n: {k: [] for k in keys} for n in names}; BD = {n: [] for n in CONF}
rng = np.random.RandomState(C.SEED)
for b in range(EV["n_boot"]):
    gs = rng.choice(groups, size=len(groups), replace=True)
    idx = np.concatenate([gi[g] for g in gs]); nr = int(sum(nref_g.get(g, 0) for g in gs)); yb = y[idx]
    for n in names:
        mm = C.metrics(yb, S[n][idx], nr)
        for k in keys:
            B[n][k].append(mm[k])
    for n in CONF:
        r = op(yb, CONF[n][idx], nr, SEL[n]["threshold"])
        BD[n].append(at_k(yb, S["risk_lgbm"][idx], nr, r["n_shown"]) - r["P"])
pc = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
ci = {n: {k: pc(B[n][k]) for k in keys} for n in names}
sc["bootstrap_ci_identical_to_eval_results"] = all(abs(ci[n][k][j] - EV["ci95"][n][k][j]) <= 1e-12
                                                   for n in ["risk_lgbm", "gec_lr"] for k in keys for j in (0, 1))
tests = {}
for cand, ref in [("risk_lgbm", "lu_proxy_tuned"), ("risk_lgbm", "lu_proxy_ratio"), ("lu_proxy_tuned", "gec_lr")]:
    t = dict(planned=False)
    for k in keys:
        diff = np.array(B[cand][k]) - np.array(B[ref][k])
        t[k] = dict(diff=point[cand][k] - point[ref][k], ci=pc(diff), p=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
    tests[f"{cand}_vs_{ref}"] = t
for n in CONF:
    OP[n]["dP_risk_lgbm_minus_proxy"] = OP[n]["risk_lgbm_P_same_n"] - OP[n]["P"]; OP[n]["dP_ci"] = pc(BD[n])

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(planned=False, status="POST HOC (unplanned) Lu-inspired proxy (no DD term), 2026-11-19, B1",
               not_a_reproduction_because="DD posteriors unavailable; ASR word (not token) confidence; GEC = BART teacher-forced min token prob",
               selection=SEL, alpha_grid=ALPHAS, seed=C.SEED, n_boot=EV["n_boot"], self_check=sc, point=point, ci95=ci,
               paired_tests=tests, operating_point_test=OP,
               inputs_sha256={k: sha(v) for k, v in {**P, "frozen_scores": f"{R}/2026-11-10_C7_B1_eval/eval_scores_frozen.csv"}.items()},
               script_sha256=sha(os.path.abspath(__file__))),
          open(f"{a.out}/lu_proxy_results.json", "w"), indent=1)
print(json.dumps(dict(sel=SEL, sc=sc, auroc={n: round(point[n]["AUROC"], 4) for n in names},
                      aurc={n: round(point[n]["AURC"], 4) for n in names}, op=OP), indent=0, default=float))

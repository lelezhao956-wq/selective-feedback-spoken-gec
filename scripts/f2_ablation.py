#!/usr/bin/env python3
"""F2 (pre-submission supplementary analysis, POST HOC; plan 08_paper/F2_analysis_plan.md). eswa-c env.
Ablation with the FINAL LightGBM settings (no inner tuning), leave-one-part-out, and a post hoc eval supplement.
Uses the unchanged c_model.py functions (lgb_fit_predict, lr_pipe, folds, metrics). Numbers only are written.
Usage: PYTHONDONTWRITEBYTECODE=1 python -I f2_ablation.py [--n_boot 1000]
Writes 06_results/F2_ablation_trdev.json, F2_lopo_trdev.json, F2_ablation_eval_posthoc.json; per-edit scores in 05_runs/2026-11-12_F2/."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C

ap = argparse.ArgumentParser(); ap.add_argument("--n_boot", type=int, default=1000)
ap.add_argument("--smoke_dir", help="write every output here instead (smoke test)"); a = ap.parse_args()
E = C.EXP; RES = a.smoke_dir or f"{E}/06_results"; RUN = a.smoke_dir or f"{E}/05_runs/2026-11-12_F2"; os.makedirs(RUN, exist_ok=True)
EV = f"{E}/05_runs/2026-11-10_C7_B1_eval"
P_LGB = dict(num_leaves=7, n_estimators=100, min_child_samples=20)      # final.yaml risk_model.params (others fixed in C.lgb_fit_predict)
C_GEC = 1                                                              # final.yaml baselines.gec_lr
KEYS = ["AURC", "P@0.3", "P@0.5", "P@0.7", "AUROC", "FFR@0.5"]
ALL = sum(C.GROUPS.values(), []); G = C.GROUPS
CONFIGS = {"all": ALL}
for g in G: CONFIGS[f"-{g}"] = [c for c in ALL if c not in G[g]]
for g in G: CONFIGS[f"only_{g}"] = list(G[g])
for g in ["EDIT", "ASR", "DD"]: CONFIGS[f"GEC+{g}"] = [c for c in ALL if c in G["GEC"] + G[g]]
CONFIGS["all-part"] = [c for c in ALL if c != "part"]
NOTE = ("F2 pre-submission supplementary analysis, specified POST HOC (after the C7 eval); plan 08_paper/F2_analysis_plan.md. "
        "LightGBM with the final fixed parameters (no inner tuning).")
T0 = time.time()
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def lgb_scores(tr, te, cols):
    return C.lgb_fit_predict(tr[cols], tr["false_feedback"], te[cols], P_LGB)[0]


def gec_scores(tr, te):
    return C.lr_pipe(G["GEC"], C_GEC).fit(tr[G["GEC"]], tr["false_feedback"]).predict_proba(te[G["GEC"]])[:, 1]


def part_of(utt):
    return utt.str.extract(r"-P(\d)")[0].radd("P")


def boot(y, grp, S, nref, pairs, n_boot):
    """Grouped bootstrap: point metrics, CI per model, paired differences a-b. nref: Series group -> reference edits."""
    groups = np.array(sorted(set(nref.index) | set(grp)))
    pos = pd.Series(np.arange(len(grp))).groupby(np.asarray(grp)).apply(lambda s: s.values).to_dict()
    gi = {g: pos.get(g, np.array([], int)) for g in groups}
    N = int(nref.sum())
    point = {m: C.metrics(y, s, N) for m, s in S.items()}
    B = {m: {k: [] for k in KEYS} for m in S}
    rng = np.random.RandomState(C.SEED)
    for _ in range(n_boot):
        gs = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([gi[g] for g in gs]); nr = int(sum(nref.get(g, 0) for g in gs))
        for m, s in S.items():
            mm = C.metrics(y[idx], s[idx], nr)
            for k in KEYS: B[m][k].append(mm[k])
    pc = lambda x: [r4(np.percentile(x, 2.5)), r4(np.percentile(x, 97.5))]
    out = {"n_edits": int(len(y)), "n_groups": int(len(groups)), "n_ref": N, "fp_rate": r4(y.mean()),
           "point": {m: {k: r4(point[m][k]) for k in KEYS} for m in S},
           "ci95": {m: {k: pc(B[m][k]) for k in KEYS} for m in S}, "diff": {}}
    for x, z in pairs:
        t = {}
        for k in KEYS:
            d = np.array(B[x][k]) - np.array(B[z][k])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            t[k] = dict(diff=r4(point[x][k] - point[z][k]), ci95=pc(d), p=r4(p))
        out["diff"][f"{x} - {z}"] = t
    return out


def holm(ps):
    ps = {k: max(v, 0.002) for k, v in ps.items()}          # p = 0 (< 1/n_boot*2) set to the resolution floor, as in E2
    order = sorted(ps, key=ps.get); m = len(ps); adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = round(run, 4)
    return adj


def nref_by_group(u, part=None):
    if part is not None: u = u[part_of(u["utt_id"]) == part]
    return u.groupby("group")["n_ref_edits"].sum()


# ------------------------------------------------------------------ data
d, nref = C.load()
U = pd.concat([pd.read_csv(f"{C.F}/utt_counts_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
y = d["false_feedback"].values; grp = d["group"].values
assert (part_of(d["utt_id"]) == d["part"]).all()
PARTS = sorted(d["part"].unique())
fl = C.folds(d)
c2 = pd.read_csv(f"{E}/05_runs/2026-11-10_C2_cv/oof_scores.csv")
assert all((c2["fold"].values[b] == k).all() for k, (_, b) in enumerate(fl)), "folds differ from C2"

# ------------------------------------------------------------------ step 2: ablation, standard grouped 5-fold CV
oof = {m: np.zeros(len(d)) for m in list(CONFIGS) + ["gec_lr"]}
for k, (tr_i, te_i) in enumerate(fl):
    tr, te = d.iloc[tr_i], d.iloc[te_i]
    for m, cols in CONFIGS.items(): oof[m][te_i] = lgb_scores(tr, te, cols)
    oof["gec_lr"][te_i] = gec_scores(tr, te)
# reference check: OOF GEC-only LR with C=1 vs C2 (C2 tuned C per fold; folds with C=1 must match exactly)
gec_chk = {}
ch = json.load(open(f"{E}/05_runs/2026-11-10_C2_cv/cv_info.json"))["chosen_params"]["gec_lr"]
for k, (_, b) in enumerate(fl):
    if ch[k]["params"]["C"] == C_GEC: gec_chk[f"fold{k}"] = float(np.abs(oof["gec_lr"][b] - c2["s_gec_lr"].values[b]).max())
pd.DataFrame({**{f"s_{m}": v for m, v in oof.items()}, "fold": c2["fold"].values}).assign(
    ds=d["ds"], utt_id=d["utt_id"], edit_idx=d["edit_idx"], group=grp, part=d["part"], false_feedback=y
).to_csv(f"{RUN}/F2_oof_trdev.csv", index=False)
print(f"[{time.time()-T0:.0f}s] step 2 OOF fitted", flush=True)

abl = boot(y, grp, oof, nref, [(m, "all") for m in list(CONFIGS)[1:] + ["gec_lr"]], a.n_boot)
cmp12 = [f"{m} - all" for m in list(CONFIGS)[1:]]
abl["holm_12_vs_all"] = {k: holm({c: abl["diff"][c][k]["p"] for c in cmp12}) for k in ["AURC", "P@0.5"]}
per_part_cv = {}
for p in PARTS:
    mk = (d["part"] == p).values
    per_part_cv[p] = boot(y[mk], grp[mk], {m: oof[m][mk] for m in ["all", "all-part", "gec_lr"]}, nref_by_group(U, p),
                          [("all", "gec_lr"), ("all-part", "gec_lr"), ("all-part", "all")], a.n_boot)
c5 = json.load(open(f"{E}/05_runs/2026-11-10_C5_ablation/ablation_lgbm.json"))["results"]
c5b = json.load(open(f"{E}/05_runs/2026-11-10_C5_ablation/ablation_boot_lgbm.json"))["results"]
c5p = json.load(open(f"{E}/05_runs/2026-11-10_C5_ablation/c5_analysis.json"))["per_part"]
json.dump(dict(note=NOTE, data="B1 train+dev, C2 outer folds (StratifiedGroupKFold 5 by submission, seed 20261110)",
               n_boot=a.n_boot, seed=C.SEED, lgbm_params=P_LGB, gec_lr_C=C_GEC, configs=CONFIGS,
               check_gec_lr_oof_equals_C2_on_folds_with_C1=gec_chk, ablation=abl, per_part_standard_cv=per_part_cv,
               C5_reference_nested_tuning=dict(note="C5: per-fold re-tuned LightGBM; reference only",
                                               metrics={m: {k: c5[m][k] for k in KEYS} for m in c5},
                                               boot_all_minus_ablation=c5b, per_part=c5p)),
          open(f"{RES}/F2_ablation_trdev.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] step 2 written", flush=True)

# ------------------------------------------------------------------ step 3: leave-one-part-out
LM = {"risk_lgbm_all": ALL, "risk_lgbm_all-part": CONFIGS["all-part"]}
lopo = {m: np.zeros(len(d)) for m in list(LM) + ["gec_lr"]}
lopo_sd = {m: np.zeros(len(d)) for m in lopo}                       # sensitivity: part-disjoint AND submission-disjoint
fold_id = c2["fold"].values
for p in PARTS:
    te_m = (d["part"] == p).values
    tr, te = d[~te_m], d[te_m]
    for m, cols in LM.items(): lopo[m][te_m] = lgb_scores(tr, te, cols)
    lopo["gec_lr"][te_m] = gec_scores(tr, te)
    for k in range(5):
        te2 = te_m & (fold_id == k); tr2 = ~te_m & (fold_id != k)
        for m, cols in LM.items(): lopo_sd[m][te2] = lgb_scores(d[tr2], d[te2], cols)
        lopo_sd["gec_lr"][te2] = gec_scores(d[tr2], d[te2])
pd.DataFrame({**{f"lopo_{m}": v for m, v in lopo.items()}, **{f"lopo_sd_{m}": v for m, v in lopo_sd.items()}}).assign(
    ds=d["ds"], utt_id=d["utt_id"], edit_idx=d["edit_idx"], group=grp, part=d["part"], false_feedback=y
).to_csv(f"{RUN}/F2_lopo_trdev.csv", index=False)
print(f"[{time.time()-T0:.0f}s] step 3 fitted", flush=True)
pairs = [("risk_lgbm_all", "gec_lr"), ("risk_lgbm_all-part", "gec_lr"), ("risk_lgbm_all", "risk_lgbm_all-part")]
cvname = {"risk_lgbm_all": "all", "risk_lgbm_all-part": "all-part", "gec_lr": "gec_lr"}
LO = {"per_part": {}, "per_part_sensitivity_submission_disjoint": {}}
for p in PARTS:
    mk = (d["part"] == p).values; nr = nref_by_group(U, p)
    LO["per_part"][p] = boot(y[mk], grp[mk], {m: v[mk] for m, v in lopo.items()}, nr, pairs, a.n_boot)
    # drop vs standard CV (same model, same metric definition): LOPO - CV
    S2 = {**{f"lopo:{m}": lopo[m][mk] for m in lopo}, **{f"cv:{m}": oof[cvname[m]][mk] for m in lopo}}
    LO["per_part"][p]["lopo_minus_standard_cv"] = boot(y[mk], grp[mk], S2, nr, [(f"lopo:{m}", f"cv:{m}") for m in lopo], a.n_boot)["diff"]
    LO["per_part_sensitivity_submission_disjoint"][p] = boot(y[mk], grp[mk], {m: v[mk] for m, v in lopo_sd.items()}, nr, pairs, a.n_boot)
LO["pooled"] = boot(y, grp, lopo, nref, pairs, a.n_boot)
S2 = {**{f"lopo:{m}": lopo[m] for m in lopo}, **{f"cv:{m}": oof[cvname[m]] for m in lopo}}
LO["pooled"]["lopo_minus_standard_cv"] = boot(y, grp, S2, nref, [(f"lopo:{m}", f"cv:{m}") for m in lopo], a.n_boot)["diff"]
LO["pooled_sensitivity_submission_disjoint"] = boot(y, grp, lopo_sd, nref, pairs, a.n_boot)
json.dump(dict(note=NOTE, design=("Leave-one-part-out on train+dev: train on all edits of the other parts, test on the held-out part. "
                                  "Pooled = the four held-out score sets concatenated (four different models). "
                                  "Sensitivity: additionally submission-disjoint (train other parts AND other C2 folds)."),
               parts=PARTS, n_boot=a.n_boot, seed=C.SEED, lgbm_params=P_LGB, gec_lr_C=C_GEC, results=LO,
               C5_per_part_reference=c5p),
          open(f"{RES}/F2_lopo_trdev.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] step 3 written", flush=True)

# ------------------------------------------------------------------ step 4: eval supplement (POST HOC, once per setting)
de = pd.read_csv(f"{EV}/features/features_eval.csv"); ue = pd.read_csv(f"{EV}/features/utt_counts_eval.csv")
fr = pd.read_csv(f"{EV}/eval_scores_frozen.csv")
assert (fr["utt_id"].values == de["utt_id"].values).all() and (fr["edit_idx"].values == de["edit_idx"].values).all()
SE = {m: lgb_scores(d, de, cols) for m, cols in CONFIGS.items()}
SE["gec_lr"] = gec_scores(d, de)
chk = {"all_vs_frozen_risk_lgbm_max_abs_diff": float(np.abs(SE["all"] - fr["s_risk_lgbm"]).max()),
       "gec_lr_vs_frozen_gec_lr_max_abs_diff": float(np.abs(SE["gec_lr"] - fr["s_gec_lr"]).max())}
assert chk["all_vs_frozen_risk_lgbm_max_abs_diff"] < 1e-9 and chk["gec_lr_vs_frozen_gec_lr_max_abs_diff"] < 1e-9, chk
pd.DataFrame({f"s_{m}": v for m, v in SE.items()}).assign(utt_id=de["utt_id"], edit_idx=de["edit_idx"], group=de["group"],
                                                          false_feedback=de["false_feedback"]).to_csv(f"{RUN}/F2_eval_scores_posthoc.csv", index=False)
ye = de["false_feedback"].values
ev = boot(ye, de["group"].values, SE, ue.groupby("group")["n_ref_edits"].sum(), [(m, "all") for m in list(CONFIGS)[1:] + ["gec_lr"]], a.n_boot)
json.dump(dict(status="post hoc", note=NOTE + " EVAL SUPPLEMENT, POST HOC: each configuration refitted on all train+dev and applied "
               "once to the B1 eval edits; nothing was chosen on eval.", n_boot=a.n_boot, seed=C.SEED, lgbm_params=P_LGB,
               gec_lr_C=C_GEC, configs=CONFIGS, check_reproduces_frozen_eval_scores=chk, results=ev),
          open(f"{RES}/F2_ablation_eval_posthoc.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] step 4 written; checks {chk}; gec_lr OOF check {gec_chk}", flush=True)

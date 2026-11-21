#!/usr/bin/env python3
"""H (POST HOC, planned=False): fixed-rule edit-confidence baseline after Qian et al. (2026, Speech Commun. 185, 103492,
Sec. 3.4, Eqs. (1)-(6); threshold rule Sec. 5.5) on the B1 cascade, compared with the frozen risk model.
No fitting, no retraining, no pipeline rerun. Reads existing feature files and the existing frozen eval scores only.
Metrics: c_model.metrics / c_model.curve (imported). Bootstrap: the submission-grouped loop of c7_eval.py:50-91 replicated
verbatim (it is not an importable function; c_model.py's own copy sits inside main()), same SEED, same n_boot, same groups.
Numbers only are written. Usage (eswa-c env): EXP_DIR=... python -I h_editconf.py --out <dir>
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd

S1 = "~/Desktop/11.12-eswa-ASR语音投稿资料/Supplementary_Material_S1/scripts"
sys.path.insert(0, S1)
import c_model as C  # module top level only defines constants/functions (main() is guarded); needs EXP_DIR

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
R = f"{C.EXP}/05_runs"
EV = json.load(open(f"{C.EXP}/06_results/eval_results.json"))
FIT = json.load(open(f"{R}/2026-11-10_C6_final/final_fit.json"))
assert a.n_boot == EV["n_boot"] and C.SEED == EV["seed"] == 20261110 and C.COVS == [0.3, 0.5, 0.7, 0.9, 1.0]

P = {"train": f"{R}/2026-11-10_C1_features/features_train.csv", "dev": f"{R}/2026-11-10_C1_features/features_dev.csv",
     "test": f"{R}/2026-11-10_C7_B1_eval/features/features_eval.csv"}
UC = {"train": f"{R}/2026-11-10_C1_features/utt_counts_train.csv", "dev": f"{R}/2026-11-10_C1_features/utt_counts_dev.csv",
      "test": f"{R}/2026-11-10_C7_B1_eval/features/utt_counts_eval.csv"}
D = {k: pd.read_csv(v) for k, v in P.items()}


# ---------------- fixed-rule confidences (no labels used anywhere in this block) ----------------
def sides(d, variant):
    """Return flt_avg, flt_min, gec_avg, gec_min as float arrays (NaN = side unavailable under this variant)."""
    M = (d["op"] == "M").values; U = (d["op"] == "U").values
    fa, fm = d["asr_conf_o_mean"].values.astype(float), d["asr_conf_o_min"].values.astype(float)
    ga, gm = np.exp(d["gec_lp_c_mean"].values.astype(float)), np.exp(d["gec_lp_c_min"].values.astype(float))
    # structural NaN only (phase0 Q6): o empty <=> M, c empty <=> U
    assert (np.isnan(fa) == M).all() and (np.isnan(fm) == M).all() and (np.isnan(ga) == U).all() and (np.isnan(gm) == U).all()
    fa, fm, ga, gm = fa.copy(), fm.copy(), ga.copy(), gm.copy()
    if variant == "Q":   # Qian Sec. 3.4.2 (p. 6): fixed 1.0 for the empty side
        fa[M] = 1.0; fm[M] = 1.0; ga[U] = 1.0; gm[U] = 1.0
    elif variant == "A":  # AUTO defaults: M -> window confidences; U -> GEC side unavailable (one-sided)
        fa[M] = d["asr_conf_win_mean"].values[M]; fm[M] = d["asr_conf_win_min"].values[M]
    else:
        raise ValueError(variant)
    return fa, fm, ga, gm


def confs(d, variant):
    fa, fm, ga, gm = sides(d, variant)
    with np.errstate(all="ignore"):
        e_avg = np.where(np.isnan(fa), ga, np.where(np.isnan(ga), fa, (fa + ga) / 2))   # one side only if the other missing
        e_min = np.fmin(fm, gm)                                                       # fmin ignores a single NaN
    return dict(edit_avg=e_avg, edit_min=e_min, flt_avg=fa, gec_avg=ga)


CONF, FILL = {}, {}
for v in ["Q", "A"]:
    raw = {ds: confs(D[ds], v) for ds in D}
    med = {k: float(np.nanmedian(raw["train"][k])) for k in raw["train"]}   # B1 train median, fallback only
    for ds in D:
        for k, x in raw[ds].items():
            n_fill = int(np.isnan(x).sum())
            FILL[f"{v}/{ds}/{k}"] = dict(n_filled=n_fill, share=round(n_fill / len(x), 4), fill_value=med[k])
            CONF[(v, ds, k)] = np.where(np.isnan(x), med[k], x)
SCORE = {key: 1.0 - c for key, c in CONF.items()}   # risk = 1 - confidence (c_model.py:159 direction)
METHODS = [(v, k) for v in ["Q", "A"] for k in ["edit_avg", "edit_min", "flt_avg", "gec_avg"]]
mname = lambda v, k: k if v == "Q" else f"{k}_A"

# score files: IDs + scores only
for ds in D:
    o = D[ds][["utt_id", "edit_idx"]].copy()
    for v, k in METHODS:
        o[f"s_{mname(v, k)}"] = SCORE[(v, ds, k)]
    o.to_csv(f"{a.out}/editconf_scores_{ds}.csv", index=False)

# ---------------- Qian-style threshold on B1 DEV only ----------------
def sweep_dev(conf, y, n_ref):
    rows = []
    for kk in range(101):
        t = kk / 100; keep = conf >= t - 1e-9; n = int(keep.sum()); tp = int(((1 - y) * keep).sum())
        Pp = tp / n if n else 0.0; Rr = tp / n_ref
        F = 1.25 * Pp * Rr / (0.25 * Pp + Rr) if (Pp + Rr) > 0 else 0.0
        rows.append((t, F, Pp, Rr, n / len(y)))
    best = max(rows, key=lambda r: (r[1], -r[0]))   # max F0.5; ties -> lowest threshold
    return dict(threshold=best[0], F05=best[1], P=best[2], R=best[3], coverage=best[4], F05_no_filter=rows[0][1])


ydev = D["dev"]["false_feedback"].values
Ndev = int(pd.read_csv(UC["dev"])["n_ref_edits"].sum())
THR = {mname(v, k): sweep_dev(CONF[(v, "dev", k)], ydev, Ndev) for v, k in METHODS}

# ---------------- B1 test ----------------
d = D["test"]
fr = pd.read_csv(f"{R}/2026-11-10_C7_B1_eval/eval_scores_frozen.csv")
m = d[["utt_id", "edit_idx"]].merge(fr, on=["utt_id", "edit_idx"], how="left", validate="1:1")
assert len(m) == len(d) and m["s_risk_lgbm"].notna().all() and (m["false_feedback"].values == d["false_feedback"].values).all()
u = pd.read_csv(UC["test"]); nref_g = u.groupby("group")["n_ref_edits"].sum(); N = int(nref_g.sum())
y = d["false_feedback"].values
S = {n: m[f"s_{n}"].values for n in ["risk_lgbm", "risk_lr", "gec_lr"]}
S["asr_conf"] = 1 - d["asr_conf_win_min"].fillna(FIT["asr_conf_fill"]).values   # c7_eval.py:44
for v, k in METHODS:
    S[mname(v, k)] = SCORE[(v, "test", k)]
names = list(S)
point = {n: C.metrics(y, S[n], N) for n in names}

# self-checks against the frozen single eval run
sc = dict(n_edits=len(d), n_edits_main=EV["n_edits"], n_ref=N, n_ref_main=EV["n_ref"],
          risk_lgbm_AUROC=point["risk_lgbm"]["AUROC"], risk_lgbm_P05=point["risk_lgbm"]["P@0.5"])
sc["auroc_ok"] = round(sc["risk_lgbm_AUROC"], 3) == 0.834
sc["p05_ok"] = round(sc["risk_lgbm_P05"], 3) == 0.616
sc["n_edits_ok"] = sc["n_edits"] == sc["n_edits_main"] and N == EV["n_ref"]
sc["frozen_point_identical"] = {n: max(abs(point[n][k] - EV["point"][n][k]) for k in EV["point"][n]) <= 1e-12
                                for n in ["risk_lgbm", "risk_lr", "gec_lr", "asr_conf"]}
if not (sc["auroc_ok"] and sc["p05_ok"]):
    open(f"{a.out}/STOPPED.md", "w").write(
        f"# STOPPED\n\nrisk_lgbm B1 test recomputed AUROC={sc['risk_lgbm_AUROC']:.6f}, P@0.5={sc['risk_lgbm_P05']:.6f}; "
        f"expected 0.834 / 0.616 (3 d.p.). Stopped before bootstrap. Score files were written; no results.\n")
    print(json.dumps(sc)); sys.exit(2)


def op_metrics(yy, conf, nr, t):
    keep = conf >= t - 1e-9; k = int(keep.sum()); tp = int(((1 - yy) * keep).sum())
    Pp = tp / k if k else float("nan"); Rr = tp / nr if nr else float("nan")
    F = 1.25 * Pp * Rr / (0.25 * Pp + Rr) if k and (Pp + Rr) > 0 else 0.0
    return dict(coverage=k / len(yy), n_shown=k, P=Pp, R=Rr, F05=F)


def at_k(yy, s, nr, k):
    cov, Pc, Rc, Fc = C.curve(yy, s, nr)
    return dict(P=float(Pc[k - 1]), R=float(Rc[k - 1]), F05=float(Fc[k - 1])) if k > 0 else dict(P=float("nan"), R=0.0, F05=0.0)


OPN = [mname(v, k) for v, k in METHODS]
opp = {}
for n in OPN:
    v = "A" if n.endswith("_A") else "Q"; k_ = n[:-2] if n.endswith("_A") else n
    c = CONF[(v, "test", k_)]; t = THR[n]["threshold"]
    r = op_metrics(y, c, N, t)
    r["no_filter"] = op_metrics(y, c, N, 0.0)
    r["risk_lgbm_same_n"] = at_k(y, S["risk_lgbm"], N, r["n_shown"])
    r["gec_lr_same_n"] = at_k(y, S["gec_lr"], N, r["n_shown"])
    opp[n] = r

# ---------------- bootstrap (c7_eval.py:50-91, unchanged logic) ----------------
groups = np.array(sorted(set(nref_g.index) | set(d["group"])))
gi = {g: d.index[d["group"] == g].values for g in groups}
keys = ["AURC", "P@0.3", "P@0.5", "P@0.7", "P@0.9", "FFR@0.3", "FFR@0.5", "FFR@0.7", "FFR@0.9", "F05@0.5", "maxF05", "AUROC", "AUPRC"]
B = {n: {k: [] for k in keys} for n in names}
BO = {n: {k: [] for k in ["coverage", "P", "R", "F05", "dP_risk_lgbm", "dP_gec_lr"]} for n in OPN}
rng = np.random.RandomState(C.SEED)
t0 = time.time()
for b in range(a.n_boot):
    gs = rng.choice(groups, size=len(groups), replace=True)
    idx = np.concatenate([gi[g] for g in gs]); nr = int(sum(nref_g.get(g, 0) for g in gs)); yb = y[idx]
    for n in names:
        mm = C.metrics(yb, S[n][idx], nr)
        for k in keys:
            B[n][k].append(mm[k])
    for n in OPN:
        v = "A" if n.endswith("_A") else "Q"; k_ = n[:-2] if n.endswith("_A") else n
        r = op_metrics(yb, CONF[(v, "test", k_)][idx], nr, THR[n]["threshold"])
        for k in ["coverage", "P", "R", "F05"]:
            BO[n][k].append(r[k])
        BO[n]["dP_risk_lgbm"].append(at_k(yb, S["risk_lgbm"][idx], nr, r["n_shown"])["P"] - r["P"])
        BO[n]["dP_gec_lr"].append(at_k(yb, S["gec_lr"][idx], nr, r["n_shown"])["P"] - r["P"])
pc = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
ci = {n: {k: pc(B[n][k]) for k in keys} for n in names}
op_ci = {n: {k: pc(v) for k, v in BO[n].items()} for n in OPN}
for n in OPN:
    opp[n]["dP_risk_lgbm_minus_edit"] = opp[n]["risk_lgbm_same_n"]["P"] - opp[n]["P"]
    opp[n]["dP_gec_lr_minus_edit"] = opp[n]["gec_lr_same_n"]["P"] - opp[n]["P"]
# bootstrap replication check: identical resamples => identical CIs to the frozen run
sc["bootstrap_ci_identical_to_eval_results"] = {n: max(abs(ci[n][k][j] - EV["ci95"][n][k][j]) for k in keys for j in (0, 1)) <= 1e-12
                                                for n in ["risk_lgbm", "risk_lr", "gec_lr", "asr_conf"]}

PAIRS = [("risk_lgbm", "edit_avg"), ("risk_lgbm", "edit_min"), ("edit_avg", "gec_lr"), ("edit_avg", "asr_conf"),
         ("edit_min", "gec_lr"), ("risk_lgbm", "flt_avg"), ("risk_lgbm", "gec_avg"),
         ("risk_lgbm", "edit_avg_A"), ("risk_lgbm", "edit_min_A"), ("edit_avg_A", "gec_lr"), ("edit_avg_A", "asr_conf")]
tests = {}
for cand, ref in PAIRS:
    t = dict(planned=False, status="post hoc (unplanned), added 2026-11-19 after the single eval run")
    for k in keys:
        diff = np.array(B[cand][k]) - np.array(B[ref][k])
        t[k] = dict(diff=point[cand][k] - point[ref][k], ci=pc(diff), p=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
    tests[f"{cand}_vs_{ref}"] = t

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
sc["test_labels_used_for_selection"] = False
sc["selection_note"] = ("Formulas fixed in phase0 from Qian (PDF) and feature definitions; fallback medians from B1 train; "
                        "thresholds from B1 dev only (sweep_dev receives dev arrays only). Test labels are read only after all "
                        "scores and thresholds are fixed, for evaluation.")
out = dict(planned=False, status="POST HOC (unplanned) baseline, 2026-11-19; Qian et al. (2026) edit confidence on B1",
           seed=C.SEED, n_boot=a.n_boot, covs=C.COVS, n_groups=len(groups), n_ref_test=N, n_ref_dev=Ndev,
           n_edits={ds: len(D[ds]) for ds in D}, primary_variant="Q (Qian Sec. 3.4.2: empty side = 1.0)",
           sensitivity_variant="A (AUTO defaults: M -> window conf; U -> FLT side only)",
           fill=FILL, self_check=sc, point=point, ci95=ci, paired_tests=tests,
           qian_operating_point=dict(rule="dev sweep t=0.00..1.00 step 0.01, keep conf>=t, max F0.5 (P=TP/kept, R=TP/n_ref_dev), ties->lowest t",
                                     dev=THR, test=opp, test_ci95=op_ci),
           script_sha256=sha(os.path.abspath(__file__)), c_model_sha256=sha(f"{S1}/c_model.py"),
           boot_seconds=round(time.time() - t0, 1))
json.dump(out, open(f"{a.out}/editconf_results.json", "w"), indent=1)
print(json.dumps(dict(self_check=sc, thr={n: THR[n]["threshold"] for n in THR},
                      auroc={n: round(point[n]["AUROC"], 4) for n in names}), indent=0))

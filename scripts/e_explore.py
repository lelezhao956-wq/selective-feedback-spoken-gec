#!/usr/bin/env python3
"""Stage E: EXPLORATORY analyses (added after the C7 eval; plan 03_configs/2026-11-10_E_exploratory_plan.yaml).
Only existing predictions: eval scores = deterministic outputs of the frozen artifacts on the existing eval features (checked against
the stored eval JSON to 1e-12), OOF = stored oof_scores.csv, thresholds = stored OOF operating points. No main model is retrained;
E5 fits small subsample models (learning curve) that are never used for anything else. Numbers only.
Usage (eswa-c env): nice -n 10 python -I e_explore.py --system B1|W|W_frozen|Q [--items E1,E2,...]
"""
import argparse, hashlib, json, os, pickle, re, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_model as C   # noqa: E402

E = C.EXP; RUNS = f"{E}/05_runs"; CORPUS = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))
B1A = f"{RUNS}/2026-11-10_C6_final"; B1O = f"{RUNS}/2026-11-10_C2_cv/oof_scores.csv"
SYS = {
    "B1": dict(eval=f"{RUNS}/2026-11-10_C7_B1_eval/features", art=B1A, oof=B1O, td=f"{RUNS}/2026-11-10_C1_features",
               res=f"{E}/06_results/eval_results.json", chain=None),
    "W": dict(eval=f"{RUNS}/2026-11-10_D_W_eval/features", art=f"{RUNS}/2026-11-10_D_W_model/05_runs/2026-11-10_C6_final",
              oof=f"{RUNS}/2026-11-10_D_W_model/05_runs/2026-11-10_C2_cv/oof_scores.csv", td=f"{RUNS}/2026-11-10_D_W_model/features",
              res=f"{E}/06_results/D_W_eval_retrained.json", chain="W"),
    "W_frozen": dict(eval=f"{RUNS}/2026-11-10_D_W_eval/features", art=B1A, oof=B1O, td=None,
                     res=f"{E}/06_results/D_W_eval_frozen.json", chain="W"),
    "Q": dict(eval=f"{RUNS}/2026-11-10_D_Q_eval/features", art=f"{RUNS}/2026-11-10_D_Q_model/05_runs/2026-11-10_C6_final",
              oof=f"{RUNS}/2026-11-10_D_Q_model/05_runs/2026-11-10_C2_cv/oof_scores.csv", td=f"{RUNS}/2026-11-10_D_Q_model/features",
              res=f"{E}/06_results/D_Q_eval_retrained.json", chain="Q"),
}
MODELS = ["risk_lgbm", "risk_lr", "gec_lr"]
ap = argparse.ArgumentParser(); ap.add_argument("--system", required=True, choices=list(SYS))
ap.add_argument("--items", default="E1,E2,E3,E4,E5,E6,E7"); ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args(); cfg = SYS[a.system]
ITEMS = [i for i in a.items.split(",") if i]
if a.system == "W_frozen":
    ITEMS = [i for i in ITEMS if i in ("E1", "E2", "E6")]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
OUT = f"{E}/06_results/E_{a.system}_exploratory.json"; FIG = f"{E}/07_figures"; os.makedirs(FIG, exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.update(status="EXPLORATORY (added after the C7 eval; 03_configs/2026-11-10_E_exploratory_plan.yaml)", system=a.system,
           seed=C.SEED, script_sha256=sha(os.path.abspath(__file__)))
T0 = time.time()


def save():
    json.dump(res, open(OUT, "w"), indent=1)


# ---------------- scores (frozen artifacts, checked against the stored eval JSON)
fit = json.load(open(f"{cfg['art']}/final_fit.json")); add = json.load(open(f"{cfg['art']}/addendum_fit.json"))
assert all(sha(f"{cfg['art']}/{f}") == h for f, h in fit["artifact_sha256"].items())
assert sha(f"{cfg['art']}/risk_lr.pkl") == add["risk_lr"]["artifact_sha256"]
ALLC = fit["feature_columns"]; CATEG = fit["models"]["risk_lgbm"]["categories"]
BOOST = lgb.Booster(model_file=f"{cfg['art']}/risk_lgbm.txt")
PK = {n: pickle.load(open(f"{cfg['art']}/{n}.pkl", "rb")) for n in ["risk_lr", "gec_lr", "asr_lr"]}
COLS = {"risk_lr": ALLC, "gec_lr": C.GROUPS["GEC"], "asr_lr": C.GROUPS["ASR"]}


def lgbX(df):
    X = df[ALLC].copy()
    for c, lev in CATEG.items():
        X[c] = X[c].astype(str).astype(pd.CategoricalDtype(lev))
    return X


def score(df):
    S = {"risk_lgbm": BOOST.predict(lgbX(df))}
    for n, m in PK.items():
        S[n] = m.predict_proba(df[COLS[n]])[:, 1]
    S["asr_conf"] = 1 - df["asr_conf_win_min"].fillna(fit["asr_conf_fill"]).values
    return S


ev = pd.read_csv(f"{cfg['eval']}/features_eval.csv")
u = pd.read_csv(f"{cfg['eval']}/utt_counts_eval.csv"); N_EV = int(u["n_ref_edits"].sum())
SE = score(ev); yE = ev["false_feedback"].values
EVJ = json.load(open(cfg["res"]))
for n in MODELS + ["asr_lr", "asr_conf"]:
    m = C.metrics(yE, SE[n], N_EV)
    bad = {k: (m[k], v) for k, v in EVJ["point"][n].items() if abs(m[k] - v) > 1e-12}
    assert not bad, ("eval scores differ from the stored eval run", n, bad)
assert len(ev) == EVJ["n_edits"]
res["eval_scores_check"] = "all point metrics identical to " + os.path.relpath(cfg["res"], E)
OOF = pd.read_csv(cfg["oof"]); yO = OOF["false_feedback"].values
OPS = {n: {k: v["threshold"] for k, v in add["operating_points"][n].items()} for n in MODELS}
res["thresholds_from_oof"] = OPS
SO = {n: OOF[f"s_{n}"].values for n in MODELS + ["asr_lr", "asr_conf"]}
pc = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
pv = lambda x: float(min(1.0, 2 * min(np.mean(np.asarray(x) <= 0), np.mean(np.asarray(x) >= 0))))


def boot_groups(groups, seed=C.SEED):
    rng = np.random.RandomState(seed); G = np.array(sorted(set(groups)))
    for _ in range(a.n_boot):
        yield rng.choice(G, size=len(G), replace=True)


# ---------------- E1 proficiency
def marks():
    m = {}
    for ds in ["train", "dev", "eval"]:
        t = pd.read_csv(f"{CORPUS}/reference-materials/sla-marks/{ds}-sla-overall.tsv", sep="\t", header=None, names=["g", "s"])
        m.update(dict(zip(t.g, t.s)))
    return m


def band(s):
    return np.where(pd.isna(s), "NO_MARK", np.where(s < 3.5, "LOW", np.where(s < 4.5, "MID", "HIGH")))


def e1_block(df, y, S, bands):
    g = df["group"].values; out = {}
    per = pd.DataFrame({"g": g, "b": bands, "y": y})
    cols = {}
    for n in MODELS:
        for op, t in OPS[n].items():
            sh = S[n] <= t
            cols[(n, op)] = sh
            per[f"sh_{n}_{op}"] = sh.astype(int); per[f"shtp_{n}_{op}"] = (sh & (y == 0)).astype(int)
            per[f"shfp_{n}_{op}"] = (sh & (y == 1)).astype(int)
    per["tp"] = (y == 0).astype(int); per["n"] = 1
    G = per.groupby(["b", "g"]).sum(numeric_only=True)

    def stats(Gb):
        s = Gb.sum(); r = dict(n_edits=int(s["n"]), fp_rate=float(s["y"] / s["n"]) if s["n"] else None)
        for n in MODELS:
            for op in OPS[n]:
                k = f"{n}_{op}"; shn = s[f"sh_{k}"]
                r[k] = dict(coverage=float(shn / s["n"]), withheld_share=float(1 - shn / s["n"]),
                            precision_shown=float(s[f"shtp_{k}"] / shn) if shn else None,
                            ffr_shown=float(1 - s[f"shtp_{k}"] / shn) if shn else None,
                            correct_withheld_share=float(1 - s[f"shtp_{k}"] / s["tp"]) if s["tp"] else None,
                            false_withheld_share=float(1 - s[f"shfp_{k}"] / s["y"]) if s["y"] else None)
        return r
    for b in ["LOW", "MID", "HIGH", "NO_MARK"]:
        if b in G.index.get_level_values(0):
            out[b] = stats(G.loc[b]); out[b]["n_submissions"] = int(len(G.loc[b]))
    # LOW - HIGH contrasts, grouped bootstrap (each submission is in one band)
    L, H = G.loc["LOW"], G.loc["HIGH"]
    keys = ["fp_rate"] + [f"{n}_{op}.{m}" for n in MODELS for op in OPS[n] for m in ["withheld_share", "correct_withheld_share", "precision_shown"]]

    def flat(r):
        f = {"fp_rate": r["fp_rate"]}
        for n in MODELS:
            for op in OPS[n]:
                for m in ["withheld_share", "correct_withheld_share", "precision_shown"]:
                    f[f"{n}_{op}.{m}"] = r[f"{n}_{op}"][m]
        return f
    fL0, fH0 = flat(stats(L)), flat(stats(H))   # NA (None) when no edit is shown at an operating point in LOW or HIGH -> contrast skipped
    pt = {k: (fL0[k] - fH0[k]) if fL0[k] is not None and fH0[k] is not None else None for k in keys}
    BS = {k: [] for k in keys}; rng = np.random.RandomState(C.SEED)
    for _ in range(a.n_boot):
        lb = L.iloc[rng.randint(0, len(L), len(L))]; hb = H.iloc[rng.randint(0, len(H), len(H))]
        fl, fh = flat(stats(lb)), flat(stats(hb))
        for k in keys:
            BS[k].append((fl[k] if fl[k] is not None else np.nan) - (fh[k] if fh[k] is not None else np.nan))
    out["LOW_minus_HIGH"] = {k: dict(diff=pt[k], ci95=pc(BS[k]), p=pv(np.array(BS[k])[~np.isnan(BS[k])])) if pt[k] is not None else
                             dict(diff=None, ci95=None, p=None, status="NA: no edits shown at this operating point in LOW or HIGH (skipped)")
                             for k in keys}
    return out


def e1():
    M = marks()
    r = {"grouping_rule": "LOW < 3.5 (A2/B1), MID 3.5-<4.5 (B2), HIGH >= 4.5 (C1) on the sla-marks overall score; NO_MARK = no overall mark",
         "bootstrap": "submissions resampled within band (bands are submission-level), 1000, seed SEED"}
    se = ev["group"].map(M).values; r["eval"] = e1_block(ev, yE, SE, band(se))
    r["eval_score_by_band_mean"] = {b: float(np.nanmean(se[band(se) == b])) for b in ["LOW", "MID", "HIGH"]}
    if a.system != "W_frozen":
        so = OOF["group"].map(M).values; r["oof_train_dev"] = e1_block(OOF, yO, SO, band(so))
        # sensitivity: tertiles of the overall score over the train+dev submissions
        sub = pd.Series(M)[pd.Series(M).index.isin(OOF["group"])]
        q1, q2 = np.quantile(sub.values, [1 / 3, 2 / 3])
        tb = lambda s: np.where(pd.isna(s), "NO_MARK", np.where(s < q1, "LOW", np.where(s < q2, "MID", "HIGH")))
        r["sensitivity_tertiles"] = dict(cuts=[float(q1), float(q2)], eval=e1_block(ev, yE, SE, tb(se)))
    res["E1_proficiency"] = r


# ---------------- E2 Holm
def holm(ps, alpha=0.05):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def e2():
    T = EVJ["paired_tests"]; pairs = ["risk_lgbm_vs_asr_conf", "risk_lgbm_vs_gec_lr", "risk_lr_vs_gec_lr"]
    floor = 2.0 / a.n_boot
    fam = {}
    for name, keys in [("primary_AURC_P@0.5", ["AURC", "P@0.5"]),
                       ("sensitivity_all_endpoints", [k for k in T[pairs[0]] if k not in ("status",)])]:
        lab, raw = [], []
        for pr in pairs:
            for k in keys:
                lab.append(f"{pr}:{k}"); raw.append(T[pr][k]["p"])
        rawf = np.maximum(np.array(raw), floor); adj = holm(rawf)
        fam[name] = {l: dict(diff=T[l.split(":")[0]][l.split(":")[1]]["diff"], p_raw=float(r0), p_raw_floored=float(rf), p_holm=float(h),
                             reject_holm_005=bool(h < 0.05))
                     for l, r0, rf, h in zip(lab, raw, rawf, adj)}
    res["E2_holm"] = dict(source=os.path.relpath(cfg["res"], E), p_floor=floor,
                          note="bootstrap p = 0 means p < 2/n_boot; set to 2/n_boot before Holm (conservative)", families=fam)


# ---------------- E3 SHAP
def e3():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    td = pd.concat([pd.read_csv(f"{cfg['td']}/features_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
    grp = {c: g for g, cs in C.GROUPS.items() for c in cs}
    r = {"method": "TreeSHAP, lightgbm pred_contrib (log-odds units)"}
    for nm, df in [("train_dev", td), ("eval", ev)]:
        sv = BOOST.predict(lgbX(df), pred_contrib=True)[:, :-1]
        imp = pd.Series(np.abs(sv).mean(0), index=ALLC).sort_values(ascending=False)
        r[nm] = dict(n=len(df), mean_abs_shap=imp.round(6).to_dict(),
                     by_group={g: float(imp[[c for c in imp.index if grp[c] == g]].sum()) for g in C.GROUPS},
                     by_group_share={g: float(imp[[c for c in imp.index if grp[c] == g]].sum() / imp.sum()) for g in C.GROUPS})
    imp = pd.Series(r["eval"]["mean_abs_shap"]).sort_values(ascending=False).head(15)[::-1]
    col = {"ASR": "#1f77b4", "DD": "#9467bd", "GEC": "#ff7f0e", "EDIT": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(imp.index, imp.values, color=[col[grp[c]] for c in imp.index])
    for g, c in col.items():
        ax.barh([], [], color=c, label=g)
    ax.legend(loc="lower right", frameon=False); ax.set_xlabel("mean |SHAP| (log-odds), eval edits")
    ax.set_title(f"{a.system}: final LightGBM, top 15 features (exploratory)", fontsize=10)
    fig.tight_layout(); p = f"{FIG}/E3_shap_{a.system}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    r["figure"] = os.path.relpath(p, E); res["E3_shap"] = r


# ---------------- E4 utility
LAMS = [round(x, 2) for x in np.arange(0.5, 5.0001, 0.25)]


def util(y, sh, lam):
    return float(((sh & (y == 0)).sum() - lam * (sh & (y == 1)).sum()) / len(y))


def best_thr(y, s, lam):
    o = np.argsort(s, kind="stable"); ss = s[o]; c = np.cumsum(np.where(y[o] == 0, 1.0, -lam))
    # only cut between distinct scores
    last = np.r_[ss[1:] != ss[:-1], True]; c2 = np.where(last, c, -np.inf)
    i = int(np.argmax(c2))
    return float(ss[i]) if c2[i] > 0 else -np.inf


def e4():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = ev["group"].values; G = np.array(sorted(set(g))); gi = {x: np.where(g == x)[0] for x in G}
    r = {"lambda": LAMS, "show_all": [], "show_none": [0.0] * len(LAMS)}
    pol = {}
    for n in MODELS:
        r[n] = dict(bayes=[], oof_threshold=[], oracle=[], bayes_coverage=[], oof_thr_coverage=[])
    for lam in LAMS:
        r["show_all"].append(util(yE, np.ones(len(yE), bool), lam))
        for n in MODELS:
            sb = SE[n] < 1 / (1 + lam); t = best_thr(yO, SO[n], lam); so = SE[n] <= t; to = best_thr(yE, SE[n], lam)
            r[n]["bayes"].append(util(yE, sb, lam)); r[n]["oof_threshold"].append(util(yE, so, lam))
            r[n]["oracle"].append(util(yE, SE[n] <= to, lam))
            r[n]["bayes_coverage"].append(float(sb.mean())); r[n]["oof_thr_coverage"].append(float(so.mean()))
            pol[(n, lam)] = so
    # bootstrap: risk_lgbm - gec_lr and risk_lgbm - show_all under the OOF threshold
    rng = np.random.RandomState(C.SEED); D1 = {l: [] for l in LAMS}; D2 = {l: [] for l in LAMS}
    for _ in range(a.n_boot):
        idx = np.concatenate([gi[x] for x in rng.choice(G, len(G), replace=True)]); yb = yE[idx]
        for lam in LAMS:
            ul = util(yb, pol[("risk_lgbm", lam)][idx], lam)
            D1[lam].append(ul - util(yb, pol[("gec_lr", lam)][idx], lam)); D2[lam].append(ul - util(yb, np.ones(len(yb), bool), lam))
    r["risk_lgbm_minus_gec_lr_oofthr"] = [dict(lam=l, diff=r["risk_lgbm"]["oof_threshold"][i] - r["gec_lr"]["oof_threshold"][i],
                                              ci95=pc(D1[l]), p=pv(D1[l])) for i, l in enumerate(LAMS)]
    r["risk_lgbm_minus_show_all_oofthr"] = [dict(lam=l, diff=r["risk_lgbm"]["oof_threshold"][i] - r["show_all"][i],
                                                ci95=pc(D2[l]), p=pv(D2[l])) for i, l in enumerate(LAMS)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(LAMS, r["show_all"], "k--", label="show all"); ax.axhline(0, color="grey", lw=0.8, label="show none")
    for n, c in zip(MODELS, ["#d62728", "#1f77b4", "#ff7f0e"]):
        ax.plot(LAMS, r[n]["oof_threshold"], color=c, label=f"{n} (OOF threshold)")
        ax.plot(LAMS, r[n]["bayes"], color=c, ls=":", lw=1)
    ax.set_xlabel("λ (cost of one false edit / benefit of one correct edit)"); ax.set_ylabel("utility per edit (eval)")
    ax.set_title(f"{a.system}: cost-sensitive utility (dotted = Bayes rule; exploratory)", fontsize=10); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); p = f"{FIG}/E4_utility_{a.system}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    r["figure"] = os.path.relpath(p, E); res["E4_utility"] = r


# ---------------- E5 learning curve
def e5():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    td = pd.concat([pd.read_csv(f"{cfg['td']}/features_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
    utd = pd.concat([pd.read_csv(f"{cfg['td']}/utt_counts_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
    N_TD = int(utd["n_ref_edits"].sum()); y = td["false_feedback"].values
    assert len(td) == len(OOF) and (td["false_feedback"].values == yO).all()
    fl = C.folds(td)
    P = {"risk_lgbm": ("lgbm", ALLC, fit["models"]["risk_lgbm"]["params"]), "risk_lr": ("lr", ALLC, add["risk_lr"]["params"]),
         "gec_lr": ("lr", C.GROUPS["GEC"], fit["models"]["gec_lr"]["params"])}
    FR = [0.25, 0.5, 0.75, 1.0]; r = {"fractions": FR, "params": {k: v[2] for k, v in P.items()}, "cv": {}, "eval": {}}
    pick = lambda m: {k: m[k] for k in ["AURC", "P@0.5", "AUROC"]}
    for n, (kind, cols, p) in P.items():
        r["cv"][n] = {}; r["eval"][n] = {}
        for f in FR:
            reps = 1 if f == 1.0 else 5; cvm, evm = [], []
            for rep in range(reps):
                rng = np.random.RandomState(C.SEED + rep)
                oof = np.zeros(len(td))
                for a_, b_ in fl:
                    tr = td.iloc[a_]; gs = np.array(sorted(tr["group"].unique()))
                    keep = set(rng.choice(gs, int(round(f * len(gs))), replace=False)); trs = tr[tr["group"].isin(keep)]
                    oof[b_] = C.fit_predict(kind, cols, p, trs, td.iloc[b_])[0]
                cvm.append(pick(C.metrics(y, oof, N_TD)))
                gs = np.array(sorted(td["group"].unique())); keep = set(rng.choice(gs, int(round(f * len(gs))), replace=False))
                se = C.fit_predict(kind, cols, p, td[td["group"].isin(keep)], ev)[0]
                evm.append(pick(C.metrics(yE, se, N_EV)))
            for tgt, L in [("cv", cvm), ("eval", evm)]:
                r[tgt][n][str(f)] = {k: dict(mean=float(np.mean([m[k] for m in L])), sd=float(np.std([m[k] for m in L], ddof=1)) if len(L) > 1 else 0.0,
                                             reps=[m[k] for m in L]) for k in L[0]}
            print("E5", n, f, r["cv"][n][str(f)]["AURC"]["mean"], r["eval"][n][str(f)]["AURC"]["mean"], flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, k in zip(axs, ["AURC", "P@0.5"]):
        for n, c in zip(MODELS, ["#d62728", "#1f77b4", "#ff7f0e"]):
            for tgt, ls in [("cv", "-"), ("eval", "--")]:
                m = [r[tgt][n][str(f)][k]["mean"] for f in FR]; s = [r[tgt][n][str(f)][k]["sd"] for f in FR]
                ax.errorbar([100 * f for f in FR], m, yerr=s, color=c, ls=ls, marker="o", ms=3, capsize=2, label=f"{n} ({tgt})")
        ax.set_xlabel("% of training submissions"); ax.set_ylabel(k)
    axs[0].legend(fontsize=7, frameon=False); fig.suptitle(f"{a.system}: learning curves (mean ± sd of 5 repeats; exploratory)", fontsize=10)
    fig.tight_layout(); p = f"{FIG}/E5_learning_curve_{a.system}.png"; fig.savefig(p, dpi=200); plt.close(fig)
    r["figure"] = os.path.relpath(p, E); res["E5_learning_curve"] = r


# ---------------- E6 ERRANT type
def e6_block(df, y, S):
    t = df["type"].astype(str).values; vc = pd.Series(t).value_counts()
    if "op" not in df.columns:   # OOF file: derive op / POS from the ERRANT type (checked on eval: type = op + ":" + type_pos)
        assert (ev["op"].astype(str) + ":" + ev["type_pos"].astype(str) == ev["type"].astype(str)).all()
        df = df.assign(op=[x.split(":", 1)[0] for x in t], type_pos=[x.split(":", 1)[1] if ":" in x else x for x in t])
    tt = np.where(pd.Series(t).map(vc).values >= 30, t, "OTHER_SMALL")
    out = {}
    for key, lab in [("type", tt), ("op", df["op"].astype(str).values), ("type_pos", df["type_pos"].astype(str).values)]:
        o = {}
        for v in sorted(set(lab)):
            m = lab == v; rr = dict(n=int(m.sum()), fp_rate=float(y[m].mean()), precision_all=float(1 - y[m].mean()))
            for n in ["risk_lgbm", "gec_lr"]:
                for op, th in OPS[n].items():
                    sh = S[n][m] <= th
                    rr[f"{n}_{op}"] = dict(withheld_share=float(1 - sh.mean()),
                                           precision_shown=float(1 - y[m][sh].mean()) if sh.any() else None, n_shown=int(sh.sum()))
            o[v] = rr
        out[key] = o
    return out


def e6():
    r = {"rule": "full ERRANT types with >= 30 edits in the set, the rest pooled as OTHER_SMALL", "eval": e6_block(ev, yE, SE)}
    if a.system != "W_frozen":
        r["oof_train_dev"] = e6_block(OOF, yO, SO)
    res["E6_errant_type"] = r


# ---------------- E7 timing
def step_secs(chain, ds):
    out = {}
    for l in open(f"{RUNS}/d_queue_status.tsv"):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 5 and f[1] == f"{chain}_{ds}" and f[3] == "done":
            m = re.match(r"(\d+) min (\d+) s", f[4])
            if m:
                out[f[2]] = int(m.group(1)) * 60 + int(m.group(2))
    return out


def e7():
    if a.system == "B1":   # 05_runs/c7_status.tsv (C7 eval run): bart 11:58:21 -> 12:04:46, features -> 12:05:15; pipeline substeps
        st = dict(teacher_forced_scoring=385, c1_features=29, dd=2569, gec=4606, asr="official Whisper-small CTM (not run here)")
    else:
        s = step_secs(cfg["chain"], "eval")
        if cfg["chain"] == "W":
            st = dict(teacher_forced_scoring=s.get("bart"), c1_features=s.get("features"), asr=s.get("asr"), dd=s.get("dd"), gec=s.get("gec"))
        else:
            st = dict(teacher_forced_scoring=s.get("qscore"), c1_features=s.get("features"), gec_llm=s.get("llm"), align=s.get("align"),
                      bart_secondary=s.get("bart2"))
    n = len(ev); nph = int(pd.read_csv(f"{cfg['eval']}/utt_counts_eval.csv").shape[0])
    fe = sum(v for k, v in st.items() if k in ("teacher_forced_scoring", "c1_features") and isinstance(v, (int, float)))
    inf = {}
    X = lgbX(ev)
    for nm, fn in [("risk_lgbm", lambda: BOOST.predict(X)), ("risk_lr", lambda: PK["risk_lr"].predict_proba(ev[ALLC])),
                   ("gec_lr", lambda: PK["gec_lr"].predict_proba(ev[C.GROUPS["GEC"]]))]:
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        inf[nm] = dict(batch_s_median=float(np.median(ts)), per_edit_ms=float(1000 * np.median(ts) / n))
    t0 = time.perf_counter(); score(ev); tot = time.perf_counter() - t0
    res["E7_timing"] = dict(n_eval_edits=n, n_eval_utterances=nph, step_seconds_eval=st,
                            feature_extraction_s_per_edit=fe / n if fe else None,
                            feature_extraction_note="teacher-forced scoring + c1_features over all eval phrases, divided by eval edits; "
                                                    "ASR confidence and DD features come from the pipeline outputs at no extra cost",
                            inference=inf, all_models_scoring_s=tot,
                            conditions="M5 Pro CPU, nice 10, while the stage-D Q queue was running (upper bounds)")


for it in ITEMS:
    t0 = time.time(); {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7}[it]()
    res.setdefault("item_seconds", {})[it] = round(time.time() - t0, 1); save(); print(it, "done", round(time.time() - t0, 1), "s", flush=True)
print(json.dumps(dict(system=a.system, items=ITEMS, out=os.path.relpath(OUT, E), total_s=round(time.time() - T0, 1))))

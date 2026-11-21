#!/usr/bin/env python3
"""Stage C2-C5: risk models, baselines, selective-feedback metrics, speaker(submission)-grouped bootstrap.
Plan: 03_configs/2026-11-10_C_plan.yaml. Run with the eswa-c env:
  python c_model.py cv        --out 05_runs/2026-11-10_C2_cv          (C2+C3: nested grouped CV, OOF scores for all methods)
  python c_model.py evaluate  --out 05_runs/2026-11-10_C2_cv          (C4: metrics, curves, bootstrap, paired tests)
  python c_model.py ablate    --out 05_runs/2026-11-10_C5_ablation    (C5: feature-group ablations, per part)
Numbers only are written. Seeds: SEED for folds, LightGBM, random baseline and bootstrap.
"""
import argparse
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 20261110
EXP = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
F = f"{EXP}/05_runs/2026-11-10_C1_features"
GROUPS = {
    "ASR": ["asr_conf_o_min", "asr_conf_o_mean", "asr_conf_win_min", "asr_conf_win_mean", "asr_conf_ctx2_min",
            "asr_dur_win_min", "asr_dur_o_mean", "utt_asr_conf_mean", "utt_asr_conf_min", "utt_asr_lowconf_frac", "utt_rate_wps"],
    "DD": ["dd_del_inside", "dd_del_adjacent", "dd_del_conf_min", "utt_dd_del_rate"],
    "GEC": ["gec_copy_lp_o_mean", "gec_copy_lp_o_min", "gec_copy_lp_win_min", "gec_lp_c_mean", "gec_lp_c_min",
            "gec_ent_c_mean", "gec_agree_c_all", "gec_lp_next", "gec_ent_next", "gec_lp_ratio",
            "phrase_lp_per_word", "phrase_copy_gap_per_word"],
    "EDIT": ["op", "type_pos", "part", "o_len", "c_len", "n_src_tokens", "rel_pos", "n_edits_utt", "phrase_len_words",
             "edit_at_phrase_edge", "utt_words"],
}
CATS = {"op", "type_pos", "part"}
COVS = [0.3, 0.5, 0.7, 0.9, 1.0]


def load():
    d = pd.concat([pd.read_csv(f"{F}/features_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
    u = pd.concat([pd.read_csv(f"{F}/utt_counts_{ds}.csv") for ds in ["train", "dev"]], ignore_index=True)
    nref = u.groupby("group")["n_ref_edits"].sum()
    return d, nref


def lr_pipe(cols, C):
    num = [c for c in cols if c not in CATS]; cat = [c for c in cols if c in CATS]
    tr = []
    if num:
        tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num))
    if cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat))
    return Pipeline([("pre", ColumnTransformer(tr)), ("lr", LogisticRegression(C=C, max_iter=5000))])


LR_GRID = [dict(C=c) for c in [0.01, 0.1, 1, 10]]
LGB_GRID = [dict(num_leaves=a, n_estimators=b, min_child_samples=c) for a, b, c in itertools.product([7, 15], [100, 300], [20, 50])]


def lgb_fit_predict(Xtr, ytr, Xte, p):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if c in CATS:
            cats = pd.CategoricalDtype(sorted(Xtr[c].astype(str).unique()))
            Xtr[c] = Xtr[c].astype(str).astype(cats); Xte[c] = Xte[c].astype(str).astype(cats)
    m = lgb.LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                           deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1, **p)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def fit_predict(kind, cols, p, tr, te):
    if kind == "lr":
        m = lr_pipe(cols, p["C"]).fit(tr[cols], tr["false_feedback"])
        return m.predict_proba(te[cols])[:, 1], m
    return lgb_fit_predict(tr[cols], tr["false_feedback"], te[cols], p)


def tune(kind, cols, tr):
    grid = LR_GRID if kind == "lr" else LGB_GRID
    inner = GroupKFold(n_splits=3)
    best, best_auc = None, -1
    for p in grid:
        oof = np.zeros(len(tr))
        for a, b in inner.split(tr, tr["false_feedback"], tr["group"]):
            oof[b] = fit_predict(kind, cols, p, tr.iloc[a], tr.iloc[b])[0]
        auc = roc_auc_score(tr["false_feedback"], oof)
        if auc > best_auc:
            best, best_auc = p, auc
    return best, best_auc


def folds(d):
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(d, d["false_feedback"], d["group"]))


def cv_scores(d, kind, cols, fl):
    oof = np.zeros(len(d)); chosen = []
    for k, (a, b) in enumerate(fl):
        tr, te = d.iloc[a], d.iloc[b]
        p, inner_auc = tune(kind, cols, tr)
        oof[b] = fit_predict(kind, cols, p, tr, te)[0]
        chosen.append(dict(fold=k, params=p, inner_auc=round(inner_auc, 4)))
    return oof, chosen


def type_prior(d, fl):
    oof = np.zeros(len(d))
    for a, b in fl:
        tr = d.iloc[a]; mu = tr["false_feedback"].mean()
        g = tr.groupby("type")["false_feedback"].agg(["sum", "count"])
        rate = (g["sum"] + mu) / (g["count"] + 1)
        oof[b] = d.iloc[b]["type"].map(rate).fillna(mu).values
    return oof


# ---------------- metrics ----------------
def curve(y, s, n_ref):
    """Edits sorted by ascending risk (ties broken by a fixed random permutation, seed SEED)."""
    rng = np.random.RandomState(SEED)
    tieb = rng.permutation(len(s))
    order = np.lexsort((tieb, s))
    yy = 1 - y[order]                              # 1 = TP
    k = np.arange(1, len(y) + 1)
    tp = np.cumsum(yy); P = tp / k; R = tp / n_ref
    F = np.where(P + R > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0.0)
    return k / len(y), P, R, F


def metrics(y, s, n_ref):
    cov, P, R, F = curve(y, s, n_ref)
    out = dict(AURC=float(np.mean(1 - P)), AUROC=float(roc_auc_score(y, s)), AUPRC=float(average_precision_score(y, s)),
               maxF05=float(F.max()), cov_at_maxF05=float(cov[F.argmax()]))
    for c in COVS:
        i = max(int(np.ceil(c * len(y))) - 1, 0)
        out[f"P@{c}"] = float(P[i]); out[f"F05@{c}"] = float(F[i]); out[f"R@{c}"] = float(R[i]); out[f"FFR@{c}"] = float(1 - P[i])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--kind", default="lgbm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, nref = load()
    y = d["false_feedback"].values
    fl = folds(d)
    if a.stage == "cv":
        allc = sum(GROUPS.values(), [])
        S, chosen = {}, {}
        d0 = d.copy()
        d0["asr_conf_win_min_f"] = d0["asr_conf_win_min"].fillna(d0["asr_conf_win_min"].median())
        S["asr_conf"] = 1 - d0["asr_conf_win_min_f"].values
        S["random"] = np.random.RandomState(SEED).rand(len(d))
        S["type_prior"] = type_prior(d, fl)
        for name, kind, cols in [("asr_lr", "lr", GROUPS["ASR"]), ("gec_lr", "lr", GROUPS["GEC"]),
                                 ("risk_lr", "lr", allc), ("risk_lgbm", "lgbm", allc)]:
            S[name], chosen[name] = cv_scores(d, kind, cols, fl)
            print(name, "OOF AUROC", round(roc_auc_score(y, S[name]), 4), flush=True)
        oof = d[["ds", "utt_id", "edit_idx", "group", "false_feedback", "asr_induced", "dd_induced", "gec_induced", "asr_induced_lit", "part", "type"]].copy()
        for k, v in S.items():
            oof[f"s_{k}"] = v
        oof["fold"] = -1
        for k, (_, b) in enumerate(fl):
            oof.loc[oof.index[b], "fold"] = k
        oof.to_csv(f"{a.out}/oof_scores.csv", index=False)
        json.dump(dict(seed=SEED, n_edits=len(d), n_groups=int(d["group"].nunique()), n_ref=int(nref.sum()),
                       fp_rate=float(y.mean()), fold_sizes=[len(b) for _, b in fl],
                       fold_fp_rates=[round(float(y[b].mean()), 4) for _, b in fl], chosen_params=chosen),
                  open(f"{a.out}/cv_info.json", "w"), indent=1)
    elif a.stage == "evaluate":
        oof = pd.read_csv(f"{a.out}/oof_scores.csv")
        y = oof["false_feedback"].values
        names = [c[2:] for c in oof.columns if c.startswith("s_")]
        N = int(nref.sum())
        point = {n: metrics(y, oof[f"s_{n}"].values, N) for n in names}
        best = max(["risk_lr", "risk_lgbm"], key=lambda n: point[n]["AUROC"])
        curves = {}
        for n in names:
            cov, P, R, Fv = curve(y, oof[f"s_{n}"].values, N)
            idx = np.unique(np.linspace(0, len(y) - 1, 201).astype(int))
            curves[n] = dict(coverage=cov[idx].round(4).tolist(), precision=P[idx].round(4).tolist(), F05=Fv[idx].round(4).tolist())
        # bootstrap over groups (all groups that have reference edits or system edits)
        groups = np.array(sorted(set(nref.index) | set(oof["group"])))
        gi = {g: oof.index[oof["group"] == g].values for g in groups}
        rng = np.random.RandomState(SEED)
        keys = ["AURC", "P@0.5", "F05@0.5", "maxF05", "AUROC", "AUPRC", "P@0.3", "P@0.7", "P@0.9", "F05@0.3", "F05@0.7", "F05@0.9",
                "FFR@0.3", "FFR@0.5", "FFR@0.7", "FFR@0.9"]
        B = {n: {k: [] for k in keys} for n in names}
        for b in range(a.n_boot):
            gs = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([gi[g] for g in gs])
            nr = int(sum(nref.get(g, 0) for g in gs))
            yb = y[idx]
            for n in names:
                m = metrics(yb, oof[f"s_{n}"].values[idx], nr)
                for k in keys:
                    B[n][k].append(m[k])
        ci = {n: {k: [float(np.percentile(B[n][k], 2.5)), float(np.percentile(B[n][k], 97.5))] for k in keys} for n in names}
        tests = {}
        # planned comparison: vs asr_conf (03_configs/2026-11-10_C_plan.yaml); vs asr_lr and gec_lr: UNPLANNED (added 2026-11-10 on user request)
        for ref_name in ["asr_conf", "asr_lr", "gec_lr"]:
            for cand in ["risk_lr", "risk_lgbm"]:
                t = {}
                for k in keys:
                    diff = np.array(B[cand][k]) - np.array(B[ref_name][k])
                    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
                    t[k] = dict(diff=point[cand][k] - point[ref_name][k], ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                p=float(min(p, 1.0)))
                tests[f"{cand}_vs_{ref_name}"] = dict(planned=(ref_name == "asr_conf"), **t)
        json.dump(dict(n_boot=a.n_boot, seed=SEED, n_groups=len(groups), n_ref=N, chosen_risk_model=best, point=point, ci95=ci,
                       paired_tests=tests), open(f"{a.out}/evaluation.json", "w"), indent=1)
        json.dump(curves, open(f"{a.out}/coverage_curves.json", "w"))
        print(json.dumps(dict(chosen=best, point={n: {k: round(v, 4) for k, v in point[n].items() if k in keys} for n in names}), indent=0))
    elif a.stage == "ablate":
        allc = sum(GROUPS.values(), [])
        res = {}
        configs = [("all", allc)] + [(f"-{g}", [c for c in allc if c not in GROUPS[g]]) for g in GROUPS] + \
                  [(f"only_{g}", GROUPS[g]) for g in GROUPS]
        kind = a.kind
        N = int(nref.sum())
        oof_all = {}
        for name, cols in configs:
            s, _ = cv_scores(d, kind, cols, fl)
            oof_all[name] = s
            res[name] = {k: round(v, 4) for k, v in metrics(y, s, N).items()}
            print(name, res[name]["AUROC"], res[name]["AURC"], res[name]["P@0.5"], flush=True)
        pd.DataFrame(oof_all).assign(group=d["group"], part=d["part"], false_feedback=y).to_csv(f"{a.out}/ablation_oof_{kind}.csv", index=False)
        json.dump(dict(kind=kind, results=res), open(f"{a.out}/ablation_{kind}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""UNPLANNED (added 2026-11-10): paired grouped bootstrap (1000, seed 20261110) of LightGBM 'all' vs each drop-one ablation,
on the ablation OOF scores. Usage: python -I c5_ablation_boot.py > 05_runs/2026-11-10_C5_ablation/ablation_boot_lgbm.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import c_model as C
E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
o = pd.read_csv(f"{E}/05_runs/2026-11-10_C5_ablation/ablation_oof_lgbm.csv")
_, nref = C.load()
y = o["false_feedback"].values
groups = np.array(sorted(set(nref.index) | set(o["group"])))
gi = {g: o.index[o["group"] == g].values for g in groups}
rng = np.random.RandomState(C.SEED); K = ["AURC", "P@0.5", "FFR@0.5", "P@0.7", "AUROC", "F05@0.5"]
cmp = ["-ASR", "-DD", "-GEC", "-EDIT"]; D = {c: {k: [] for k in K} for c in cmp}
for b in range(1000):
    gs = rng.choice(groups, size=len(groups), replace=True); idx = np.concatenate([gi[g] for g in gs]); nr = int(sum(nref.get(g, 0) for g in gs))
    ma = C.metrics(y[idx], o["all"].values[idx], nr)
    for c in cmp:
        mc = C.metrics(y[idx], o[c].values[idx], nr)
        for k in K: D[c][k].append(ma[k] - mc[k])
N = int(nref.sum()); pa = C.metrics(y, o["all"].values, N)
out = {c: {k: dict(diff_all_minus_ablation=round(pa[k] - C.metrics(y, o[c].values, N)[k], 4),
                   ci95=[round(float(np.percentile(D[c][k], 2.5)), 4), round(float(np.percentile(D[c][k], 97.5)), 4)],
                   p=round(float(min(1, 2 * min((np.array(D[c][k]) <= 0).mean(), (np.array(D[c][k]) >= 0).mean()))), 4)) for k in K} for c in cmp}
print(json.dumps(dict(UNPLANNED=True, n_boot=1000, seed=C.SEED, results=out), indent=1))

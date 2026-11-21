"""F1 analysis: run after annotation. Usage: python3 f_human_eval.py [--re]  (writes 06_results/F_human_eval.json)
Stratified design (2x2: auto label x risk-model decision at cov0.5), estimates weighted by population/sample per cell.
'?' items are excluded from the main analysis and counted separately."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
H = os.path.expanduser("$EXP_DIR") if os.path.isdir(os.path.expanduser("$EXP_DIR")) else os.path.expanduser("~/mnt/EXP_DIR")
V = f"{H}/11_human_validation"; B, SEED = 1000, 20261110
def read_sheet(p, idcol):
    x = pd.read_excel(p, sheet_name="标注", dtype=str); x = x.iloc[:, [0, 5]]; x.columns = [idcol, "h"]
    x["h"] = x.h.fillna("").str.strip(); return x
key = pd.read_csv(f"{V}/key_DO_NOT_OPEN.csv"); samp = json.load(open(f"{V}/sampling.json"))
pop = samp["population_per_cell"]
a = key.merge(read_sheet(f"{V}/annotation_sheet.xlsx", "item_id"), on="item_id")
out = {"n_items": len(a), "n_blank": int((a.h == "").sum()), "n_unsure": int((a.h == "?").sum())}
assert out["n_blank"] == 0, "annotation incomplete"
m = a[a.h.isin(["0", "1"])].copy(); m["h_ff"] = 1 - m.h.astype(int)   # human false feedback
nc = m.cell.value_counts(); m["w"] = m.cell.map(lambda c: pop[c] / nc[c])
def stats(df):
    w = df.w; r = {}
    r["human_ffr_weighted"] = float((w * df.h_ff).sum() / w.sum())
    r["auto_ffr_weighted"] = float((w * df.false_feedback).sum() / w.sum())
    r["agreement_weighted"] = float((w * (df.h_ff == df.false_feedback)).sum() / w.sum())
    for mdl, col in [("risk", "s_risk_lgbm"), ("gec_only", "s_gec_lr")]:
        r[f"auroc_{mdl}_human"] = float(roc_auc_score(df.h_ff, df[col], sample_weight=w)) if df.h_ff.nunique() == 2 else np.nan
    for mdl, col in [("risk", "kept_risk"), ("gec_only", "kept_gec")]:
        k = df[col] == 1; r[f"prec_kept_{mdl}_human"] = float((w[k] * (1 - df.h_ff[k])).sum() / w[k].sum())
    r["d_auroc"] = r["auroc_risk_human"] - r["auroc_gec_only_human"]; r["d_prec_kept"] = r["prec_kept_risk_human"] - r["prec_kept_gec_only_human"]
    return r
pt = stats(m); rng = np.random.default_rng(SEED); bs = []
for _ in range(B):
    bs.append(stats(pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in m.groupby("cell")])))
bs = pd.DataFrame(bs)
out["estimates"] = {k: {"est": round(v, 4), "ci95": [round(float(bs[k].quantile(.025)), 4), round(float(bs[k].quantile(.975)), 4)]} for k, v in pt.items()}
for d_ in ["d_auroc", "d_prec_kept"]:
    out["estimates"][d_]["p_two_sided"] = float(min(1, 2 * min((bs[d_] <= 0).mean(), (bs[d_] >= 0).mean())))
out["per_cell"] = {c: {"n": int(len(g)), "human_ff_rate": round(float(g.h_ff.mean()), 3), "agree_with_auto": round(float((g.h_ff == g.false_feedback).mean()), 3)} for c, g in m.groupby("cell")}
out["confusion_unweighted"] = pd.crosstab(m.false_feedback, m.h_ff).rename_axis(index="auto_ff", columns="human_ff").to_dict()
out["auto_ff_by_cause"] = {c: {"n": int(len(g)), "human_confirms_ff": round(float(g.h_ff.mean()), 3)} for c, g in m[m.false_feedback == 1].groupby("cause")}
out["by_type_top"] = m[m.false_feedback == 1].groupby("type").h_ff.agg(["size", "mean"]).sort_values("size", ascending=False).head(10).round(3).to_dict("index")
if "--re" in sys.argv and os.path.exists(f"{V}/reannotation_50.xlsx"):
    rk = pd.read_csv(f"{V}/key_reannotation_DO_NOT_OPEN.csv"); r2 = read_sheet(f"{V}/reannotation_50.xlsx", "re_id").merge(rk, on="re_id")
    j = r2.merge(a[["item_id", "h"]], on="item_id", suffixes=("_2", "_1")); jj = j[j.h_1.isin(["0", "1"]) & j.h_2.isin(["0", "1"])]
    out["intra_rater"] = {"n": int(len(jj)), "agreement": round(float((jj.h_1 == jj.h_2).mean()), 3), "cohen_kappa": round(float(cohen_kappa_score(jj.h_1, jj.h_2)), 3)}
if "--inter" in sys.argv and os.path.exists(f"{V}/annotator2_50.xlsx"):
    rk = pd.read_csv(f"{V}/key_reannotation_DO_NOT_OPEN.csv"); r2 = read_sheet(f"{V}/annotator2_50.xlsx", "re_id").merge(rk, on="re_id")
    j = r2.merge(a[["item_id", "h"]], on="item_id", suffixes=("_2", "_1")); jj = j[j.h_1.isin(["0", "1"]) & j.h_2.isin(["0", "1"])]
    out["inter_rater"] = {"n": int(len(jj)), "n_unsure_annotator2": int((j.h_2 == "?").sum()), "agreement": round(float((jj.h_1 == jj.h_2).mean()), 3),
                          "cohen_kappa": round(float(cohen_kappa_score(jj.h_1, jj.h_2)), 3)}
if "--inter2" in sys.argv and os.path.exists(f"{V}/annotator2_round2_50.xlsx"):
    rk = pd.read_csv(f"{V}/key_round2_DO_NOT_OPEN.csv"); r2 = read_sheet(f"{V}/annotator2_round2_50.xlsx", "s_id").merge(rk, on="s_id")
    j = r2.merge(a[["item_id", "h"]], on="item_id", suffixes=("_2", "_1")); jj = j[j.h_1.isin(["0", "1"]) & j.h_2.isin(["0", "1"])]
    assert (j.h_2 == "").sum() == 0, "round 2 incomplete"
    out["inter_rater_round2"] = {"n": int(len(jj)), "n_unsure_annotator2": int((j.h_2 == "?").sum()), "agreement": round(float((jj.h_1 == jj.h_2).mean()), 3),
                                 "cohen_kappa": round(float(cohen_kappa_score(jj.h_1, jj.h_2)), 3)}
if "--inter" in sys.argv and "--inter2" in sys.argv:
    # annotator B over both rounds (100 items): weighted inappropriate rate, confirmations, model comparison; round-2 kappa CI
    def rdB(path, keyf, idc):
        x = read_sheet(path, idc).merge(pd.read_csv(keyf).rename(columns={idc: idc}), on=idc); return x[["item_id", "h"]]
    B = pd.concat([rdB(f"{V}/annotator2_50.xlsx", f"{V}/key_reannotation_DO_NOT_OPEN.csv", "re_id"),
                   rdB(f"{V}/annotator2_round2_50.xlsx", f"{V}/key_round2_DO_NOT_OPEN.csv", "s_id")])
    jb = key.merge(B, on="item_id").merge(a[["item_id", "h"]], on="item_id", suffixes=("_B", "_A"))
    jb = jb[jb.h_B.isin(["0", "1"])].copy(); jb["h_ff"] = 1 - jb.h_B.astype(int)
    nb = jb.cell.value_counts(); jb["w"] = jb.cell.map(lambda c: pop[c] / nb[c])
    ptB = stats(jb); rngB = np.random.default_rng(SEED)
    bsB = pd.DataFrame([stats(pd.concat([g.sample(len(g), replace=True, random_state=int(rngB.integers(1e9))) for _, g in jb.groupby("cell")])) for _ in range(1000)])
    out["annotator_B"] = {"n": int(len(jb)), "per_cell_n": {c: int(v) for c, v in nb.items()},
        "estimates": {k: {"est": round(v, 4), "ci95": [round(float(bsB[k].quantile(.025)), 4), round(float(bsB[k].quantile(.975)), 4)]} for k, v in ptB.items()},
        "auto_ff_by_cause": {c: {"n": int(len(g)), "B_confirms_ff": round(float(g.h_ff.mean()), 3), "A_confirms_ff": round(float((g.h_A == "0").mean()), 3)}
                             for c, g in jb[jb.false_feedback == 1].groupby("cause")},
        "B_flags_not_flagged_by_A": int(((jb.h_B == "0") & (jb.h_A == "1")).sum()), "B_flags_total": int((jb.h_B == "0").sum())}
    rk2 = pd.read_csv(f"{V}/key_round2_DO_NOT_OPEN.csv"); r2 = read_sheet(f"{V}/annotator2_round2_50.xlsx", "s_id").merge(rk2, on="s_id")
    j2 = r2.merge(a[["item_id", "h"]], on="item_id", suffixes=("_2", "_1")); rng2 = np.random.default_rng(SEED)
    ks = [cohen_kappa_score(*j2.sample(len(j2), replace=True, random_state=int(rng2.integers(1e9)))[["h_1", "h_2"]].T.values) for _ in range(1000)]
    out["inter_rater_round2"]["kappa_ci95"] = [round(float(np.nanpercentile(ks, 2.5)), 3), round(float(np.nanpercentile(ks, 97.5)), 3)]
out["note"] = ("Annotator A (author, language background): 200 stratified eval edits (seed 20261112). Annotator B (architecture PhD, non-author): "
               "round 1 = 50 items (annotator2_50), round 2 = 50 new items after rule clarification (annotator2_round2_50). Post hoc pre-submission analysis.")
json.dump(out, open(f"{H}/06_results/F_human_eval.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(out["estimates"], indent=1))

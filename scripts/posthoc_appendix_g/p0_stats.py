"""Phase 0 structural stats only (counts, NaN shares, ranges). No corpus text printed."""
import json, os, sys
import numpy as np, pandas as pd
R = os.environ["EXP_DIR"] + "/05_runs"
P = {"train": f"{R}/2026-11-10_C1_features/features_train.csv", "dev": f"{R}/2026-11-10_C1_features/features_dev.csv",
     "test": f"{R}/2026-11-10_C7_B1_eval/features/features_eval.csv"}
F6 = ["gec_lp_c_mean", "gec_lp_c_min", "asr_conf_o_mean", "asr_conf_o_min", "asr_conf_win_mean", "asr_conf_win_min"]
out = {}
for ds, p in P.items():
    d = pd.read_csv(p)
    r = dict(n=len(d), unique_ids=bool(not d.duplicated(["utt_id", "edit_idx"]).any()), op_counts=d["op"].value_counts().to_dict(),
             fp_rate=round(float(d["false_feedback"].mean()), 4))
    r["nan_share_by_op"] = {op: {c: round(float(g[c].isna().mean()), 4) for c in F6} for op, g in d.groupby("op")}
    r["nan_share_all"] = {c: round(float(d[c].isna().mean()), 4) for c in F6}
    r["ranges"] = {c: [round(float(d[c].min()), 4), round(float(d[c].max()), 4)] for c in F6}
    r["o_len0_by_op"] = {op: int((g["o_len"] == 0).sum()) for op, g in d.groupby("op")}
    r["c_len0_by_op"] = {op: int((g["c_len"] == 0).sum()) for op, g in d.groupby("op")}
    r["gec_lp_c_mean_le_min_violations"] = int((d["gec_lp_c_mean"] < d["gec_lp_c_min"] - 1e-9).sum())
    r["asr_o_nan_given_olen_gt0"] = int((d["asr_conf_o_mean"].isna() & (d["o_len"] > 0)).sum())
    r["gec_c_nan_given_clen_gt0"] = int((d["gec_lp_c_mean"].isna() & (d["c_len"] > 0)).sum())
    r["both_win_and_gec_nan"] = int((d["asr_conf_win_mean"].isna() & d["gec_lp_c_mean"].isna()).sum())
    out[ds] = r
ev = json.load(open(f"{R}/2026-11-10_C2_cv/evaluation.json"))
out["C2_chosen_risk_model"] = ev["chosen_risk_model"]
print(json.dumps(out, indent=1))

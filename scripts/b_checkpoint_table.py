#!/usr/bin/env python3
"""B checkpoint: one table per dataset with the three conditions (numbers only) -> 06_results/b_checkpoint_summary[SUFFIX].csv/.md
Usage: python3 -I b_checkpoint_table.py [--v4]   (--v4: label_stats_v4, B2b v2 runs, suffix _v4)"""
import os
import sys
V4 = "--v4" in sys.argv
SUF = "_v4" if V4 else ""
RUN = lambda c, ds: f"2026-11-10_{c}_{ds}" + ("_v2" if V4 and c == "B2b" else "")
import csv
import json
from pathlib import Path

EXP = Path(os.environ.get("EXP_DIR") or sys.exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
rows = []
for ds in ["dev", "train"]:
    st = list(csv.DictReader(open(EXP / f"06_results/label_stats{SUF}_{ds}.csv")))
    for cond in ["B1", "B2b", "B2"]:
        m = json.load(open(EXP / f"05_runs/{RUN(cond, ds)}/metrics.json"))
        for d in ["primary", "literal"]:
            o = [x for x in st if x["condition"] == cond and x["asr_err_definition"] == d and x["grouping"] == "overall"][0]
            rows.append(dict(dataset=ds, condition=cond, label_definition=d, gec_wer=m["gec_wer"]["err"],
                             P=m["sgecf_overall"]["P"], R=m["sgecf_overall"]["R"], F05=m["sgecf_overall"]["F05"],
                             n_hyp_edits=o["n_hyp_edits"], n_tp=o["n_tp"], n_fp=o["n_fp"], fp_rate=o["fp_rate"],
                             fp_asr=o["n_asr_induced"], fp_dd=o["n_dd_induced"], fp_gec=o["n_gec_induced"], fp_mixed=o["n_mixed"],
                             fp_asr_share=o["asr_share_of_fp"], fp_dd_share=o["dd_share_of_fp"], fp_gec_share=o["gec_share_of_fp"],
                             n_ref_edits=o["n_ref_edits"], n_missed=o["n_missed"], missed_asr=o["n_missed_asr"],
                             missed_dd=o["n_missed_dd"], missed_gec=o["n_missed_gec"]))
with open(EXP / f"06_results/b_checkpoint_summary{SUF}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
keys = list(rows[0])
md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
(EXP / f"06_results/b_checkpoint_summary{SUF}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

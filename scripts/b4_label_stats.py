#!/usr/bin/env python3
"""B4: label statistics -> 06_results/label_stats_{ds}.csv

One row per (condition, asr_err_definition, grouping, group).
Conditions: B1 (Whisper ASR + DD + GEC), B2b (manual verbatim + DD + GEC), B2 (manual fluent + GEC).
asr_err_definition:
  primary : ASR omission of disfluencies that the annotators also removed is NOT an ASR error (labels v3, main analysis)
  literal : any mismatch between pre-DD ASR tokens and the manual verbatim tokens (sensitivity analysis, *_lit columns)
Groupings: overall, part (P1/P3/P4/P5), op (M/R/U), errant_type.
Columns: n_hyp_edits, n_tp, n_fp, fp_rate (= share of system edits that are false feedback), n_asr_induced, n_dd_induced,
n_gec_induced, n_mixed (subset of asr_induced), n_disfl_only_fp (FP whose window has only disfluency-omission "ASR errors"),
asr/dd/gec shares of FP; n_ref_edits, n_missed, n_missed_asr/dd/gec, n_missed_mixed, n_disfl_only_missed, shares of missed.
Usage: python3 -I b4_label_stats.py --ds dev --cond B1=<labels_dir> --cond B2b=<labels_dir> --cond B2=<labels_dir> --out <csv>
"""
import argparse
import csv
from collections import defaultdict

LABELS = ("asr_induced", "dd_induced", "gec_induced", "mixed", "missed_asr", "missed_dd", "missed_gec")

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--cond", action="append", required=True, help="NAME=labels_dir")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def s(rows, k):
    return sum(int(x[k]) for x in rows)


def sh(n, d):
    return round(n / d, 4) if d else ""


out = []
for c in a.cond:
    name, d = c.split("=", 1)
    hyp = list(csv.DictReader(open(f"{d}/hyp_edit_labels_{a.ds}.csv")))
    ref = list(csv.DictReader(open(f"{d}/ref_edit_labels_{a.ds}.csv")))
    for definition, suf in [("primary", ""), ("literal", "_lit")]:
        L = lambda k: k + suf if k in LABELS else k
        for grouping, key in [("overall", lambda x: "all"), ("part", lambda x: x["part"]),
                              ("op", lambda x: x["op"]), ("errant_type", lambda x: x["type"])]:
            H, R = defaultdict(list), defaultdict(list)
            for x in hyp:
                H[key(x)].append(x)
            for x in ref:
                R[key(x)].append(x)
            for g in sorted(set(H) | set(R)):
                h, r = H.get(g, []), R.get(g, [])
                fp_rows = [x for x in h if x["false_feedback"] == "1"]
                ms_rows = [x for x in r if x["missed"] == "1"]
                nfp, nm = len(fp_rows), len(ms_rows)
                na, nd, ng = s(h, L("asr_induced")), s(h, L("dd_induced")), s(h, L("gec_induced"))
                ma, md, mg = s(r, L("missed_asr")), s(r, L("missed_dd")), s(r, L("missed_gec"))
                out.append(dict(dataset=a.ds, condition=name, asr_err_definition=definition, grouping=grouping, group=g,
                                n_hyp_edits=len(h), n_tp=len(h) - nfp, n_fp=nfp, fp_rate=sh(nfp, len(h)),
                                n_asr_induced=na, n_dd_induced=nd, n_gec_induced=ng, n_mixed=s(fp_rows, L("mixed")),
                                n_disfl_only_fp=s(fp_rows, "asr_err_disfl_only"),
                                asr_share_of_fp=sh(na, nfp), dd_share_of_fp=sh(nd, nfp), gec_share_of_fp=sh(ng, nfp),
                                n_ref_edits=len(r), n_missed=nm, n_missed_asr=ma, n_missed_dd=md, n_missed_gec=mg,
                                n_missed_mixed=s(ms_rows, L("mixed")), n_disfl_only_missed=s(ms_rows, "asr_err_disfl_only"),
                                asr_share_of_missed=sh(ma, nm), dd_share_of_missed=sh(md, nm), gec_share_of_missed=sh(mg, nm)))
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"wrote {len(out)} rows to {a.out}")

#!/usr/bin/env python3
"""Aggregate summaries for results/ (numbers only; no corpus text, no file/utterance IDs). Added 2026-11-12 after the
reproduction audit (docs/AUDIT_REPORT.md, defects 7, 8 and the Figure 4 data). Reads existing run outputs; recomputes nothing
that was not already computed, except plain means and ratios of stored values.
Writes into $EXP_DIR/06_results/:
  system_metrics_summary.json      ASR WER (official flow), GEC WER (sclite 1-decimal and exact from raw counts), SGECF P/R/F0.5,
                                   hypothesis edits / TP / FP per system (B1, W, Q) and set (dev, train, eval; Q also dev-subset)
  qwen_score_summary.json          Qwen teacher-forced score means per set (+ Q generation format-error rate)
  C2_cv_evaluation.json, C2_cv_cv_info.json, C2_cv_coverage_curves.json   copies of the stage C2 cross-validation outputs (OOF)
  fig4_precision_at_coverage_eval.csv   data of manuscript Figure 4 (held-out precision of shown feedback vs coverage, B1)
Usage: EXP_DIR=... RUN_DIR=... python3 -I pkg_results_summary.py"""
import csv, json, os, shutil, statistics

E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
RUN = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))
RUNS, RES = f"{E}/05_runs", f"{E}/06_results"


def raw_sum(p):  # sclite *.raw "Sum" line: n_snt n_wrd corr sub del ins err s_err nce  (counts)
    if not os.path.exists(p): return None
    for l in open(p):
        f = l.replace("|", " ").split()
        if f and f[0] == "Sum":
            n_snt, n_wrd, corr, sub, dl, ins, err = (int(x) for x in f[1:8])
            pct = lambda x: round(100 * x / n_wrd, 3)
            return {"n_snt": n_snt, "n_wrd": n_wrd, "err_exact": pct(err), "sub_exact": pct(sub), "del_exact": pct(dl), "ins_exact": pct(ins)}
    return None


def labels(d):
    p = f"{d}/labels_v4/b3_verification.json"
    if not os.path.exists(p): return None
    o = json.load(open(p))["overall_from_labels"]
    return {"n_hyp_edits": o["tp"] + o["fp"], "tp": o["tp"], "fp": o["fp"], "fp_rate": round(o["fp"] / (o["tp"] + o["fp"]), 4)}


RUNDIRS = {("B1", "dev"): ("2026-11-10_B1_dev", "stageB/B1_dev"), ("B1", "train"): ("2026-11-10_B1_train", "stageB/B1_train"),
           ("B1", "eval"): ("2026-11-10_C7_B1_eval", "stageB/B1_eval"),
           ("Q", "dev-subset"): ("D_qrehearsal/2026-11-10_D_Q_dev-subset", "stageD_qrehearsal/Q_dev-subset")}
for s in ("W", "Q"):
    for ds in ("dev", "train", "eval"): RUNDIRS[(s, ds)] = (f"2026-11-10_D_{s}_{ds}", f"stageD/{s}_{ds}")
out = {"note": "Numbers only, aggregated from existing outputs by scripts/pkg_results_summary.py. gec_wer = sclite Sum/Avg "
               "(1 decimal, as reported by the official scorer); *_exact = recomputed from the sclite raw counts. ASR WER: official "
               "flow (run-asr-to-task-input --task asr + run-wer-scoring) on the decoded gec-subset files; 'small' = official "
               "Whisper-small ASR (B1), 'large_v3' = W. Labels: v4 primary (labels_v4/b3_verification.json).", "systems": {}}
for (s, ds), (rd, sd) in sorted(RUNDIRS.items()):
    m = f"{RUNS}/{rd}/metrics.json"
    if not os.path.exists(m): continue
    M = json.load(open(m)); g = M["gec_wer"]; o = M["sgecf_overall"]
    r = {"gec_wer": g["err"], "gec_sub": g["sub"], "gec_del": g["del"], "gec_ins": g["ins"],
         "gec_wer_raw": raw_sum(f"{RUN}/{sd}/gec/wer-scoring/hyp/gec.ctm.filt.raw"),
         "sgecf_P": o["P"], "sgecf_R": o["R"], "sgecf_F05": o["F05"], "labels_v4": labels(f"{RUNS}/{rd}")}
    if s == "W":
        r["asr_wer_large_v3_raw"] = raw_sum(f"{RUN}/{sd}/asr_wer/W/wer/hyp/asr.ctm.filt.raw")
        r["asr_wer_small_same_subset_raw"] = raw_sum(f"{RUN}/{sd}/asr_wer/small/wer/hyp/asr.ctm.filt.raw")
    out["systems"].setdefault(s, {})[ds] = r
json.dump(out, open(f"{RES}/system_metrics_summary.json", "w"), indent=1)

# Qwen teacher-forced score means (phrase-level: seq_lp, seq_copy_lp, eos_lp; word-level: t_lp, t_ent, s_lp, t_agree)
q = {"note": "Means over phrases (seq_*, eos) and over target words (t_*, s_lp, t_agree) of the Qwen2.5-7B-Instruct teacher-forced "
             "scores (scripts/d_qwen_score.py); format errors from the generation summary (d_llm_gec.py llm_summary.json).", "sets": {}}
for ds, rd, sd in [("dev-subset", "D_qrehearsal/2026-11-10_D_Q_dev-subset", "stageD_qrehearsal/Q_dev-subset"),
                   ("dev", "2026-11-10_D_Q_dev", "stageD/Q_dev"), ("train", "2026-11-10_D_Q_train", "stageD/Q_train"),
                   ("eval", "2026-11-10_D_Q_eval", "stageD/Q_eval")]:
    p = f"{RUNS}/{rd}/features/bart_scores_{ds}.jsonl"
    if not os.path.exists(p): continue
    R = [json.loads(l) for l in open(p)]
    W = lambda k: [x for r in R for x in r[k]]
    d = {"phrases": len(R), "words": len(W("t_lp")),
         "mean_seq_lp": round(statistics.fmean(r["seq_lp"] for r in R), 6), "mean_seq_copy_lp": round(statistics.fmean(r["seq_copy_lp"] for r in R), 6),
         "mean_eos_lp": round(statistics.fmean(r["eos_lp"] for r in R), 6), "mean_word_t_lp": round(statistics.fmean(W("t_lp")), 6),
         "mean_word_t_ent": round(statistics.fmean(W("t_ent")), 6), "mean_word_s_lp": round(statistics.fmean(W("s_lp")), 6),
         "word_t_agree": round(statistics.fmean(W("t_agree")), 6)}
    ls = f"{RUN}/{sd}/llm/llm_summary.json"
    if os.path.exists(ls):
        L = json.load(open(ls)); d["generation"] = {k: L[k] for k in L if k in ("phrases", "format_errors", "format_error_rate", "empty_fallback", "truncated")}
    q["sets"][ds] = d
json.dump(q, open(f"{RES}/qwen_score_summary.json", "w"), indent=1)

for n in ("evaluation.json", "cv_info.json", "coverage_curves.json"):
    shutil.copy2(f"{RUNS}/2026-11-10_C2_cv/{n}", f"{RES}/C2_cv_{n}")

# Figure 4: held-out (B1 eval) precision of shown feedback at coverage 0.3/0.5/0.7/0.9 (point estimates and 95% CIs from
# eval_results.json) and at coverage 1.0 (all edits shown: 1 - fp_rate, identical for every model)
ER = json.load(open(f"{RES}/eval_results.json"))
with open(f"{RES}/fig4_precision_at_coverage_eval.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "coverage", "precision", "ci95_low", "ci95_high", "source"])
    for mdl, pt in ER["point"].items():
        for c in ("0.3", "0.5", "0.7", "0.9"):
            ci = ER["ci95"].get(mdl, {}).get(f"P@{c}", ["", ""])
            w.writerow([mdl, c, pt[f"P@{c}"], ci[0], ci[1], f"eval_results.json point/ci95 P@{c}"])
        w.writerow([mdl, "1.0", 1 - ER["fp_rate"], "", "", "1 - eval_results.json fp_rate"])
print("written: system_metrics_summary.json qwen_score_summary.json C2_cv_*.json fig4_precision_at_coverage_eval.csv")

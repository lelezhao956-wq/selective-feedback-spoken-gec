# Corpus licence and corpus-derived files in this package

## The licence requirement
This package was produced with the **Speak & Improve Corpus 2025**: Cambridge University Press & Assessment (2025).
Speak & Improve Corpus 2025 [Data set], https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025
(registration required; non-commercial use only). Corpus paper: Knill et al. (2025), https://doi.org/10.17863/CAM.114333
(SLaTE 2025 version: https://doi.org/10.21437/SLaTE.2025-34). Relevant terms of that licence agreement (dataset page, checked
2026-11-11; quoted):

- "Do not release items (e.g. models, data statistics) derived from the corpus without prior approval of CUP&A."
- "Do not provide the corpus (full or partial) to others in any way, even if they have also signed the licence agreement."
- Use is "for non-commercial research and educational purposes only".
- The licensee "may publish excerpts of less than 100 words from the licensed dataset".
- Publications must acknowledge and cite the corpus (see the README, section "Corpus").

**Status: requested, pending.** The corpus-derived files listed below carry no open licence: they are subject to the Speak &
Improve Corpus 2025 licence and are released subject to CUP&A approval (requested 2026-11-11). Until the approval is recorded
here, this package is for review only and must not be redistributed.

| item | status | date | reference |
|---|---|---|---|
| release of the corpus-derived files listed below | requested, pending | 2026-11-11 (requested) | |

## What the package does NOT contain
No audio, no transcripts, no manual annotations, no per-utterance or per-edit labels, features, scores or model outputs, and no
trained model files (only their SHA-256 hashes in `configs/final.yaml`). No corpus file ID occurs anywhere in the package (checked
against all IDs of the official file lists; number of IDs: [corpus-derived value omitted; see the paper]).

## Corpus-derived files and the level of information they contain
Levels: **A** = aggregate statistics (counts, rates, metrics, confidence intervals over a set or a group); **A-curve** = aggregate
curve on a fixed grid; **P** = parameters / hashes of models trained on corpus-derived labels (no data); **N** = numbers quoted in
prose. No file is at the per-utterance or per-edit level.

| file | content | level |
|---|---|---|
| `results/reproduction.json` | official baseline check on the dev-subset: WER, SGECF P/R/F0.5, SLA scores | A |
| `results/official_baseline_comparison.json` | B1 dev vs official baseline: overall, sub/del/ins, per-error-type F0.5 | A |
| `results/system_metrics_summary.json` | ASR WER, GEC WER, SGECF per system and set | A |
| `results/b_checkpoint_summary{,_v4}.{csv,md}` | per system x set: GEC WER, P/R/F0.5, edit / TP / FP counts (12 rows each) | A |
| `results/label_stats{,_v4}_{dev,train}.csv` | edit, TP, FP, ASR-induced counts and FP rates by overall / test part / operation / ERRANT type (rows per file, and cells per file with fewer than 5 edits: [corpus-derived value omitted; see the paper]) | A (small cells) |
| `results/oracle_upper_bounds{,_v4}.json` | oracle filtering upper bounds per set | A |
| `results/asr_conf_diagnostics.json` | AUROC, FP rates by ASR-error window and by confidence quartile (with cut points) | A |
| `results/C2_cv_cv_info.json` | CV design: edit / group counts, fold sizes and FP rates, chosen hyper-parameters | A, P |
| `results/C2_cv_evaluation.json` | OOF metrics with bootstrap CIs and paired tests | A |
| `results/C2_cv_coverage_curves.json` | precision vs coverage per model, 201-point grid | A-curve |
| `results/eval_results.json`, `results/eval_posthoc_equal_precision.json` | held-out eval metrics, CIs, paired tests, FP by cause | A |
| `results/fig4_precision_at_coverage_eval.csv` | data of Figure 4: precision at coverage with CIs (35 rows) | A-curve |
| `results/D_W_eval_{frozen,retrained}.json`, `results/D_Q_eval_retrained{,_bartfeat}.json` | W / Q eval metrics | A |
| `results/qwen_score_summary.json` | mean Qwen teacher-forced scores, format-error rates per set | A |
| `results/E_{B1,W,W_frozen,Q}_exploratory.json` | exploratory analyses: proficiency groups, utility curves (19 points), Holm tests | A, A-curve |
| `results/GRECO_{B1,W,Q}_eval.json` | GRECO comparison metrics, chosen hyper-parameters | A, P |
| `configs/final.yaml` | frozen configuration: hyper-parameters, thresholds, SHA-256 of the trained model files (not the models) | P |
| `configs/2026-11-10_*.yaml`, `configs/post_freeze_changes.md` | analysis plans; some quote counts and rates observed before the plan | N |
| `README.md`, `docs/AUDIT_REPORT.md`, `environment/mac_changes.md` | counts, timings and metrics quoted in prose | N |

Not corpus-derived: `scripts/` (code; the three few-shot examples in `scripts/d_qwen_prompt*.py` are self-written, not taken
from the corpus), `environment/` exports and hashes of public models, `environment/patches/` (official-code portability changes,
see `environment/patches/README.md`), `LICENSE`, `MANIFEST.sha256`.

## Excerpt scan (transcripts)
`scripts/pkg_excerpt_scan.py` finds, in every text file, maximal runs of consecutive words whose every 3-word window occurs in
the manual reference transcripts (all sets, asr / fluent / gec STMs) and reports file, line and run length only. Result for
package version 5 (2026-11-11): longest run and number of runs of 5 or more words: [corpus-derived value omitted; see the paper]; none of 100 words or more. The runs are ordinary
English shared by chance: in the self-written few-shot examples (`scripts/d_qwen_prompt.py` lines 14–19,
`scripts/d_qwen_prompt_v2.py` lines 11–16), in SCTK build logs and the quoted SCTK disclaimer
(`environment/patches/sctk-2.4.9_*.diff`), in the quoted SCTK disclaimer in `environment/patches/README.md`, and in
`README.md`, `configs/final.yaml`, `configs/2026-11-10_D_plan.yaml` and `environment/mac_changes.md` (prose). The stricter
`scripts/pkg_leak_scan.py` (6-word sequences) finds 0 files with hits.

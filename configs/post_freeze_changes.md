# Changes to frozen scripts after the freeze (`final.yaml` script_sha256)

All results in `results/` were produced by the frozen versions (sha256 in `final.yaml`). The distributed scripts differ from them
only by the changes below. Summary and new hashes: `final.yaml`, key `post_freeze_notes`.

| script | frozen sha256 | distributed sha256 | change |
|---|---|---|---|
| `b3_edit_labels.py` | 6656ac13… | ec543496… (v4; v3: a9063d27…) | (1) 2026-11-11 self-check uses the official `compute_p_r_f05`; (2) 2026-11-12 path constant; (3) package v4: no default run root |
| `c1_features.py` | 2d397750… | 3cccc279… (v4; v3: 4c65fb4f…) | (2) 2026-11-12 path constant; (3) package v4: no default run root |
| `c_model.py` | 0c6f2e89… | b8ff529b… (v4; v3: f044045f…) | (2) 2026-11-12 path constant; (3) package v4: EXP_DIR required |
| `b_pipeline2.sh` | f2b94607… | a70db7a5… | 2026-11-12 two comment lines |
| `c1_bart_score.py`, `c6_final.py`, `stm_to_ctm.py`, `b_norm_verbatim.sh` | — | unchanged | — |

Verification: (a) the author re-generated the B1 and W labels with change (1): byte-identical; (b) the reproduction audit
(`docs/AUDIT_REPORT.md`; a pre-submission reproduction audit, run in a separate workspace with freshly created environments and following only the package documentation; the audit was carried out with an AI coding assistant — see the declaration of generative AI use in the manuscript) reproduced B1 labels, label statistics, OOF metrics, the five frozen-model sha256 and the C7 eval exactly;
(c) after change (2), `c_model.py cv/evaluate`, `c6_final.py`, `c6b_addendum.py` were re-run on the same features: the five
frozen-model sha256 equal `final.yaml` and the CV outputs equal `results/C2_cv_*.json` (difference 0); and the Q dev-subset chain
was re-run with the refactored shell scripts in a fresh run root: all 18 result files byte-identical.

Why change (1): the self-check compared the labels' P/R/F0.5 with the official `result.txt` at 3 decimals using a re-implemented
formula; a different floating-point operation order changed one Q-dev per-type value at a %.3f tie, so the check failed although
the labels were correct. The labelling itself (alignment, edit matching, cause attribution) was not touched.

## (1) `b3_edit_labels.py`, 2026-11-11
```diff
@@ -291,8 +291,7 @@ def main():
     norm = lambda d: {k: int(v) for k, v in d.items() if v}
     ver["counts_equal_official"] = {"tp": norm(tp) == norm(o_tp), "fp": norm(fp) == norm(o_fp), "fn": norm(fn) == norm(o_fn)}
     T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
-    P, R = T / (T + F), T / (T + N)
-    F05 = 1.25 * P * R / (0.25 * P + R) if (P + R) else 0.0
+    P, R, F05 = SE.compute_p_r_f05(T, F, N)   # official formula (float operation order matters for %.3f ties)
     txt = open(f"{rd}/gec/sgecf-scoring/result.txt").read()
     o = re.search(r"Overall results:\nP:\tR:\tF0\.5\n([\d.]+)\t([\d.]+)\t([\d.]+)", txt).groups()
     ver["overall_from_labels"] = {"tp": T, "fp": F, "fn_official": N, "P": round(P, 3), "R": round(R, 3), "F05": round(F05, 3)}
@@ -302,7 +301,7 @@ def main():
     for t, vals in re.findall(r"Edit Type:\t(\S+)\nP:\tR:\tF0\.5\n([\d.]+\t[\d.]+\t[\d.]+)", txt):
         tt, ff, nn = tp[t], fp[t], fn.get(t, 0)
         try:
-            p_, r_ = tt / (tt + ff), tt / (tt + nn); f_ = 1.25 * p_ * r_ / (0.25 * p_ + r_)
+            p_, r_, f_ = SE.compute_p_r_f05(tt, ff, nn)   # official formula, as print_results
             per_type_ok &= ("%.3f\t%.3f\t%.3f" % (p_, r_, f_)) == vals
         except ZeroDivisionError:
             pass
```

## (2) path constants and comments, 2026-11-12
```diff
diff --git a/04_scripts/b3_edit_labels.py b/04_scripts/b3_edit_labels.py
--- a/04_scripts/b3_edit_labels.py
+++ b/04_scripts/b3_edit_labels.py
@@ -3,7 +3,7 @@
 OFFICIAL Spoken ERRANT code (scoring-tools/sgecf-tools/spoken_errant.py, create_files_for_errant.py).
 
 Run with the sandi-all env from the corpus root:
-  python b3_edit_labels.py --run_dir stageB/B1_dev --ds dev --out_dir <<local project>/05_runs/...>/labels
+  python b3_edit_labels.py --run_dir stageB/B1_dev --ds dev --out_dir <$EXP_DIR/05_runs/...>/labels
 
 Correctness proofs written to <out_dir>/b3_verification.json (the script exits non-zero if any fails):
   (1) utterance-ID alignment: the four aligned texts re-created with create_files_for_errant.py logic are identical,
@@ -45,7 +45,7 @@ from collections import defaultdict
 
 import pandas as pd
 
-CORPUS = os.path.expanduser("<local corpus path>")
+CORPUS = (os.environ.get("RUN_DIR") or os.environ["CORPUS_DIR"])
 SG = os.path.join(CORPUS, "scoring-tools/sgecf-tools")
 sys.path.insert(0, SG)
 import spoken_errant as SE            # noqa: E402  official module
diff --git a/04_scripts/b_pipeline2.sh b/04_scripts/b_pipeline2.sh
--- a/04_scripts/b_pipeline2.sh
+++ b/04_scripts/b_pipeline2.sh
@@ -2,7 +2,7 @@
 # Stage B pipeline v2 (B1 = official Whisper-small transcripts, B2 = manual fluent transcripts,
 # B2b = manual verbatim transcripts + real DD). v2 = v1 (b_pipeline.sh, used for B1 dev) + MODE verbatim;
 # the asr and fluent branches are unchanged.
-# Run from the corpus root:  cd <local corpus path> && caffeinate -i bash <this> MODE DS FLIST OUT_DIR
+# Run from the corpus root:  cd "$RUN_DIR" && caffeinate -i bash <this> MODE DS FLIST OUT_DIR
 #   MODE   : asr    -> input reference-materials/pre-norm/${DS}-gec.ctm, official DD + GEC (run-gec.sh)
 #            fluent -> input reference-materials/stms/${DS}-fluent.stm converted by stm_to_ctm.py,
 #                      DD skipped (identity_dd.py), GEC called with the same arguments as run-gec.sh
@@ -11,7 +11,7 @@
 #   DS     : dev | train | eval (eval is locked until C7) | dev-subset (format/plumbing tests)
 #   FLIST  : file list, normally reference-materials/flists.flac/${DS}-gec.tsv
 #   OUT_DIR: relative path under the corpus root (scoring writes reference text there, so outputs
-#            stay in the corpus tree; derived files are copied to <local project>/05_runs by collect script)
+#            stay in the corpus tree; derived files are copied to $EXP_DIR/05_runs by collect script)
 # Every step prints:  STEP <name> <start_epoch> <end_epoch> rc=<rc>
 set -u
 MODE=$1; DS=$2; FLIST=$3; OUT=$4
diff --git a/04_scripts/c1_features.py b/04_scripts/c1_features.py
--- a/04_scripts/c1_features.py
+++ b/04_scripts/c1_features.py
@@ -23,7 +23,7 @@ from collections import defaultdict
 
 import pandas as pd
 
-CORPUS = os.path.expanduser("<local corpus path>")
+CORPUS = (os.environ.get("RUN_DIR") or os.environ["CORPUS_DIR"])
 sys.path.insert(0, os.path.join(CORPUS, "scoring-tools/sgecf-tools"))
 import spoken_errant as SE            # noqa: E402
 import create_files_for_errant as CF  # noqa: E402
diff --git a/04_scripts/c_model.py b/04_scripts/c_model.py
--- a/04_scripts/c_model.py
+++ b/04_scripts/c_model.py
@@ -26,7 +26,7 @@ import lightgbm as lgb
 
 warnings.filterwarnings("ignore")
 SEED = 20261110
-EXP = os.path.expanduser("<local project path>")
+EXP = (os.environ.get("EXP_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 F = f"{EXP}/05_runs/2026-11-10_C1_features"
 GROUPS = {
     "ASR": ["asr_conf_o_min", "asr_conf_o_mean", "asr_conf_win_min", "asr_conf_win_mean", "asr_conf_ctx2_min",
```

(Old local paths in the removed lines are shown as `<local corpus path>` / `<local project path>`.)

## (3) package version 4: no silent default for output locations
In package version 4 the path constants no longer fall back to a default: `EXP_DIR` (and, for the scripts that read the corpus
run root, `RUN_DIR`) must be set, otherwise the script stops with a message. `final.yaml` (`post_freeze_notes`, written for
version 3) is a historical record and is not edited (one exception, package version 6: in the distributed copy only, two text strings no longer describe the audit as
third-party: `post_freeze_notes.independent_check` now begins "Reproduction audit (pre-submission, AI-assisted; see
docs/AUDIT_REPORT.md; ..." and the POST-FREEZE NOTE header says "after the reproduction audit"; the research copy, every frozen value and every hash are unchanged); the distributed sha256 of the three scripts are those in the table above
(also in `MANIFEST.sha256`). Verification: `c_model.py cv/evaluate`, `c6_final.py`, `c6b_addendum.py` re-run with the version-4
scripts on the same features again give the five frozen-model sha256 of `final.yaml` (see docs/AUDIT_REPORT.md, appendix).
```diff
--- c_model.py (v3)
+++ c_model.py (v4)
-EXP = (os.environ.get("EXP_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+EXP = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
--- b3_edit_labels.py, c1_features.py (v3)
+++ b3_edit_labels.py, c1_features.py (v4)
-CORPUS = (os.environ.get("RUN_DIR") or os.environ["CORPUS_DIR"])
+CORPUS = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))
```

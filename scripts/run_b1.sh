#!/bin/bash
# B1 (official S&I 2025 baseline) end to end for the main system, in the order used for the paper:
#   B1 train+dev pipelines -> collect -> norm_verbatim v3/v4 -> edit labels v3/v4 -> label stats -> BART scores -> features (labels v4)
#   -> c_model.py cv -> c_model.py evaluate -> c6_final.py (frozen models) -> c6b_addendum.py.
# The held-out eval is NOT run here: run it once afterwards with scripts/c7_driver.sh (it refuses to run twice).
# Optional (needed by the Q dev-subset rehearsal): STEP=devsubset runs only the B1 dev-subset pipeline (stageB/test_dev-subset_asr).
# Added 2026-11-12 after the reproduction audit (docs/AUDIT_REPORT.md, defects 4-6, 9); the same sequence reproduced
# every B1 number of results/ exactly (labels, label stats, OOF, frozen-model sha256, C7 eval).
# Resumable: each step writes $EXP_DIR/05_runs/run_b1_state/<step>.ok and is skipped when it exists.
# Run (long; keeps the machine awake):
#   cd "$RUN_DIR" && nohup caffeinate -ims bash "$SCRIPTS_DIR/run_b1.sh" > run_b1.log 2>&1 & disown
# Measured (Apple M5 Pro, CPU; dev and train pipelines in parallel): pipelines 136 min, everything after them 16 min.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1
EXP=$EXP_DIR; SC=$SCRIPTS_DIR; ST=$EXP/05_runs/run_b1_state; mkdir -p $ST $EXP/05_runs $EXP/06_results
PYC=$CONDA_BASE/envs/$ENV_ESWA_C/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$RUN_DIR" || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # name cmd...   (skips if .ok exists; stops on failure)
  local n=$1; shift
  [ -f $ST/$n.ok ] && { log "SKIP $n (done)"; return 0; }
  log "START $n"; local t0=$(date +%s)
  "$@"; local rc=$?
  local s=$(( $(date +%s)-t0 ))
  if [ $rc -ne 0 ]; then log "FAILED $n rc=$rc after ${s}s"; exit $rc; fi
  echo "$s" > $ST/$n.ok; log "DONE $n ${s}s"
}
in_all() { ( source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL && "$@" ); }
mkdir -p stageB

if [ "${STEP:-}" = devsubset ]; then
  step pipe_devsubset bash -c "bash $SC/b_pipeline2.sh asr dev-subset reference-materials/flists.flac/dev-subset-gec.tsv stageB/test_dev-subset_asr > stageB/test_dev-subset_asr.log 2>&1 && grep -q '^DONE ' stageB/test_dev-subset_asr.log"
  exit 0
fi
pipe() { bash $SC/b_pipeline2.sh asr $1 reference-materials/flists.flac/$1-gec.tsv stageB/B1_$1 > stageB/B1_$1.log 2>&1 && grep -q '^DONE ' stageB/B1_$1.log; }
collect() { bash $SC/b_collect.sh asr 2026-11-10_B1_$1 stageB/B1_$1 stageB/B1_$1.log; }
normv() { bash $SC/b_norm_verbatim.sh $1 reference-materials/flists.flac/$1-gec.tsv stageB/norm_verbatim$2_$1 $3; }
labels() { in_all python $SC/b3_edit_labels.py --run_dir stageB/B1_$1 --ds $1 --mode asr \
             --verbatim_norm_ctm stageB/norm_verbatim$2_$1/gec/norm/gec.ctm --out_dir $EXP/05_runs/2026-11-10_B1_$1/$3 > stageB/labels$2_B1_$1.log 2>&1; }
F=$EXP/05_runs/2026-11-10_C1_features
bart() { mkdir -p $F && in_all python $SC/c1_bart_score.py --run_dir stageB/B1_$1 --out $F/bart_scores_$1.jsonl > stageB/bart_$1.log 2>&1; }
feat() { in_all python $SC/c1_features.py --ds $1 --run_dir stageB/B1_$1 --labels $EXP/05_runs/2026-11-10_B1_$1/labels_v4 \
             --bart $F/bart_scores_$1.jsonl --out_dir $F > stageB/features_$1.log 2>&1; }

# 1. B1 dev and train pipelines (official ASR -> DD -> GEC -> scoring), in parallel (independent; default thread counts)
if [ ! -f $ST/pipe_dev.ok ] || [ ! -f $ST/pipe_train.ok ]; then
  ( step pipe_dev pipe dev ) & P1=$!
  ( step pipe_train pipe train ) & P2=$!
  wait $P1; r1=$?; wait $P2; r2=$?
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || { log "pipeline failure dev=$r1 train=$r2"; exit 1; }
fi
# 2. collect metrics; verbatim reference CTMs (v3 = with markup, v4 = --drop_markup); edit labels v3 (labels/) and v4 (labels_v4/, primary)
for ds in dev train; do
  step collect_$ds collect $ds
  step normv3_$ds normv $ds "" ""
  step normv4_$ds normv $ds _v4 --drop_markup
  step labels3_$ds labels $ds "" labels
  step labels4_$ds labels $ds _v4 labels_v4
done
# 3. label statistics (results/label_stats{,_v4}_{dev,train}.csv, condition B1)
for ds in dev train; do
  step b4_v3_$ds python3 -I $SC/b4_label_stats.py --ds $ds --cond B1=$EXP/05_runs/2026-11-10_B1_$ds/labels --out $EXP/06_results/label_stats_$ds.csv
  step b4_v4_$ds python3 -I $SC/b4_label_stats.py --ds $ds --cond B1=$EXP/05_runs/2026-11-10_B1_$ds/labels_v4 --out $EXP/06_results/label_stats_v4_$ds.csv
done
# 4. BART teacher-forced scores and features (labels v4)
for ds in dev train; do step bart_$ds bart $ds; done
for ds in dev train; do step feat_$ds feat $ds; done
# 5. models: 5-fold grouped CV with inner model selection (OOF), evaluation (bootstrap), final fit (frozen artifacts), addendum (risk_lr, operating points)
step cv       $PYC -I $SC/c_model.py cv --out $EXP/05_runs/2026-11-10_C2_cv
step evaluate $PYC -I $SC/c_model.py evaluate --out $EXP/05_runs/2026-11-10_C2_cv
step c6       $PYC -I $SC/c6_final.py --out $EXP/05_runs/2026-11-10_C6_final
step c6b      $PYC -I $SC/c6b_addendum.py --out $EXP/05_runs/2026-11-10_C6_final
log "B1 train/dev done; check the artifact sha256 against $CONFIGS_DIR/final.yaml, then run scripts/c7_driver.sh once"

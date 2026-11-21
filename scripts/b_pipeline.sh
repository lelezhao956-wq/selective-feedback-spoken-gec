#!/bin/bash
# Stage B pipeline (B1 = official Whisper-small transcripts, B2 = manual fluent transcripts).
# Run from the corpus root:  cd "$RUN_DIR" && caffeinate -i bash <this> MODE DS FLIST OUT_DIR
#   MODE   : asr    -> input reference-materials/pre-norm/${DS}-gec.ctm, official DD + GEC (run-gec.sh)
#            fluent -> input reference-materials/stms/${DS}-fluent.stm converted by stm_to_ctm.py,
#                      DD skipped (identity_dd.py), GEC called with the same arguments as run-gec.sh
#   DS     : dev | train | eval (eval is locked until C7) | dev-subset (format/plumbing tests)
#   FLIST  : file list, normally reference-materials/flists.flac/${DS}-gec.tsv
#   OUT_DIR: relative path under the corpus root (scoring writes reference text there, so outputs
#            stay in the corpus tree; derived files are copied to $EXP_DIR/05_runs by collect script)
# Every step prints:  STEP <name> <start_epoch> <end_epoch> rc=<rc>
set -u
MODE=$1; DS=$2; FLIST=$3; OUT=$4
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
STMDS=$DS
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 100; }
mkdir -p "$OUT"

step() { local name=$1; shift; local t0=$(date +%s); "$@"; local rc=$?; echo "STEP $name $t0 $(date +%s) rc=$rc"; [ $rc -eq 0 ] || { echo "ABORT at $name"; exit $rc; }; }

if [ "$MODE" = asr ]; then
  IN_CTM=reference-materials/pre-norm/${DS}-gec.ctm
  [ "$DS" = dev-subset ] && IN_CTM=reference-materials/pre-norm/dev-gec.ctm   # select_files.py restricts to FLIST
elif [ "$MODE" = fluent ]; then
  IN_CTM=$OUT/input/${DS}-fluent.pre-norm.ctm
  mkdir -p $OUT/input
  step stm_to_ctm python3 -I "$SCRIPTS/stm_to_ctm.py" --stm reference-materials/stms/${STMDS}-fluent.stm --flist $FLIST --out $IN_CTM
else
  echo "unknown MODE $MODE"; exit 100
fi

step asr-to-task-input_gec ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $IN_CTM \
     --flist $FLIST --task gec --out_dir $OUT/gec

if [ "$MODE" = asr ]; then
  # official DD + GEC; DD/GEC split is reported from STEP-less timestamps written below
  t0=$(date +%s)
  step run-gec_DD+GEC ./baseline-pipeline/run-scripts/run-gec.sh --ctm $OUT/gec/norm/gec.ctm --dd_model sandi-models/dd/dd_model \
       --gec_model sandi-models/gec/gec_model --out_dir $OUT/gec
  echo "SUBSTEP dd $t0 $(stat -f %m $OUT/gec/fluent.ctm)"
  echo "SUBSTEP gec $(stat -f %m $OUT/gec/fluent.ctm) $(stat -f %m $OUT/gec/gec.ctm)"
else
  mkdir -p $OUT/gec/dd $OUT/gec/LOGs
  step identity_dd python3 -I "$SCRIPTS/identity_dd.py" --input_ctm $OUT/gec/norm/gec.ctm \
       --fluent_ctm $OUT/gec/fluent.ctm --gec_tsv $OUT/gec/dd/pre-gec.tsv
  # same call as run-gec.sh lines 131-136 (GEC_ENV = envs/sandi-all-env.sh)
  gec_only() { source ./envs/sandi-all-env.sh && python ./baseline-pipeline/gec-tools/gec-plus-align.py \
       --input_file $OUT/gec/dd/pre-gec.tsv --gec_model sandi-models/gec/gec_model \
       --gec_ctm $OUT/gec/gec.ctm 2>&1 | tee $OUT/gec/LOGs/gec.LOG; [ -f $OUT/gec/gec.ctm ]; }
  step gec_only gec_only
fi

step run-wer-scoring_gec ./scoring-tools/run-scripts/run-wer-scoring.sh --stm reference-materials/stms/${STMDS}-gec.stm \
     --ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/wer-scoring
step run-sgecf-scoring ./scoring-tools/run-scripts/run-sgecf-scoring.sh --edit_detail \
     --flt_stm reference-materials/stms/${STMDS}-fluent.stm --gec_stm reference-materials/stms/${STMDS}-gec.stm \
     --flt_ctm $OUT/gec/fluent.ctm --gec_ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/sgecf-scoring
echo "DONE $MODE $DS"

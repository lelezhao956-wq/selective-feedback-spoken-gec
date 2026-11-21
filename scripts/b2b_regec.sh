#!/bin/bash
# B2b fix run (2026-11-10).  The queue's B2b runs failed in the official gec-plus-align.py because DD reduced some
# all-disfluent phrases to empty strings (see b2b_filter_empty.py).  This script REUSES the DD output of the failed run
# (left untouched in stageB/B2b_<ds>/) and runs only: filter -> GEC -> GEC WER -> SGECF, in a new dir stageB/B2b_<ds>_regec/.
# GEC and scoring calls are identical to b_pipeline2.sh / official run-gec.sh lines 131-136.
# Run from the corpus root:  caffeinate -i bash b2b_regec.sh DS
set -u
DS=$1; SRC=stageB/B2b_$DS/gec; OUT=stageB/B2b_${DS}_regec
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 100; }
step() { local name=$1; shift; local t0=$(date +%s); "$@"; local rc=$?; echo "STEP $name $t0 $(date +%s) rc=$rc"; [ $rc -eq 0 ] || { echo "ABORT at $name"; exit $rc; }; }
mkdir -p $OUT/gec/dd $OUT/gec/LOGs
# reused (copied, not moved) from the failed run: normalised pre-DD input and the DD log
cp -Rp $SRC/norm $OUT/gec/norm
cp -p $SRC/LOGs/dd.LOG $SRC/LOGs/select_files.gec.log $SRC/LOGs/add_phrase_markers.gec.log $SRC/LOGs/ctm_norm.gec.log $OUT/gec/LOGs/
echo "REUSED_DD_FROM $SRC pre-gec.tsv sha256 $(shasum -a 256 $SRC/dd/pre-gec.tsv | cut -c1-16) fluent.ctm sha256 $(shasum -a 256 $SRC/fluent.ctm | cut -c1-16)"
step filter_empty python3 -I "$SCRIPTS/b2b_filter_empty.py" $SRC $OUT/gec
echo "FILTERED_IDS $(tail -n +2 $OUT/gec/dd/filtered_empty_phrases.tsv | tr '\n' ' ')"
gec_only() { source ./envs/sandi-all-env.sh && python ./baseline-pipeline/gec-tools/gec-plus-align.py \
     --input_file $OUT/gec/dd/pre-gec.tsv --gec_model sandi-models/gec/gec_model \
     --gec_ctm $OUT/gec/gec.ctm 2>&1 | tee $OUT/gec/LOGs/gec.LOG; [ -f $OUT/gec/gec.ctm ]; }
step gec_only gec_only
step run-wer-scoring_gec ./scoring-tools/run-scripts/run-wer-scoring.sh --stm reference-materials/stms/${DS}-gec.stm \
     --ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/wer-scoring
step run-sgecf-scoring ./scoring-tools/run-scripts/run-sgecf-scoring.sh --edit_detail \
     --flt_stm reference-materials/stms/${DS}-fluent.stm --gec_stm reference-materials/stms/${DS}-gec.stm \
     --flt_ctm $OUT/gec/fluent.ctm --gec_ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/sgecf-scoring
echo "DONE verbatim $DS (regec)"

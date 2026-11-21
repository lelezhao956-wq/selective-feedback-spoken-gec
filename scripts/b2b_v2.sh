#!/bin/bash
# B2b v2 (2026-11-10): manual verbatim transcript with transcription markup removed (stm_to_ctm.py --drop_markup)
# -> official task input -> official DD -> empty-phrase filter (b2b_filter_empty.py) -> official GEC -> GEC WER -> SGECF.
# DD and GEC are called exactly as in the official run-gec.sh (lines 113-136: same scripts, envs, models, arguments);
# run-gec.sh itself is not used because the empty-phrase filter has to sit between DD and GEC (official code unchanged).
# Output in a NEW dir stageB/B2b_<ds>_v2 (v1 dirs untouched).  Run from corpus root:  caffeinate -i bash b2b_v2.sh DS
set -u
DS=$1; FLIST=reference-materials/flists.flac/${DS}-gec.tsv; OUT=stageB/B2b_${DS}_v2
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 100; }
step() { local name=$1; shift; local t0=$(date +%s); "$@"; local rc=$?; echo "STEP $name $t0 $(date +%s) rc=$rc"; [ $rc -eq 0 ] || { echo "ABORT at $name"; exit $rc; }; }
mkdir -p $OUT/input $OUT/dd_raw/dd $OUT/gec/dd $OUT/gec/LOGs
step stm_to_ctm python3 -I "$SCRIPTS/stm_to_ctm.py" --stm reference-materials/stms/${DS}-asr.stm --flist $FLIST \
     --out $OUT/input/${DS}-verbatim.pre-norm.ctm --drop_in_ignore_of reference-materials/stms/${DS}-fluent.stm --drop_markup
step asr-to-task-input_gec ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $OUT/input/${DS}-verbatim.pre-norm.ctm \
     --flist $FLIST --task gec --out_dir $OUT/gec
dd_only() { source ./envs/sandi-dd-env.sh && python ./baseline-pipeline/gec-tools/dd.py --input_ctm $OUT/gec/norm/gec.ctm \
     --disfluency_model sandi-models/dd/dd_model --fluent_ctm $OUT/dd_raw/fluent.ctm --gec_tsv $OUT/dd_raw/dd/pre-gec.tsv \
     2>&1 | tee $OUT/gec/LOGs/dd.LOG; [ -f $OUT/dd_raw/fluent.ctm ] && [ -f $OUT/dd_raw/dd/pre-gec.tsv ]; }
step dd_only dd_only
step filter_empty python3 -I "$SCRIPTS/b2b_filter_empty.py" $OUT/dd_raw $OUT/gec
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
echo "DONE verbatim $DS (v2, markup dropped)"

#!/bin/bash
# Normalised manual VERBATIM tokens for the B3 three-way labels (ASR vs DD vs GEC):
# reference-materials/stms/${DS}-asr.stm -> stm_to_ctm.py (segment phrases, words in fluent IGNORE regions dropped)
# -> official run-asr-to-task-input.sh --task gec -> $OUT/gec/norm/gec.ctm  (same route as the B2b input).
# Usage from corpus root: bash b_norm_verbatim.sh DS FLIST OUT [EXTRA_STM_TO_CTM_OPTION, e.g. --drop_markup (v4, 2026-11-10)]
set -eu
DS=$1; FLIST=$2; OUT=$3; EXTRA=${4:-}; SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 100; }
mkdir -p $OUT/input
python3 -I "$SCRIPTS/stm_to_ctm.py" --stm reference-materials/stms/${DS}-asr.stm --flist $FLIST \
   --out $OUT/input/${DS}-verbatim.pre-norm.ctm --drop_in_ignore_of reference-materials/stms/${DS}-fluent.stm $EXTRA
./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $OUT/input/${DS}-verbatim.pre-norm.ctm \
   --flist $FLIST --task gec --out_dir $OUT/gec > $OUT/task_input.log 2>&1
echo "verbatim norm words: $(wc -l < $OUT/gec/norm/gec.ctm)"

#!/bin/bash
# Copy the licence-safe outputs of one stage-B run into $EXP_DIR/05_runs/<RUN_NAME>/, extract metrics,
# and commit.  Usage (from corpus root):  bash b_collect.sh MODE RUN_NAME CORPUS_OUT_DIR LOGFILE
#   MODE asr    : copies model outputs derived from Whisper (norm/gec.ctm, fluent.ctm, gec.ctm, dd/pre-gec.tsv)
#   MODE fluent : copies NO text files (inputs/outputs are near-copies of the manual transcripts);
#                 only sha256 + paths of the files that stay in the corpus tree.
# Never copied (they contain reference annotation text): wer-scoring/ref, *.pra, *.sgml, *.dtl, *.lur,
#   *.filt, sgecf-scoring/errant/*, input/*.ctm of fluent runs.
set -u
MODE=$1; RUN=$2; OUT=$3; LOG=$4
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR
R=$EXP/05_runs/$RUN
mkdir -p $R/outputs
cp "$LOG" $R/pipeline.log
grep -E '^(STEP|SUBSTEP|DONE|ABORT)' "$LOG" > $R/step_timings.txt
if [ "$MODE" = asr ]; then
  for f in gec/norm/gec.ctm gec/fluent.ctm gec/gec.ctm gec/dd/pre-gec.tsv; do
    mkdir -p $R/outputs/$(dirname $f); cp $OUT/$f $R/outputs/$f
  done
fi
cp $OUT/gec/sgecf-scoring/result.txt $R/sgecf_result_edit_detail.txt
grep -E 'Sum/Avg|SPKR' $OUT/gec/wer-scoring/hyp/gec.ctm.filt.sys | head -2 > $R/gec_wer_sum_avg.txt
( cd $OUT && find . -type f \( -name '*.ctm' -o -name '*.tsv' -o -name 'result.txt' -o -name '*.sys' \) -exec shasum -a 256 {} + ) > $R/corpus_tree_files.sha256
echo "corpus_out_dir: $(cd $OUT && pwd)" > $R/location.txt
python3 -I $SC/b_metrics.py --run_dir $R --mode $MODE
bash $SC/git_commit_locked.sh "Stage B: $RUN (pipeline outputs + metrics)"

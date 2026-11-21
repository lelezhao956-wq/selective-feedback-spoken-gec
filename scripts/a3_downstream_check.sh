#!/bin/bash
# A3 part 2: run the official downstream pipeline (SLA, DD+GEC, GEC WER, SGECF) on the OFFICIAL
# Whisper output (dev-subset.reference/asr/decode/dev-subset-asr/asr.ctm) instead of our own ASR.
# Commands are copied from check-pipeline.sh; only the input CTM and the output directory differ.
# Run from the corpus root: cd "$RUN_DIR" && bash <this script>
# Per-step timing: each step prints "STEP <name> <start_epoch> <end_epoch> rc=<rc>" to stdout.
set -u
IN=dev-subset.reference/asr/decode/dev-subset-asr/asr.ctm
OUT=dev-subset.downstream-check
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 100; }

step() { local name=$1; shift; local t0=$(date +%s); "$@"; local rc=$?; echo "STEP $name $t0 $(date +%s) rc=$rc"; }

for p in P1 P3 P4 P5; do
  step "asr-to-task-input_sla-$p" ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $IN \
      --flist reference-materials/flists.flac/dev-subset-sla-$p.tsv --task sla-$p --out_dir $OUT/sla
done
step run-sla ./baseline-pipeline/run-scripts/run-sla.sh --ctm_dir $OUT/sla/norm --model_dir sandi-models/bert-sla \
    --out_dir $OUT/sla --test dev-subset
step run-sla-scoring ./scoring-tools/run-scripts/run-sla-scoring.sh --pred $OUT/sla/sla.tsv --test dev-subset-sla-overall \
    --out_dir $OUT/sla
step asr-to-task-input_gec ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $IN \
    --flist reference-materials/flists.flac/dev-subset-gec.tsv --task gec --out_dir $OUT/gec
step run-gec_DD+GEC ./baseline-pipeline/run-scripts/run-gec.sh --ctm $OUT/gec/norm/gec.ctm --dd_model sandi-models/dd/dd_model \
    --gec_model sandi-models/gec/gec_model --out_dir $OUT/gec
step run-wer-scoring_gec ./scoring-tools/run-scripts/run-wer-scoring.sh --stm reference-materials/stms/dev-subset-gec.stm \
    --ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/wer-scoring
step run-sgecf-scoring ./scoring-tools/run-scripts/run-sgecf-scoring.sh --flt_stm reference-materials/stms/dev-subset-fluent.stm \
    --gec_stm reference-materials/stms/dev-subset-gec.stm --flt_ctm $OUT/gec/fluent.ctm \
    --gec_ctm $OUT/gec/gec.ctm --out_dir $OUT/gec/sgecf-scoring

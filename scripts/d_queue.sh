#!/bin/bash
# Stage D queue (plan: 03_configs/2026-11-10_D_plan.yaml). Sequential, resumable, with heartbeat and stall detection.
# Start (only via d_resume.sh, which does):  cd "$RUN_DIR" && nohup caffeinate -ims bash d_queue.sh PROFILE > LOG 2>&1 & disown
#   PROFILE: rehearsal (dev-subset; outputs stageD_rehearsal/, 05_runs/D_rehearsal/) | full (dev/train/eval + modelling)
# Steps (per chain): W = Whisper large-v3 -> official DD -> official BART GEC;  Q = B1 DD output -> Qwen2.5-7B GEC.
#   W: asr, ctm (format check + official task input + ASR WER), dd, filter, gec, score (+collect), labels, bart, features
#   Q: llm, align, score (+collect), labels, bart, features;   M (full only): frozen-model eval and retrain-CV-eval per system
# Each step: if a valid .done marker exists (all listed products present with the recorded sha256) -> skipped.
#   Otherwise existing products are renamed to <product>.partial-<YYYYmmdd-HHMMSS> (never deleted), the step runs, and on
#   success + verification a .done marker (finish time, seconds, sha256 of every product) is written.
#   A failed step marks its chain failed: the chain's later steps are skipped (SKIPPED_DEP); other chains continue.
# Heartbeat every 5 min -> 05_runs/d_heartbeat[_rehearsal].tsv; no progress (probe count unchanged and < 60 s CPU of the step's
#   process tree) for 30 min -> STALLED in the status file (the step is NOT killed).
set -u
PROFILE=${1:?profile}
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; CORPUS=$RUN_DIR
cd $CORPUS || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [ "$PROFILE" = rehearsal ]; then SUF=_rehearsal; SD=stageD_rehearsal; RUNS=$EXP/05_runs/D_rehearsal; SETS="dev-subset"
elif [ "$PROFILE" = full ]; then SUF=""; SD=stageD; RUNS=$EXP/05_runs; SETS="dev train eval"
else echo "unknown profile"; exit 100; fi
STATE=$EXP/05_runs/D_state$SUF; STATUS=$EXP/05_runs/d_queue_status$SUF.tsv; HB=$EXP/05_runs/d_heartbeat$SUF.tsv
LOCK=$STATE/queue.lock; mkdir -p $STATE $SD $RUNS
if [ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "queue already running (pid $(cat $LOCK))"; exit 101; fi
echo $$ > $LOCK
[ -f $STATUS ] || printf 'time\tchain\tstep\tphase\tdetail\n' > $STATUS
[ -f $HB ] || printf 'time\tchain\tstep\tprogress\tcpu_s_tree\tcpu_pct_tree\tgpu_util_pct\tstalled\n' > $HB
st() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${1:-}" "${2:-}" "${3:-}" "${4:-}" >> $STATUS; echo "[$(date '+%F %T')] ${1:-} ${2:-} ${3:-} ${4:-}"; }
st QUEUE - start "pid $$ profile $PROFILE sets: $SETS"
date '+%F %T' >> $STATE/starts

# ---------------- environments
CONDA=$CONDA_BASE/etc/profile.d/conda.sh
PY_D=$CONDA_BASE/envs/$ENV_ESWA_D/bin/python; PY_C=$CONDA_BASE/envs/$ENV_ESWA_C/bin/python
DPATH=$CONDA_BASE/envs/$ENV_ESWA_D/bin:$PATH   # ffmpeg/ffprobe for mlx-whisper (asr step only)
in_all() { ( source $CONDA && conda activate $ENV_SANDI_ALL && "$@" ); }

# ---------------- markers
hash_path() {  # sha256 of a file, or of the sorted per-file sha256 list of a directory
  if [ -d "$1" ]; then ( cd "$1" && find . -type f ! -name '.DS_Store' -print0 | sort -z | xargs -0 shasum -a 256 ) | shasum -a 256 | cut -d' ' -f1
  elif [ -f "$1" ]; then shasum -a 256 "$1" | cut -d' ' -f1; else echo MISSING; fi; }
marker() { echo "$STATE/$1.$2.done"; }
valid_done() {  # chain step products...
  local m=$(marker $1 $2); [ -f $m ] || return 1; shift 2
  local p; for p in "$@"; do sed -n 's/^sha256 //p' $m | grep -qxF "$p $(hash_path $p)" || return 1; done; return 0; }
write_done() {  # chain step seconds products...
  local m=$(marker $1 $2) c=$1 s=$2 sec=$3; shift 3
  { echo "chain $c"; echo "step $s"; echo "finished $(date '+%F %T')"; echo "seconds $sec"; local p; for p in "$@"; do echo "sha256 $p $(hash_path $p)"; done; } > $m.tmp && mv $m.tmp $m; }

# ---------------- heartbeat (background)
CUR=$STATE/current
cpu_tree() {  # total CPU seconds and %CPU of a process tree
  ps -A -o pid=,ppid=,time=,%cpu= | awk -v root=$1 '
    { pid[NR]=$1; pp[NR]=$2; t=$3; n=split(t,a,"[:.-]"); s=0
      if (t ~ /-/) { s=a[1]*86400+a[2]*3600+a[3]*60+a[4] } else if (n==4) { s=a[1]*3600+a[2]*60+a[3] } else { s=a[1]*60+a[2] }
      sec[$1]=s; pc[$1]=$4; par[$1]=$2 }
    END { inT[root]=1; changed=1; while (changed) { changed=0; for (p in par) if (!(p in inT) && (par[p] in inT)) { inT[p]=1; changed=1 } }
          for (p in inT) if (p in sec) { S+=sec[p]; P+=pc[p] }; printf "%d\t%.0f", S, P }'; }
heartbeat() {
  local last_prog="" last_cpu=0 since=$(date +%s) flagged=0 last_step=""
  while kill -0 $$ 2>/dev/null; do
    if [ -f $CUR ]; then
      IFS=$'\t' read -r c s pid probe < $CUR
      [ "$c/$s" != "$last_step" ] && { last_step="$c/$s"; last_prog=""; last_cpu=0; since=$(date +%s); flagged=0; }
      local prog=$(eval "$probe" 2>/dev/null | tr -d ' '); prog=${prog:-0}
      local ct=$(cpu_tree $pid); local cs=${ct%%$'\t'*}
      local gpu=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
      if [ "$prog" != "$last_prog" ] || [ $(( cs - last_cpu )) -ge 60 ]; then last_prog=$prog; last_cpu=$cs; since=$(date +%s); flagged=0; fi
      local stalled=0
      if [ $(( $(date +%s) - since )) -ge 1800 ]; then stalled=1
        [ $flagged = 0 ] && { st $c $s STALLED "no progress for $(( ($(date +%s)-since)/60 )) min (progress $prog, cpu_s $cs); not killed"; flagged=1; }; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$c" "$s" "$prog" "$ct" "${gpu:-NA}" "$stalled" >> $HB
    fi
    sleep ${HB_INTERVAL:-300}
  done; }
heartbeat & HBPID=$!
trap 'rc=$?; kill $HBPID 2>/dev/null; rm -f $CUR; st QUEUE - exit "rc=$rc"; rm -f $LOCK' EXIT

# ---------------- step runner
FAILED_CHAINS=" "
run_step() {  # chain step "products" "probe" est_min cmd...
  local c=$1 s=$2 prods=$3 probe=$4 est=$5; shift 5
  if [[ "$FAILED_CHAINS" == *" $c "* ]]; then st $c $s SKIPPED_DEP "earlier step of $c failed"; return 1; fi
  if valid_done $c $s $prods; then st $c $s SKIP_DONE "valid .done marker"; return 0; fi
  local ts=$(date '+%Y%m%d-%H%M%S') p moved=""
  for p in $prods; do [ -e $p ] && { mv $p $p.partial-$ts; moved="$moved $p.partial-$ts"; }; done
  [ -f $(marker $c $s) ] && mv $(marker $c $s) $(marker $c $s).invalid-$ts
  st $c $s start "est ${est} min${moved:+; renamed partial:$moved}"
  local log=$STATE/$c.$s.log t0=$(date +%s)
  ( "$@" ) > $log 2>&1 &
  local pid=$!; printf '%s\t%s\t%s\t%s\n' $c $s $pid "$probe" > $CUR
  wait $pid; local rc=$?; rm -f $CUR
  local sec=$(( $(date +%s) - t0 ))
  local missing=""; for p in $prods; do [ -e $p ] || missing="$missing $p"; done
  if [ $rc -ne 0 ] || [ -n "$missing" ]; then
    st $c $s FAILED "rc=$rc ${missing:+missing:$missing }after $((sec/60)) min (est $est); log $log"; FAILED_CHAINS="$FAILED_CHAINS$c "; return 1; fi
  write_done $c $s $sec $prods
  st $c $s done "$((sec/60)) min $((sec%60)) s (est $est min); $(grep -E '^\{' $log | tail -1 | cut -c1-250)"
}
leak() { python3 -I $SC/leak_scan.py > $STATE/leak_scan_last.log 2>&1; local rc=$?; st $1 - leak_scan "rc=$rc $(tail -1 $STATE/leak_scan_last.log)"; }
commit() { bash $SC/git_commit_locked.sh "$1" >> $STATE/git.log 2>&1; }

# ---------------- step bodies
small_ctm() { [ $1 = dev-subset ] && echo reference-materials/pre-norm/dev-asr.ctm || echo reference-materials/pre-norm/$1-asr.ctm; }
smallgec_ctm() { [ $1 = dev-subset ] && echo reference-materials/pre-norm/dev-gec.ctm || echo reference-materials/pre-norm/$1-gec.ctm; }
wer_line() { grep -E 'Sum/Avg' $1 | head -1 | tr -s ' |' ' '; }
s_ctm() {  # ds OUT
  local ds=$1 O=$2 F=reference-materials/flists.flac/$1-gec.tsv
  mkdir -p $O/asr_wer
  python3 -I $SC/d_ctm_check.py --ctm $O/decode/asr.ctm --flist $F --ref_ctm $(smallgec_ctm $ds) \
     --asr_stm reference-materials/stms/$ds-asr.stm --out_stm $O/asr_wer/ref-asr.stm || return 3
  ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $O/decode/asr.ctm --flist $F --task gec --out_dir $O/gec || return 4
  [ -s $O/gec/norm/gec.ctm ] || return 4
  local sys; for sys in W small; do
    local in=$O/decode/asr.ctm; [ $sys = small ] && in=$(small_ctm $ds)
    ./baseline-pipeline/run-scripts/run-asr-to-task-input.sh --input_ctms $in --flist $F --task asr --out_dir $O/asr_wer/$sys || return 5
    ./scoring-tools/run-scripts/run-wer-scoring.sh --stm $O/asr_wer/ref-asr.stm --ctm $O/asr_wer/$sys/asr.ctm --out_dir $O/asr_wer/$sys/wer || return 5
  done
  echo "{\"asr_wer_large_v3\": \"$(wer_line $O/asr_wer/W/wer/hyp/asr.ctm.filt.sys)\", \"asr_wer_small\": \"$(wer_line $O/asr_wer/small/wer/hyp/asr.ctm.filt.sys)\"}"
}
s_dd() { source ./envs/sandi-dd-env.sh && mkdir -p $1/dd_raw/dd $1/gec/LOGs && python ./baseline-pipeline/gec-tools/dd.py --input_ctm $1/gec/norm/gec.ctm \
     --disfluency_model sandi-models/dd/dd_model --fluent_ctm $1/dd_raw/fluent.ctm --gec_tsv $1/dd_raw/dd/pre-gec.tsv 2>&1 | tee $1/gec/LOGs/dd.LOG
  [ -f $1/dd_raw/fluent.ctm ] && [ -f $1/dd_raw/dd/pre-gec.tsv ]; }
s_filter() { mkdir -p $1/gec/dd && python3 -I $SC/b2b_filter_empty.py $1/dd_raw $1/gec; }
s_gec() { source ./envs/sandi-all-env.sh && python ./baseline-pipeline/gec-tools/gec-plus-align.py --input_file $1/gec/dd/pre-gec.tsv \
     --gec_model sandi-models/gec/gec_model --gec_ctm $1/gec/gec.ctm 2>&1 | tee $1/gec/LOGs/gec.LOG; [ -s $1/gec/gec.ctm ]; }
s_score() {  # ds OUT RUNNAME
  local ds=$1 O=$2
  ./scoring-tools/run-scripts/run-wer-scoring.sh --stm reference-materials/stms/$ds-gec.stm --ctm $O/gec/gec.ctm --out_dir $O/gec/wer-scoring || return 1
  ./scoring-tools/run-scripts/run-sgecf-scoring.sh --edit_detail --flt_stm reference-materials/stms/$ds-fluent.stm --gec_stm reference-materials/stms/$ds-gec.stm \
     --flt_ctm $O/gec/fluent.ctm --gec_ctm $O/gec/gec.ctm --out_dir $O/gec/sgecf-scoring || return 1
  [ -s $O/gec/sgecf-scoring/result.txt ] || return 1
  local L=$O/collect_markers.txt; cat $STATE/*.done > $L 2>/dev/null; echo "DONE D $O" >> $L
  bash $SC/b_collect.sh asr ${3#$EXP/05_runs/} $O $L || return 2
  python3 -I -c "import json;d=json.load(open('$3/metrics.json'));o=d['sgecf_overall'];print(json.dumps(dict(gec_wer=d['gec_wer']['err'],P=o['P'],R=o['R'],F05=o['F05'])))"; }
V_of() { [ $1 = dev-subset ] && echo $SD/norm_verbatim_v4_dev-subset/gec/norm/gec.ctm || echo stageB/norm_verbatim_v4_$1/gec/norm/gec.ctm; }
s_labels() { in_all python $SC/b3_edit_labels.py --run_dir $2 --ds $1 --mode asr --verbatim_norm_ctm $(V_of $1) --out_dir $3/labels_v4; }
s_bart() { mkdir -p $3/features && in_all python $SC/c1_bart_score.py --run_dir $2 --out $3/features/bart_scores_$1.jsonl && cat $3/features/bart_scores_$1.diag.json | tr -d '\n '; echo; }
s_feat() { in_all python $SC/c1_features.py --ds $1 --run_dir $2 --labels $3/labels_v4 --bart $3/features/bart_scores_$1.jsonl --out_dir $3/features; }
s_qprep() {  # ds OUT B1dir: Q uses the B1 (official Whisper-small + DD) output up to DD, unchanged
  mkdir -p $2/gec/dd && cp -R $3/gec/norm $2/gec/norm && cp $3/gec/fluent.ctm $2/gec/fluent.ctm && cp $3/gec/dd/pre-gec.tsv $2/gec/dd/pre-gec.tsv &&
  in_all python $SC/d_llm_align.py --pre_gec $2/gec/dd/pre-gec.tsv --llm $2/llm/llm_gec.tsv --gec_ctm $2/gec/gec.ctm; }

D=2026-11-10_D
# ---------------- pre-steps
if [ "$PROFILE" = rehearsal ]; then
  run_step V dev-subset "$SD/norm_verbatim_v4_dev-subset" "echo 0" 1 \
    bash $SC/b_norm_verbatim.sh dev-subset reference-materials/flists.flac/dev-subset-gec.tsv $SD/norm_verbatim_v4_dev-subset --drop_markup
  [[ "$FAILED_CHAINS" == *" V "* ]] && FAILED_CHAINS="$FAILED_CHAINS W_dev-subset Q_dev-subset "
fi
# ---------------- W chains (large-v3 ASR -> official DD -> official BART GEC)
est() { case $1:$2 in asr:dev) echo 61;; asr:train) echo 38;; asr:eval) echo 60;; dd:dev) echo 35;; dd:train) echo 23;; dd:eval) echo 34;;
  gec:dev) echo 66;; gec:train) echo 43;; gec:eval) echo 65;; llm:dev) echo 43;; llm:train) echo 28;; llm:eval) echo 42;; *) echo 1;; esac; }
for ds in $SETS; do
  c=W_$ds; O=$SD/W_$ds; R=$RUNS/${D}_W_$ds; mkdir -p $O
  run_step $c asr "$O/decode" "ls $O/decode/json 2>/dev/null | grep -c 'json\$'" $(est asr $ds) \
    env PATH=$DPATH $PY_D -I $SC/d_asr.py --flist reference-materials/flists.flac/$ds-gec.tsv --out_dir $O/decode
  run_step $c ctm "$O/gec/norm $O/asr_wer" "echo 0" 2 s_ctm $ds $O
  run_step $c dd "$O/dd_raw" "echo 0" $(est dd $ds) s_dd $O
  run_step $c filter "$O/gec/dd $O/gec/fluent.ctm" "echo 0" 1 s_filter $O
  run_step $c gec "$O/gec/gec.ctm" "echo 0" $(est gec $ds) s_gec $O
  run_step $c score "$O/gec/wer-scoring $O/gec/sgecf-scoring $R/outputs $R/metrics.json" "echo 0" 2 s_score $ds $O $R
  run_step $c labels "$R/labels_v4" "echo 0" 3 s_labels $ds $O $R
  run_step $c bart "$R/features/bart_scores_$ds.jsonl $R/features/bart_scores_$ds.diag.json" "echo 0" 7 s_bart $ds $O $R
  run_step $c features "$R/features/features_$ds.csv $R/features/utt_counts_$ds.csv" "echo 0" 1 s_feat $ds $O $R
  commit "Stage D: chain $c ($PROFILE)"; leak $c
done
# ---------------- Q chains (B1 DD output -> Qwen2.5-7B-Instruct GEC)
for ds in $SETS; do
  c=Q_$ds; O=$SD/Q_$ds; R=$RUNS/${D}_Q_$ds; mkdir -p $O
  B1=stageB/B1_$ds; [ $ds = dev-subset ] && B1=stageB/test_dev-subset_asr
  run_step $c llm "$O/llm" "wc -l < $O/llm/gen.jsonl" $(est llm $ds) $PY_D -I $SC/d_llm_gec.py --input $B1/gec/dd/pre-gec.tsv --out_dir $O/llm
  run_step $c align "$O/gec/norm $O/gec/fluent.ctm $O/gec/dd $O/gec/gec.ctm" "echo 0" 1 s_qprep $ds $O $B1
  run_step $c score "$O/gec/wer-scoring $O/gec/sgecf-scoring $R/outputs $R/metrics.json" "echo 0" 2 s_score $ds $O $R
  run_step $c labels "$R/labels_v4" "echo 0" 3 s_labels $ds $O $R
  run_step $c bart "$R/features/bart_scores_$ds.jsonl $R/features/bart_scores_$ds.diag.json" "echo 0" 7 s_bart $ds $O $R
  run_step $c features "$R/features/features_$ds.csv $R/features/utt_counts_$ds.csv" "echo 0" 1 s_feat $ds $O $R
  commit "Stage D: chain $c ($PROFILE)"; leak $c
done
# ---------------- M: modelling (full only). (a) frozen C6 models on the new eval; (b) retrain on new train+dev, eval once.
if [ "$PROFILE" = full ]; then
  for S in W Q; do
    need_ok() { local x; for x in "$@"; do [[ "$FAILED_CHAINS" == *" $x "* ]] && return 1; done; return 0; }
    EF=$RUNS/${D}_${S}_eval/features; EL=$RUNS/${D}_${S}_eval/labels_v4
    if need_ok ${S}_eval; then
      run_step M_$S frozen "$EXP/06_results/D_${S}_eval_frozen.json" "echo 0" 1 \
        $PY_C -I $SC/d_model.py frozen --system $S --eval_features $EF --eval_labels $EL --out $EXP/06_results/D_${S}_eval_frozen.json
    else st M_$S frozen SKIPPED_DEP "${S}_eval failed"; fi
    if need_ok ${S}_dev ${S}_train ${S}_eval; then
      run_step M_$S retrain "$EXP/06_results/D_${S}_eval_retrained.json $RUNS/${D}_${S}_model" "echo 0" 35 \
        $PY_C -I $SC/d_model.py retrain --system $S --features_dirs $RUNS/${D}_${S}_train/features $RUNS/${D}_${S}_dev/features \
        --eval_features $EF --eval_labels $EL --root $RUNS/${D}_${S}_model --out $EXP/06_results/D_${S}_eval_retrained.json
    else st M_$S retrain SKIPPED_DEP "a ${S} chain failed"; fi
    commit "Stage D: modelling $S"; leak M_$S
  done
fi
st QUEUE - DONE "failed chains:${FAILED_CHAINS}"

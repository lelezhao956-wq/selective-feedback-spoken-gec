#!/bin/bash
# Stage B overnight queue.  Start (from anywhere):
#   cd "$RUN_DIR" && nohup caffeinate -i bash $SCRIPTS_DIR/b_queue.sh 85841 \
#        > stageB/b_queue.log 2>&1 &
# 1. waits until PID $1 (the running B1 dev pipeline, b_pipeline.sh v1) has exited;
# 2. finishes B1 dev (collect + B3 labels), then runs in order B1 train -> B2 dev -> B2 train -> B2b dev -> B2b train
#    with b_pipeline2.sh; after each part: collect (+commit), then B3 labels in the background (+commit) while the next
#    part's pipeline already runs;
# 3. dependencies: collect and labels of a part depend only on that part's pipeline; B4 for a dataset needs the labels of
#    B1, B2b and B2 for that dataset (with fewer conditions it still runs and is marked PARTIAL). A failed part is logged
#    and only its dependents are skipped; all other parts continue.
# 4. after every commit a leakage scan (scripts/leak_scan.py) is run and logged.
# Status: one line per event in $EXP_DIR/05_runs/b_queue_status.tsv (time, part, phase, detail); full log stageB/b_queue.log.
WAIT_PID=${1:-85841}
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; C=$RUN_DIR
STATUS=$EXP/05_runs/b_queue_status.tsv
cd $C || exit 1
[ -f $STATUS ] || printf 'time\tpart\tphase\tdetail\n' > $STATUS
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $STATUS; echo "[$(date '+%F %T')] $1 | $2 | $3"; }
VERB=stageB/norm_verbatim

# estimated minutes (from the 200-utterance throughput test) for calibration in the lab notebook
est() { case $1 in B1_dev) echo 110;; B1_train) echo 72;; B2_dev) echo 61;; B2_train) echo 41;; B2b_dev) echo 95;; B2b_train) echo 61;; *) echo NA;; esac; }  # bash 3.2: no assoc arrays
LABEL_PIDS=()

leak() { python3 -I $SC/leak_scan.py > stageB/leak_scan_last.log 2>&1; local rc=$?; st "$1" leak_scan "rc=$rc $(tail -1 stageB/leak_scan_last.log)"; }

label_bg() {   # $1 part  $2 mode  $3 ds  $4 run_name
  local part=$1 mode=$2 ds=$3 run=$4
  (
    t0=$(date +%s)
    source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL
    python $SC/b3_edit_labels.py --run_dir stageB/$part --ds $ds --mode $mode \
        --verbatim_norm_ctm ${VERB}_$ds/gec/norm/gec.ctm --out_dir $EXP/05_runs/$run/labels \
        > stageB/labels_$part.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch stageB/.labels_ok_$part
      st $part labels_ok "$(( $(date +%s)-t0 ))s; $(tail -1 stageB/labels_$part.log | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); o=d["overall_from_labels"]; print("TP",o["tp"],"FP",o["fp"],"P/R/F",o["P"],o["R"],o["F05"])' 2>/dev/null)"
      bash $SC/git_commit_locked.sh "Stage B3: labels $run" >> stageB/b_queue_git.log 2>&1
      leak $part
    else
      st $part labels_FAILED "rc=$rc (3 = verification against official scorer failed); see stageB/labels_$part.log; B4 for $ds will be PARTIAL"
    fi
  ) &
  LABEL_PIDS+=($!)
}

finish_part() {  # $1 part  $2 mode  $3 ds ; pipeline already ran, log stageB/$part.log
  local part=$1 mode=$2 ds=$3 run=2026-11-10_$1
  if grep -q '^DONE ' stageB/$part.log; then
    st $part pipeline_ok "$(grep -E '^(STEP|SUBSTEP)' stageB/$part.log | awk '{printf "%s=%ss ", $2, $4-$3}')"
    if bash $SC/b_collect.sh $mode $run stageB/$part stageB/$part.log >> stageB/b_queue_git.log 2>&1; then
      st $part collect_ok "05_runs/$run ($(python3 -I -c "import json;d=json.load(open('$EXP/05_runs/$run/metrics.json'));o=d['sgecf_overall'];print('GEC WER',d['gec_wer']['err'],'SGECF',o['P'],o['R'],o['F05'])"))"
      leak $part
      label_bg $part $mode $ds $run
    else
      st $part collect_FAILED "see stageB/b_queue_git.log; labels for $part skipped"
    fi
  else
    st $part pipeline_FAILED "$(grep -E '^ABORT' stageB/$part.log | tail -1); collect+labels for $part skipped"
  fi
}

st queue start "waiting for PID $WAIT_PID (B1 dev, b_pipeline.sh v1)"
while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
st B1_dev pipeline_end "est $(est B1_dev) min; wallclock $(tr '\n' ' ' < stageB/B1_dev.wallclock 2>/dev/null)"
finish_part B1_dev asr dev

for spec in "B1_train asr train" "B2_dev fluent dev" "B2_train fluent train" "B2b_dev verbatim dev" "B2b_train verbatim train"; do
  set -- $spec; part=$1; mode=$2; ds=$3
  if [ -e stageB/$part ]; then st $part SKIPPED "stageB/$part already exists (not overwritten)"; continue; fi
  st $part pipeline_start "est $(est $part) min"
  t0=$(date +%s)
  bash $SC/b_pipeline2.sh $mode $ds reference-materials/flists.flac/$ds-gec.tsv stageB/$part > stageB/$part.log 2>&1
  st $part pipeline_end "rc=$? actual $(( ($(date +%s)-t0)/60 )) min vs est $(est $part) min"
  finish_part $part $mode $ds
done

st queue waiting_for_labels "label jobs: ${LABEL_PIDS[*]:-none}"
for p in ${LABEL_PIDS[*]:-}; do wait $p; done

for ds in dev train; do
  conds=(); missing=()
  for pc in "B1 B1_$ds" "B2b B2b_$ds" "B2 B2_$ds"; do
    set -- $pc
    if [ -f stageB/.labels_ok_$2 ]; then conds+=(--cond "$1=$EXP/05_runs/2026-11-10_$2/labels"); else missing+=($1); fi
  done
  if [ ${#conds[@]} -eq 0 ]; then st B4_$ds SKIPPED "no labels available"; continue; fi
  python3 -I $SC/b4_label_stats.py --ds $ds "${conds[@]}" --out $EXP/06_results/label_stats_$ds.csv > stageB/b4_$ds.log 2>&1
  rc=$?
  if [ ${#missing[@]} -gt 0 ]; then tag="PARTIAL (missing: ${missing[*]})"; else tag="complete (B1, B2b, B2)"; fi
  st B4_$ds "b4_rc=$rc" "$tag -> 06_results/label_stats_$ds.csv"
done
bash $SC/git_commit_locked.sh "Stage B4: label statistics (queue)" >> stageB/b_queue_git.log 2>&1
leak B4
st queue DONE "all parts processed; see this file for failures"

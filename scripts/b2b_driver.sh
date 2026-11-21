#!/bin/bash
# B2b fix driver (2026-11-10). Start from anywhere:
#   cd "$RUN_DIR" && nohup caffeinate -i bash $SCRIPTS_DIR/b2b_driver.sh \
#        > stageB/b2b_driver.log 2>&1 & disown
# For dev then train: b2b_regec.sh (filter empty phrases -> GEC -> GEC WER -> SGECF, reusing the DD output of the failed
# queue run) -> collect + commit -> B3 labels + commit -> leak scan. Then B4 for dev and train with B1, B2b, B2 -> commit.
# Status lines are appended to $EXP_DIR/05_runs/b_queue_status.tsv (same format as b_queue.sh).
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; C=$RUN_DIR
STATUS=$EXP/05_runs/b_queue_status.tsv
cd $C || exit 1
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $STATUS; echo "[$(date '+%F %T')] $1 | $2 | $3"; }
leak() { python3 -I $SC/leak_scan.py > stageB/leak_scan_last.log 2>&1; local rc=$?; st "$1" leak_scan "rc=$rc $(tail -1 stageB/leak_scan_last.log)"; }
est() { case $1 in dev) echo 35;; train) echo 23;; esac; }
st B2b_fix start "regec dev+train (reuse DD of failed queue runs, empty phrases filtered)"
for ds in dev train; do
  part=B2b_${ds}_regec; run=2026-11-10_B2b_$ds; t0=$(date +%s)
  st $part pipeline_start "est GEC $(est $ds) min + scoring ~1 min"
  bash $SC/b2b_regec.sh $ds > stageB/$part.log 2>&1
  st $part pipeline_end "rc=$? actual $(( ($(date +%s)-t0)/60 )) min"
  if ! grep -q '^DONE ' stageB/$part.log; then st $part pipeline_FAILED "$(grep '^ABORT' stageB/$part.log | tail -1)"; continue; fi
  st $part pipeline_ok "$(grep -E '^STEP' stageB/$part.log | awk '{printf "%s=%ss ", $2, $4-$3}')"
  mkdir -p $EXP/05_runs/$run
  cp stageB/$part/gec/dd/filtered_empty_phrases.tsv $EXP/05_runs/$run/filtered_empty_phrases.tsv
  grep -E '^\{' stageB/$part.log > $EXP/05_runs/$run/filter_summary.json
  if bash $SC/b_collect.sh verbatim $run stageB/$part stageB/$part.log >> stageB/b_queue_git.log 2>&1; then
    st $part collect_ok "05_runs/$run ($(python3 -I -c "import json;d=json.load(open('$EXP/05_runs/$run/metrics.json'));o=d['sgecf_overall'];print('GEC WER',d['gec_wer']['err'],'SGECF',o['P'],o['R'],o['F05'])"))"
    leak $part
  else st $part collect_FAILED "see stageB/b_queue_git.log"; continue; fi
  t1=$(date +%s)
  ( source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL && \
    python $SC/b3_edit_labels.py --run_dir stageB/$part --ds $ds --mode verbatim \
      --verbatim_norm_ctm stageB/norm_verbatim_$ds/gec/norm/gec.ctm --out_dir $EXP/05_runs/$run/labels ) > stageB/labels_$part.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch stageB/.labels_ok_B2b_$ds
    st $part labels_ok "$(( $(date +%s)-t1 ))s; $(tail -1 stageB/labels_$part.log | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); o=d["overall_from_labels"]; print("TP",o["tp"],"FP",o["fp"],"P/R/F",o["P"],o["R"],o["F05"])' 2>/dev/null)"
    bash $SC/git_commit_locked.sh "Stage B3: labels $run (regec)" >> stageB/b_queue_git.log 2>&1
    leak $part
  else st $part labels_FAILED "rc=$rc; see stageB/labels_$part.log"; fi
done
for ds in dev train; do
  conds=(); missing=()
  for pc in "B1 B1_$ds" "B2b B2b_$ds" "B2 B2_$ds"; do
    set -- $pc
    if [ -f stageB/.labels_ok_$2 ]; then conds+=(--cond "$1=$EXP/05_runs/2026-11-10_$2/labels"); else missing+=($1); fi
  done
  python3 -I $SC/b4_label_stats.py --ds $ds "${conds[@]}" --out $EXP/06_results/label_stats_$ds.csv > stageB/b4_$ds.log 2>&1
  rc=$?
  if [ ${#missing[@]} -gt 0 ]; then tag="PARTIAL (missing: ${missing[*]})"; else tag="complete (B1, B2b, B2)"; fi
  st B4_$ds "b4_rc=$rc" "$tag -> 06_results/label_stats_$ds.csv"
done
bash $SC/git_commit_locked.sh "Stage B4: label statistics with B2b (regec)" >> stageB/b_queue_git.log 2>&1
leak B4
st B2b_fix DONE "see this file for failures"
bash $SC/git_commit_locked.sh "Queue status: B2b fix done" >> stageB/b_queue_git.log 2>&1

#!/bin/bash
# B2b v2 driver (2026-11-10). Start:  cd "$RUN_DIR" && nohup caffeinate -i bash <this> > stageB/b2b_v2_driver.log 2>&1 & disown
# dev then train: b2b_v2.sh -> collect + commit -> B3 labels (reference V v4 = stageB/norm_verbatim_v4_<ds>) + commit -> leak scan.
# Status lines appended to $EXP_DIR/05_runs/b_queue_status.tsv.
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; C=$RUN_DIR; STATUS=$EXP/05_runs/b_queue_status.tsv
cd $C || exit 1
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $STATUS; echo "[$(date '+%F %T')] $1 | $2 | $3"; }
leak() { python3 -I $SC/leak_scan.py > stageB/leak_scan_last.log 2>&1; st "$1" leak_scan "rc=$? $(tail -1 stageB/leak_scan_last.log)"; }
est() { case $1 in dev) echo 92;; train) echo 60;; esac; }
st B2b_v2 start "markup-free verbatim input; dev then train"
for ds in dev train; do
  part=B2b_${ds}_v2; run=2026-11-10_B2b_${ds}_v2; t0=$(date +%s)
  st $part pipeline_start "est $(est $ds) min (DD+GEC+scoring)"
  bash $SC/b2b_v2.sh $ds > stageB/$part.log 2>&1
  st $part pipeline_end "rc=$? actual $(( ($(date +%s)-t0)/60 )) min vs est $(est $ds) min"
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
      --verbatim_norm_ctm stageB/norm_verbatim_v4_$ds/gec/norm/gec.ctm --out_dir $EXP/05_runs/$run/labels ) > stageB/labels_$part.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch stageB/.labels_ok_$part
    st $part labels_ok "$(( $(date +%s)-t1 ))s; $(tail -1 stageB/labels_$part.log | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); o=d["overall_from_labels"]; print("TP",o["tp"],"FP",o["fp"],"P/R/F",o["P"],o["R"],o["F05"])' 2>/dev/null)"
    bash $SC/git_commit_locked.sh "Stage B3: labels $run (v4 reference)" >> stageB/b_queue_git.log 2>&1
    leak $part
  else st $part labels_FAILED "rc=$rc; see stageB/labels_$part.log"; fi
done
st B2b_v2 DONE "see this file for failures"
bash $SC/git_commit_locked.sh "Queue status: B2b v2 done" >> stageB/b_queue_git.log 2>&1

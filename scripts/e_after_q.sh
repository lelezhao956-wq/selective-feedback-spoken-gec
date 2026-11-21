#!/bin/bash
# After the stage-D Q queue: (1) stage E exploratory analyses for Q, (2) GRECO comparison (only if env + clone + checkpoint present).
# Start: nohup caffeinate -ims nice -n 10 bash e_after_q.sh [all|greco] > 05_runs/e_after_q.log 2>&1 & disown
# Never touches the Q queue: it only waits for the Q queue process (pid in $EXP_DIR/05_runs/D_state/queue_Q.lock) to end;
# stops with an error if that lock file is missing (run the Q queue first: d_resume.sh Q). Stops with an error if a GRECO prerequisite is missing.
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; RUNS=$EXP/05_runs; ST=$RUNS/e_status.tsv; CORPUS=$RUN_DIR
PYC=$CONDA_BASE/envs/$ENV_ESWA_C/bin/python; PYG=$CONDA_BASE/envs/$ENV_ESWA_G/bin/python
GDIR=${GRECO_DIR:-$HOME/greco_code}; CKPT=$GDIR/models/checkpoint.bin; CKPT_SHA=1a11360c12bdfb20660555fe1e4a71e28b4c21e0acd0b4077d8af80a8f096884
LM=$HF_HUB_CACHE/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f
GD=$RUNS/2026-11-10_G_greco; MODE=${1:-all}
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $ST; }
commit() { bash $SC/git_commit_locked.sh "$1" >> $RUNS/e_git.log 2>&1; }
leak() { python3 -I $SC/leak_scan.py > $RUNS/e_leak_last.log 2>&1; st "$1" leak_scan "rc=$? $(tail -1 $RUNS/e_leak_last.log)"; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
st AFTER_Q start "pid $$ mode $MODE"
# ---- wait for the Q queue
LOCKF=$RUNS/D_state/queue_Q.lock
[ -f $LOCKF ] || { st AFTER_Q FAILED "Q queue lock file not found ($LOCKF): run the Q queue first (d_resume.sh Q)"; echo "ERROR: $LOCKF not found" >&2; exit 1; }
QP=$(grep -oE '[0-9]+' $LOCKF | head -1)
[ -n "$QP" ] || { st AFTER_Q FAILED "no pid in $LOCKF"; echo "ERROR: no pid in $LOCKF" >&2; exit 1; }
while kill -0 $QP 2>/dev/null; do sleep 60; done
st AFTER_Q q_queue_ended "pid $QP"
# ---- (1) stage E for Q
if [ $MODE = all ]; then
  if [ -f $EXP/06_results/D_Q_eval_retrained.json ] && [ -f $RUNS/2026-11-10_D_Q_model/05_runs/2026-11-10_C2_cv/oof_scores.csv ]; then
    st Q start "stage E"; t0=$(date +%s)
    if nice -n 10 $PYC -I $SC/e_explore.py --system Q > $RUNS/e_Q.log 2>&1; then st Q done "$(( $(date +%s)-t0 )) s; $(tail -1 $RUNS/e_Q.log | cut -c1-200)"
    else st Q FAILED "rc=$? see 05_runs/e_Q.log"; fi
    leak Q; commit "Stage E (exploratory): Q"
  else st Q SKIPPED "D_Q_eval_retrained.json or the Q model OOF missing (Q chain failed or incomplete)"; fi
fi
# ---- (2) GRECO
miss=""
[ -x $PYG ] || miss="$miss env eswa-g;"
[ -f $GDIR/models.py ] || miss="$miss GRECO clone ($GDIR);"
[ -d $LM ] || miss="$miss deberta-v3-large snapshot;"
if [ -f $CKPT ]; then [ "$(shasum -a 256 $CKPT | cut -d' ' -f1)" = $CKPT_SHA ] || miss="$miss checkpoint sha256 mismatch;"; else miss="$miss checkpoint;"; fi
if [ -n "$miss" ]; then
  st GRECO FAILED "missing:$miss (see README, GRECO; then run 'bash $SC/e_after_q.sh greco')"
  echo "ERROR: GRECO prerequisites missing:$miss" >&2; exit 1
fi
mkdir -p $GD
gscore() {  # system ds run_dir labels [extra]
  local s=$1 ds=$2 rd=$3 lb=$4; shift 4
  ( cd $CORPUS && nice -n 10 $PYG -I $SC/g_greco_score.py --greco_dir $GDIR --checkpoint $CKPT --lm $LM --run_dir $rd --labels $lb --ds $ds \
      --out $GD/greco_${s}_${ds}$([ $# -gt 0 ] && echo _test).csv "$@" ) > $GD/greco_${s}_${ds}.log 2>&1
}
# dev-subset validation + speed
if gscore B1 dev stageB/B1_dev $RUNS/2026-11-10_B1_dev/labels_v4 --subset_flist reference-materials/flists.flac/dev-subset-gec.tsv; then
  st GRECO dev_subset_test "$(tail -1 $GD/greco_B1_dev.log | cut -c1-250)"
  $PYC -I -c "
import json,sys; d=json.load(open('$GD/greco_B1_dev_test.diag.json'))
sys.exit(0 if d['token_count_mismatch']==0 and d['scored_now']+d['kept']==d['n_edits'] else 4)" || { st GRECO FAILED "dev-subset validation failed -> GRECO abandoned"; commit "GRECO: dev-subset validation failed"; exit 0; }
else st GRECO FAILED "dev-subset scoring crashed (see $GD/greco_B1_dev.log) -> GRECO abandoned"; commit "GRECO: dev-subset crash"; exit 0; fi
# full scoring (resumable) + model, one system at a time (B1, W as planned; Q added after the eval, supplement amendment 2026-11-11)
for s in ${GRECO_SYSTEMS:-B1 W Q}; do
  sf=0
  for ds in train dev eval; do
    case $s in
      B1) rd=stageB/B1_$ds; lb=$RUNS/2026-11-10_B1_$ds/labels_v4; [ $ds = eval ] && lb=$RUNS/2026-11-10_C7_B1_eval/labels_v4;;
      W)  rd=stageD/W_$ds; lb=$RUNS/2026-11-10_D_W_$ds/labels_v4;;
      Q)  rd=stageD/Q_$ds; lb=$RUNS/2026-11-10_D_Q_$ds/labels_v4;;
    esac
    t0=$(date +%s)
    if gscore $s $ds $rd $lb; then st GRECO score_${s}_${ds} "$(( ($(date +%s)-t0)/60 )) min; $(tail -1 $GD/greco_${s}_${ds}.log | cut -c1-200)"
    else st GRECO FAILED "scoring $s $ds rc=$? (see $GD/greco_${s}_${ds}.log)"; sf=1; fi
  done
  commit "GRECO: scores $s"
  if [ $sf = 1 ]; then st GRECO SKIPPED_DEP "model $s: scoring failed"; continue; fi
  out=$EXP/06_results/GRECO_${s}_eval.json
  if [ -f $out ]; then st GRECO model_$s "exists (eval scored once) -> not rerun"; continue; fi
  if nice -n 10 $PYC -I $SC/g_greco_model.py --system $s --out $out > $GD/model_$s.log 2>&1; then st GRECO model_$s "$(tail -1 $GD/model_$s.log | cut -c1-250)"
  else st GRECO FAILED "model $s rc=$? (see $GD/model_$s.log)"; fi
  leak GRECO_$s; commit "GRECO comparison $s (added after the C7 eval)"
done
st AFTER_Q end "done"

#!/bin/bash
# Start or resume a stage D queue (same command for the first start and after any interruption).
# Usage: bash $SCRIPTS_DIR/d_resume.sh PROFILE
#   W            : W chain (Whisper large-v3) dev/train/eval + W modelling     -> d_queue_w.sh
#   Q            : Q chain (Qwen GEC) dev/train/eval + Q modelling; waits until no W queue is running -> d_queue_q.sh
#   W_rehearsal / Q_rehearsal : dev-subset rehearsals (separate state, status and heartbeat files)
#   rehearsal    : the original combined rehearsal (d_queue.sh, kept for the record)
# Steps with a valid .done marker are skipped; unfinished outputs are renamed *.partial-<time> by the queue (never deleted).
PROFILE=${1:?profile}; source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR
case $PROFILE in
  W) Q=d_queue_w.sh; SUF=""; L=queue_W.lock;;  W_rehearsal) Q=d_queue_w.sh; SUF=_wrehearsal; L=queue_W.lock;;
  Q) Q=d_queue_q.sh; SUF=""; L=queue_Q.lock;;  Q_rehearsal) Q=d_queue_q.sh; SUF=_qrehearsal; L=queue_Q.lock;;
  rehearsal) Q=d_queue.sh; SUF=_rehearsal; L=queue.lock;;
  *) echo "unknown profile $PROFILE"; exit 100;;
esac
STATE=$EXP/05_runs/D_state$SUF; LOCK=$STATE/$L; mkdir -p $STATE
if [ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "$PROFILE queue already running (pid $(cat $LOCK)) - not starting a second one"; exit 1; fi
cd "$RUN_DIR" || exit 1
LOG=$STATE/queue_${PROFILE}_$(date '+%Y%m%d-%H%M%S').log
nohup caffeinate -ims bash $SC/$Q $PROFILE > $LOG 2>&1 &
CPID=$!; disown
sleep 3
echo "started $PROFILE: pid $CPID, log $LOG"
pmset -g assertions | grep -E 'PreventUserIdleSystemSleep|PreventSystemSleep|PreventDiskIdle|caffeinate' | tee $STATE/pmset_assertions_${PROFILE}_$(date '+%Y%m%d-%H%M%S').txt

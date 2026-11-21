#!/bin/bash
# Stage E (exploratory) runner. Usage: nohup caffeinate -ims nice -n 10 bash e_run.sh B1 W W_frozen > log 2>&1 & disown
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; PY=$CONDA_BASE/envs/$ENV_ESWA_C/bin/python; ST=$EXP/05_runs/e_status.tsv
[ -f $ST ] || printf 'time\tsystem\tphase\tdetail\n' > $ST
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $ST; }
cd $EXP
for s in "$@"; do
  st $s start "pid $$"; t0=$(date +%s)
  if nice -n 10 $PY -I $SC/e_explore.py --system $s > $EXP/05_runs/e_${s}.log 2>&1; then st $s done "$(( $(date +%s)-t0 )) s; $(tail -1 $EXP/05_runs/e_${s}.log | cut -c1-200)"
  else st $s FAILED "rc=$? after $(( $(date +%s)-t0 )) s; see 05_runs/e_${s}.log"; fi
  python3 -I $SC/leak_scan.py > $EXP/05_runs/e_leak_last.log 2>&1; st $s leak_scan "rc=$? $(tail -1 $EXP/05_runs/e_leak_last.log)"
  bash $SC/git_commit_locked.sh "Stage E (exploratory): $s" >> $EXP/05_runs/e_git.log 2>&1
done
st ALL end "$*"

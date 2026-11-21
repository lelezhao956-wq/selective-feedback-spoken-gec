#!/bin/bash
# Morning check of the stage D queues.  Usage: bash $SCRIPTS_DIR/d_check.sh [all|W|Q|rehearsal]   (default all)
PROFILE=${1:-all}; PATHS_NO_CORPUS=1 source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR
if [ $PROFILE = rehearsal ]; then SUF=_rehearsal; else SUF=""; fi
STATE=$EXP/05_runs/D_state$SUF; STATUS=$EXP/05_runs/d_queue_status$SUF.tsv; HB=$EXP/05_runs/d_heartbeat$SUF.tsv
echo "== now $(date '+%F %T')"
echo "== queue processes"
for q in W Q; do L=$STATE/queue_$q.lock
  if [ -f $L ] && kill -0 "$(cat $L)" 2>/dev/null; then echo "$q: RUNNING (pid $(cat $L))"; else echo "$q: not running"; fi; done
echo "   last queue events:"; grep -E '	QUEUE' $STATUS | tail -4 | cut -f1-5 | cut -c1-160
echo "== steps"
python3 -I - "$STATUS" "$STATE" "$PROFILE" <<'PY'
import sys, os, datetime as dt
status, state, prof = sys.argv[1:]
sets = ["dev", "train", "eval"]
W = ["asr", "ctm", "dd", "filter", "gec", "score", "labels", "bart", "features"]
Q = ["llm", "align", "score", "labels", "qscore", "features", "bart2", "features2"]
oW = [f"W_{s}/{x}" for s in sets for x in W] + ["M_W/frozen", "M_W/retrain"]
oQ = [f"Q_{s}/{x}" for s in sets for x in Q] + ["M_Q/retrain", "M_Q/retrain_bart"]
order = oW if prof == "W" else oQ if prof == "Q" else oW + oQ
E = {"asr": dict(dev=66, train=42, eval=66), "dd": dict(dev=35, train=23, eval=34), "gec": dict(dev=66, train=43, eval=65),
     "llm": dict(dev=54, train=36, eval=53), "qscore": dict(dev=25, train=16, eval=24), "bart": 7, "bart2": 7, "labels": 3, "score": 2,
     "ctm": 2, "retrain": 30, "retrain_bart": 30, "align": 1}
def est(k):
    c, s = k.split("/"); ds = c.split("_", 1)[1]; v = E.get(s, 1)
    return v.get(ds, 1) if isinstance(v, dict) else v
last = {}
for l in open(status).read().splitlines()[1:]:
    t, c, s, ph, det = (l.split("\t") + [""] * 5)[:5]
    if not c.startswith("QUEUE"):
        last[f"{c}/{s}"] = (t, ph, det)
rem, now = 0, dt.datetime.now(); stt = {}
for k in order:
    c, s = k.split("/"); ok = os.path.exists(f"{state}/{c}.{s}.done"); t, ph, det = last.get(k, ("", "pending", ""))
    st = "done" if ok and ph in ("done", "SKIP_DONE") else "running" if ph == "start" else "STALLED" if ph == "STALLED" else \
         "FAILED" if ph == "FAILED" else "skipped (dependency failed)" if ph == "SKIPPED_DEP" else "pending"
    if st in ("pending", "running", "STALLED"):
        e = est(k)
        if st != "pending" and t:
            e = max(e - (now - dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")).seconds // 60, 1)
        rem += e
    print(f"{k:20s} {st:12s} {t[11:]:8s} {det[:120]}")
print(f"== remaining (planning estimates, running steps pro rata): {rem} min -> ETA {(now + dt.timedelta(minutes=rem)):%m-%d %H:%M}")
PY
echo "== last heartbeats"; tail -6 $HB 2>/dev/null | column -t -s $'\t'
S=$(head -1 $STATE/starts 2>/dev/null)
echo "== sleep/wake events since the first queue start ($S)"
if [ -n "$S" ]; then n=$(pmset -g log | grep -E "Sleep|Wake|DarkWake" | awk -v s="$S" '/^20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] / && substr($0,1,19) >= s' | grep -vE 'Assertions|Kernel Idle sleep preventers' | tee /dev/stderr | wc -l); echo "   ($n events)"; fi
echo "== caffeinate assertions now"; pmset -g assertions | grep -A1 caffeinate | grep -E 'Details' | sort | uniq -c

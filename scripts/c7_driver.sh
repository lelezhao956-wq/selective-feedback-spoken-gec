#!/bin/bash
# C7 (eval, run ONCE) following 03_configs/final.yaml eval_procedure_C7 + 2026-11-10_eval_plan_supplement.yaml.
# Start: cd "$RUN_DIR" && nohup caffeinate -i bash <this> > stageB/c7_driver.log 2>&1 & disown
# Stops at the first failure (no step is retried automatically). Refuses to run if an eval result already exists.
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; EXP=$EXP_DIR; SC=$SCRIPTS_DIR; C=$RUN_DIR
STATUS=$EXP/05_runs/c7_status.tsv; RUN=2026-11-10_C7_B1_eval; R=$EXP/05_runs/$RUN; RES=$EXP/06_results/eval_results.json
cd $C || exit 1
[ -e $RES ] && { echo "ERROR: $RES exists - eval already scored"; exit 100; }
[ -e stageB/B1_eval ] && { echo "ERROR: stageB/B1_eval exists"; exit 100; }
[ -f $STATUS ] || printf 'time\tstep\tphase\tdetail\n' > $STATUS
st() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> $STATUS; echo "[$(date '+%F %T')] $1 | $2 | $3"; }
leak() { python3 -I $SC/leak_scan.py > stageB/leak_scan_last.log 2>&1; st "$1" leak_scan "rc=$? $(tail -1 stageB/leak_scan_last.log)"; }
die() { st "$1" FAILED "$2"; exit 1; }
st C7 start "eval run once; final.yaml + eval_plan_supplement"
t0=$(date +%s)
bash $SC/b_pipeline2.sh asr eval reference-materials/flists.flac/eval-gec.tsv stageB/B1_eval > stageB/B1_eval.log 2>&1 || die pipeline "see stageB/B1_eval.log"
grep -q '^DONE ' stageB/B1_eval.log || die pipeline "no DONE line"
st pipeline ok "$(( ($(date +%s)-t0)/60 )) min; $(grep -E '^(STEP|SUBSTEP)' stageB/B1_eval.log | awk '{printf "%s=%ss ", $2, $4-$3}')"
bash $SC/b_collect.sh asr $RUN stageB/B1_eval stageB/B1_eval.log >> stageB/b_queue_git.log 2>&1 || die collect "see stageB/b_queue_git.log"
st collect ok "$(python3 -I -c "import json;d=json.load(open('$R/metrics.json'));o=d['sgecf_overall'];print('GEC WER',d['gec_wer']['err'],'SGECF',o['P'],o['R'],o['F05'])")"
leak collect
bash $SC/b_norm_verbatim.sh eval reference-materials/flists.flac/eval-gec.tsv stageB/norm_verbatim_v4_eval --drop_markup > stageB/norm_verbatim_v4_eval.log 2>&1 || die norm_verbatim "see log"
st norm_verbatim ok "$(tail -1 stageB/norm_verbatim_v4_eval.log)"
source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL
python $SC/b3_edit_labels.py --run_dir stageB/B1_eval --ds eval --mode asr --verbatim_norm_ctm stageB/norm_verbatim_v4_eval/gec/norm/gec.ctm \
   --out_dir $R/labels_v4 > stageB/labels_B1_eval_v4.log 2>&1 || die labels "rc=$? (3 = verification vs official scorer failed)"
st labels ok "$(tail -1 stageB/labels_B1_eval_v4.log | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); o=d["overall_from_labels"]; print("TP",o["tp"],"FP",o["fp"],"P/R/F",o["P"],o["R"],o["F05"])')"
mkdir -p $R/features
python $SC/c1_bart_score.py --run_dir stageB/B1_eval --out $R/features/bart_scores_eval.jsonl > stageB/bart_eval.log 2>&1 || die bart "see stageB/bart_eval.log"
st bart ok "$(cat $R/features/bart_scores_eval.diag.json | tr -d '\n ')"
python $SC/c1_features.py --ds eval --run_dir stageB/B1_eval --labels $R/labels_v4 --bart $R/features/bart_scores_eval.jsonl \
   --out_dir $R/features > stageB/features_eval.log 2>&1 || die features "rc=$? (3 = verification failed)"
st features ok "$(tail -1 stageB/features_eval.log | cut -c1-300)"
bash $SC/git_commit_locked.sh "Stage C7: B1 eval pipeline, labels v4, features (before scoring)" >> stageB/b_queue_git.log 2>&1
leak features
conda deactivate
$CONDA_BASE/envs/$ENV_ESWA_C/bin/python -I $SC/c7_eval.py --features $R/features --labels $R/labels_v4 --out $RES > stageB/c7_eval.log 2>&1 || die eval_scoring "see stageB/c7_eval.log"
st eval_scoring ok "$(tail -1 stageB/c7_eval.log)"
bash $SC/git_commit_locked.sh "Stage C7: eval results (run once)" >> stageB/b_queue_git.log 2>&1
leak eval_scoring
st C7 DONE "total $(( ($(date +%s)-t0)/60 )) min"

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL
for ds in dev train; do t0=$(date +%s); python $SCRIPTS_DIR/c1_bart_score.py --run_dir stageB/B1_$ds --out $EXP_DIR/05_runs/2026-11-10_C1_features/bart_scores_$ds.jsonl; echo "BART_DONE $ds rc=$? $(( $(date +%s)-t0 ))s"; done

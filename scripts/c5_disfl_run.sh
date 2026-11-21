source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1; source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_SANDI_ALL
for ds in dev train; do python $SCRIPTS_DIR/c5_b2b_residual_disfl.py --ds $ds --out $EXP_DIR/05_runs/2026-11-10_C5_extra/b2b_residual_disfl_$ds.json; echo "DISFL_DONE $ds rc=$?"; done

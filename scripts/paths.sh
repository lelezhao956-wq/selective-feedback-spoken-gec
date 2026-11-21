# Shared path configuration, sourced by every shell script in this directory (no hard-coded local paths, no silent defaults
# for output locations). Set before running (README, Setup):
#   EXP_DIR      your work directory for research outputs (05_runs/, 06_results/, 07_figures/); must exist; must NOT be the
#                package itself                                                                                    [required]
#   CORPUS_DIR   root of the Speak & Improve Corpus 2025 (read-only inputs)                                          [required]
#   RUN_DIR      run root with the official layout (baseline-pipeline/, scoring-tools/, envs/, check-pipeline.sh, data/,
#                reference-materials/, sandi-models/), built with scripts/make_run_root.sh; pipeline outputs and logs are
#                written under $RUN_DIR (stageB/, stageD/, ...)                                                     [required]
#   CONDA_BASE   conda installation                                                                     [default: $HOME/miniconda3]
#   ENV_SANDI_ALL, ENV_SANDI_DD, ENV_ESWA_C, ENV_ESWA_D, ENV_ESWA_G   conda environment names   [default: sandi-all, sandi-dd, eswa-c, eswa-d, eswa-g]
#   HF_HUB_CACHE (or HF_HOME)   Hugging Face hub cache                          [default: ${HF_HOME:-$HOME/.cache/huggingface}/hub]
#   GIT_COMMIT   set to 1 to let git_commit_locked.sh commit $EXP_DIR                                      [default: never commit]
# Set by this file (do not set them yourself): SCRIPTS_DIR (this directory), PKG_DIR (the package root),
#   CONFIGS_DIR (= $PKG_DIR/configs), PUBLISHED_RESULTS_DIR (= $PKG_DIR/results, the published results; read only).
# Callers that need no corpus / no run root set PATHS_NO_CORPUS=1 / PATHS_NO_RUN=1 before sourcing.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPTS_DIR")"; CONFIGS_DIR="$PKG_DIR/configs"; PUBLISHED_RESULTS_DIR="$PKG_DIR/results"
if [ -z "${EXP_DIR:-}" ]; then echo "EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)" >&2; return 1 2>/dev/null || exit 1; fi
if [ ! -d "$EXP_DIR" ]; then echo "EXP_DIR does not exist: $EXP_DIR (create it first)" >&2; return 1 2>/dev/null || exit 1; fi
if [ "$(cd "$EXP_DIR" && pwd -P)" = "$(cd "$PKG_DIR" && pwd -P)" ]; then echo "EXP_DIR must not be the package directory ($PKG_DIR)" >&2; return 1 2>/dev/null || exit 1; fi
if [ -z "${PATHS_NO_CORPUS:-}" ]; then
  if [ -z "${CORPUS_DIR:-}" ]; then echo "CORPUS_DIR is not set: export CORPUS_DIR=<corpus root> (README, Setup)" >&2; return 1 2>/dev/null || exit 1; fi
  if [ ! -d "$CORPUS_DIR" ]; then echo "CORPUS_DIR does not exist: $CORPUS_DIR" >&2; return 1 2>/dev/null || exit 1; fi
fi
if [ -z "${PATHS_NO_CORPUS:-}" ] && [ -z "${PATHS_NO_RUN:-}" ]; then
  if [ -z "${RUN_DIR:-}" ]; then echo "RUN_DIR is not set: build a run root with scripts/make_run_root.sh and export RUN_DIR (README, Setup)" >&2; return 1 2>/dev/null || exit 1; fi
  if [ ! -d "$RUN_DIR" ]; then echo "RUN_DIR does not exist: $RUN_DIR" >&2; return 1 2>/dev/null || exit 1; fi
  if [ "$(cd "$RUN_DIR" && pwd -P)" = "$(cd "$CORPUS_DIR" && pwd -P)" ]; then
    echo "WARNING: RUN_DIR is the corpus directory: outputs and logs will be written into the corpus (use make_run_root.sh)" >&2; fi
fi
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_SANDI_ALL="${ENV_SANDI_ALL:-sandi-all}"; ENV_SANDI_DD="${ENV_SANDI_DD:-sandi-dd}"
ENV_ESWA_C="${ENV_ESWA_C:-eswa-c}"; ENV_ESWA_D="${ENV_ESWA_D:-eswa-d}"; ENV_ESWA_G="${ENV_ESWA_G:-eswa-g}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
export SCRIPTS_DIR PKG_DIR CONFIGS_DIR PUBLISHED_RESULTS_DIR EXP_DIR CORPUS_DIR RUN_DIR CONDA_BASE HF_HUB_CACHE \
       ENV_SANDI_ALL ENV_SANDI_DD ENV_ESWA_C ENV_ESWA_D ENV_ESWA_G

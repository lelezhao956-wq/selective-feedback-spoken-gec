#!/bin/bash
# Build a separate run root (RUN_DIR) of symlinks so that pipeline outputs (stageB/, stageD/, ...) are not written into the corpus.
# Usage: CORPUS_DIR=/path/to/corpus bash make_run_root.sh RUN_DIR [OFFICIAL_DIR]
#   corpus inputs (data/, reference-materials/, sandi-models/, README.txt) are linked from $CORPUS_DIR (read-only use);
#   official code (baseline-pipeline/, scoring-tools/, envs/, check-pipeline.sh) is linked from OFFICIAL_DIR, i.e. the
#   directory where the official zips were unpacked, patched and rebuilt (README, Setup); default: $CORPUS_DIR.
# Then: export RUN_DIR=<that directory>. Never deletes or overwrites anything (refuses existing entries).
set -u
PATHS_NO_RUN=1 source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1
R=${1:?RUN_DIR}; O=${2:-$CORPUS_DIR}
mkdir -p "$R" || exit 1
for n in data reference-materials sandi-models README.txt; do
  [ -e "$CORPUS_DIR/$n" ] || { echo "missing in corpus: $n"; exit 1; }
  [ -e "$R/$n" ] || [ -L "$R/$n" ] && { echo "exists, left as is: $R/$n"; continue; }
  ln -s "$CORPUS_DIR/$n" "$R/$n"
done
for n in baseline-pipeline scoring-tools envs check-pipeline.sh; do
  [ -e "$O/$n" ] || { echo "missing official code: $O/$n (unpack, patch and build first)"; exit 1; }
  [ -e "$R/$n" ] || [ -L "$R/$n" ] && { echo "exists, left as is: $R/$n"; continue; }
  ln -s "$O/$n" "$R/$n"
done
echo "run root ready: export RUN_DIR=$R"

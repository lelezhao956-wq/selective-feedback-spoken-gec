#!/bin/bash
# Optional snapshot commit of $EXP_DIR under a mkdir-based lock (queue label jobs and collect steps can overlap).
# DEFAULT: does nothing. It commits only when GIT_COMMIT=1 is set and $EXP_DIR is itself a git work tree; it never commits
# into any other repository. Usage: GIT_COMMIT=1 bash git_commit_locked.sh "message"
PATHS_NO_CORPUS=1 source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || exit 1
if [ "${GIT_COMMIT:-0}" != 1 ]; then echo "git_commit_locked: GIT_COMMIT not set - not committing ($1)"; exit 0; fi
[ "$(git -C "$EXP_DIR" rev-parse --show-toplevel 2>/dev/null)" = "$(cd "$EXP_DIR" && pwd -P)" ] || { echo "git_commit_locked: $EXP_DIR is not a git work tree root - not committing"; exit 0; }
LOCK=$EXP_DIR/.git/commit.lockdir
for i in $(seq 1 600); do mkdir "$LOCK" 2>/dev/null && break; sleep 1; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
cd "$EXP_DIR" && git add -A && { git diff --cached --quiet && echo "nothing to commit: $1" || git commit -q -m "$1"; } && git log --oneline | head -1

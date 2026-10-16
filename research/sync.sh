#!/bin/bash
# Commit code, research scripts, writeup and finished logs, push, and copy the logs into
# the main checkout's logs/ (where the user looks). Usage: research/sync.sh "<message>"
# Logs of still-running processes are skipped (no trailing completion marker yet is fine:
# only logs whose writer pid is gone are committed).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
MSG=${1:-"Research progress"}
for f in logs/*.log; do
  pid=${f##*_}; pid=${pid%.log}
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    continue  # Still being written.
  fi
  git add "$f"
  cp --update=none "$f" /workspace/autoresearch/logs/ 2>/dev/null
done
git add train_cifar.py test_train_cifar.py research/*.py research/*.sh 2>/dev/null
[ -f interval_search_writeup.txt ] && git add interval_search_writeup.txt
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git -c user.name="Roland Gao" -c user.email="41932972+RolandGao@users.noreply.github.com" \
    commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
fi
git push -q 2>&1 | tail -2
git log --oneline -1

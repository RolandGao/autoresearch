#!/bin/bash
# Rebuild and push cooldown_study.pdf whenever a new held-out result lands, until no job matching
# PATTERN is pending or running. Usage: research/watch_study.sh <pattern> (run in the background).
PATTERN=$1
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
last=""
while true; do
  now=$(grep -h "^heldout_result run=\|^start_heldout_result run=" logs/cifar_baseline_*.log 2>/dev/null | wc -l)
  active=$(ls queue/pending queue/running 2>/dev/null | grep -c "$PATTERN")
  if [ "$now" != "$last" ]; then
    /venv/main/bin/python research/make_cooldown_study.py > /dev/null && git add -f cooldown_study.pdf plots/*.png \
      && SYNC_DATA_ONLY=1 research/sync.sh "cooldown_study.pdf: updated ($now held-out results)" > /dev/null \
      && git -C /workspace/autoresearch pull -q --ff-only
    echo "$(date -u +%T) rebuilt with $now held-out results"
    last=$now
  fi
  [ "$active" -eq 0 ] && break
  sleep 120
done
echo "done"

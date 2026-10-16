#!/bin/bash
# Rebuild and push cooldown_study.pdf whenever new probe results land, until no probe job is
# pending or running. Usage: research/watch_study.sh (run in the background).
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
last=""
while true; do
  now=$(cat logs/research_cooldown_search_probe_*.log 2>/dev/null | grep -c '"kind": "per_group"')
  active=$(ls queue/pending queue/running 2>/dev/null | grep -c probe)
  if [ "$now" != "$last" ]; then
    /venv/main/bin/python research/make_cooldown_study.py > /dev/null && git add -f cooldown_study.pdf plots/cooldown_study.png \
      && research/sync.sh "cooldown_study.pdf: $now of 28 candidate results" > /dev/null \
      && git -C /workspace/autoresearch pull -q --ff-only
    echo "$(date -u +%T) rebuilt with $now candidate results"
    last=$now
  fi
  [ "$active" -eq 0 ] && break
  sleep 60
done
echo "done"

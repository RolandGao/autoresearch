#!/bin/bash
# Rebuild and push lookahead_joint_study.pdf every INTERVAL seconds (default 1200) while any 020_/021_ job
# (hparam_lookahead, joint_search, cooldown_shapes) is pending or running, then once more and exit.
# Usage: nohup research/watch_lookahead_report.sh [interval] >> queue/report_watch.log 2>&1 &
INTERVAL=${1:-1200}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
while true; do
  active=$(ls queue/pending queue/running 2>/dev/null | grep -cE "^02[01]_")
  git pull -q --rebase --autostash origin master
  /venv/main/bin/python research/make_lookahead_joint_report.py > /dev/null \
    && git add -f lookahead_joint_study.pdf plots/lookahead_joint/*.png \
    && SYNC_DATA_ONLY=1 research/sync.sh "lookahead_joint_study.pdf: updated ($active jobs active)" > /dev/null
  echo "$(date -u +%T) rebuilt ($active jobs active)"
  [ "$active" -eq 0 ] && break
  sleep "$INTERVAL"
done
echo "done"

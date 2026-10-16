#!/bin/bash
# Rebuild and push cooldown_study.pdf whenever new probe results land, until no probe job is
# pending or running. Usage: research/watch_study.sh (run in the background).
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
last=""
while true; do
  now=$( (cat logs/research_cooldown_search_probe_*.log logs/research_hparam_importance_*.log 2>/dev/null | grep -c '"kind": "\(per_group\|sweep\|center\)"'; grep -h "^heldout_result run=auto_gfam\|^start_heldout_result run=auto_gfam\|^seed_summary run=transformed_a_rampdown" logs/cifar_baseline_*.log 2>/dev/null | wc -l) | paste -sd+ | bc)
  active=$(ls queue/pending queue/running 2>/dev/null | grep -c "probe\|importance\|gfam\|rampdown")
  if [ "$now" != "$last" ]; then
    /venv/main/bin/python research/make_cooldown_study.py > /dev/null && git add -f cooldown_study.pdf plots/cooldown_study.png \
      && research/sync.sh "cooldown_study.pdf: updated ($now result lines)" > /dev/null \
      && git -C /workspace/autoresearch pull -q --ff-only
    echo "$(date -u +%T) rebuilt with $now candidate results"
    last=$now
  fi
  [ "$active" -eq 0 ] && break
  sleep 60
done
echo "done"

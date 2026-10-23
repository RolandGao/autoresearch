#!/bin/bash
# For up to 12 h: start research/worker.sh (by UUID) on every GPU that has had no compute process for
# 5 minutes in a row and has no worker of ours yet. Exits once all 8 GPUs have workers. Log: queue/claim.log.
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$REPO/queue/claim.log
end=$(( $(date +%s) + 43200 ))
declare -A idle claimed
while [ "$(date +%s)" -lt "$end" ] && [ "${#claimed[@]}" -lt 8 ]; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  for i in 0 1 2 3 4 5 6 7; do
    [ -n "${claimed[$i]}" ] && continue
    u=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader)
    if echo "$busy" | grep -q "$u"; then idle[$i]=0; else idle[$i]=$(( ${idle[$i]:-0} + 1 )); fi
    if [ "${idle[$i]}" -ge 5 ]; then
      nohup "$REPO/research/worker.sh" "$u" "hl$i" > /dev/null 2>&1 &
      claimed[$i]=$u
      echo "$(date -u +%FT%TZ) started worker hl$i on GPU $i $u" >> "$LOG"
    fi
  done
  echo "$(date -u +%FT%TZ) idle-minutes: ${idle[*]} claimed: ${!claimed[*]}" >> "$LOG"
  sleep 60
done
echo "$(date -u +%FT%TZ) claimer exits (claimed: ${!claimed[*]})" >> "$LOG"

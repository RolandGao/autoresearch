#!/bin/bash
# Queue worker: runs job scripts from queue/pending (sorted by name) on one GPU.
# Usage: research/worker.sh <gpu-uuid> <worker-name>
# A job is a bash script run from the repo root; its output goes to queue/out/<job>.out.
# Claiming is an atomic mv. Fast failures (< 60 s) back off and requeue; after 3 in a
# row the worker exits (a dead GPU must not drain the queue). queue/STOP stops workers.
GPU=$1
NAME=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
Q=$REPO/queue
cd "$REPO" || exit 1
fails=0
while true; do
  [ -f "$Q/STOP" ] && exit 0
  job=$(ls "$Q/pending" 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 10; continue; fi
  mv "$Q/pending/$job" "$Q/running/$job" 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) $NAME start $job" >> "$Q/worker.log"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU WORKER_NAME=$NAME bash "$Q/running/$job" > "$Q/out/$job.out" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  echo "$(date -u +%FT%TZ) $NAME end $job rc=$rc ${dur}s" >> "$Q/worker.log"
  if [ $rc -eq 0 ]; then
    mv "$Q/running/$job" "$Q/done/$job"
    fails=0
  elif [ $dur -lt 60 ]; then
    fails=$((fails + 1))
    if [ $fails -ge 3 ]; then
      mv "$Q/running/$job" "$Q/failed/$job"
      echo "$(date -u +%FT%TZ) $NAME exiting after $fails fast failures" >> "$Q/worker.log"
      exit 1
    fi
    mv "$Q/running/$job" "$Q/pending/$job"
    sleep $((30 * fails))
  else
    mv "$Q/running/$job" "$Q/failed/$job"
    fails=0
  fi
done

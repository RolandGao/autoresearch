#!/bin/bash
# Wait until N more jobs reach queue/done or queue/failed, or until a timeout (seconds).
# Usage: research/wait_jobs.sh <n> <timeout_seconds>. Prints the queue state at the end.
REPO=$(cd "$(dirname "$0")/.." && pwd)
Q=$REPO/queue
N=$1
LIMIT=$2
count() { echo $(( $(ls "$Q/done" | wc -l) + $(ls "$Q/failed" | wc -l) )); }
start=$(count)
end=$(( $(date +%s) + LIMIT ))
while [ "$(count)" -lt $((start + N)) ] && [ "$(date +%s)" -lt "$end" ]; do
  sleep 20
done
echo "finished $(( $(count) - start )) jobs; pending $(ls "$Q/pending" | wc -l); running: $(ls "$Q/running" | tr '\n' ' ')"
echo "failed: $(ls "$Q/failed" | tr '\n' ' ')"
tail -n 8 "$Q/worker.log"

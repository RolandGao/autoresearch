#!/bin/bash
# Enqueue train_cifar.py runs. Usage:
#   research/enqueue.sh <priority> <experiments,comma,separated> [seeds,comma,separated]
# e.g. research/enqueue.sh 10 baseline_bs500 100,101,102,103,104,105,106,107
# Lower priority numbers run first. Each job runs one process (all its experiments in order).
REPO=$(cd "$(dirname "$0")/.." && pwd)
PRIO=$1
EXPS=$2
SEEDS=${3:-}
NAME="$(printf %03d "$PRIO")_$(date -u +%H%M%S%N)_${EXPS//,/+}"
NAME=${NAME:0:180}
{
  echo "CIFAR_EXPERIMENTS=$EXPS CIFAR_SEEDS=$SEEDS /venv/main/bin/python train_cifar.py"
} > "$REPO/queue/pending/$NAME.sh"
echo "$NAME"

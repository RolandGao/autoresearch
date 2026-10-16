"""Enqueue train_cifar.py runs, one job per (experiment, seed group).

python research/enqueue.py <priority> <exp> [<exp> ...] --seeds 0 1 [--together]
  --seeds: seeds; one job per seed unless --together (one job running all seeds).
Lower priorities run first.
"""
import argparse
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("priority", type=int)
parser.add_argument("experiments", nargs="+")
parser.add_argument("--seeds", nargs="+", default=["0"])
parser.add_argument("--together", action="store_true")
args = parser.parse_args()
groups = [args.seeds] if args.together else [[s] for s in args.seeds]
for experiment in args.experiments:
    for seeds in groups:
        name = f"{args.priority:03d}_{time.time_ns()}_{experiment}_s{'-'.join(seeds)}"[:180]
        with open(os.path.join(REPO, "queue", "pending", name + ".sh"), "x") as job:
            job.write(f"CIFAR_EXPERIMENTS={experiment} CIFAR_SEEDS={','.join(seeds)} "
                      "/venv/main/bin/python train_cifar.py\n")
        print(name)

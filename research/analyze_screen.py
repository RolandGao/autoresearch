"""Analyze overfit_screen logs: python research/analyze_screen.py

Per batch size, mode, group and step count: the LR with the lowest seed-mean loss, the
loss there, and the loss change one x sqrt(10) step either side (sensitivity proxy),
next to the known full-run optimum of the baseline (initial LR of baseline_bs{BS}).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_cifar as t  # noqa: E402

data = defaultdict(lambda: defaultdict(list))  # (bs, mode, group, k) -> lr -> [loss]
for path in glob.glob(os.path.join(REPO, "logs", "research_overfit_screen_bs*.log")):
    bs = int(re.search(r"screen_bs(\d+)", path).group(1))
    for line in open(path):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "losses" not in r:
            continue
        for k, loss in r["losses"].items():
            data[(bs, r["mode"], r["group"], int(k))][r["lr"]].append(
                loss if loss is not None else float("inf"))
        # Diverged runs have no entries for later checkpoints: count them as inf.
        for k in (1, 2, 4, 8, 16, 32):
            if str(k) not in r["losses"] and r["losses"]:
                data[(bs, r["mode"], r["group"], k)][r["lr"]].append(float("inf"))
for key in sorted(data):
    bs, mode, group, k = key
    curve = {lr: sum(v) / len(v) for lr, v in data[key].items()}
    lrs = sorted(curve)
    best = min(lrs, key=lambda lr: curve[lr])
    i = lrs.index(best)
    left = curve[lrs[i - 1]] - curve[best] if i > 0 else float("nan")
    right = curve[lrs[i + 1]] - curve[best] if i + 1 < len(lrs) else float("nan")
    base = t.RUN_CONFIGS[f"baseline_bs{bs}"]["param_groups"][group]["lr_scheduler"][0][1]
    print(f"bs{bs:<5d} {mode:5s} {group:11s} k={k:<3d} best_lr={best:<8g} loss={curve[best]:.4f} "
          f"d(-)={left:.4f} d(+)={right:.4f}  full-run={base:g}  ratio={best/base:.3g}")

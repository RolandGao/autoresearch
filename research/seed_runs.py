"""Per-seed results of plain train_cifar.py runs (no search), from logs/cifar_baseline_*.log.

per_seed(name, since): {seed: (tta %, val_acc %)} for run name over every log whose UTC timestamp
(in the file name) is at least since ("YYYYMMDD_HHMMSS"); later logs override earlier ones.
"""
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_seed(name, since):
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "logs", "cifar_baseline_*.log"))):
        if os.path.basename(path)[len("cifar_baseline_"):][:15] < since:
            continue
        current = None
        for line in open(path, errors="replace"):
            m = re.match(r"config run=(\S+)_seed(\d+)$", line.strip())
            if m:
                current = int(m.group(2)) if m.group(1) == name else None
                continue
            m = re.match(r"val_acc=(\S+) tta_val_acc=(\S+)", line)
            if m and current is not None:
                out[current] = (100 * float(m.group(2)), 100 * float(m.group(1)))
                current = None
    return out

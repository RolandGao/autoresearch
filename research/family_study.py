"""Data of Experiment 3 (global_family searches, auto_gfam_* logs): per family and condition,
the search's own score, cost, evaluations, held-out TTA of the start (oracle projected onto the
family) and of the best schedule, and the start / best parameter values."""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = ("intervals5", "linear", "power", "knots2", "knots3", "knots5")
FAMILY_LABELS = {"intervals5": "5 constant intervals", "linear": "Linear decay (1 param)",
                 "power": "Power decay (lr, p)", "knots2": "2 knots (t = 0, 0.5)",
                 "knots3": "3 knots (t = 0, 1/3, 2/3)", "knots5": "5 knots (t = 0, 0.2, ..., 0.8)"}
CONDITIONS = ("", "_ramp", "_rampdown", "_wave")
CONDITION_LABELS = {"": "No transform", "_ramp": "Ramp", "_rampdown": "Reverse ramp", "_wave": "Wave"}


def load(prefix):
    """Searches named <prefix>_<family>_bs2000<condition> (auto_gfam: Experiment 3, auto_nfam: 9)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "logs", "cifar_baseline_*.log"))):
        text = open(path, errors="replace").read()
        m = re.search(rf"^auto_result run=({prefix}_\S+)_seed0 .*?search_seconds=(\S+) evaluations=(\d+)",
                      text, re.M)
        if not m:
            continue
        name = m.group(1)
        own = re.search(rf"^auto_result run={name}_seed0 .*?tta_val_acc=(\S+)", text, re.M)
        row = dict(seconds=float(m.group(2)), evaluations=int(m.group(3)), own=100 * float(own.group(1)))
        stop = re.search(rf"^search_stop run={name}_seed0 reason=(\S+) moves=(\d+)", text, re.M)
        if stop:
            row["stop"], row["moves"] = stop.group(1), int(stop.group(2))
        seeds = re.search(rf"^auto_result run={name}_seed0 .*?search_seeds=(\d+)", text, re.M)
        if seeds:
            row["search_seeds"] = int(seeds.group(1))
        for label, key in (("start", "start_heldout_result"), ("final", "heldout_result")):
            h = re.search(rf"^{key} run={name}_seed0 .*?tta_val_acc_mean=(\S+) tta_val_acc_std=(\S+)",
                          text, re.M)
            if h:
                row[label] = (100 * float(h.group(1)), 100 * float(h.group(2)))
            v = re.search(rf"^  {label} conv.initial_lr: (.*)$", text, re.M)
            if v:
                row[f"{label}_values"] = json.loads(v.group(1))
        out[name] = row
    return out

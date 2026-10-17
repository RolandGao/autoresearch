"""Data of the hparam range experiment (research/hparam_range.py logs).

For each main interval length L: the center (Experiment 2's joint-search center), its TTA and the
stop threshold, and per group: TTA at each probed grid offset and, per side, the offset where
the TTA first fell to the threshold (None: no drop within the max steps; missing: running).
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
LENGTHS = (1, 2, 5, 10, 40)


def load(length):
    out = dict(center=None, probe={g: {} for g in GROUPS}, stop={g: {} for g in GROUPS}, done=set())
    for path in glob.glob(os.path.join(REPO, "logs", f"research_hparam_range_bs2000_L{length}_*.log")):
        groups = None
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r["kind"] == "center":
                out["center"] = r
            elif r["kind"] == "range":
                groups = r["group"]
                tta = 100 * r["tta"] if r["tta"] != float("-inf") else 0.0  # Diverged: no accuracy.
                out["probe"][r["group"]][r["offset"]] = (r["value"], tta)
            elif r["kind"] == "side":
                out["stop"][r["group"]][r["direction"]] = r["stop"]
            elif r["kind"] == "done" and groups:
                out["done"].add(groups)
    return out


def all_lengths():
    return {length: load(length) for length in LENGTHS}


def largest_within(probe, center_tta, margin, max_side):
    """train_cifar.directional_search's initial_lr_search on recorded probes (TTA in %): from the
    center, step up, then down, until a score falls below the best so far minus margin (or after
    max_side steps); return the largest offset scoring within margin of the best, and whether a
    side ended at the probe limit. None if the probes end before the rule would stop."""
    points = {**probe, 0: (None, center_tta)}
    probes, best, at_limit = [(0, center_tta)], center_tta, False
    for direction in (1, -1):
        for k in range(1, max_side + 1):
            offset = direction * k
            if offset not in points:
                return None
            probes.append((offset, points[offset][1]))
            if points[offset][1] < best - margin:
                break
            best = max(best, points[offset][1])
        else:
            at_limit = at_limit or direction == 1
    return max(o for o, s in probes if s >= best - margin), at_limit

"""Data of the hparam importance experiment (research/hparam_importance.py logs).

For each main interval length L: the joint-search center (four LRs), its score, and the
one-at-a-time sweep: per group, score at grid offsets -5..+5 around the center.
"""
import glob
import json
import os
import re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
LENGTHS = (1, 2, 5, 10, 40)


def load(length):
    paths = glob.glob(os.path.join(REPO, "logs", f"research_hparam_importance_bs2000_L{length}_*.log"))
    out = dict(center=None, sweep=defaultdict(dict), evaluations=0, done=False)
    for path in paths:
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r["kind"] == "center":
                out["center"] = r
            elif r["kind"] == "sweep":
                out["sweep"][r["group"]][r["offset"]] = (r["value"], 100 * r["tta"], r["loss"])
            elif r["kind"] == "eval":
                out["evaluations"] += 1
            elif r["kind"] == "done":
                out["done"] = True
    return out


def all_lengths():
    return {length: load(length) for length in LENGTHS}

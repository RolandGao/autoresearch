"""Analysis of research/level_horizon.py: per group and level, the mean over seeds of the score
after the first interval (none), after a one-interval cooldown (short) and after the rest of the
run (rest_lin).

python research/level_study.py <batch_size>
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ("none", "short", "rest_lin")


def load(batch_size):
    data = defaultdict(lambda: defaultdict(list))  # group -> (offset, value) -> [scores dicts]
    for path in glob.glob(os.path.join(REPO, "logs", f"research_level_horizon_bs{batch_size}_*.log")):
        for line in open(path):
            if line.startswith("{"):
                r = json.loads(line)
                if r["kind"] == "eval":
                    data[r["group"]][(r["offset"], r["value"])].append({k: 100 * r[k] for k in KINDS})
    return {g: {key: {k: statistics.mean(s[k] for s in runs) for k in KINDS} | {"n": len(runs)}
                for key, runs in sorted(levels.items())} for g, levels in data.items()}


if __name__ == "__main__":
    for g, levels in load(int(sys.argv[1])).items():
        print(f"\n{g}")
        for (offset, value), m in levels.items():
            print(f"  {offset:+d} {value:<10g} " + " ".join(f"{k}={m[k]:6.2f}" for k in KINDS) + f"  n={m['n']}")

"""Analysis of research/lookahead_noise.py: per interval k and lookahead, each candidate's mean
score over seeds, the SD over seeds of (candidate - plan) scored on the same seed (the noise of a
paired single-seed comparison), and the decision quality of picking a candidate by the lookahead
on one seed: how often it picks the candidate with the best mean truth, and the mean regret in
truth (pp) of its pick.

python research/lookahead_study.py <batch_size>
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(batch_size):
    data = defaultdict(dict)  # (k, offset) -> seed -> scores
    plan = None
    for path in glob.glob(os.path.join(REPO, "logs", f"research_lookahead_noise_bs{batch_size}_s*.log")):
        seed = None
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r["kind"] == "setup":
                seed, plan = r["seed"], r["plan"]
            elif r["kind"] == "eval":
                data[(r["k"], r["offset"])][seed] = {key: 100 * v for key, v in r.items()
                                                     if key not in ("kind", "k", "offset", "c", "time")}
                data[(r["k"], r["offset"])][seed]["c"] = r["c"]
    return plan, data


def summary(batch_size):
    plan, data = load(batch_size)
    out = {}
    for k in sorted({k for k, _ in data}):
        offsets = sorted(o for kk, o in data if kk == k)
        seeds = sorted(set.intersection(*(set(data[(k, o)]) for o in offsets)))
        if not seeds:
            continue
        variants = [v for v in data[(k, offsets[0])][seeds[0]] if v != "c"]
        truth_mean = {o: statistics.mean(data[(k, o)][s]["truth"] for s in seeds) for o in offsets}
        best = max(offsets, key=lambda o: truth_mean[o])
        rows = {}
        for v in variants:
            means = {o: statistics.mean(data[(k, o)][s][v] for s in seeds) for o in offsets}
            paired = {o: statistics.stdev([data[(k, o)][s][v] - data[(k, 0)][s][v] for s in seeds])
                      if len(seeds) > 1 else 0.0 for o in offsets if o != 0}
            picks = [max(offsets, key=lambda o: data[(k, o)][s][v]) for s in seeds]
            rows[v] = dict(means=means, paired_sd=paired, hit=sum(p == best for p in picks) / len(seeds),
                           regret=statistics.mean(truth_mean[best] - truth_mean[p] for p in picks),
                           mean_pick=max(offsets, key=lambda o: means[o]))
        out[k] = dict(offsets=offsets, seeds=seeds, best=best, truth_mean=truth_mean, rows=rows,
                      c={o: data[(k, o)][seeds[0]]["c"] for o in offsets})
    return plan, out


if __name__ == "__main__":
    plan, out = summary(int(sys.argv[1]))
    print("plan", plan)
    for k, r in out.items():
        print(f"\ninterval {k}: seeds {len(r['seeds'])}, candidates {[r['c'][o] for o in r['offsets']]}, "
              f"best by mean truth {r['c'][r['best']]}")
        for v, row in r["rows"].items():
            means = " ".join(f"{row['means'][o]:6.2f}" for o in r["offsets"])
            sd = " ".join(f"{row['paired_sd'].get(o, 0):.2f}" for o in r["offsets"])
            print(f"  {v:9} means {means} | paired SD {sd} | mean-pick {r['c'][row['mean_pick']]:<6} "
                  f"1-seed hit {row['hit']:.2f} regret {row['regret']:.3f}")

"""Analysis of research/hparam_lookahead.py: per main group, interval k and lookahead, each candidate's
mean score over seeds, which candidate the lookahead's mean picks against the truth's, and the decision
quality of picking by the lookahead on one seed (hit rate, mean regret in truth, pp). Also the searched
cooldown multipliers (how often each value is chosen).

python research/hparam_lookahead_study.py [batch_size]
"""
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ("truth", "lin_main", "lin_all", "s_main", "s_tied", "s_each")


def load(batch_size=2000):
    data = defaultdict(dict)  # (group, k, offset) -> seed -> record
    for path in glob.glob(os.path.join(REPO, "logs", f"research_hparam_lookahead_*_bs{batch_size}_s*.log")):
        seed = None
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r["kind"] == "setup":
                seed = r["seed"]
            elif r["kind"] == "eval":
                rec = {v: 100 * r[v] for v in VARIANTS if v in r}
                rec.update(c=r["c"], searched=r.get("searched", {}))
                data[(r["group"], r["k"], r["offset"])][seed] = rec
    return data


def summary(batch_size=2000):
    data = load(batch_size)
    out = {}
    for g, k in sorted({(g, k) for g, k, _ in data}):
        offsets = sorted(o for gg, kk, o in data if (gg, kk) == (g, k))
        seeds = sorted(set.intersection(*(set(data[(g, k, o)]) for o in offsets)))
        if not seeds:
            continue
        cell = {o: data[(g, k, o)] for o in offsets}
        truth = {o: statistics.mean(cell[o][s]["truth"] for s in seeds) for o in offsets}
        best = max(offsets, key=truth.get)
        rows = {}
        for v in VARIANTS:
            if not all(v in cell[o][s] for o in offsets for s in seeds):
                continue
            means = {o: statistics.mean(cell[o][s][v] for s in seeds) for o in offsets}
            picks = [max(offsets, key=lambda o: cell[o][s][v]) for s in seeds]
            rows[v] = dict(means=means, mean_pick=max(offsets, key=means.get),
                           hit=sum(p == best for p in picks) / len(seeds),
                           regret=statistics.mean(truth[best] - truth[p] for p in picks))
        mults = {v: Counter(tuple(sorted(cell[o][s]["searched"][v].items())) for o in offsets for s in seeds
                            if v in cell[o][s]["searched"]) for v in ("s_main", "s_tied", "s_each")}
        out[(g, k)] = dict(offsets=offsets, seeds=seeds, best=best, rows=rows, mults=mults,
                           c={o: cell[o][seeds[0]]["c"] for o in offsets})
    return out


if __name__ == "__main__":
    for (g, k), r in summary(int(sys.argv[1]) if len(sys.argv) > 1 else 2000).items():
        print(f"\n{g} interval {k + 1}: {len(r['seeds'])} seeds, candidates {[r['c'][o] for o in r['offsets']]}, "
              f"best by mean truth {r['c'][r['best']]}")
        for v, row in r["rows"].items():
            means = " ".join(f"{row['means'][o]:6.2f}" for o in r["offsets"])
            print(f"  {v:8} means {means} | mean-pick {r['c'][row['mean_pick']]:<6} "
                  f"1-seed hit {row['hit']:.2f} regret {row['regret']:.3f}")
        for v, counts in r["mults"].items():
            if counts:
                top = ", ".join(f"{dict(m)} x{n}" for m, n in counts.most_common(4))
                print(f"  {v} searched m (top): {top}")

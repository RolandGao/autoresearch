"""Analyze cooldown_fidelity logs: python research/analyze_fidelity.py [bs] [group] [cool]

Per interval k: truth = mean final TTA per candidate over seeds. For each proxy variant
(none, or searched mult x others mode): per-seed regret = truth(best) - truth(argmax of
that seed's proxy), and the Spearman correlation of the seed-mean proxy with the truth.
Variants are ranked by mean regret over intervals and seeds.
"""
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bs = sys.argv[1] if len(sys.argv) > 1 else "2000"
group = sys.argv[2] if len(sys.argv) > 2 else "conv"
cool = sys.argv[3] if len(sys.argv) > 3 else ""  # e.g. "rest" or "2" for _cool<tag> logs
pattern = f"research_cooldown_fidelity_bs{bs}_{group}_k[0-9]_{'cool' + cool + '_' if cool else ''}2026*.log"
records = defaultdict(list)  # k -> records
for path in glob.glob(os.path.join(REPO, "logs", pattern)):
    k = int(path.split("_k")[1].split("_")[0])
    for line in open(path):
        if line.startswith("{"):
            records[k].append(json.loads(line))


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


summary = defaultdict(lambda: dict(regret=[], rho=[]))
for k, recs in sorted(records.items()):
    truth = defaultdict(list)
    proxy = defaultdict(lambda: defaultdict(dict))  # variant -> seed -> c -> score
    for r in recs:
        if r.get("kind") == "truth" and r["score"] is not None:
            truth[r["c"]].append(r["score"])
        elif r.get("kind") == "none":
            proxy["none"][r["seed"]][r["c"]] = r["score"] if r["score"] is not None else -1
        elif r.get("kind") == "proxy":
            proxy[f"s{r['ms']}|o{r['mo']}"][r["seed"]][r["c"]] = (
                r["score"] if r["score"] is not None else -1)
    cands = sorted(truth)
    if not cands:
        continue
    tmean = {c: sum(v) / len(v) for c, v in truth.items()}
    best = max(tmean.values())
    print(f"k={k} truth: " + " ".join(f"{c:g}:{100*tmean[c]:.2f}(n{len(truth[c])})" for c in cands))
    for variant, seeds in proxy.items():
        regrets = []
        for seed, scores in seeds.items():
            if set(scores) >= set(cands):
                pick = max(cands, key=lambda c: scores[c])
                regrets.append(best - tmean[pick])
        complete = [s for s in seeds.values() if set(s) >= set(cands)]
        if not complete:
            continue
        pmean = [sum(s[c] for s in complete) / len(complete) for c in cands]
        summary[variant]["regret"] += regrets
        summary[variant]["rho"].append(spearman(pmean, [tmean[c] for c in cands]))
        summary[variant].setdefault("per_k", {})[k] = sum(regrets) / len(regrets)
print()
print(f"{'variant':24s} regret(pp)  rho   per-interval regret")
for variant, s in sorted(summary.items(), key=lambda kv: sum(kv[1]["regret"]) / len(kv[1]["regret"])):
    regret = 100 * sum(s["regret"]) / len(s["regret"])
    rho = sum(s["rho"]) / len(s["rho"])
    per_k = " ".join(f"k{k}:{100*v:.2f}" for k, v in sorted(s["per_k"].items()))
    print(f"{variant:24s} {regret:6.3f}    {rho:5.2f}  {per_k}  (n={len(s['regret'])})")

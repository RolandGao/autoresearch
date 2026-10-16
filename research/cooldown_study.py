"""Data of the cooldown lookahead study (research/cooldown_search_probe.py logs).

For each interval k: per method, the seed-mean score of every candidate conv LR (TTA %), the LR
its seed-mean curve picks, the regret of that pick (truth at the best LR minus truth at the
pick, pp), the mean per-seed regret (each seed picks from its own scores, as a single-seed
search would), and the cooldown multipliers the searched methods chose.
"""
import glob
import json
import os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ("truth", "none", "lin_conv", "lin_all", "conv_only", "shared", "per_group")
LABELS = {
    "truth": "Ground truth (final TTA)",
    "none": "No lookahead",
    "lin_conv": "Linear to 0: conv only",
    "lin_all": "Linear to 0: all four groups",
    "conv_only": "1 searched cooldown: conv only, others on ground truth",
    "shared": "1 searched cooldown: shared by all four groups",
    "per_group": "4 searched cooldowns (one per group)",
}


def load(k):
    scores = defaultdict(lambda: defaultdict(dict))  # method -> c -> seed -> score
    chosen = defaultdict(lambda: defaultdict(dict))  # method -> c -> seed -> mults
    setup = None
    for path in glob.glob(os.path.join(REPO, "logs", f"research_cooldown_search_probe_bs2000_k{k}_*.log")):
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "setup":
                setup = r
            elif kind in METHODS and r.get("score") is not None:
                scores[kind][r["c"]][r["seed"]] = 100 * r["score"]
                if kind in ("shared", "conv_only"):
                    chosen[kind][r["c"]][r["seed"]] = r["m"]
                elif kind == "per_group":
                    chosen[kind][r["c"]][r["seed"]] = tuple(r["mults"])
    return setup, scores, chosen


def summary(k):
    setup, scores, chosen = load(k)
    if "truth" not in scores:
        return None
    seeds = sorted({s for by_c in scores["truth"].values() for s in by_c})
    cands = sorted(scores["truth"])
    complete = [c for c in cands if all(len(scores[m][c]) == len(seeds) for m in METHODS)]
    if not complete:
        return None  # Still running: no candidate has every method on every seed yet.
    curves = {m: {c: sum(scores[m][c].values()) / len(scores[m][c]) for c in complete} for m in METHODS}
    truth = curves["truth"]
    best = max(truth.values())
    out = dict(k=k, setup=setup, seeds=seeds, cands=complete, curves=curves, picks={}, regret={},
               seed_regret={}, seed_picks={}, chosen={})
    for m in METHODS:
        pick = max(complete, key=lambda c: curves[m][c])
        out["picks"][m] = pick
        out["regret"][m] = best - truth[pick]
        per_seed = [max(complete, key=lambda c: scores[m][c][s]) for s in seeds]
        out["seed_picks"][m] = per_seed
        out["seed_regret"][m] = sum(best - truth[p] for p in per_seed) / len(per_seed)
    for m in ("conv_only", "shared", "per_group"):
        out["chosen"][m] = {c: Counter(chosen[m][c].values()) for c in complete}
    return out


if __name__ == "__main__":
    for k in range(4):
        s = summary(k)
        if s is None:
            print(k, "no data")
            continue
        print(f"interval {k + 1}: seeds {s['seeds']}, candidates {s['cands']}")
        for m in METHODS:
            print(f"  {LABELS[m]:38s} pick {s['picks'][m]:<6g} regret {s['regret'][m]:.2f} "
                  f"per-seed regret {s['seed_regret'][m]:.2f}")

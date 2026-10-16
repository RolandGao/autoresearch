"""Collect the data tables of the PDF report from logs/ (JSON on stdout or as a module).

fidelity(k, cool_tag)    seed-mean truth and proxy scores per candidate conv LR
fixed_runs()             seed_summary of every plain (non-search) run: name -> (n, mean, sd)
search_runs()            summarize_auto rows: experiment -> (n, seconds, evals, own, heldout, range)
interval_scores(run)     per-interval selected scores of an auto search run (pass replay check)
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")


def fidelity(k, cool_tag):
    pattern = f"research_cooldown_fidelity_bs2000_conv_k{k}_{cool_tag + '_' if cool_tag else ''}2026*.log"
    data = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(LOGS, pattern)):
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("kind") in ("truth", "none"):
                data[r["kind"]][r["c"]].append(r["score"])
            elif r.get("kind") == "proxy":
                data[f"s{r['ms']}|o{r['mo']}"][r["c"]].append(r["score"])
    return {variant: {c: 100 * sum(v) / len(v) for c, v in by_c.items() if v}
            for variant, by_c in data.items()}


def fixed_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(LOGS, "cifar_baseline_*.log"))):
        for line in open(path, errors="replace"):
            if line.startswith("seed_summary "):
                fields = dict(re.findall(r"(\S+)=(\[[^\]]*\]|\S+)", line))
                name = fields["run"]
                if name.startswith(("auto_", "global_", "interval_")):
                    continue
                seeds = json.loads(fields["seeds"])
                if min(seeds) < 100:
                    continue  # Held-out seeds only.
                stamp = os.path.basename(path)[len("cifar_baseline_"):][:15]
                era = "new" if stamp >= "20261016_183000" else "old"  # BS 500/125 baselines
                runs[(name, len(seeds), min(seeds), era)] = (
                    len(seeds), 100 * float(fields["tta_val_acc_mean"]),
                    100 * float(fields["tta_val_acc_std"]))
    return runs


def search_runs(since="", until=""):
    args = [sys.executable, os.path.join(REPO, "research", "summarize_auto.py")]
    if since:
        args += ["--since", since]
    if until:
        args += ["--until", until]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    rows = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 7 and parts[0].startswith("auto_") and parts[1].isdigit():
            rows[parts[0]] = dict(n=int(parts[1]), seconds=float(parts[2]), evals=float(parts[3]),
                                  own=float(parts[4]), heldout=float(parts[5]),
                                  spread=float(line.split("range ")[1].rstrip(")")))
    return rows


def screen(k=16, mode="fresh"):
    """Overfit/fresh few-step screen: per (bs, group) best LR and the loss rise one x sqrt(10)
    step below / above it (seed means)."""
    data = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(LOGS, "research_overfit_screen_bs*.log")):
        bs = int(re.search(r"screen_bs(\d+)", path).group(1))
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("mode") != mode:
                continue
            loss = r["losses"].get(str(k))
            data[(bs, r["group"])][r["lr"]].append(loss if loss is not None else float("inf"))
    out = {}
    for key, curve in data.items():
        means = {lr: sum(v) / len(v) for lr, v in curve.items()}
        lrs = sorted(means)
        best = min(lrs, key=lambda lr: means[lr])
        i = lrs.index(best)
        below = means[lrs[i - 1]] - means[best] if i > 0 else float("nan")
        above = means[lrs[i + 1]] - means[best] if i + 1 < len(lrs) else float("nan")
        out[key] = (best, below, above)
    return out


def final_schedule(run, path_name):
    """Knots (segment starts + last end) of an auto search's committed schedule."""
    for path in glob.glob(os.path.join(LOGS, "cifar_baseline_*.log")):
        text = open(path, errors="replace").read()
        if f"auto_result run={run} " not in text:
            continue
        m = re.search(rf"^  final {re.escape(path_name)}: (.*)$", text, re.M)
        if m:
            lines = json.loads(m.group(1))
            return [round(line[1], 4) for line in lines] + [round(lines[-1][-1], 4)]
    return None


def interval_scores(run):
    for path in glob.glob(os.path.join(LOGS, "cifar_baseline_*.log")):
        scores, found = [], False
        for line in open(path, errors="replace"):
            if line.startswith(f"auto_interval run={run} "):
                found = True
                fields = dict(re.findall(r"(\S+)=(\S+)", line))
                scores.append((int(fields["interval"]), int(fields["evaluations"]),
                               100 * float(fields["tta_val_acc"]), float(fields["conv.initial_lr"])))
        if found:
            return scores
    return []


if __name__ == "__main__":
    print(json.dumps({"fidelity_k1": fidelity(1, ""), "interval": interval_scores(sys.argv[1])
                      if len(sys.argv) > 1 else None}, indent=1, default=str)[:3000])


def fidelity_methods(k):
    """Interval k (0-based) of the conv cooldown probe: seed-mean curves (TTA %, per candidate
    conv LR) of truth, no lookahead, and linear-to-0 / searched constant cooldowns with a
    one-interval or rest-of-training lookahead. "Searched constant" takes, per seed and
    candidate, the best constant cooldown mult x candidate over the logged mults (as a search
    on that seed would see it), then averages over seeds. The last interval's rest is one
    interval, so both lookahead lengths coincide there."""
    def collect(pattern):
        truth, none = defaultdict(dict), defaultdict(dict)
        lin, const = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
        for path in glob.glob(os.path.join(LOGS, pattern)):
            for line in open(path):
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                kind = r.get("kind")
                if kind == "truth":
                    truth[r["c"]][r["seed"]] = r["score"]
                elif kind == "none":
                    none[r["c"]][r["seed"]] = r["score"]
                elif kind == "proxy" and r["mo"] == "cont":
                    if r["ms"] == "lin0":
                        lin[r["c"]][r["seed"]] = r["score"]
                    else:
                        const[r["c"]][r["seed"]][r["ms"]] = r["score"]
        mean = lambda by_seed: 100 * sum(by_seed.values()) / len(by_seed)  # noqa: E731
        return (
            {c: mean(v) for c, v in truth.items()}, {c: mean(v) for c, v in none.items()},
            {c: mean(v) for c, v in lin.items()},
            {c: mean({s: max(m.values()) for s, m in v.items()}) for c, v in const.items()},
        )

    one = collect(f"research_cooldown_fidelity_bs2000_conv_k{k}_2026*.log")
    rest = collect(f"research_cooldown_fidelity_bs2000_conv_k{k}_coolrest_2026*.log")
    if k == 3:
        rest = one
    curves = {"truth": one[0] or rest[0], "no lookahead": one[1] or rest[1],
              "linear to 0, one interval": one[2], "searched constant, one interval": one[3],
              "linear to 0, to the end": rest[2], "searched constant, to the end": rest[3]}
    return {name: curve for name, curve in curves.items() if curve}


def pick_and_regret(curves):
    """Per method: the LR it picks (argmax of its curve) and the truth lost by that pick (pp)."""
    truth = curves["truth"]
    best = max(truth.values())
    out = {}
    for name, curve in curves.items():
        common = [c for c in curve if c in truth]
        pick = max(common, key=lambda c: curve[c])
        out[name] = (pick, best - truth[pick])
    return out

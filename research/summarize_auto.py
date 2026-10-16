"""Summarize auto-search logs: python research/summarize_auto.py [--since YYYYMMDD_HHMMSS] [name ...]

One row per search run (experiment, seed): search seconds, evaluations, the search's own
final TTA, the held-out mean over seeds 100-107 and the committed schedules; then the
mean per experiment over its search seeds.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
since = ""  # --since YYYYMMDD_HHMMSS: only logs started at or after this time
if "--since" in args:
    i = args.index("--since")
    since = args[i + 1]
    args = args[:i] + args[i + 2:]
filters = args
rows = []
for path in sorted(glob.glob(os.path.join(REPO, "logs", "cifar_baseline_*.log"))):
    if since and os.path.basename(path)[len("cifar_baseline_"):][:15] < since:
        continue
    run = result = heldout = None
    finals = {}
    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if line.startswith("auto_result "):
                fields = dict(re.findall(r"(\S+)=(\S+)", line))
                run = fields["run"]
                result = dict(seconds=float(fields["search_seconds"]),
                              evals=int(fields["evaluations"]),
                              tta=float(fields["tta_val_acc"]))
            elif line.startswith("  final ") and run:
                name, lines = line[len("  final "):].split(": ", 1)
                finals[name.split(".")[0]] = json.loads(lines)
            elif line.startswith("heldout_result ") and run:
                fields = dict(re.findall(r"(\S+)=(\[[^\]]*\]|\S+)", line))
                heldout = float(fields["tta_val_acc_mean"])
                rows.append(dict(file=os.path.basename(path), run=run, heldout=heldout,
                                 finals=finals, **result))
                run, finals = None, {}
if filters:
    rows = [r for r in rows if any(f in r["run"] for f in filters)]
by_exp = defaultdict(list)
for r in rows:
    exp, seed = r["run"].rsplit("_seed", 1)
    by_exp[exp].append(r)
    sched = " ".join(f"{g}=" + ",".join(f"{seg[1]:g}" for seg in lines)
                     for g, lines in r["finals"].items())
    print(f"{exp:42s} s{seed:<3s} {r['seconds']:7.0f}s {r['evals']:4d}ev own {100*r['tta']:.2f} "
          f"heldout {100*r['heldout']:.2f}  {sched}")
print()
print(f"{'experiment':42s} n  search_s  evals  own    heldout(mean of search seeds)")
for exp, rs in sorted(by_exp.items()):
    n = len(rs)
    mean = lambda key: sum(r[key] for r in rs) / n
    spread = (max(r["heldout"] for r in rs) - min(r["heldout"] for r in rs)) * 100
    print(f"{exp:42s} {n}  {mean('seconds'):7.0f}  {mean('evals'):5.1f}  {100*mean('tta'):.2f}  "
          f"{100*mean('heldout'):.2f} (range {spread:.2f})")

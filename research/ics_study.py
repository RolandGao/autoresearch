"""Data of the ICS (run_ics_search) campaign: per version, cell (scenario, batch size, transform)
and search seed, the held-out TTA (seeds 100-107), the search's own final score, its cost in
units of a full run (train steps / total steps), whether it hit the budget, and the committed
schedules; plus the reference runs (baselines and do-nothing transforms, seeds 100-107) logged
since the campaign started.
"""
import glob
import json
import os
import re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZES = (2000, 500, 125)
SCENARIOS = ("a", "b")
SCENARIO_LABELS = {"a": "conv only", "b": "4 LRs"}
TRANSFORMS = ("", "_ramp", "_rampdown", "_wave")
TRANSFORM_LABELS = {"": "none", "_ramp": "ramp", "_rampdown": "reverse ramp", "_wave": "wave"}
SINCE = "20261017_070000"


def logs_since(since):
    for path in sorted(glob.glob(os.path.join(REPO, "logs", "cifar_baseline_*.log"))):
        if os.path.basename(path)[len("cifar_baseline_"):][:15] >= since:
            yield path


def load(version):
    """{(scenario, batch_size, transform): {seed: row}} for runs named ics_<version>_..."""
    out = defaultdict(dict)
    pattern = re.compile(rf"^auto_result run=ics_{version}_([ab])_bs(\d+)(\S*?)_seed(\d+) .*?tta_val_acc=(\S+) "
                         r"search_seconds=(\S+) evaluations=(\d+) train_steps=(\d+) budget_factor=(\S+) "
                         r"budget_hit=(\S+)", re.M)
    for path in logs_since("20261017_000000"):
        text = open(path, errors="replace").read()
        for m in pattern.finditer(text):
            scenario, bs, transform, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            name = f"ics_{version}_{scenario}_bs{bs}{transform}_seed{seed}"
            held = re.search(rf"^heldout_result run={name} .*?tta_val_acc_mean=(\S+) tta_val_acc_std=(\S+)",
                             text, re.M)
            schedules = {g: json.loads(v) for g, v in
                         re.findall(r"^  final (\w+)\.initial_lr: (\[.*\])$", text[m.end():], re.M)}
            row = dict(own=100 * float(m.group(5)), seconds=float(m.group(6)), evaluations=int(m.group(7)),
                       cost=float(m.group(9)), budget_hit=m.group(10) == "true", schedules=schedules)
            if held:
                row["heldout"] = 100 * float(held.group(1))
                row["heldout_sd"] = 100 * float(held.group(2))
            out[(scenario, bs, transform)][seed] = row
    return out


def references():
    """{run name: held-out mean TTA} for seed_summary lines on seeds 100-107 since the campaign began."""
    refs = {}
    for path in logs_since(SINCE):
        for line in open(path, errors="replace"):
            if line.startswith("seed_summary ") and "seeds=[100, 101, 102, 103, 104, 105, 106, 107]" in line:
                m = re.search(r"run=(\S+) .*?tta_val_acc_mean=(\S+)", line)
                refs[m.group(1)] = 100 * float(m.group(2))
    return refs


def target(refs, batch_size):
    return refs.get(f"baseline_bs{batch_size}")


def do_nothing(refs, scenario, batch_size, transform):
    if not transform:
        return refs.get(f"baseline_bs{batch_size}")
    return refs.get(f"transformed_{scenario}{transform}_bs{batch_size}")

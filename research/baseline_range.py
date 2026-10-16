"""Hparam range around the baseline: how far each LR's initial value can move before the full
run's score drops by 10%.

baseline_bs2000, full 200-step runs. One LR at a time (the others on the baseline), its whole
baseline schedule is scaled so the initial LR steps along the 0.6^k grid (+1 = one step larger,
x1/0.6), on each side until the test TTA is at most (1 - drop) x the baseline's TTA on the same
seed, or after max_side steps. A diverged or crashed run counts as a drop.
"""
import argparse
import copy
from math import isfinite

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--groups", required=True, help="comma-separated LRs to probe")
parser.add_argument("--directions", default="-1,1", help="sides to probe (-1 smaller, 1 larger)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--drop", type=float, default=0.10, help="relative TTA drop that stops a side")
parser.add_argument("--max-side", type=int, default=20, help="max grid steps on each side")
args = parser.parse_args()
log = open_log(f"baseline_range_bs2000_{args.groups.replace(',', '-')}_d{args.directions.replace(',', '')}",
               args, __file__)

MULT = 0.6
config = base_config(2000)
model = new_model()


def run(name, cfg):
    try:
        result = t.run_seeded(name, model, cfg, args.seed, debug=False)
        tta, acc = result["tta_val_acc"], result["val_acc"]
    except Exception as error:  # A huge LR may crash the run; it counts as diverged.
        emit(log, kind="error", run=name, error=repr(error)[:300])
        return 0.0, 0.0
    return (tta, acc) if isfinite(tta) else (0.0, 0.0)


def scaled(group, initial):
    lines = config["param_groups"][group]["lr_scheduler"]
    factor = initial / lines[0][1]
    return [[line[0], *(v * factor for v in line[1:])] for line in lines]


center_tta, center_acc = run("baseline_range_center", config)
threshold = (1 - args.drop) * center_tta
emit(log, kind="center", tta=center_tta, val_acc=center_acc, threshold=threshold, seed=args.seed)
for g in args.groups.split(","):
    initial = config["param_groups"][g]["lr_scheduler"][0][1]
    index = t.lr_grid_index(initial, MULT)
    for direction in map(int, args.directions.split(",")):
        stop = None
        for k in range(1, args.max_side + 1):
            value = t.round_hparam(MULT ** (index - direction * k))
            cfg = copy.deepcopy(config)
            cfg["param_groups"][g]["lr_scheduler"] = scaled(g, value)
            tta, acc = run(f"baseline_range_{g}_{direction * k:+d}", cfg)
            emit(log, kind="range", group=g, offset=direction * k, value=value, tta=tta, val_acc=acc)
            if tta <= threshold:
                stop = direction * k
                break
        emit(log, kind="side", group=g, direction=direction, stop=stop)
emit(log, kind="done")

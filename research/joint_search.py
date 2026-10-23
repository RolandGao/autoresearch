"""Capacity of the 5-constant-interval conv schedule: a joint neighbourhood search over full runs.

Conv LR only (the other groups on the baseline), conv's specified schedule = 5 constant intervals on
the x0.6 grid, under a transform of conv's applied LR. One search seed. Each iteration evaluates all
3^5 - 1 = 242 neighbours (every interval moved by -1, 0 or +1 grid step, jointly) as full runs, moves
to the best one if it beats the current schedule, and repeats until no neighbour is better. Every
evaluated schedule is logged (kind=eval; cached, so each schedule is trained once); the winner's
search score is the result (no replay). Start: the oracle (baseline conv LR divided by the
transform) reduced to its 5 interval means, snapped to the grid. References on the same seed:
baseline (the untransformed baseline; only with --transform none) and do_nothing (the baseline's
specified schedule under the transform).

python research/joint_search.py --transform ramp --seed 0
"""
import argparse
import copy
import itertools

from common import emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--transform", default="none", choices=["none", *t.LR_TRANSFORMS])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--intervals", type=int, default=5)
parser.add_argument("--max-iters", type=int, default=50)
args = parser.parse_args()
log = open_log(f"joint_search_{args.transform}_bs{args.batch_size}_s{args.seed}", args, __file__)

raw = copy.deepcopy(t.RUN_CONFIGS[f"baseline_bs{args.batch_size}"])
if args.transform != "none":
    raw["param_groups"]["conv"]["lr_transform"] = copy.deepcopy(t.LR_TRANSFORMS[args.transform])
config = t.normalize_config(raw, key=None)
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
base = trainer.base_schedules
total = trainer.total_steps
length = total // args.intervals
assert length * args.intervals == total
start_state = trainer.snapshot()
transform = config["param_groups"]["conv"].get("lr_transform")


def factor(step):
    return 1.0 if transform is None else t.lr_transform_factor(transform, step, total)


def run(conv_lines):
    trainer.restore(start_state)
    sched = dict(base)
    sched["conv.initial_lr"] = conv_lines
    ok = trainer.train(sched, total)
    return 100 * trainer.score("tta_val_acc") if ok else 0.0


cache = {}


def evaluate(point):
    if point not in cache:
        values = [t.round_hparam(0.6 ** i) for i in point]
        cache[point] = run([(length, v) for v in values])
        emit(log, kind="eval", values=values, score=cache[point] / 100, n=len(cache))
    return cache[point]


refs = {"do_nothing": run(base["conv.initial_lr"])}
if transform is None:
    refs["baseline"] = refs["do_nothing"]
emit(log, kind="setup", seed=args.seed, total=total, length=length, transform=transform,
     **{k: v / 100 for k, v in refs.items()})

oracle = [t.schedule_value_at_step(base["conv.initial_lr"], s) / factor(s) for s in range(total)]
current = tuple(t.lr_grid_index(sum(oracle[i * length:(i + 1) * length]) / length, 0.6)
                for i in range(args.intervals))
current_score = evaluate(current)
emit(log, kind="move", iteration=0, values=[t.round_hparam(0.6 ** i) for i in current], score=current_score / 100)
for iteration in range(1, args.max_iters + 1):
    neighbours = [tuple(c + d for c, d in zip(current, delta))
                  for delta in itertools.product((-1, 0, 1), repeat=args.intervals) if any(delta)]
    best = max(neighbours, key=evaluate)
    if evaluate(best) <= current_score:
        break
    current, current_score = best, evaluate(best)
    emit(log, kind="move", iteration=iteration, values=[t.round_hparam(0.6 ** i) for i in current],
         score=current_score / 100)
emit(log, kind="done", values=[t.round_hparam(0.6 ** i) for i in current], score=current_score / 100,
     evaluations=len(cache), train_steps=trainer.train_steps)

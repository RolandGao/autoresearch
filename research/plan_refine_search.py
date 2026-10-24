"""The 5-constant-interval conv schedule refined interval by interval (ICS pass 2), against the joint search.

Same setting as research/joint_search.py: conv LR only (the other groups on the baseline), conv's specified
schedule = 5 constant intervals on the x0.6 grid, under a transform of conv's applied LR, one search seed.
Plan = joint_search.py's start: the oracle (baseline conv LR divided by the transform) reduced to its 5
interval means, snapped to the grid (no shift). For each interval k in order, interval 0 included, from the
state after the committed intervals 0..k-1: score the plan's value, then one grid step smaller and, only if
that is not better, one grid step larger (ICS pass 2's climb, refine_side_steps=1); a candidate's score is
interval k at the candidate followed by the rest of the run on the plan (ICS rest_fixed lookahead), so every
score is a full run's tta_val_acc. The best is committed and its state kept. No averaged continuations for
the last interval. The committed schedule's score is its own last search score (no replay).
--passes 2: a second pass from initialisation whose plan (candidates' centre and lookahead tail) is the first
pass's committed schedule; logs as plan2_*.
--climb: score both one-step neighbours of the plan's value, then keep stepping toward the better one while
the score improves (no step limit); logs as plan_climb_*.

python research/plan_refine_search.py --transform ramp --seed 0
"""
import argparse
import copy

from common import emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--transform", default="none", choices=["none", *t.LR_TRANSFORMS])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--intervals", type=int, default=5)
parser.add_argument("--climb", action="store_true",
                    help="score both one-step neighbours, then keep stepping toward the better one while it improves")
parser.add_argument("--passes", type=int, default=1,
                    help="refinement passes; each later pass restarts from initialisation with the previous "
                         "pass's committed schedule as its plan")
args = parser.parse_args()
log = open_log(f"plan{args.passes if args.passes > 1 else ''}_{'climb' if args.climb else 'refine'}_{args.transform}"
               f"_bs{args.batch_size}_s{args.seed}", args, __file__)

raw = copy.deepcopy(t.RUN_CONFIGS[f"baseline_bs{args.batch_size}"])
if args.transform != "none":
    raw["param_groups"]["conv"]["lr_transform"] = copy.deepcopy(t.LR_TRANSFORMS[args.transform])
config = t.normalize_config(raw, key=None)
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
base = trainer.base_schedules
total = trainer.total_steps
n = args.intervals
length = total // n
assert length * n == total
transform = config["param_groups"]["conv"].get("lr_transform")


def factor(step):
    return 1.0 if transform is None else t.lr_transform_factor(transform, step, total)


def grid(index):
    return t.round_hparam(0.6 ** index)


oracle = [t.schedule_value_at_step(base["conv.initial_lr"], s) / factor(s) for s in range(total)]
plan = [t.lr_grid_index(sum(oracle[i * length:(i + 1) * length]) / length, 0.6) for i in range(n)]
emit(log, kind="setup", seed=args.seed, total=total, length=length, transform=transform,
     plan=[grid(i) for i in plan])

start_state = trainer.snapshot()


def refine_pass(search_pass, plan):
    """One refinement pass from initialisation over plan (grid indices); returns (committed, score)."""
    committed = []
    state = start_state
    for k in range(n):
        scores, states = {}, {}

        def evaluate(index):
            """Interval k at grid index `index` from the committed prefix, then the plan to the end."""
            if index not in scores:
                trainer.restore(state)
                sched = dict(base)
                sched["conv.initial_lr"] = [(length, grid(i)) for i in committed + [index] + plan[k + 1:]]
                ok = trainer.train(sched, length)
                states[index] = trainer.snapshot() if ok else None
                if ok and k < n - 1:
                    ok = trainer.train(sched, total - (k + 1) * length)
                scores[index] = 100 * trainer.score("tta_val_acc") if ok else 0.0
                emit(log, kind="eval", search_pass=search_pass, interval=k, value=grid(index), offset=index - plan[k],
                     values=[grid(i) for i in committed + [index] + plan[k + 1:]], score=scores[index] / 100,
                     train_steps=trainer.train_steps)
            return scores[index]

        best = plan[k]
        evaluate(best)
        if args.climb:  # both neighbours first, then climb the better side, unbounded
            direction = max((1, -1), key=lambda d: evaluate(plan[k] + d))
            while evaluate(best + direction) > scores[best]:
                best += direction
        else:
            for direction in (1, -1):  # grid index +1 = one step smaller: smaller first, as in ICS's climb
                if evaluate(plan[k] + direction) > scores[best]:
                    best = plan[k] + direction
                    break
        committed.append(best)
        state = states[best]
        emit(log, kind="commit", search_pass=search_pass, interval=k, value=grid(best), offset=best - plan[k],
             score=scores[best] / 100, tried={grid(i): v / 100 for i, v in scores.items()},
             train_steps=trainer.train_steps)
    return committed, scores[committed[-1]]


current = plan
for search_pass in range(args.passes):
    current, score = refine_pass(search_pass, current)
    emit(log, kind="pass_done", search_pass=search_pass, values=[grid(i) for i in current], score=score / 100,
         train_steps=trainer.train_steps, runs=trainer.train_steps / total)
emit(log, kind="done", values=[grid(i) for i in current], score=score / 100, plan=[grid(i) for i in plan],
     passes=args.passes, train_steps=trainer.train_steps, runs=trainer.train_steps / total)

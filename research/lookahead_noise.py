"""How noisy and how biased is a lookahead's comparison of interval candidates?

Constant-interval conv schedules (5 intervals), the other groups on the baseline. A plan (one
constant conv LR per interval; default C4, which matches the baseline at BS 2000) is followed
up to interval k; there, candidate constant LRs c = the plan's value moved by grid offsets
(x0.6 grid) are trained for one interval from the same state, then scored by:
  truth     the rest of the run on the plan (the final score the decision affects)
  rest_lin  the rest of the run with conv decaying linearly from c to 0
  c40_<m>   one interval (T/5 steps) with conv constant at m x c
  l40       one interval with conv decaying linearly from c to 0
  a40_<m>   one interval with conv constant at m x the plan's value for interval k (the same
            cooldown for every candidate; only with --abs-mults)
Every seed is its own run from initialization (data order and augmentation follow the seed), so
the spread over seeds of a candidate's score, and of the difference between two candidates
scored on the same seed, are the noise a single-seed search decision faces.
"""
import argparse

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--plan", required=True, help="comma-separated constant conv LR per interval")
parser.add_argument("--offsets", default="-2,-1,0,1,2")
parser.add_argument("--mults", default="0.6,0.36,0.22")
parser.add_argument("--abs-mults", default="", help="common-level cooldown multipliers (of the plan's value)")
parser.add_argument("--tag", default="", help="log name suffix (a separate probe series)")
args = parser.parse_args()
log = open_log(f"lookahead_noise{args.tag}_bs{args.batch_size}_s{args.seed}", args, __file__)

config = base_config(args.batch_size)
plan = [float(v) for v in args.plan.split(",")]
offsets = [int(o) for o in args.offsets.split(",")]
mults = [float(m) for m in args.mults.split(",")]
abs_mults = [float(m) for m in args.abs_mults.split(",")] if args.abs_mults else []
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
base = trainer.base_schedules
total = trainer.total_steps
length = total // len(plan)


def conv_schedule(head, rest):
    """Constant intervals for the given head values, then rest (a list of schedule lines)."""
    return [(length, v) for v in head] + rest


def run_with(conv_lines, steps):
    sched = dict(base)
    sched["conv.initial_lr"] = conv_lines
    ok = trainer.train(sched, steps)
    return trainer.score("tta_val_acc") if ok else 0.0


start = trainer.snapshot()
emit(log, kind="setup", plan=plan, total=total, length=length, seed=args.seed)
prefix_state = start
for k in range(len(plan)):
    remaining = total - (k + 1) * length
    for offset in offsets:
        c = t.round_hparam(0.6 ** (t.lr_grid_index(plan[k], 0.6) - offset))
        trainer.restore(prefix_state)
        head = plan[:k] + [c]
        run_with(conv_schedule(head, conv_schedule(plan[k + 1:], [])), length)
        mid = trainer.snapshot()
        scores = {}
        scores["truth"] = run_with(conv_schedule(head, conv_schedule(plan[k + 1:], [])), remaining) if remaining \
            else trainer.score("tta_val_acc")
        if remaining:
            trainer.restore(mid)
            scores["rest_lin"] = run_with(conv_schedule(head, [(remaining, c, 0.0)]), remaining)
            for m in mults:
                trainer.restore(mid)
                scores[f"c40_{m:g}"] = run_with(conv_schedule(head, [(length, m * c), (remaining - length, 0.0)]),
                                                 length)
            trainer.restore(mid)
            scores["l40"] = run_with(conv_schedule(head, [(length, c, 0.0), (remaining - length, 0.0)]), length)
            for m in abs_mults:
                trainer.restore(mid)
                scores[f"a40_{m:g}"] = run_with(conv_schedule(head, [(length, m * plan[k]), (remaining - length, 0.0)]),
                                                length)
        emit(log, kind="eval", k=k, offset=offset, c=c, **scores)
    trainer.restore(prefix_state)
    run_with(conv_schedule(plan, []), length)
    prefix_state = trainer.snapshot()
emit(log, kind="done", train_steps=trainer.train_steps)

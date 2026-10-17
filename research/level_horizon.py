"""How does each LR's level, set from scratch in the first interval, score at short and long
horizons?

From initialization, the four trained LRs are constant over the first interval (T/5 steps) at a
context (conv 0.13 and the others at their baseline initial LRs by default); one group's level is
varied over a x7.7 grid (0.6^4) around its context value. Each level is scored:
  none      right after the interval
  short     after one more interval with every group at 0.22 x its interval value
  rest_lin  after the rest of the run with every group decaying linearly from its interval value
            to 0 (a full run's horizon)
A level choice made at a short horizon is only as good as its agreement with rest_lin.
"""
import argparse

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--group", required=True)
parser.add_argument("--offsets", default="-4,-3,-2,-1,0,1,2,3,4,5")
parser.add_argument("--conv", type=float, default=0.13)
args = parser.parse_args()
log = open_log(f"level_horizon_bs{args.batch_size}_{args.group}_s{args.seed}", args, __file__)

GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
config = base_config(args.batch_size)
context = {g: config["param_groups"][g]["lr_scheduler"][0][1] for g in GROUPS}
context["conv"] = args.conv
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
base = trainer.base_schedules
total = trainer.total_steps
length = total // 5
start = trainer.snapshot()
emit(log, kind="setup", context=context, total=total, length=length, seed=args.seed)
for offset in [int(o) for o in args.offsets.split(",")]:
    values = dict(context)
    values[args.group] = t.round_hparam(0.6 ** (t.lr_grid_index(context[args.group], 0.6) - 4 * offset))
    scores = {}
    for kind, steps in (("none", 0), ("short", length), ("rest_lin", total - length)):
        trainer.restore(start)
        sched = dict(base)
        for g in GROUPS:
            if kind == "short":
                sched[f"{g}.initial_lr"] = [(length, values[g]), (length, 0.22 * values[g]), (total - 2 * length, 0.0)]
            elif kind == "rest_lin":
                sched[f"{g}.initial_lr"] = [(length, values[g]), (total - length, values[g], 0.0)]
            else:
                sched[f"{g}.initial_lr"] = [(length, values[g]), (total - length, 0.0)]
        ok = trainer.train(sched, length + steps)
        scores[kind] = trainer.score("tta_val_acc") if ok else 0.0
    emit(log, kind="eval", group=args.group, offset=offset, value=values[args.group], **scores)
emit(log, kind="done", train_steps=trainer.train_steps)

"""Cooldown proxy fidelity (scenario A: one searched group, the rest on the baseline).

For main interval k (length L = T/5) and candidate constant LRs c of the searched group:
  truth:  baseline until kL, c for L steps, then the baseline schedule to the end -> final metric.
  proxy:  same prefix and main interval, then a C-step lookahead cooldown -> metric.
  none:   the metric right after the main interval.
Cooldown variants: searched group constant m_s * c (or linear c -> 0, "lin0");
other groups continue the baseline ("cont"), decay linearly from their current value to 0
("lin0"), or are constant at m_o * their mean LR over the main interval.
A good cooldown ranks the candidates like the truth does.
"""
import argparse
import itertools

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--config", default=None)
parser.add_argument("--group", default="conv")
parser.add_argument("--interval", type=int, required=True)
parser.add_argument("--seeds", default="0,1,2,3")
parser.add_argument("--offsets", default="-3,-2,-1,0,1,2,3")
parser.add_argument("--searched-mults", default="1,0.6,0.36,0.22,0.13,0.047,0,lin0")
parser.add_argument("--other-mults", default="cont,lin0,1,0.6,0.36,0.22,0.13,0.047,0")
parser.add_argument("--fraction", type=int, default=5)
parser.add_argument("--transform", default=None, help="json lr_transform for the searched group")
args = parser.parse_args()
log = open_log(f"cooldown_fidelity_bs{args.batch_size}_{args.group}_k{args.interval}", args, __file__)

import json  # noqa: E402

config = base_config(args.batch_size, args.config)
if args.transform:
    config["param_groups"][args.group]["lr_transform"] = json.loads(args.transform)


def parse(values):
    return [v if v in ("cont", "lin0") else float(v) for v in values.split(",")]


searched_mults, other_mults = parse(args.searched_mults), parse(args.other_mults)
model = new_model()
path = f"{args.group}.initial_lr"
candidates = None
for seed in [int(s) for s in args.seeds.split(",")]:
    t.TRAINING_SEED = seed
    trainer = t.SegmentTrainer(model, config)
    base = trainer.base_schedules
    total = trainer.total_steps
    length = total // args.fraction
    start, cool = args.interval * length, length
    if start + length + cool > total:
        raise SystemExit("interval has no room for a cooldown")
    if candidates is None:
        centre = t.lr_grid_index(t.window_mean(base[path], start, length), 0.6)
        candidates = [t.round_hparam(0.6 ** (centre + o)) for o in map(int, args.offsets.split(","))]
        emit(log, kind="setup", total=total, length=length, start=start, candidates=candidates,
             mean_lr=t.window_mean(base[path], start, length))
    trainer.train(base, start)
    prefix = trainer.snapshot()
    for c in candidates:
        trainer.restore(prefix)
        main = dict(base)
        main[path] = t.constant_window(base[path], start, length, c)
        ok = trainer.train(main, length)
        main_state = trainer.snapshot()
        emit(log, kind="none", seed=seed, c=c, score=trainer.score("tta_val_acc") if ok else None)
        if ok:
            ok = trainer.train(main, total - start - length)
        emit(log, kind="truth", seed=seed, c=c, score=trainer.score("tta_val_acc") if ok else None)
        cool_start = start + length
        for ms, mo in itertools.product(searched_mults, other_mults):
            trainer.restore(main_state)
            sched = dict(main)
            sched[path] = (t.linear_window(main[path], cool_start, cool, c, 0.0) if ms == "lin0"
                           else t.constant_window(main[path], cool_start, cool, ms * c))
            for other, lines in base.items():
                if other == path or not other.endswith(".initial_lr") or mo == "cont":
                    continue
                if mo == "lin0":
                    first = t.schedule_value_at_step(lines, cool_start)
                    sched[other] = t.linear_window(lines, cool_start, cool, first, 0.0)
                else:
                    sched[other] = t.constant_window(
                        lines, cool_start, cool, mo * t.window_mean(lines, start, length))
            ok = trainer.train(sched, cool)
            emit(log, kind="proxy", seed=seed, c=c, ms=ms, mo=mo,
                 score=trainer.score("tta_val_acc") if ok else None)
emit(log, kind="done", train_steps=trainer.train_steps)

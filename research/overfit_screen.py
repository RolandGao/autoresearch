"""Screening by a few steps: does a short run locate each group's LR scale and sensitivity?

For each group, sweep its constant LR over a x sqrt(10) grid (other groups constant at
their baseline initial LR) and train K steps from initialization:
  fixed: overfit one fixed (augmented) training batch; score = loss on that batch.
  fresh: normal training batches; score = loss on a fixed batch of 2000 test images.
Scores are logged after 1, 2, 4, ..., K steps.
"""
import argparse

import torch

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--config", default=None)
parser.add_argument("--groups", default="conv,head,norm_bias,whiten_bias")
parser.add_argument("--exponents", default="-8,10", help="LR = 10**(e/2) for e in [lo, hi]")
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--seeds", default="0,1")
parser.add_argument("--modes", default="fixed,fresh")
args = parser.parse_args()
log = open_log(f"overfit_screen_bs{args.batch_size}", args, __file__)

lo, hi = map(int, args.exponents.split(","))
lrs = [t.round_hparam(10 ** (e / 2)) for e in range(lo, hi + 1)]
checkpoints = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.steps]
model = new_model()
for mode in args.modes.split(","):
    config = base_config(args.batch_size, args.config)
    if mode == "fixed":
        config.update(overfit=True, num_epochs=args.steps)
    # Constant schedules at each group's initial LR, over the whole (short) run.
    for group in config["param_groups"].values():
        group["lr_scheduler"] = [(args.steps if mode == "fixed" else
                                  sum(line[0] for line in group["lr_scheduler"]),
                                  group["lr_scheduler"][0][1])]
    for seed in [int(s) for s in args.seeds.split(",")]:
        t.TRAINING_SEED = seed
        trainer = t.SegmentTrainer(model, config)
        held = None
        if mode == "fresh":
            images, labels = next(iter(trainer.test_loader))
            held = t.FixedBatchLoader(images, labels)
        start = trainer.snapshot()
        for name in args.groups.split(","):
            path = f"{name}.initial_lr"
            for lr in lrs:
                trainer.restore(start)
                sched = dict(trainer.base_schedules)
                sched[path] = [(trainer.total_steps, lr)]
                losses, done = {}, 0
                for k in checkpoints:
                    ok = trainer.train(sched, k - done)
                    done = k
                    if not ok:
                        break
                    loss = (t.batch_loss(model, held) if held is not None
                            else -trainer.score("neg_train_loss"))
                    losses[k] = loss if loss == loss else None
                emit(log, mode=mode, seed=seed, group=name, lr=lr, losses=losses)
emit(log, kind="done")

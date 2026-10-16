"""Hparam importance from scratch: main-interval search of the four LRs, then one-at-a-time sweeps.

Batch size 2000, baseline_bs2000 except the four trained LRs (conv, head, norm_bias,
whiten_bias), which are constant for the first L training steps from initialization; the score
is taken right after those L steps (no lookahead).
  1. Joint main-interval search from scratch: every LR starts at 1.0; scan each LR over
     x10^k, k = -3..3 (others fixed), keep the best, snap to the 0.6^k grid, then coordinate
     climb on the grid (one step up / down, move while the score improves) until no LR moves.
  2. Sweep: from that center, vary each LR alone over 5 grid points on each side.
Score: test-set TTA accuracy; the test cross-entropy loss is logged too.
"""
import argparse

import torch
import torch.nn.functional as F

from common import base_config, emit, new_model, open_log, t

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True, help="main interval length L")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--side", type=int, default=5, help="grid points on each side in the sweep")
args = parser.parse_args()
log = open_log(f"hparam_importance_bs2000_L{args.steps}", args, __file__)

GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
MULT = 0.6
config = base_config(2000, None)
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
start = trainer.snapshot()
base = trainer.base_schedules
cache = {}


def test_loss():
    model.train()  # Train-mode BatchNorm, as in every evaluation.
    with torch.inference_mode():
        images, labels = trainer.test_loader.normalized_images(), trainer.test_loader.labels
        total = sum(F.cross_entropy(model(x).float(), y, reduction="sum")
                    for x, y in zip(images.split(2000), labels.split(2000)))
    return (total / len(labels)).item()


def evaluate(values, phase):
    key = tuple(values[g] for g in GROUPS)
    if key not in cache:
        trainer.restore(start)
        sched = dict(base)
        for g in GROUPS:
            sched[f"{g}.initial_lr"] = t.constant_window(base[f"{g}.initial_lr"], 0, args.steps, values[g])
        ok = trainer.train(sched, args.steps)
        tta = trainer.score("tta_val_acc") if ok else float("-inf")
        loss = test_loss() if ok else float("inf")
        cache[key] = (tta, loss)
        emit(log, kind="eval", phase=phase, values=dict(values), tta=tta, loss=loss)
    return cache[key][0]


def grid(value, offset):
    return t.round_hparam(MULT ** (t.lr_grid_index(value, MULT) - offset))


# 1. Joint main-interval search from scratch.
current = {g: 1.0 for g in GROUPS}
for g in GROUPS:  # Coarse scan over decades.
    options = [t.round_hparam(current[g] * 10.0 ** k) for k in range(-3, 4)]
    scores = {v: evaluate(dict(current, **{g: v}), "scan") for v in options}
    current[g] = max(options, key=lambda v: scores[v])
current = {g: grid(v, 0) for g, v in current.items()}
best = evaluate(current, "climb")
for _ in range(20):  # Coordinate climb on the grid.
    moved = False
    for g in GROUPS:
        for direction in (1, -1):
            while True:
                trial = dict(current, **{g: grid(current[g], direction)})
                score = evaluate(trial, "climb")
                if score <= best:
                    break
                current, best, moved = trial, score, True
    if not moved:
        break
emit(log, kind="center", values=dict(current), tta=best, loss=cache[tuple(current[g] for g in GROUPS)][1],
     evaluations=len(cache))

# 2. One-at-a-time sweeps around the center.
for g in GROUPS:
    for offset in range(-args.side, args.side + 1):
        trial = dict(current, **{g: grid(current[g], offset)})
        evaluate(trial, "sweep")
        tta, loss = cache[tuple(trial[h] for h in GROUPS)]
        emit(log, kind="sweep", group=g, offset=offset, value=trial[g], tta=tta, loss=loss)
emit(log, kind="done", evaluations=len(cache), train_steps=trainer.train_steps)

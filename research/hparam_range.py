"""Hparam range from scratch: how far each LR can move from the Experiment 2 center before the
score drops by 10%.

Same setup as research/hparam_importance.py (batch size 2000, baseline_bs2000 except the four
trained LRs, constant for the first L training steps from initialization, scored right after
those L steps). The center is that experiment's joint-search center for L. From it, one LR
at a time steps along the 0.6^k grid on each side (+1 = one step larger, x1/0.6) until the test
TTA is at most (1 - drop) x the center's TTA, or after max_side steps. A diverged run counts as
a drop. Every evaluated point is logged with its test loss.
"""
import argparse

import torch
import torch.nn.functional as F

from common import base_config, emit, new_model, open_log, t
import importance_study

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True, help="main interval length L")
parser.add_argument("--groups", required=True, help="comma-separated LRs to probe")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--drop", type=float, default=0.10, help="relative TTA drop that stops a side")
parser.add_argument("--max-side", type=int, default=20, help="max grid steps on each side")
args = parser.parse_args()
log = open_log(f"hparam_range_bs2000_L{args.steps}_{args.groups.replace(',', '-')}", args, __file__)

GROUPS = ("conv", "head", "norm_bias", "whiten_bias")
MULT = 0.6
config = base_config(2000, None)
model = new_model()
t.TRAINING_SEED = args.seed
trainer = t.SegmentTrainer(model, config)
start = trainer.snapshot()
base = trainer.base_schedules


def test_loss():
    model.train()  # Train-mode BatchNorm, as in every evaluation.
    with torch.inference_mode():
        images, labels = trainer.test_loader.normalized_images(), trainer.test_loader.labels
        total = sum(F.cross_entropy(model(x).float(), y, reduction="sum")
                    for x, y in zip(images.split(2000), labels.split(2000)))
    return (total / len(labels)).item()


def evaluate(values, phase):
    trainer.restore(start)
    sched = dict(base)
    for g in GROUPS:
        sched[f"{g}.initial_lr"] = t.constant_window(base[f"{g}.initial_lr"], 0, args.steps, values[g])
    ok = trainer.train(sched, args.steps)
    tta = trainer.score("tta_val_acc") if ok else float("-inf")
    loss = test_loss() if ok else float("inf")
    emit(log, kind="eval", phase=phase, values=dict(values), tta=tta, loss=loss)
    return tta, loss


def grid(value, offset):
    return t.round_hparam(MULT ** (t.lr_grid_index(value, MULT) - offset))


center = importance_study.load(args.steps)["center"]["values"]
center_tta, center_loss = evaluate(center, "center")
threshold = (1 - args.drop) * center_tta
emit(log, kind="center", values=center, tta=center_tta, loss=center_loss, threshold=threshold)
evaluations = 1
for g in args.groups.split(","):
    for direction in (-1, 1):
        stop = None
        for k in range(1, args.max_side + 1):
            trial = dict(center, **{g: grid(center[g], direction * k)})
            tta, loss = evaluate(trial, "range")
            evaluations += 1
            emit(log, kind="range", group=g, offset=direction * k, value=trial[g], tta=tta, loss=loss)
            if tta <= threshold:
                stop = direction * k
                break
        emit(log, kind="side", group=g, direction=direction, stop=stop)
emit(log, kind="done", evaluations=evaluations, train_steps=trainer.train_steps)

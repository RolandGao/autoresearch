"""Plot every LR transform of the interval-search study (train_cifar.LR_TRANSFORMS) to plots/lr_transforms.pdf.

Rows: one per transform, x = step / total steps. Columns: the multiplier on conv's applied LR; the
applied conv LR when the BS 2000 baseline's linear decay (0.22 -> 0) is kept under the transform
(do nothing); the specified schedule that gives the baseline back (baseline / multiplier, the oracle)
and its 5 interval means snapped to the x0.6 grid as constant intervals (the joint search's start).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_cifar as t  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#1f1f1e", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})

TOTAL, INTERVALS = 200, 5
TITLES = {
    "ramp": "ramp: x 4^(2x-1), x0.25 -> x4",
    "rampdown": "reverse ramp: x 0.25^(2x-1), x4 -> x0.25",
    "wave": "wave: x 2^sin(2 pi x), up first",
    "waveinv": "inverted wave: x 2^sin(2 pi (x+1/2)), down first (held back)",
    "ramp8": "steep ramp: x 8^(2x-1), x1/8 -> x8 (held back)",
}
steps = np.arange(TOTAL)
x = steps / TOTAL
baseline = 0.22 * (1 - x)
length = TOTAL // INTERVALS

fig, axes = plt.subplots(len(t.LR_TRANSFORMS), 3, figsize=(10, 2.15 * len(t.LR_TRANSFORMS)),
                         layout="constrained", sharex=True)
for row, (name, spec) in enumerate(t.LR_TRANSFORMS.items()):
    f = np.array([t.lr_transform_factor(spec, s, TOTAL) for s in steps])
    oracle = baseline / f
    stairs = [t.round_hparam(0.6 ** t.lr_grid_index(oracle[i * length:(i + 1) * length].mean(), 0.6))
              for i in range(INTERVALS)]

    ax = axes[row, 0]
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.plot(x, f, color=BLUE)
    ax.set_yscale("log", base=2)
    ticks = [v for v in (0.125, 0.25, 0.5, 1, 2, 4, 8) if f.min() / 1.05 <= v <= f.max() * 1.05]
    ax.set_yticks(ticks, labels=[f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_ylabel("multiplier")
    ax.set_title(TITLES.get(name, name))

    ax = axes[row, 1]
    ax.plot(x, baseline, color=INK, lw=1.6, label="baseline (target)")
    ax.plot(x, baseline * f, color=ORANGE, label="baseline schedule, transformed")
    ax.set_ylabel("applied conv LR")
    if row == 0:
        ax.set_title("applied LR if the schedule is left unchanged")

    ax = axes[row, 2]
    ax.plot(x, oracle, color=BLUE, label="oracle: baseline / multiplier")
    ax.stairs(stairs, np.linspace(0, 1, INTERVALS + 1), color=AQUA, lw=2, ls="--", baseline=None,
              label="its 5 interval means on the x0.6 grid")
    for i, v in enumerate(stairs):
        ax.annotate(f"{v:g}", ((i + 0.5) / INTERVALS, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.5, color=INK2)
    ax.set_ylabel("specified conv LR")
    if row == 0:
        ax.set_title("schedule a search must find")
for ax in axes[-1]:
    ax.set_xlabel("fraction of training")
for ax in axes.flat:
    ax.set_xlim(0, 1)
handles = [h for ax in axes[0, 1:] for h in ax.get_legend_handles_labels()[0]]
labels = [lab for ax in axes[0, 1:] for lab in ax.get_legend_handles_labels()[1]]
fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="outside lower center", ncol=4)
fig.suptitle("LR transforms on conv's applied LR (BS 2000, baseline 0.22 -> 0 over 200 steps)",
             fontsize=10, color=INK)
out = os.path.join(REPO, "plots", "lr_transforms.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print(out)

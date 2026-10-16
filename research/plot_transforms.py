"""Plot the two LR transforms of the interval-search study to plots/lr_transforms.pdf.

Rows: ramp (x 4**(2x-1)) and wave (x 2**sin(2 pi x)), x = step / total steps.
Columns: the multiplier; the applied conv LR when the BS 2000 baseline's linear decay
(0.22 -> 0) is kept under the transform; the spec schedule that gives the baseline back
(baseline / multiplier) and the grid-snapped 5-knot oracle used in the experiments.
"""
import os
import sys
from math import log

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_cifar as t  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})

x = np.linspace(0, 1, 401)
baseline = 0.22 * (1 - x)


def factor(spec, x):
    if spec["kind"] == "ramp":
        return spec["factor"] ** (2 * x - 1)
    return spec["factor"] ** np.sin(2 * np.pi * (x + spec["phase"]))


fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
titles = {"ramp": "ramp: x 4^(2x-1)  (x0.25 -> x4)", "wave": "wave: x 2^sin(2 pi x)  (x0.5 .. x2)"}
for row, (name, spec) in enumerate(t.LR_TRANSFORMS.items()):
    f = factor(spec, x)
    # 1. The multiplier.
    ax = axes[row, 0]
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=":")
    ax.plot(x, f, color=BLUE)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.25, 0.5, 1, 2, 4], labels=["0.25", "0.5", "1", "2", "4"])
    ax.set_ylabel("LR multiplier")
    ax.set_title(titles[name], loc="left", fontsize=10, color=TEXT)
    # 2. Applied LR if the baseline spec is kept.
    ax = axes[row, 1]
    ax.plot(x, baseline, color=BLUE, label="baseline (no transform)")
    ax.plot(x, baseline * f, color=ORANGE, label="baseline spec, transformed")
    ax.set_ylabel("applied conv LR")
    ax.set_title("applied LR, schedule left unchanged", loc="left", fontsize=10, color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    # 3. The spec that recovers the baseline, and the 5-knot grid oracle.
    ax = axes[row, 2]
    ax.plot(x, baseline, color=BLUE, label="baseline spec")
    ax.plot(x, baseline / f, color=ORANGE, label="spec giving the baseline back")
    knots_x = np.linspace(0, 1, 6)
    knots = [round(0.6 ** round(log(0.22 * (1 - k) / factor(spec, k)) / log(0.6)), 4)
             if k < 1 else 0.0 for k in knots_x]
    ax.plot(knots_x, knots, color=AQUA, linestyle="--", marker="o", markersize=5,
            label="5-knot grid oracle")
    ax.annotate("oracle", (knots_x[1], knots[1]), textcoords="offset points", xytext=(6, 6),
                color=MUTED, fontsize=8)
    ax.set_ylabel("specified conv LR")
    ax.set_title("schedule a search must find", loc="left", fontsize=10, color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
for ax in axes.flat:
    ax.set_xlabel("fraction of training")
    ax.set_xlim(0, 1)
fig.suptitle("LR transforms applied on top of the specified schedule (conv, BS 2000 baseline 0.22 -> 0)",
             fontsize=11, color=TEXT)
out = os.path.join(REPO, "plots", "lr_transforms.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=130)
print(out)

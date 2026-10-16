"""Finding F1 across all intervals -> plots/f1_intervals.pdf / .png.

Batch size 2000, conv searched; for each interval (40 steps), the seed-mean score of each
constant candidate conv LR under: the truth (final TTA when the baseline resumes), no
lookahead, linear-to-0 and searched constant cooldowns, with a one-interval or a
rest-of-training lookahead (research/report_data.fidelity_methods). Each curve is shown
relative to its own best; the marker at 0 is the LR it picks.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
import report_data as d  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # method -> (color, linestyle, marker)
    "truth": (TEXT, "-", "o"),
    "no lookahead": ("#8c8b86", ":", "x"),
    "linear to 0, one interval": ("#2a78d6", "-", "s"),
    "searched constant, one interval": ("#eb6834", "-", "^"),
    "linear to 0, to the end": ("#2a78d6", "--", "s"),
    "searched constant, to the end": ("#eb6834", "--", "^"),
}
plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.7,
})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.0), layout="constrained", sharey=True)
for k, ax in zip(range(4), axes.flat):
    curves = d.fidelity_methods(k)
    ax.set_title(f"interval {k + 1} (steps {40 * k}-{40 * k + 40} of 200)", loc="left", fontsize=9.5)
    if "truth" not in curves:
        ax.text(0.5, 0.5, "probe running", transform=ax.transAxes, ha="center", color=MUTED)
        continue
    for name, (color, style, marker) in STYLE.items():
        if name not in curves or (k == 3 and "to the end" in name):
            continue  # The last interval's rest is one interval: same as the solid lines.
        curve = curves[name]
        lrs = sorted(curve)
        best = max(curve.values())
        ax.plot(lrs, [curve[c] - best for c in lrs], color=color, linestyle=style, marker=marker,
                markersize=4, label=name)
    ax.set_xscale("log")
    lrs = sorted(curves["truth"])
    ax.set_xticks(lrs, labels=[f"{v:g}" for v in lrs])
    ax.minorticks_off()
    ax.set_ylim(-1.6, 0.15)
    ax.set_xlabel("constant conv LR during the interval")
for ax in axes[:, 0]:
    ax.set_ylabel("score minus the curve's best (pp)")
handles = {}
for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(label, handle)
fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, frameon=False,
           fontsize=8)
fig.suptitle("Which LR does each lookahead pick? (batch size 2000, conv; 4 seeds per point)",
             fontsize=10.5)
out = os.path.join(REPO, "plots", "f1_intervals.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=140)
print(out)

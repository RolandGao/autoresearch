"""Plot finding F1 (short lookaheads prefer small LRs) to plots/lookahead_bias.pdf / .png.

Data: research/cooldown_fidelity.py at BS 2000, interval 2 (steps 40-80), conv searched;
seed means (4 seeds) per candidate constant conv LR. Truth = final TTA when the baseline
schedule resumes after the interval; proxies = TTA after a lookahead of the given length
(constant cooldown at 0.13-0.22 x candidate). Each curve is shown relative to its own best
value, so the peak marks the LR it would choose.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
lrs = [0.028, 0.047, 0.078, 0.13, 0.22, 0.36, 0.6]
curves = [  # (label, values, color, style)
    ("final accuracy (truth)", [93.63, 93.73, 93.83, 93.94, 93.98, 93.57, 93.26], "#0b0b0b", "-"),
    ("40-step lookahead", [91.74, 92.36, 92.66, 92.55, 91.64, 89.37, 86.07], "#eb6834", "-"),
    ("80-step lookahead", [91.90, 92.58, 93.01, 93.02, 92.73, 91.78, 90.09], "#2a78d6", "--"),
    ("lookahead to the end", [92.17, 92.79, 93.21, 93.33, 93.23, 92.42, 90.80], "#1baf7a", "-."),
]
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})
fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
short = {"final accuracy (truth)": "truth", "40-step lookahead": "40-step",
         "80-step lookahead": "80-step", "lookahead to the end": "to end"}
peaks = {}  # chosen LR -> short names of the curves that choose it
for label, values, color, style in curves:
    best = max(values)
    ax.plot(lrs, [v - best for v in values], color=color, linestyle=style, marker="o",
            markersize=5, label=label)
    peaks.setdefault(lrs[values.index(best)], []).append(short[label])
for row, (peak, names) in enumerate(sorted(peaks.items())):
    ax.annotate(f"{', '.join(names)}: {peak:g}", (peak, 0.12 + 0.22 * (row % 2)),
                ha="center", fontsize=7, color=MUTED)
ax.set_xscale("log")
ax.set_xticks(lrs, labels=[f"{v:g}" for v in lrs])
ax.minorticks_off()
ax.set_ylim(-2.2, 0.6)
ax.set_xlabel("constant conv LR during interval 2 (steps 40-80 of 200)")
ax.set_ylabel("TTA minus the curve's best (pp)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
out = os.path.join(REPO, "plots", "lookahead_bias.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print(out)

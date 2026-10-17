"""Build cooldown_study.pdf: which 40-step lookahead cooldown ranks interval candidates like the
final result? Data: research/cooldown_search_probe.py logs via research/cooldown_study.py.
python research/make_cooldown_study.py"""
import glob
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
sys.path.insert(0, REPO)
import cooldown_study as cs  # noqa: E402
import importance_study as imp  # noqa: E402
import family_study as fam  # noqa: E402
import range_study as rng  # noqa: E402
import seed_runs as sr  # noqa: E402
import report_data  # noqa: E402
import train_cifar as tc  # noqa: E402

OUT = os.path.join(REPO, "cooldown_study.pdf")
OBSERVATIONS = os.path.join(REPO, "research", "cooldown_study_observations.txt")
INK, MUTED, RULE, HEAD_BG = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d9d8d3", "#f1f0ec"))
base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontSize=16, leading=20, alignment=TA_LEFT,
                            textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=8.5, textColor=MUTED, spaceAfter=8),
    "h1": ParagraphStyle("h1", parent=base["Heading2"], fontSize=12, leading=15, textColor=INK,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=base["Heading3"], fontSize=10, leading=12.5, textColor=INK,
                         spaceBefore=6, spaceAfter=2, keepWithNext=1),
    "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.3, leading=12.6, textColor=INK,
                           spaceAfter=4),
    "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8, leading=10.4, textColor=MUTED,
                            spaceAfter=4),
    "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.2, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8.2, leading=10.2, textColor=INK,
                            fontName="Helvetica-Bold"),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def keep(items):
    """KeepTogether with nested KeepTogethers flattened (nesting makes reportlab break pages)."""
    flat = []
    for item in items:
        flat += item._content if isinstance(item, KeepTogether) else [item]
    return KeepTogether(flat)


def table(rows, widths):
    data = [[Paragraph(str(c), S["cellb"] if r == 0 else S["cell"]) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([t, Spacer(1, 6)])


def figure(summaries, path):
    styles = {"truth": ("#0b0b0b", "-", "o"), "none": ("#8c8b86", ":", "x"),
              "lin_conv": ("#2a78d6", "-", "s"), "lin_all": ("#2a78d6", "--", "s"),
              "conv_only": ("#1baf7a", "-", "D"), "shared": ("#eb6834", "-", "^"),
              "per_group": ("#eb6834", "--", "v")}
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#e4e3df", "axes.grid": True,
                         "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "lines.linewidth": 1.7})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0), layout="constrained", sharey=True)
    for k, ax in enumerate(axes.flat):
        s = summaries[k]
        ax.set_title(f"interval {k + 1} (steps {40 * k}-{40 * k + 40} of 200)", loc="left", fontsize=9.5)
        if s is None:
            ax.text(0.5, 0.5, "probe running", transform=ax.transAxes, ha="center", color="#52514e")
            continue
        for m in cs.METHODS:
            color, style, marker = styles[m]
            curve = s["curves"][m]
            top = max(curve.values())
            ax.plot(s["cands"], [curve[c] - top for c in s["cands"]], color=color, linestyle=style,
                    marker=marker, markersize=4, label=cs.LABELS[m])
        ax.set_xscale("log")
        ax.set_xticks(s["cands"], labels=[f"{c:g}" for c in s["cands"]])
        ax.minorticks_off()
        ax.set_ylim(-1.5, 0.12)
        ax.set_xlabel("constant conv LR during the interval")
    for ax in axes[:, 0]:
        ax.set_ylabel("score minus the curve's best (pp)")
    handles = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, frameon=False,
               fontsize=8)
    fig.savefig(path, dpi=140)


summaries = [cs.summary(k) for k in range(4)]
fig_path = os.path.join(REPO, "plots", "cooldown_study.png")
figure(summaries, fig_path)
img = Image(fig_path)
img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth

story = [
    P("Interval search experiments", "title"),
    P(f"Built {datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d %H:%M} PT"
      f"{'' if all(x is not None and len(x['cands']) == 7 for x in summaries) else ' (runs in progress; partial results are marked)'}. "
      "Experiment design by Roland Gao; runs and analysis by Claude. Code: "
      "research/cooldown_search_probe.py (runs), research/cooldown_study.py (analysis), "
      "research/make_cooldown_study.py (this PDF). Logs: logs/research_cooldown_search_probe_*, "
      "logs/research_hparam_importance_*. Batch size 2000 throughout.", "sub"),
    P("Experiment 1. Cooldown study: which 40-step lookahead ranks interval candidates like the final "
      "result?", "h1"),
    P("Question", "h2"),
    P("An interval search picks each interval's LR by training the interval with a candidate LR, then "
      "a short lookahead (the cooldown), and scoring the result. Which cooldown ranks the candidates the "
      "way the final accuracy does? Compared, all with a 40-step lookahead (one interval at batch size "
      "2000): no lookahead; linear decay to 0 (conv only, or all four groups); one searched constant "
      "cooldown, either for conv only with the other groups on their ground-truth schedules, or shared by "
      "all four groups; and four searched constant cooldowns (one per group)."),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Model / data", "CIFAR-10 airbench, batch size 2000, 200 steps, baseline_bs2000 (conv Muon2 0.22 -> 0; "
                         "head, norm_bias, whiten_bias SGD on their baseline schedules)"],
        ["Intervals", "4 intervals of 40 steps: steps 0-40, 40-80, 80-120, 120-160 (the last interval has no "
                      "room for a 40-step lookahead)"],
        ["Candidates", "7 constant conv LRs for the interval: the 0.6^k grid point nearest the baseline's mean conv "
                       "LR in that interval, and 3 grid steps either side; other groups on the baseline"],
        ["Ground truth", "Final test TTA when training continues on the baseline schedule after the interval"],
        ["No lookahead", "Test TTA right after the interval"],
        ["Linear to 0", "Conv only: conv decays linearly from the candidate to 0 over 40 steps, the other groups "
                        "continue their schedules. All groups: every group decays linearly to 0 from its current "
                        "value"],
        ["1 searched cooldown: conv only", "Conv constant at m x candidate for 40 steps; the other groups follow "
                                           "their baseline (ground-truth) schedules; m chosen from {1, 0.6, 0.36, "
                                           "0.22, 0.13, 0.078, 0.047, 0} by the best score on that seed"],
        ["1 searched cooldown: shared", "Constant LRs for 40 steps: conv at m x candidate, each other group at m x "
                                        "its mean LR over the interval; one m for all groups, chosen as above"],
        ["4 searched cooldowns", "As above but one m per group; coordinate search from the best shared m, one "
                                 "group at a time over the same 8 values, until no group changes (at most 3 "
                                 "passes)"],
        ["Seed", "1 training seed (0); every method uses the same prefix and data order"],
        ["Score", "Test-set TTA accuracy (%)"],
    ], [3.3 * cm, 13.7 * cm]),
    P("Results", "h2"),
    P("Picked LR = the candidate with the highest score. Regret = ground truth at its best candidate "
      "minus ground truth at the picked one (pp). One seed: single-run noise is about 0.13 pp, so "
      "differences of that size between candidates are not resolved.", "small"),
]
rows = [["Method"] + [f"Interval {k + 1}" for k in range(4)]]
for m in cs.METHODS:
    cells = [cs.LABELS[m]]
    for s in summaries:
        partial = "" if s is None or len(s["cands"]) == 7 else f" [{len(s['cands'])}/7 candidates]"
        cells.append("running" if s is None else f"{s['picks'][m]:g} ({s['regret'][m]:.2f}){partial}")
    rows.append(cells)
story += [P("Picked LR (regret, pp)", "h2"),
          table(rows, [5.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]),
          keep([img, P("Figure 1. Each method's score per candidate, relative to its own best "
                               "(the point at 0 is its pick).", "small")])]
for s in summaries:
    if s is None:
        continue
    k = s["k"]
    rows = [["Method"] + [f"{c:g}" for c in s["cands"]]]
    for m in cs.METHODS:
        curve = s["curves"][m]
        top = max(curve.values())
        rows.append([cs.LABELS[m]] + [f"<b>{curve[c]:.2f}</b>" if curve[c] == top else f"{curve[c]:.2f}"
                                      for c in s["cands"]])
    width = (17 - 5.2) / len(s["cands"])
    partial = "" if len(s["cands"]) == 7 else f" - PARTIAL, {len(s['cands'])} of 7 candidates so far"
    story += [keep([P(f"Interval {k + 1}: score per candidate conv LR (bold: each row's best){partial}",
                              "h2"), table(rows, [5.2 * cm] + [width * cm] * len(s["cands"]))])]
    conv_only, shared, per = (s["chosen"][m] for m in ("conv_only", "shared", "per_group"))
    rows = [["Candidate", "Conv only: chosen m (seeds)", "Shared: chosen m (seeds)",
             "4 searched: chosen (conv, head, norm_bias, whiten_bias) (seeds)"]]
    for c in s["cands"]:
        rows.append([f"{c:g}", ", ".join(f"{m:g} x{n}" for m, n in sorted(conv_only[c].items())),
                     ", ".join(f"{m:g} x{n}" for m, n in sorted(shared[c].items())),
                     "; ".join(f"({', '.join(f'{v:g}' for v in ms)}) x{n}" for ms, n in sorted(per[c].items()))])
    story += [table(rows, [1.8 * cm, 3.6 * cm, 3.6 * cm, 8.0 * cm])]
if os.path.exists(OBSERVATIONS):
    story += [P("Observations", "h2")] + [P(line) for line in open(OBSERVATIONS).read().split("\n\n") if line.strip()]

# Experiment 2: hparam importance from scratch.
runs = imp.all_lengths()
story += [
    P("Experiment 2. Hparam importance from scratch", "h1"),
    P("Question", "h2"),
    P("How important is each of the four LRs when they are searched jointly from scratch, and does the answer "
      "depend on the main interval length?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Model / data", "baseline_bs2000 except the four trained LRs (conv, head, norm_bias, whiten_bias), which "
                         "are constant for the first L training steps from initialization"],
        ["Main interval L", "1, 2, 5, 10, 40 steps; the score is taken right after the L steps (no lookahead)"],
        ["Joint search", "From scratch: every LR starts at 1.0. Scan each LR over x10^k, k = -3..3 (others fixed), "
                         "keep the best, snap to the 0.6^k grid, then coordinate-climb on the grid (one step up / down, "
                         "move while the score improves) until no LR moves: the center"],
        ["Sweep", "From the center, each LR alone over 5 grid points on each side (others at the center). Offset +1 "
                  "= one grid step larger (x1/0.6), -1 = one step smaller (x0.6)"],
        ["Seed / score", "1 seed (0); test-set TTA accuracy (%); the test cross-entropy loss is logged too"],
    ], [3.3 * cm, 13.7 * cm]),
]
rows = [["L (steps)", "Center: conv, head, norm_bias, whiten_bias", "Center TTA", "Evaluations"]]
for length, r in runs.items():
    c = r["center"]
    rows.append([str(length), "running" if c is None else ", ".join(f"{c['values'][g]:g}" for g in imp.GROUPS),
                 "-" if c is None else f"{100 * c['tta']:.2f}", str(r["evaluations"])])
story += [P("Joint-search centers", "h2"), table(rows, [2.0 * cm, 8.0 * cm, 2.6 * cm, 2.6 * cm])]


def drops(sweep):
    center = sweep[0][1]
    side = lambda k: center - (sweep[-k][1] + sweep[k][1]) / 2  # noqa: E731
    return side(1), side(2), center - min(v[1] for v in sweep.values())


rows = [["LR"] + [f"L = {length}" for length in runs]]
for g in imp.GROUPS:
    cells = [g]
    for length, r in runs.items():
        sw = r["sweep"].get(g, {})
        cells.append(f"{drops(sw)[0]:.2f} / {drops(sw)[1]:.2f} / {drops(sw)[2]:.2f}"
                     if len(sw) == 11 else "running")
    rows.append(cells)
story += [P("Importance: TTA drop from the center (pp) at 1 step / 2 steps (mean of both sides) / worst within "
            "5 steps", "h2"), table(rows, [2.6 * cm] + [2.88 * cm] * len(runs))]

fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), layout="constrained", sharey=True)
colors_g = {"conv": "#2a78d6", "head": "#eb6834", "norm_bias": "#1baf7a", "whiten_bias": "#6250d6"}
for ax, (length, r) in zip(axes.flat, runs.items()):
    ax.set_title(f"L = {length} step{'s' if length > 1 else ''}", loc="left", fontsize=9.5)
    ax.axhline(0, color="#8c8b86", linewidth=0.8, linestyle=":")
    for g in imp.GROUPS:
        sw = r["sweep"].get(g, {})
        if 0 not in sw:
            continue
        offs = sorted(sw)
        ax.plot(offs, [sw[o][1] - sw[0][1] for o in offs], color=colors_g[g], marker="o", markersize=3.5, label=g)
    ax.set_xlabel("grid offset from the center (+ = larger LR)")
    ax.set_xticks(range(-5, 6))
for ax in axes[:, 0]:
    ax.set_ylabel("TTA minus the center's (pp)")
axes.flat[-1].axis("off")
handles = {}
for ax in axes.flat:
    for h, label in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(label, h)
axes.flat[-1].legend(handles.values(), handles.keys(), loc="center", frameon=False, fontsize=9)
imp_path = os.path.join(REPO, "plots", "importance_study.png")
fig.savefig(imp_path, dpi=140)
plt.close(fig)
imp_img = Image(imp_path)
imp_img.drawWidth, imp_img.drawHeight = 17 * cm, 17 * cm * imp_img.imageHeight / imp_img.imageWidth
story += [keep([imp_img, P("Figure 2. One-at-a-time sweeps around each L's joint-search center.",
                                   "small")])]
for length, r in runs.items():
    if len(r["sweep"]) < 4:
        continue
    rows = [["LR"] + [f"{o:+d}" for o in range(-5, 6)]]
    for g in imp.GROUPS:
        sw = r["sweep"][g]
        rows.append([f"{g} ({sw[0][0]:g})"] + [f"{sw[o][1] - sw[0][1]:+.2f}" if o in sw else "-"
                                              for o in range(-5, 6)])
    story += [keep([P(f"L = {length}: TTA minus the center's (pp) at each grid offset (center LR in "
                              f"brackets; center TTA {100 * r['center']['tta']:.2f})", "h2"),
                            table(rows, [3.4 * cm] + [1.23 * cm] * 11)])]


# Experiment 3: representation quality of schedule families.
families = fam.load("auto_gfam")
fixed = report_data.fixed_runs()


def fixed_ref(name):
    r = fixed.get((name, 8, 100, "old")) or fixed.get((name, 8, 100, "new"))
    return f"{r[1]:.2f}" if r else "running"


story += [
    P("Experiment 3. Representation quality of schedule families", "h1"),
    P("Question", "h2"),
    P("How close to the baseline can each family of conv LR schedules get, under no transform and under "
      "the ramp, reverse ramp and wave transforms, when searched by the same global search?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Model / data", "baseline_bs2000; only the conv LR schedule is searched, the other groups stay on the "
                         "baseline. The transform multiplies the conv LR the search specifies"],
        ["Conditions", "No transform; ramp x 4^(2x-1) (x0.25 -> x4); reverse ramp x 4^(1-2x) (x4 -> x0.25); wave "
                       "x 2^sin(2 pi x) (x1 -> x2 -> x0.5 -> x1); x = fraction of training"],
        ["Families", "5 constant intervals (5 values); linear decay to 0 (initial LR); power decay lr x (1-t)^p "
                     "(lr, p; p = 1 is linear); 2 knots at t = 0, 0.5; 3 knots at t = 0, 1/3, 2/3; 5 knots at "
                     "t = 0, 0.2, 0.4, 0.6, 0.8 (knot values; linear between knots, 0 at the end)"],
        ["Search", "Generalized global_neighbour (global_family): coordinate search over the family's parameters on "
                   "multiplicative grids (LR values x0.6, p x0.8), one grid step at a time, moving while the score "
                   "improves; every candidate is a full training run on seed 0 from the same initial state"],
        ["Start", "The oracle spec (baseline conv schedule divided by the transform) projected onto the family: "
                  "interval means or knot values snapped to the grid; linear / power start at 0.22 with p = 1"],
        ["Evaluation", "Start and best schedules retrained on held-out seeds 100-107 (mean test TTA; not part of "
                       "the search)"],
    ], [3.3 * cm, 13.7 * cm]),
]
ref = {"": "93.92", "_ramp": fixed_ref("transformed_a_ramp_bs2000"),
       "_rampdown": fixed_ref("transformed_a_rampdown_bs2000"), "_wave": fixed_ref("transformed_a_wave_bs2000")}
story += [P("References (held-out TTA, seeds 100-107)", "h2"),
          table([["", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)],
                 ["Baseline linear decay, no transform (the target)", *(["93.92"] * 4)],
                 ["Baseline spec kept under the transform (do-nothing)", *(ref[c] for c in fam.CONDITIONS)]],
                [6.6 * cm] + [2.6 * cm] * 4)]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = families.get(f"auto_gfam_{f}_bs2000{c}")
        if r is None or "final" not in r:
            cells.append("running")
        else:
            cells.append(f"<b>{r['final'][0]:.2f}</b> (start {r['start'][0]:.2f}; {r['evaluations']} runs, "
                         f"{r['seconds']:.0f} s)")
    rows.append(cells)
story += [P("Held-out TTA of the best schedule (start = the projected oracle before search; search cost)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4)]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = families.get(f"auto_gfam_{f}_bs2000{c}")
        cells.append("running" if r is None or "final_values" not in r else
                     ", ".join(f"{v:g}" for v in r["final_values"].values()))
    rows.append(cells)
story += [P("Best parameter values (LR values in order; power: lr, p)", "h2"), table(rows, [3.6 * cm] + [3.35 * cm] * 4)]

fam_colors = {"intervals5": "#2a78d6", "linear": "#8c8b86", "power": "#eb6834", "knots2": "#1baf7a",
              "knots3": "#6250d6", "knots5": "#e34948"}


def family_figure(results, prefix, path):
    """Applied conv LR (spec x transform) of each family's best schedule, one panel per condition."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained", sharey=True)
    steps = list(range(200))
    for ax, c in zip(axes.flat, fam.CONDITIONS):
        ax.set_title(fam.CONDITION_LABELS[c], loc="left", fontsize=9.5)
        ax.plot(steps, [0.22 * (1 - st / 200) for st in steps], color="#0b0b0b", linewidth=2.2,
                label="target: baseline applied LR")
        transform = tc.LR_TRANSFORMS.get(c.lstrip("_"))
        for f in fam.FAMILIES:
            r = results.get(f"{prefix}_{f}_bs2000{c}")
            if r is None or "final_values" not in r:
                continue
            lines = tc.family_schedule(tc.GLOBAL_FAMILIES[f], r["final_values"], 200)
            applied = [tc.schedule_value_at_step(lines, st) * (tc.lr_transform_factor(transform, st, 200)
                                                               if transform else 1.0) for st in steps]
            ax.plot(steps, applied, color=fam_colors[f], linewidth=1.3, label=fam.FAMILY_LABELS[f])
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("applied conv LR")
    handles = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


fam_path = os.path.join(REPO, "plots", "family_study.png")
family_figure(families, "auto_gfam", fam_path)
fam_img = Image(fam_path)
fam_img.drawWidth, fam_img.drawHeight = 17 * cm, 17 * cm * fam_img.imageHeight / fam_img.imageWidth
story += [keep([fam_img, P("Figure 3. Applied conv LR (spec x transform) of each family's best schedule; "
                                   "the black line is the baseline's applied LR, which a perfect search would "
                                   "reproduce.", "small")])]
fam_obs = os.path.join(REPO, "research", "family_study_observations.txt")
if os.path.exists(fam_obs):
    story += [P("Observations", "h2")] + [P(x) for x in open(fam_obs).read().split("\n\n") if x.strip()]

# Experiment 4: hparam range (Experiment 2 redone with probes to a 10% drop).
ranges = rng.all_lengths()
story += [
    P("Experiment 4. Hparam range from scratch (Experiment 2 redone)", "h1"),
    P("Question", "h2"),
    P("Experiment 2's +-5 sweeps depend on where the center landed. How far can each LR move from the center, "
      "on each side, before the score drops by 10%? A range that ends on both sides marks an LR with an optimum; "
      "a side that never drops marks an LR the score is insensitive to in that direction."),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Model / score", "As in Experiment 2: baseline_bs2000 except the four LRs, constant for the first L steps "
                          "from initialization; test TTA right after the L steps; 1 seed (0)"],
        ["Center", "Experiment 2's joint-search center for each L (re-evaluated; it reproduces Experiment 2's TTA)"],
        ["Probe", "One LR at a time (others at the center), one 0.6^k grid step at a time on each side (+1 = x1/0.6, "
                  "larger), until TTA <= 0.9 x the center's TTA (a 10% relative drop) or 20 steps. A diverged run "
                  "counts as a drop"],
    ], [3.3 * cm, 13.7 * cm]),
]


def stop_text(stop, side):
    if side not in stop:
        return "running"
    return f"{stop[side]:+d}" if stop[side] is not None else (">20" if side > 0 else "<-20")


rows = [["LR"] + [f"L = {length}" for length in ranges]]
for g in rng.GROUPS:
    cells = [g]
    for length, r in ranges.items():
        stop, probe = r["stop"][g], r["probe"][g]
        text = f"{stop_text(stop, -1)} / {stop_text(stop, 1)}"
        if -1 in stop and 1 in stop:  # LR range kept within 10%: one step inside each stop.
            lo = probe.get((stop[-1] or -21) + 1, (r["center"]["values"][g], 0))[0]
            hi = probe.get((stop[1] or 21) - 1, (r["center"]["values"][g], 0))[0]
            text += f"<br/>{lo:g} - {hi:g}"
        cells.append(text)
    rows.append(cells)
centers = ", ".join(f"L = {length}: {100 * r['center']['tta']:.2f}" for length, r in ranges.items() if r["center"])
story += [P("Stop offsets (smaller side / larger side) and the LR range within 10% of the center", "h2"),
          table(rows, [2.6 * cm] + [2.88 * cm] * len(ranges)),
          P(f"&lt;-20 / &gt;20: no 10% drop within 20 steps on that side (the range shown then ends at the last "
            f"probed point). Center TTA: {centers}.", "small")]

fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), layout="constrained", sharey=True)
floor = -40
for ax, (length, r) in zip(axes.flat, ranges.items()):
    ax.set_title(f"L = {length} step{'s' if length > 1 else ''}", loc="left", fontsize=9.5)
    ax.axhline(0, color="#8c8b86", linewidth=0.8, linestyle=":")
    ax.axhline(-10, color="#0b0b0b", linewidth=0.8, linestyle="--")
    if r["center"] is None:
        continue
    center_tta = 100 * r["center"]["tta"]
    for g in rng.GROUPS:
        probe = {**r["probe"][g], 0: (r["center"]["values"][g], center_tta)}
        offs = sorted(probe)
        ax.plot(offs, [max(floor, 100 * (probe[o][1] / center_tta - 1)) for o in offs], color=colors_g[g],
                marker="o", markersize=3, linewidth=1.3, label=g)
    ax.set_xlabel("grid offset from the center (+ = larger LR)")
    ax.set_xlim(-21, 21)
    ax.set_ylim(floor - 2, 3)
for ax in axes[:, 0]:
    ax.set_ylabel("TTA change vs center (%)")
axes.flat[-1].axis("off")
axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=9)
rng_path = os.path.join(REPO, "plots", "range_study.png")
fig.savefig(rng_path, dpi=140)
plt.close(fig)
rng_img = Image(rng_path)
rng_img.drawWidth, rng_img.drawHeight = 17 * cm, 17 * cm * rng_img.imageHeight / rng_img.imageWidth
story += [keep([rng_img, P(f"Figure 4. Relative TTA change along each probe; the dashed line is the 10% stop. "
                           f"Points below {floor}% are drawn at {floor}%.", "small")])]
rng_obs = os.path.join(REPO, "research", "range_study_observations.txt")
if os.path.exists(rng_obs):
    story += [P("Observations", "h2")] + [P(x) for x in open(rng_obs).read().split("\n\n") if x.strip()]


def observations(name):
    path = os.path.join(REPO, "research", name)
    if not os.path.exists(path):
        return []
    return [P("Observations", "h2")] + [P(x) for x in open(path).read().split("\n\n") if x.strip()]


def mean_se(values):
    m = sum(values) / len(values)
    sd = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return m, sd, sd / len(values) ** 0.5


# Experiment 5: which non-conv interval schedules help (conv fixed). Runs from 2026-10-16 23:00 UTC.
SINCE = "20261016_225000"
HELDOUT = list(range(100, 108))
noise = sr.per_seed("baseline_bs2000", SINCE)
mix = [("mix_c4_ctl_bs2000", "C4", "baseline", "baseline", "baseline"),
       ("mix_c4_head_bs2000", "C4", "searched", "baseline", "baseline"),
       ("mix_c4_norm_bias_bs2000", "C4", "baseline", "searched", "baseline"),
       ("mix_c4_whiten_bias_bs2000", "C4", "baseline", "baseline", "searched"),
       ("schedule_interval_lrs_bs2000", "C4", "searched", "searched", "searched"),
       ("mix_lin_others_bs2000", "baseline (0.22 -> 0)", "searched", "searched", "searched"),
       ("baseline_bs2000", "baseline (0.22 -> 0)", "baseline", "baseline", "baseline")]
mix_data = {name: (noise if name == "baseline_bs2000" else sr.per_seed(name, SINCE)) for name, *_ in mix}
control = mix_data["mix_c4_ctl_bs2000"]
rows = [["Conv", "head", "norm_bias", "whiten_bias", "Held-out TTA (mean +- SE)", "vs first row (paired)"]]
for name, *cells in mix:
    data = {s: v[0] for s, v in mix_data[name].items() if s in HELDOUT}
    if len(data) < len(HELDOUT):
        rows.append([*cells, f"running ({len(data)}/8)", "-"])
        continue
    m, _, se = mean_se(list(data.values()))
    diff = "-"
    if name != "mix_c4_ctl_bs2000" and all(s in control for s in HELDOUT):
        dm, _, dse = mean_se([data[s] - control[s][0] for s in HELDOUT])
        diff = f"{dm:+.2f} +- {dse:.2f}"
    rows.append([*cells, f"{m:.2f} +- {se:.2f}", diff])
story += [
    P("Experiment 5. Which non-conv interval schedules help", "h1"),
    P("Question", "h2"),
    P("The two original 5-interval searches scored 93.77 (conv only) and 93.92 (all four LRs) on held-out seeds, "
      "but their conv schedules differ in the last two intervals (0.078, 0.017 vs 0.047, 0.01). With conv fixed, do "
      "the other groups' searched intervals help, and which group?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Schedules", "C4 = conv constant 0.13, 0.13, 0.078, 0.047, 0.01 (5 x 40 steps; the four-LR interval search's "
                      "conv). Searched intervals: head 2100, 2100, 2100, 2100, 1300; norm_bias 770, 770, 770, 460, "
                      "280; whiten_bias 460, 460, 280, 280, 280. Baseline: conv 0.22 -> 0, head 1300 -> 0, norm_bias "
                      "99 -> 0 (linear over 200 steps), whiten_bias 36 -> 0 over 75 steps, then 0"],
        ["Runs", "Plain training runs (no search) on held-out seeds 100-107, current code; the paired difference "
                 "is per seed against the first row"],
    ], [3.3 * cm, 13.7 * cm]),
    P("Held-out TTA", "h2"),
    table(rows, [3.0 * cm, 1.7 * cm, 1.8 * cm, 2.2 * cm, 4.0 * cm, 3.3 * cm]),
    P("For reference, the original conv-only interval search's schedule (conv 0.13, 0.13, 0.078, 0.078, 0.017, the "
      "others on the baseline) scored 93.77 on the same seeds (older code).", "small"),
] + observations("mix_study_observations.txt")

# Experiment 6: seed noise of the baseline.
story += [
    P("Experiment 6. Seed noise of the baseline", "h1"),
    P("Question", "h2"),
    P("How much does the baseline's score vary across training seeds, and how many seeds does a comparison need?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Runs", "baseline_bs2000, plain training runs on seeds 0-127 (seed 0 is the search seed; 100-107 are the "
                    "held-out seeds); the seed sets the initialization, data order and augmentation"],
           ["Score", "Test TTA accuracy (%) and test accuracy without TTA"]], [3.3 * cm, 13.7 * cm]),
]
if len(noise) >= 2:
    tta = [v[0] for v in noise.values()]
    acc = [v[1] for v in noise.values()]
    m, sd, se = mean_se(tta)
    am, asd, ase = mean_se(acc)
    held = [noise[s][0] for s in HELDOUT if s in noise]
    rows = [["Metric", "Seeds", "Mean", "SD (one run)", "SE of the mean", "Min", "Max"],
            ["TTA", str(len(tta)), f"{m:.3f}", f"{sd:.3f}", f"{se:.3f}", f"{min(tta):.2f}", f"{max(tta):.2f}"],
            ["No TTA", str(len(acc)), f"{am:.3f}", f"{asd:.3f}", f"{ase:.3f}", f"{min(acc):.2f}", f"{max(acc):.2f}"]]
    story += [P(f"Baseline across seeds{'' if len(noise) == 128 else f' (PARTIAL: {len(noise)}/128)'}", "h2"),
              table(rows, [2.4 * cm, 1.6 * cm, 2.2 * cm, 2.6 * cm, 2.9 * cm, 2.2 * cm, 2.2 * cm])]
    order = sorted(noise)
    rows = [["Seeds per mean (n)", "SD of an n-seed mean: SD/sqrt(n)", "Measured (disjoint blocks of n)",
             "Smallest difference two n-seed means resolve (95%)"]]
    block_sd = {}
    for n in (1, 2, 4, 8, 16, 32):
        blocks = [order[i:i + n] for i in range(0, len(order) - n + 1, n)]
        means = [sum(noise[s][0] for s in b) / n for b in blocks]
        block_sd[n] = mean_se(means)[1] if len(means) > 2 else None
        rows.append([str(n), f"{sd / n ** 0.5:.3f}",
                     f"{block_sd[n]:.3f} ({len(means)} blocks)" if block_sd[n] is not None else "-",
                     f"{1.96 * sd * (2 / n) ** 0.5:.2f} pp"])
    story += [P("Averaging over seeds", "h2"), table(rows, [3.4 * cm, 4.4 * cm, 4.4 * cm, 4.8 * cm])]
    if 0 in noise and len(held) == 8:
        story += [P(f"Seed 0 (the search seed): {noise[0][0]:.2f}. Held-out seeds 100-107: mean "
                    f"{sum(held) / 8:.2f}. All seeds: {m:.2f}.", "small")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    ax = axes[0]
    ax.hist(tta, bins=16, color="#2a78d6", edgecolor="white", linewidth=1.5)
    ax.axvline(m, color="#0b0b0b", linewidth=1.5, label=f"mean {m:.2f}")
    if 0 in noise:
        ax.axvline(noise[0][0], color="#eb6834", linewidth=1.5, linestyle="--", label=f"seed 0: {noise[0][0]:.2f}")
    ax.set_xlabel("test TTA (%)")
    ax.set_ylabel("seeds")
    ax.set_title("Baseline TTA across seeds", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ns = [n for n in block_sd if block_sd[n] is not None]
    ax.plot(ns, [sd / n ** 0.5 for n in ns], color="#8c8b86", linewidth=2, label="SD / sqrt(n)")
    ax.plot(ns, [block_sd[n] for n in ns], color="#2a78d6", marker="o", markersize=5, linewidth=1.3,
            label="measured (disjoint blocks)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("seeds per mean (n)")
    ax.set_ylabel("SD of the n-seed mean (pp)")
    ax.set_title("Noise of an n-seed mean", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    noise_path = os.path.join(REPO, "plots", "noise_study.png")
    fig.savefig(noise_path, dpi=140)
    plt.close(fig)
    img = Image(noise_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P("Figure 5. Left: the baseline's TTA over seeds. Right: the spread of an n-seed mean, "
                           "predicted from the single-run SD and measured on disjoint blocks of seeds.", "small")])]
story += observations("noise_study_observations.txt")

# Experiment 7: hparam range around the baseline (full runs).
brange = dict(center=None, probe={g: {} for g in rng.GROUPS}, stop={g: {} for g in rng.GROUPS})
for path in glob.glob(os.path.join(REPO, "logs", "research_baseline_range_bs2000_*.log")):
    for line in open(path):
        if line.startswith("{"):
            r = json.loads(line)
            if r["kind"] == "center":
                brange["center"] = r
            elif r["kind"] == "range":
                brange["probe"][r["group"]][r["offset"]] = (r["value"], 100 * r["tta"])
            elif r["kind"] == "side":
                brange["stop"][r["group"]][r["direction"]] = r["stop"]
base_cfg = tc.normalize_config(tc.RUN_CONFIGS["baseline_bs2000"], key=None)
story += [
    P("Experiment 7. Hparam range around the baseline", "h1"),
    P("Question", "h2"),
    P("Starting from the baseline instead of a from-scratch center, how far can each LR's initial value move, on "
      "each side, before the full run's score drops by 10%?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Runs", "baseline_bs2000, full 200-step runs, seed 0. One LR at a time (the others on the baseline)"],
           ["Probe", "The LR's whole baseline schedule is scaled so its initial value steps along the 0.6^k grid "
                     "(+1 = x1/0.6, larger), on each side until TTA <= 0.9 x the baseline's TTA on seed 0, or 20 "
                     "steps. A diverged run counts as a drop"]], [3.3 * cm, 13.7 * cm]),
]
if brange["center"] is not None:
    rows = [["LR", "Baseline initial LR", "Stop (smaller / larger)", "Initial LR range within 10%"]]
    for g in rng.GROUPS:
        stop, probe = brange["stop"][g], brange["probe"][g]
        initial = base_cfg["param_groups"][g]["lr_scheduler"][0][1]
        text, span = f"{stop_text(stop, -1)} / {stop_text(stop, 1)}", "-"
        if -1 in stop and 1 in stop:
            lo = probe.get((stop[-1] or -21) + 1, (initial, 0))[0]
            hi = probe.get((stop[1] or 21) - 1, (initial, 0))[0]
            span = f"{lo:g} - {hi:g} (x{hi / lo:.3g})"
        rows.append([g, f"{initial:g}", text, span])
    story += [P(f"Stop offsets (baseline TTA on seed 0: {100 * brange['center']['tta']:.2f}; stop at "
                f"{100 * brange['center']['threshold']:.2f})", "h2"),
              table(rows, [2.8 * cm, 3.4 * cm, 4.2 * cm, 6.6 * cm])]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    center_tta = 100 * brange["center"]["tta"]
    ax.axhline(0, color="#8c8b86", linewidth=0.8, linestyle=":")
    ax.axhline(-10, color="#0b0b0b", linewidth=0.8, linestyle="--")
    for g in rng.GROUPS:
        probe = {**brange["probe"][g], 0: (None, center_tta)}
        offs = sorted(probe)
        ax.plot(offs, [max(floor, 100 * (probe[o][1] / center_tta - 1)) for o in offs], color=colors_g[g],
                marker="o", markersize=3.5, linewidth=1.3, label=g)
    ax.set_xlim(-21, 21)
    ax.set_ylim(floor - 2, 3)
    ax.set_xlabel("grid steps of the initial LR from the baseline (+ = larger)")
    ax.set_ylabel("TTA change vs baseline (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    brange_path = os.path.join(REPO, "plots", "baseline_range.png")
    fig.savefig(brange_path, dpi=140)
    plt.close(fig)
    img = Image(brange_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P(f"Figure 6. Relative TTA change of full runs as one LR's initial value moves from the "
                           f"baseline; dashed: the 10% stop. Points below {floor}% are drawn at {floor}%.",
                           "small")])]

    # 95% confidence set for the best probed value (Gupta's subset selection), single runs with the
    # seed SD of Experiment 6.
    sigma = mean_se([v[0] for v in noise.values()])[1]
    mc = np.random.default_rng(0)

    def gupta_q(k):
        z = mc.standard_normal((400000, k))
        return float(np.quantile((z[:, 1:] - z[:, :1]).max(axis=1), 0.95))

    conf, conf_range, rows = {}, {}, [["LR", "Probed points k", "Cutoff c (pp)", "Best observed", "95% set (initial LR)",
                       "Points in set"]]
    for g in rng.GROUPS:
        initial = base_cfg["param_groups"][g]["lr_scheduler"][0][1]
        probe = {**brange["probe"][g], 0: (initial, center_tta)}
        c = sigma * gupta_q(len(probe))
        best = max(probe, key=lambda o: probe[o][1])
        members = sorted(o for o in probe if probe[o][1] >= probe[best][1] - c)
        ends = []
        for side, edge in ((-1, members[0]), (1, members[-1])):
            unstopped = brange["stop"][g].get(side, 0) is None
            ends.append(unstopped and abs(edge) >= 19)  # Reaches the probe limit: not bounded there.
        conf[g] = (c, best, members)
        conf_range[g] = (probe[members[0]][0], probe[members[-1]][0], ends[0], ends[1])
        low = f"(open) {probe[members[0]][0]:g}" if ends[0] else f"{probe[members[0]][0]:g}"
        high = f"{probe[members[-1]][0]:g} (open)" if ends[1] else f"{probe[members[-1]][0]:g}"
        gaps = "" if members == list(range(members[0], members[-1] + 1)) else ", with gaps"
        rows.append([g, str(len(probe)), f"{c:.2f}", f"{probe[best][0]:g} ({probe[best][1]:.2f})",
                     f"{low} - {high}" if members[0] != members[-1] else f"{probe[best][0]:g}",
                     f"{len(members)} of {len(probe)}{gaps}"])
    story += [P("95% confidence set for the best initial LR", "h2"),
              P(f"Each probed value is one run on seed 0, so its score carries the single-run noise of Experiment 6 "
                f"(SD {sigma:.3f} pp, assumed the same at every value and independent between values). A value is "
                f"kept when its score is within c of the best observed score, with c = q x SD and q the 95% "
                f"quantile of max_j (Z_j - Z_1) over the k probed values (Z independent standard normals; Monte "
                f"Carlo). This is Gupta's subset selection: the set contains the best probed value with probability "
                f"at least 95%, even in the hardest case where all values are equally good. (open) marks an end that "
                f"reaches the probe limit on a side without a 10% drop, so the set may continue beyond it."),
              table(rows, [2.4 * cm, 2.2 * cm, 2.1 * cm, 3.0 * cm, 4.6 * cm, 2.7 * cm])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), layout="constrained", sharey=True)
    low_y = 92.4
    for ax, g in zip(axes.flat, rng.GROUPS):
        initial = base_cfg["param_groups"][g]["lr_scheduler"][0][1]
        probe = {**brange["probe"][g], 0: (initial, center_tta)}
        c, best, members = conf[g]
        offs = sorted(probe)
        ys = [max(low_y, probe[o][1]) for o in offs]
        ax.plot(offs, ys, color=colors_g[g], linewidth=1.0, alpha=0.6)
        inside = [o for o in offs if o in members]
        outside = [o for o in offs if o not in members]
        ax.scatter(inside, [max(low_y, probe[o][1]) for o in inside], color=colors_g[g], s=22, zorder=3)
        ax.scatter(outside, [max(low_y, probe[o][1]) for o in outside], facecolor="white", edgecolor=colors_g[g],
                   s=22, zorder=3)
        ax.axhline(probe[best][1] - c, color="#0b0b0b", linewidth=0.8, linestyle="--")
        ax.set_title(f"{g} (baseline {initial:g})", loc="left", fontsize=9.5)
        ax.set_xlim(-21, 21)
        ax.set_ylim(low_y - 0.1, 94.45)
        ax.set_xlabel("grid steps of the initial LR from the baseline")
    for ax in axes[:, 0]:
        ax.set_ylabel("test TTA (%)")
    legend_items = [plt.Line2D([], [], marker="o", linestyle="", color="#52514e", label="in the 95% set"),
                    plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color="#52514e",
                               label="excluded"),
                    plt.Line2D([], [], color="#0b0b0b", linestyle="--", linewidth=0.8, label="best - c")]
    fig.legend(handles=legend_items, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    conf_path = os.path.join(REPO, "plots", "baseline_range_confidence.png")
    fig.savefig(conf_path, dpi=140)
    plt.close(fig)
    img = Image(conf_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P(f"Figure 7. The probes near the top: filled points are in the 95% set (score at least "
                           f"best - c, dashed); points below {low_y} are drawn at {low_y}.", "small")])]
story += observations("baseline_range_observations.txt")

# Experiment 8: the interval search's first-interval rule (largest LR within a margin of the best),
# applied to Experiment 4's probes.
MARGINS = (2.0, 1.0, 0.5)
story += [
    P("Experiment 8. Largest LR within the margin, per L (from Experiment 4)", "h1"),
    P("Question", "h2"),
    P("The interval search picks each LR in its first interval as the largest LR scoring within a margin of the "
      "best (directional_search with initial_lr_search, initial_dropoff_margin = 0.02). Applied at main interval "
      "lengths L = 1, 2, 5, 10, 40 from scratch, does the pick at L = 1 predict the pick at L = 40?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Data", "Experiment 4's probes (no new runs): for each L, one LR at a time on the 0.6^k grid around "
                    "Experiment 2's center, the other three LRs at the center, TTA right after L steps, seed 0. "
                    "The probes extend past every point where the rule stops"],
           ["Rule", "As in train_cifar.py: from the center, step up while the score stays within the margin of the "
                    "best so far, then down the same way (at most 20 steps per side); pick the largest LR within "
                    "the margin of the best. Margin 0.02 TTA = 2 pp (the code's value); 1 and 0.5 pp shown too"]],
          [3.3 * cm, 13.7 * cm]),
]
rule_rows = [["L (steps)", "Center TTA", *rng.GROUPS]]
picks = {}


def versus_reference(g, value):
    """Where a pick lies relative to Experiment 7's full-run 95% range for the best initial LR."""
    lo, hi, open_lo, open_hi = conf_range[g]
    if value > hi * 1.001 and not open_hi:
        return "above"
    if value < lo * 0.999 and not open_lo:
        return "below"
    return "in"


def range_text(g):
    lo, hi, open_lo, open_hi = conf_range[g]
    return f"{'(open) ' if open_lo else ''}{lo:g} - {hi:g}{' (open)' if open_hi else ''}"


for length, r in ranges.items():
    center_tta = 100 * r["center"]["tta"]
    cells = [str(length), f"{center_tta:.2f}"]
    for g in rng.GROUPS:
        for margin in MARGINS:
            out = rng.largest_within(r["probe"][g], center_tta, margin, 20)
            offset, limit = out
            value = r["center"]["values"][g] if offset == 0 else r["probe"][g][offset][0]
            picks[(length, g, margin)] = (value, offset, limit)
        value, offset, limit = picks[(length, g, 2.0)]
        cells.append(f"{value:g} ({offset:+d}){'*' if limit else ''}, {versus_reference(g, value)}")
    rule_rows.append(cells)
rule_rows.append(["Exp. 7", "full run", *(range_text(g) for g in rng.GROUPS)])
story += [P("Picks with the code's margin (2 pp): LR (grid steps from the center), and where it lies relative to "
            "Experiment 7's full-run 95% range for the best initial LR (last row)", "h2"),
          table(rule_rows, [1.8 * cm, 1.8 * cm, 3.1 * cm, 3.3 * cm, 3.5 * cm, 3.5 * cm]),
          P("* the upward probe reached the 20-step limit without falling out of the margin, so the pick is the "
            "probe limit. in / above / below: inside, above or below the Experiment 7 range ((open): that end of "
            "the range is not bounded by the data). Experiment 7 is a different context (full 200-step runs, the "
            "other LRs on the baseline schedules), the target a short-horizon pick should predict.", "small")]
rows = [["L (steps)", *(f"conv, margin {m:g} pp" for m in MARGINS)]]
for length in ranges:
    rows.append([str(length), *(f"{picks[(length, 'conv', m)][0]:g}, {versus_reference('conv', picks[(length, 'conv', m)][0])}"
                                for m in MARGINS)])
rows.append(["Exp. 7", *([range_text("conv")] * len(MARGINS))])
story += [P("Conv pick by margin", "h2"), table(rows, [2.6 * cm] + [4.0 * cm] * len(MARGINS))]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
lengths = list(ranges)
for ax, g in zip(axes.flat, rng.GROUPS):
    lo, hi, open_lo, open_hi = conf_range[g]
    values = ([picks[(n, g, m)][0] for n in lengths for m in MARGINS] + [lo, hi]
              + [ranges[n]["center"]["values"][g] for n in lengths])
    ymin, ymax = min(values) / 3, max(values) * 3
    ax.axhspan(ymin if open_lo else lo, ymax if open_hi else hi, color=colors_g[g], alpha=0.12, linewidth=0,
               label="Exp. 7 full-run 95% range")
    if g == "conv":
        for margin, style in zip(MARGINS, ("-", "--", ":")):
            ax.plot(lengths, [picks[(n, g, margin)][0] for n in lengths], color=colors_g[g], linestyle=style,
                    marker="o", markersize=4, linewidth=1.3, label=f"pick, margin {margin:g} pp")
    else:
        ax.plot(lengths, [picks[(n, g, 2.0)][0] for n in lengths], color=colors_g[g], marker="o", markersize=4,
                linewidth=1.3, label="pick, margin 2 pp")
    ax.plot(lengths, [ranges[n]["center"]["values"][g] for n in lengths], color="#8c8b86", marker="s",
            markersize=3.5, linewidth=1.0, label="Exp. 2 center (best score)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(lengths, [str(n) for n in lengths])
    ax.set_xlabel("main interval length L (steps)")
    ax.set_ylabel(f"{g} LR")
    ax.set_title(g + (" (range open below)" if open_lo else ""), loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
rule_path = os.path.join(REPO, "plots", "largest_within.png")
fig.savefig(rule_path, dpi=140)
plt.close(fig)
img = Image(rule_path)
img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
story += [keep([img, P("Figure 8. The rule's pick per L (conv at three margins, the others at the code's 2 pp), "
                       "Experiment 2's best-score center, and the shaded reference: Experiment 7's full-run 95% "
                       "range for the best initial LR.", "small")])]
story += observations("largest_within_observations.txt")


# Experiment 9: Experiment 3 redone with the noise-aware global_neighbour search (noisy_family).
nfam = fam.load("auto_nfam")
story += [
    P("Experiment 9. Representation quality of schedule families, noise-aware search", "h1"),
    P("Question", "h2"),
    P("Experiment 3's search scored every candidate on one run (seed 0), so it climbed on seed noise and stopped "
      "at schedules that were lucky on seed 0 (e.g. 5 constant intervals: 93.99 on seed 0, 93.62 held-out). With a "
      "search that only moves to a neighbour that is statistically better given the seed noise, and adds seeds "
      "until it can tell, how close does each family get?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["As Experiment 3", "baseline_bs2000, conv only (the others on the baseline); the same families, conditions "
                            "(no transform, ramp, reverse ramp, wave), grids (LR x0.6, p x0.8) and starts (the "
                            "oracle projected onto the family); start and final retrained on held-out seeds 100-107"],
        ["Search", "Global neighbour (noisy_family): each round compares the current schedule with all its "
                   "neighbours (each parameter one grid step up or down; k = 2 for linear, 4 for power and 2 knots, "
                   "6 for 3 knots, 10 for 5 intervals and 5 knots). Every point's score is its mean TTA over full "
                   "runs on search seeds 0, 1, 2, ..."],
        ["Decision", "d = neighbour mean - current mean, SE = 0.12 pp x sqrt(1/n_neighbour + 1/n_current) (the "
                     "single-run SD of Experiment 6, runs treated as independent). One-sided 80% bounds, "
                     "Bonferroni over the k neighbours: z = 1.28 / 1.64 / 1.83 / 2.05 for k = 2 / 4 / 6 / 10. "
                     "Better: d - z SE &gt; 0. Ruled out: d + z SE &lt; 0. Otherwise undecided"],
        ["Moves and seeds", "If any neighbour is better, move to the better one with the highest mean. Else run "
                            "one more seed on the current schedule and on every undecided neighbour, and repeat. "
                            "Stop when every neighbour is ruled out, or at 8 seeds per point (undecided neighbours "
                            "then count as not better)"],
    ], [3.3 * cm, 13.7 * cm]),
]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = nfam.get(f"auto_nfam_{f}_bs2000{c}")
        old = families.get(f"auto_gfam_{f}_bs2000{c}")
        if r is None or "final" not in r:
            cells.append("running")
        else:
            cells.append(f"<b>{r['final'][0]:.2f}</b> (Exp. 3: {old['final'][0]:.2f}; start {r['start'][0]:.2f}; "
                         f"{r.get('moves', '?')} moves, {r['evaluations']} runs)")
    rows.append(cells)
story += [P("Held-out TTA of the final schedule (Experiment 3's for comparison; the start; search cost)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4),
          P("Target: the baseline's 93.92 on the held-out seeds; do-nothing references as in Experiment 3 "
            "(ramp 93.42, reverse ramp 93.57, wave 93.72).", "small")]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = nfam.get(f"auto_nfam_{f}_bs2000{c}")
        cells.append("running" if r is None or "final_values" not in r else
                     ", ".join(f"{v:g}" for v in r["final_values"].values())
                     + f" ({r.get('search_seeds', '?')} seeds; {r.get('stop', '?').replace('_', ' ')})")
    rows.append(cells)
story += [P("Final parameter values (seeds behind the final mean; why the search stopped)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4)]
if any("final_values" in r for r in nfam.values()):
    nfam_path = os.path.join(REPO, "plots", "family_study_noisy.png")
    family_figure(nfam, "auto_nfam", nfam_path)
    img = Image(nfam_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P("Figure 9. Applied conv LR of each family's final schedule (noise-aware search); the black "
                           "line is the baseline's applied LR.", "small")])]
story += observations("noisy_family_observations.txt")


# Experiment 10: Experiment 3 redone with the guided search (guided_family).
xfam = fam.load("auto_xfam")
story += [
    P("Experiment 10. Representation quality of schedule families, guided search", "h1"),
    P("Question", "h2"),
    P("Experiment 3's single-seed search walked far but climbed on seed noise; Experiment 9's significance test "
      "removed the noise but rarely moved more than one grid step. Can a search that explores deeper on cheap "
      "single runs, pools them with a smooth fit, jumps several steps (and several parameters) at once, and "
      "accepts a move only on fresh seeds, get both?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["As Experiment 3", "baseline_bs2000, conv only (the others on the baseline); the same families, conditions, "
                            "grids (LR x0.6, p x0.8) and starts; start and final retrained on held-out seeds 100-107"],
        ["1. Explore", "From the current schedule, along each parameter and direction, one grid step at a time, one "
                       "run each on seed 0, while the run is not clearly worse than the best run on that line "
                       "(score &gt;= best - 0.84 x 0.12 pp x sqrt(2), one-sided 80%), up to 3 steps"],
        ["2. Fit", "Per parameter, a least-squares quadratic in the grid offset through that line's seed-0 runs "
                   "(the current schedule's included): the smoothness prior pools 4-7 runs. Predicted gain of an "
                   "offset = fit there - fit at the current schedule"],
        ["3. Propose", "Each parameter moved alone to its best fitted offset, and the joint move of every "
                       "parameter with a positive predicted gain (gains summed, a separability prior); the 3 "
                       "largest predicted gains"],
        ["4. Verify", "On fresh seeds 1-8 (never seed 0, which chose the candidates): better if the lower one-sided "
                      "80% bound of (candidate - current) &gt; 0, ruled out if the upper bound &lt; 0, Bonferroni "
                      "over the candidates (z = 1.50 for 3), SE = 0.12 pp x sqrt(1/n_c + 1/n_current); otherwise one "
                      "more seed for the current schedule and the undecided candidates, up to 8"],
        ["Moves", "Move to the better candidate with the highest mean and repeat; stop when no parameter is "
                  "predicted to gain or no candidate proves better"],
    ], [3.3 * cm, 13.7 * cm]),
]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = xfam.get(f"auto_xfam_{f}_bs2000{c}")
        e3 = families.get(f"auto_gfam_{f}_bs2000{c}")
        e9 = nfam.get(f"auto_nfam_{f}_bs2000{c}")
        if r is None or "final" not in r:
            cells.append("running")
        else:
            cells.append(f"<b>{r['final'][0]:.2f}</b> (Exp. 3: {e3['final'][0]:.2f}; Exp. 9: {e9['final'][0]:.2f}; "
                         f"start {r['start'][0]:.2f}; {r.get('moves', '?')} moves, {r['evaluations']} runs)")
    rows.append(cells)
story += [P("Held-out TTA of the final schedule (Experiments 3 and 9 for comparison; the start; search cost)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4),
          P("Target: the baseline's 93.92 on the held-out seeds; do-nothing references as in Experiment 3 "
            "(ramp 93.42, reverse ramp 93.57, wave 93.72).", "small")]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = xfam.get(f"auto_xfam_{f}_bs2000{c}")
        seeds = "not verified" if r and r.get("search_seeds") == 0 else f"{r.get('search_seeds', '?') if r else '?'} seeds"
        cells.append("running" if r is None or "final_values" not in r else
                     ", ".join(f"{v:g}" for v in r["final_values"].values())
                     + f" ({seeds}; {r.get('stop', '?').replace('_', ' ')})")
    rows.append(cells)
story += [P("Final parameter values (fresh seeds behind the final mean, 'not verified' when the search stopped "
            "before testing any candidate; why it stopped)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4)]
if any("final_values" in r for r in xfam.values()):
    xfam_path = os.path.join(REPO, "plots", "family_study_guided.png")
    family_figure(xfam, "auto_xfam", xfam_path)
    img = Image(xfam_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P("Figure 10. Applied conv LR of each family's final schedule (guided search); the black "
                           "line is the baseline's applied LR.", "small")])]
story += observations("guided_family_observations.txt")


def attach_headings(flowables):
    """Keep each subheading on the same page as the table / figure that follows it."""
    out = []
    for item in flowables:
        if (isinstance(item, KeepTogether) and out and isinstance(out[-1], Paragraph)
                and out[-1].style.name == "h2"):
            out[-1] = keep([out[-1], item])
        else:
            out.append(item)
    return out


story = attach_headings(story)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Interval search experiments (batch size 2000)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Cooldown study").build(story, onFirstPage=footer,
                                                                       onLaterPages=footer)
print(OUT)


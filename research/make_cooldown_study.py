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
import lookahead_study as las  # noqa: E402
import ics_study as ics  # noqa: E402
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


# Experiment 11: the guided search with uncertainty-aware candidate selection (selection = "lcb").
yfam = fam.load("auto_yfam")
story += [
    P("Experiment 11. Representation quality of schedule families, guided search with uncertainty-aware "
      "selection", "h1"),
    P("Question", "h2"),
    P("Experiment 10's weak point was candidate selection: quadratic fits through noisy single runs followed "
      "noise to the end of the explored range, and a joint move summing those noisy gains crowded out the real "
      "improvements. Does ranking by the fit's lower confidence bound, building the joint move only from clearly "
      "positive parameters, and testing single moves first fix it?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["As Experiment 10", "Same families, conditions, starts, exploration (one seed-0 run per point, up to 3 steps "
                             "per direction), quadratic fit per parameter, fresh-seed verification (seeds 1-8, "
                             "one-sided 80%, Bonferroni over the candidates raced together) and held-out evaluation"],
        ["Fit uncertainty", "The SE of each fitted gain (fit at the offset - fit at the current schedule) from the "
                            "regression with 0.12 pp noise per run: large where the fit extrapolates or rests on "
                            "few points"],
        ["Offset choice", "Per parameter, the offset (other than the current one) with the highest lower bound, "
                          "gain - 0.84 SE"],
        ["Candidates", "Single-parameter moves with a positive fitted gain, ranked by that lower bound; the best 3 "
                       "are raced first. Only if none proves better: the joint move of the parameters whose lower "
                       "bound is above 0 (at least 2), raced alone"],
    ], [3.3 * cm, 13.7 * cm]),
]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = yfam.get(f"auto_yfam_{f}_bs2000{c}")
        e9 = nfam.get(f"auto_nfam_{f}_bs2000{c}")
        e10 = xfam.get(f"auto_xfam_{f}_bs2000{c}")
        if r is None or "final" not in r:
            cells.append("running")
        else:
            cells.append(f"<b>{r['final'][0]:.2f}</b> (Exp. 9: {e9['final'][0]:.2f}; Exp. 10: {e10['final'][0]:.2f}; "
                         f"start {r['start'][0]:.2f}; {r.get('moves', '?')} moves, {r['evaluations']} runs)")
    rows.append(cells)
story += [P("Held-out TTA of the final schedule (Experiments 9 and 10 for comparison; the start; search cost)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4),
          P("Target: the baseline's 93.92 on the held-out seeds; do-nothing references as in Experiment 3 "
            "(ramp 93.42, reverse ramp 93.57, wave 93.72).", "small")]
rows = [["Family", *(fam.CONDITION_LABELS[c] for c in fam.CONDITIONS)]]
for f in fam.FAMILIES:
    cells = [fam.FAMILY_LABELS[f]]
    for c in fam.CONDITIONS:
        r = yfam.get(f"auto_yfam_{f}_bs2000{c}")
        seeds = "not verified" if r and r.get("search_seeds") == 0 else f"{r.get('search_seeds', '?') if r else '?'} seeds"
        cells.append("running" if r is None or "final_values" not in r else
                     ", ".join(f"{v:g}" for v in r["final_values"].values())
                     + f" ({seeds}; {r.get('stop', '?').replace('_', ' ')})")
    rows.append(cells)
story += [P("Final parameter values (fresh seeds behind the final mean, 'not verified' when the search stopped "
            "before testing any candidate; why it stopped)", "h2"),
          table(rows, [3.6 * cm] + [3.35 * cm] * 4)]
if any("final_values" in r for r in yfam.values()):
    yfam_path = os.path.join(REPO, "plots", "family_study_guided_lcb.png")
    family_figure(yfam, "auto_yfam", yfam_path)
    img = Image(yfam_path)
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
    story += [keep([img, P("Figure 11. Applied conv LR of each family's final schedule (guided search, "
                           "uncertainty-aware selection); the black line is the baseline's applied LR.", "small")])]
story += observations("guided_lcb_observations.txt")


# Introduction to the 2-day campaign (Experiments 12 on).
story += [
    P("The 2-day campaign: constant intervals with a cooldown search (Experiments 12 on)", "h1"),
    P("Goal (set by the user on 2026-10-17): a search over constant-interval LR schedules with a cooldown "
      "lookahead that matches the baseline's held-out TTA at batch sizes 2000, 500 and 125, under the transforms "
      "(ramp, reverse ramp, wave) and with either conv's LR (other groups on the baseline) or all four LRs (from "
      "1.0) searched, using at most 20 full runs' worth of training steps (every trained step counts; held-out "
      "evaluation does not). Targets on held-out seeds 100-107: 93.92 (BS 2000), 94.00 (BS 500), 93.86 (BS 125); "
      "one held-out mean has a standard error of about 0.045 pp, so 'matching' here means within about 0.1 pp."),
    P("The searches are run_ics_search in train_cifar.py (configs ics_v&lt;N&gt;_&lt;a|b&gt;_bs&lt;BS&gt;[_transform], "
      "a = conv only, b = 4 LRs); each version is run with 2 search seeds per cell. Experiments 12, 16, 17, 18 and 20 "
      "are probes that shaped the versions; 13-15, 19, 21 and 22 compare the versions."),
]
FINAL_VERSION = "v29"
FINAL_SOURCES = {"a": "v29", "b": "v29"}
final_by_scenario = {sc: ics.load(v) for sc, v in FINAL_SOURCES.items()}
ics_refs = ics.references()
story += [
    P(f"Campaign summary: the current algorithm (ICS {FINAL_VERSION})", "h2"),
    table([
        ["Step", "What it does (budget in full runs; it scales with the run's length)"],
        ["Representation", "During the search each searched LR is 5 constant intervals (spec values; a transform "
                           "applies on top), on the x0.6 grid. The committed schedule joins the 5 values piecewise "
                           "linearly through the interval midpoints, the end half-intervals continuing the slope "
                           "(Experiment 24)"],
        ["Two searches", "The search below runs twice, each from initialization with its own data order and half of "
                         "the budget; the two 5-interval schedules are averaged (geometric mean). Steps are kept back "
                         "so each search always finishes within its share"],
        ["Levels (4 LRs only)", "First search, interval 1, from 1.0: conv on a x7.7 grid (best score), then head and "
                                "both biases together on a x7.7 grid, keeping the largest level within 1 pp of the "
                                "best, each scored after the interval plus one interval at 0.22 x the value (~4-5 "
                                "runs); the second search reuses these levels. Later, head and the biases follow "
                                "conv's interval-to-interval profile"],
        ["Pass 1", "Each interval in order: climb conv from the previous interval's value (smaller values first, "
                   "moving while the score improves); a candidate's score is the interval plus one interval at 0.22 "
                   "x the candidate. This finds the schedule's shape cheaply, transform included, but is biased "
                   "toward small LRs (Experiment 12)"],
        ["Shift", "Pass 1's conv schedule, except its last interval, is moved one grid step up (the bias correction; "
                  "with 4 LRs the other groups keep their scanned levels)"],
        ["Pass 2", "From the initial state again: each interval, the first included, climbs at most one grid step "
                   "from the shifted plan, a candidate scored by the interval plus the rest of the run on the plan "
                   "(the plan's tail unscaled); the last interval averages 2 continuations with fresh data order when "
                   "the budget allows"],
        ["Budget", "Every trained step counts; at most 20 full runs (the searches use 20)"],
    ], [3.3 * cm, 13.7 * cm]),
]
rows = [["Batch size", *(f"{ics.TRANSFORM_LABELS[tr]}" for tr in ics.TRANSFORMS)]]
for bs in ics.BATCH_SIZES:
    base = ics.target(ics_refs, bs)
    for sc in ics.SCENARIOS:
        cells = [f"BS {bs}, {ics.SCENARIO_LABELS[sc]} ({'-' if base is None else f'{base:.2f}'})"]
        for tr in ics.TRANSFORMS:
            held = [r["heldout"] for r in final_by_scenario[sc].get((sc, bs, tr), {}).values() if "heldout" in r]
            cells.append("-" if not held else f"<b>{sum(held) / len(held):.2f}</b> ({sum(held) / len(held) - base:+.2f}; "
                         f"{len(held)} seeds)")
        rows.append(cells)
story += [P(f"Held-out TTA of ICS {FINAL_VERSION} (mean over search seeds; vs the baseline in brackets)", "h2"),
          table(rows, [4.2 * cm] + [3.2 * cm] * 4),
          P("Previous summaries: v24 (4 LRs) and v19 (conv only), 10 constant intervals, one search; see Experiments "
            "22-25 for the steps between.", "small")]

# Experiment 12: lookahead fidelity and noise for constant-interval decisions (2-day campaign).
look_by_bs = {bs: las.summary(bs, "") for bs in (2000, 500, 125)}
look_by_bs = {bs: v for bs, v in look_by_bs.items() if v[1]}
story += [
    P("Experiment 12. Which lookahead decides constant intervals like the final result, and how noisy is it?",
      "h1"),
    P("Question", "h2"),
    P("An interval search commits each constant interval by scoring candidates with a lookahead from the same "
      "trained state. Which lookahead ranks the candidates like their true effect on the final score, and how "
      "noisy is one seed's comparison of two candidates scored from the same state?"),
    P("Setup", "h2"),
    table([
        ["Item", "Value"],
        ["Plans", "conv as 5 constant intervals, the other groups on the baseline. BS 2000: C4 (0.13, 0.13, 0.078, "
                  "0.047, 0.01; matches the baseline, Experiment 5); BS 500 and 125: the q = 0.75 staircases of "
                  "Experiment 16 (0.11, 0.085, 0.058, 0.033, 0.0065 and 0.04, 0.031, 0.021, 0.012, 0.0024)"],
        ["Candidates", "At interval k, from the plan's trained state: the plan's value moved -2..+2 grid steps (x0.6), "
                       "trained for the interval"],
        ["Scores", "truth: the rest of the run on the plan (what the decision affects); rest_lin: the rest of the "
                   "run with conv decaying linearly from the candidate to 0; c40_m: one interval (a fifth of the "
                   "run) at m x candidate (m = 0.6, 0.36, 0.22); l40: one interval decaying linearly to 0"],
        ["Seeds", "BS 2000 and 500: 8 (0-7), BS 125: 4; each seed is its own run from initialization"],
        ["Decision quality", "1-seed hit: how often one seed's best candidate by the score is the candidate with "
                             "the best mean truth; regret: the mean, over seeds, of the mean truth lost by that pick "
                             "(pp)"],
    ], [3.3 * cm, 13.7 * cm]),
]
variants12 = ["truth", "rest_lin", "c40_0.22", "c40_0.36", "c40_0.6", "l40"]
for bs, (plan12, look) in look_by_bs.items():
    rows = [["Interval (best by truth)", *variants12]]
    for k, r in look.items():
        cells = [f"{k + 1} ({r['c'][r['best']]:g})"]
        for v in variants12:
            row = r["rows"].get(v)
            cells.append("-" if row is None else f"{r['c'][row['mean_pick']]:g}; {row['hit']:.2f}; {row['regret']:.3f}")
        rows.append(cells)
    seeds12 = len(next(iter(look.values()))["seeds"])
    story += [P(f"BS {bs} ({seeds12} seeds): mean pick; 1-seed hit rate; 1-seed regret (pp of final TTA)", "h2"),
              table(rows, [3.4 * cm] + [2.27 * cm] * len(variants12))]
plan12, look = look_by_bs[2000]
rows = [["Interval", "Noise of a paired comparison at BS 2000 (SD over seeds of candidate - plan value, pp): truth / rest_lin"]]
for k, r in look.items():
    tr = r["rows"]["truth"]["paired_sd"]
    rl = r["rows"].get("rest_lin", {"paired_sd": {}})["paired_sd"]
    rows.append([str(k + 1), "; ".join(f"{r['c'][o]:g}: {tr[o]:.2f} / " + (f"{rl[o]:.2f}" if o in rl else "-")
                                       for o in r["offsets"] if o != 0)])
story += [P("Noise of one seed's comparison of two candidates", "h2"), table(rows, [2.0 * cm, 15.0 * cm])]
abs_by_bs = {bs: las.summary(bs, "_abs") for bs in (2000, 500)}
abs_by_bs = {bs: v for bs, v in abs_by_bs.items() if v[1]}
if abs_by_bs:
    abs_variants = ["c40_0.22", "a40_0.22", "a40_0.6", "a40_1", "rest_lin"]
    rows = [["Batch size", *abs_variants]]
    for bs, (_, look_abs) in abs_by_bs.items():
        cells = [f"BS {bs}"]
        for v in abs_variants:
            regs = [r["rows"][v]["regret"] for k, r in look_abs.items() if v in r["rows"]]
            cells.append(" / ".join(f"{x:.3f}" for x in regs) + (f" (mean {sum(regs) / len(regs):.3f})" if regs else ""))
        rows.append(cells)
    story += [P("A common cooldown level (second probe series, same plans and seeds): 1-seed regret per interval "
                "(intervals 1-4) and mean, pp", "h2"),
              table(rows, [2.0 * cm] + [3.0 * cm] * len(abs_variants)),
              P("a40_m: one interval at m x the plan's value for the interval, the same for every candidate (a "
                "candidate-relative cooldown, c40_m, also scales the cooldown with the candidate).", "small")]
fig, axes = plt.subplots(len(look_by_bs), 5, figsize=(12, 2.6 * len(look_by_bs)), layout="constrained", sharey=True,
                         squeeze=False)
styles12 = {"truth": ("#0b0b0b", "-"), "rest_lin": ("#2a78d6", "-"), "c40_0.22": ("#1baf7a", "--"),
            "c40_0.36": ("#eb6834", "--"), "l40": ("#6250d6", ":")}
for row_axes, (bs, (_, look_bs)) in zip(axes, look_by_bs.items()):
    for ax, (k, r) in zip(row_axes, look_bs.items()):
        xs = [r["c"][o] for o in r["offsets"]]
        for v, (color, ls) in styles12.items():
            row = r["rows"].get(v)
            if row is None:
                continue
            top = max(row["means"].values())
            ax.plot(xs, [max(-1.5, row["means"][o] - top) for o in r["offsets"]], color=color, linestyle=ls,
                    marker="o", markersize=3, linewidth=1.3, label=v)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(xs, [f"{x:g}" for x in xs], rotation=45, fontsize=7)
        ax.set_title(f"BS {bs}, interval {k + 1}", loc="left", fontsize=8.5)
    row_axes[0].set_ylabel("mean minus best (pp)")
axes[0][-1].legend(frameon=False, fontsize=7, loc="lower left")
la_path = os.path.join(REPO, "plots", "lookahead_fidelity.png")
fig.savefig(la_path, dpi=140)
plt.close(fig)
img = Image(la_path)
img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * img.imageHeight / img.imageWidth
story += [keep([img, P("Figure 12. Each score's mean per candidate over seeds, relative to its own best (points below -1.5 "
                       "drawn at -1.5). A lookahead decides like the truth when its curve peaks where the truth's does.",
                       "small")])]
story += observations("lookahead_fidelity_observations.txt")


# The 2-day campaign's ICS experiments (run_ics_search): one section per version.
ics_refs = ics.references()


def ics_section(version, number, title, question, setup_rows, obs_file):
    data = ics.load(version)
    story.extend([P(f"Experiment {number}. {title}", "h1"), P("Question", "h2"), P(question), P("Setup", "h2"),
                  table([["Item", "Value"], *setup_rows], [3.3 * cm, 13.7 * cm])])
    rows = [["Batch size, scenario (baseline)", *(ics.TRANSFORM_LABELS[t] for t in ics.TRANSFORMS)]]
    matched = total_cells = 0
    for bs in ics.BATCH_SIZES:
        base = ics.target(ics_refs, bs)
        for sc in ics.SCENARIOS:
            cells = [f"BS {bs}, {ics.SCENARIO_LABELS[sc]} ({base:.2f})" if base else f"BS {bs}, {ics.SCENARIO_LABELS[sc]}"]
            for tr in ics.TRANSFORMS:
                seeds = data.get((sc, bs, tr), {})
                held = [r["heldout"] for r in seeds.values() if "heldout" in r]
                if not held:
                    cells.append("running")
                    continue
                mean = sum(held) / len(held)
                cost = max(r["cost"] for r in seeds.values())
                dn = ics.do_nothing(ics_refs, sc, bs, tr)
                delta = f"{mean - base:+.2f}" if base else "?"
                if base:
                    total_cells += 1
                    matched += mean >= base - 0.1
                cells.append(f"<b>{mean:.2f}</b> ({delta}; {'/'.join(f'{h:.2f}' for h in held)}); "
                             f"{cost:.1f}x{'*' if any(r['budget_hit'] for r in seeds.values()) else ''}"
                             + (f"; do-nothing {dn:.2f}" if dn and tr else ""))
            rows.append(cells)
    story.extend([P("Held-out TTA (mean over search seeds; vs the baseline; per search seed); cost in full runs", "h2"),
                  table(rows, [3.4 * cm] + [3.4 * cm] * 4),
                  P(f"Cells within 0.1 pp of the baseline (about 2 SE of an 8-seed mean) or above: {matched} of "
                    f"{total_cells} finished. * = the search hit its step budget. Do-nothing: the baseline schedule "
                    "kept under the transform.", "small")])
    rows = [["Cell", "Committed schedules (search seed 0)"]]
    for bs in ics.BATCH_SIZES:
        for sc in ics.SCENARIOS:
            for tr in ics.TRANSFORMS:
                r = data.get((sc, bs, tr), {}).get(0)
                if r and r["schedules"]:
                    rows.append([f"BS {bs}, {ics.SCENARIO_LABELS[sc]}, {ics.TRANSFORM_LABELS[tr]}",
                                 "; ".join(f"{g}: {', '.join(f'{v:g}' for v in vals)}"
                                           for g, vals in r["schedules"].items())])
    if len(rows) > 1:
        story.extend([P("Committed schedules (search seed 0)", "h2"), table(rows, [4.6 * cm, 12.4 * cm])])
    story.extend(observations(obs_file))


ics_section(
    "v1", 13, "ICS v1: constant intervals with a rest-of-run lookahead, all batch sizes and transforms",
    "Does the interval-cooldown search, with the lookahead Experiment 12 found faithful, match the baseline at batch "
    "sizes 2000, 500 and 125, with and without transforms, searching conv only or all four LRs, within 20 full runs?",
    [["Representation", "Each searched LR: 5 constant intervals (spec values; the transform applies on top), on the "
                        "x0.6 grid"],
     ["Per interval", "Climb the lead LR (conv) from the previous interval's value, smaller values first, moving "
                      "while the score improves. A candidate's score: train the interval from the committed state, "
                      "then the rest of the run with every searched LR decaying linearly from its candidate value to "
                      "0 (rest_lin); the last interval is scored by the final TTA. Commit the best and continue from "
                      "its trained state"],
     ["Conv only (a)", "conv from its baseline initial LR; head, norm_bias, whiten_bias on the baseline"],
     ["4 LRs (b)", "All four from 1.0. Interval 1: each LR's level is scanned on a x7.7 grid (both neighbours first, "
                   "then the better side while improving), scored after the interval plus one interval at 0.36 x the "
                   "value; then conv is refined on the x0.6 grid with rest_lin. Later intervals: conv is climbed and "
                   "the other three follow its profile (value_k = value_1 x conv_k / conv_1)"],
     ["Budget", "At most 20 full runs of train steps (every trained step counts); the transform applies to the "
                "searched groups"],
     ["Evaluation", "2 search seeds (0, 1) per cell; each committed schedule retrained on held-out seeds 100-107"]],
    "ics_v1_observations.txt")
ics_section(
    "v2", 14, "ICS v2: a short-cooldown pass, then a rest-of-run pass on its plan",
    "v1's rest-of-run linear lookahead is defined on the spec schedule, so under a transform its applied shape is "
    "distorted and it misjudges early intervals (v1 under the ramp: 93.11-93.19, below doing nothing). Does a first "
    "pass with a short, local cooldown, whose schedule then serves as the second pass's lookahead, fix that?",
    [["Pass 1", "Every interval in order: climb conv from the previous interval's value, each candidate scored after "
                "the interval plus one interval at 0.22 x its value (the least biased short cooldown, Experiment 12; "
                "local, so a transform's global shape does not distort it)"],
     ["Pass 2", "From the initial state again: each interval's climb starts at pass 1's value; a candidate is scored "
                "after the rest of the run on pass 1's schedule, its tail scaled by candidate / pass-1 value "
                "(rest_plan); the last interval by the final TTA"],
     ["Other", "As v1 (5 constant intervals, x0.6 grid, budget 20 full runs, 2 search seeds, held-out seeds "
               "100-107); conv only: v2's 4-LR variant is replaced by v3/v4 (level rule)"]],
    "ics_v2_observations.txt")
ics_section(
    "v5", 15, "ICS v5: v2 plus kept first interval and replicated late decisions",
    "The late intervals' decisions are worth ~0.1 pp each and one seed gets them wrong about half the time "
    "(Experiment 12), while the first interval's rest-of-run candidates cost a full run each. Does keeping pass 1's "
    "first interval and scoring the late candidates by several continuations close the remaining gap within budget?",
    [["As v2", "Pass 1 with the 0.22 x short cooldown, pass 2 with the rest_plan lookahead"],
     ["Kept interval", "Pass 2 keeps pass 1's first interval (value and trained state, at no cost)"],
     ["Replicates", "In pass 2, a candidate for interval 4 is scored by the mean of 2 continuations and for interval "
                    "5 of 3: the first on the run's own data stream (its state is committed), the others restore the "
                    "pre-interval state, reseed the data order and augmentation (fresh epoch) and train interval + "
                    "lookahead again"],
     ["4 LRs", "Levels: x7.7 scan right after interval 1 (conv: best score; head and biases: largest within 1 pp of "
               "the best), then as conv only with the other groups tied to conv's profile"]],
    "ics_v5_observations.txt")

# Experiment 16: representation of constant intervals (rep_* fixed runs, seeds 100-107).
story += [
    P("Experiment 16. How well do constant intervals represent the baseline?", "h1"),
    P("Question", "h2"),
    P("Before asking a search to find a constant-interval schedule, can such a schedule match the baseline at all, "
      "at each batch size and under each transform, with 5 or 10 intervals?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Schedules", "conv's schedule replaced by n equal constant intervals, each at the oracle spec (the "
                         "baseline's applied LR divided by the transform) a fraction q of the way through the "
                         "interval; the last interval at min(q, 0.95) (the oracle reaches 0 at the end). q = 0.5 is "
                         "about the interval mean, q = 1 the end value. Exact values (no grid); the other groups on "
                         "the baseline"],
           ["Runs", "Plain training runs on held-out seeds 100-107 (one standard error about 0.045 pp)"]],
          [3.3 * cm, 13.7 * cm]),
]
rows = [["Batch size, transform", "n = 5: q 0.5 / 0.75 / 1", "n = 10: q 0.5 / 0.75 / 1", "Baseline (do-nothing)"]]
for bs in ics.BATCH_SIZES:
    for tr in ics.TRANSFORMS:
        def rep(n):
            vals = [ics_refs.get(f"rep_n{n}_q{q}_bs{bs}{tr}") for q in (50, 75, 100)]
            return " / ".join("-" if v is None else f"{v:.2f}" for v in vals)
        base = ics.target(ics_refs, bs)
        dn = ics.do_nothing(ics_refs, "a", bs, tr)
        rows.append([f"BS {bs}, {ics.TRANSFORM_LABELS[tr]}", rep(5), rep(10),
                     ("-" if base is None else f"{base:.2f}") + ("" if not tr or dn is None else f" ({dn:.2f})")])
story += [P("Held-out TTA of the oracle staircases", "h2"), table(rows, [4.0 * cm, 4.3 * cm, 4.3 * cm, 4.4 * cm])]
story += observations("rep_observations.txt")

# Experiment 17: averaging two searches' schedules.
avg_runs = ics.averaged()
story += [
    P("Experiment 17. Averaging two searches' schedules", "h1"),
    P("Question", "h2"),
    P("Two search seeds of the same ICS version often commit different schedules, one or two grid steps apart in a "
      "few intervals, and score 0.1-0.4 pp apart. Their decision errors are partly independent; does the geometric "
      "mean of the two schedules, interval by interval, score better than the two do on average?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Schedules", "For each finished cell of ICS v1, v2 and v5 (conv only): the geometric mean of search seeds "
                         "0 and 1's committed conv values, interval by interval (rounded to 2 significant figures, "
                         "not snapped to the grid)"],
           ["Runs", "Plain training runs on held-out seeds 100-107"]], [3.3 * cm, 13.7 * cm]),
]
rows = [["Cell", "Averaged", "Parents (seed 0 / 1)", "Gain over their mean", "Baseline"]]
gains = []
for name in sorted(avg_runs):
    if not name.startswith("avg_"):
        continue
    version = name.split("_")[1]
    rest = name[len(f"avg_{version}_"):]
    sc, bs = rest[0], int(rest.split("_bs")[1].split("_")[0])
    tr = rest.split(f"bs{bs}")[1]
    parents = ics.load(version).get((sc, bs, tr), {})
    held = [parents[s]["heldout"] for s in sorted(parents) if "heldout" in parents[s]]
    if len(held) < 2:
        continue
    mean = sum(held) / len(held)
    gains.append(avg_runs[name][0] - mean)
    base = ics.target(ics_refs, bs)
    rows.append([f"{version}, BS {bs}, {ics.TRANSFORM_LABELS[tr]}", f"<b>{avg_runs[name][0]:.2f}</b>",
                 " / ".join(f"{h:.2f}" for h in held), f"{avg_runs[name][0] - mean:+.2f}",
                 "-" if base is None else f"{base:.2f}"])
if gains:
    story += [P("Held-out TTA", "h2"), table(rows, [4.6 * cm, 2.2 * cm, 3.6 * cm, 3.4 * cm, 3.2 * cm]),
              P(f"Mean gain over the parents' mean: {sum(gains) / len(gains):+.3f} pp over {len(gains)} cells "
                f"({sum(g > 0 for g in gains)} positive).", "small")]
story += observations("averaging_observations.txt")

# Experiment 18: expanding 5-interval schedules to 10 intervals.
exp_rows = [["5-interval schedule", "As 5 intervals", "Expanded to 10", "Gain", "Baseline"]]
exp_gains = []
for name, (mean, sched) in sorted(avg_runs.items()):
    if not name.startswith("exp_"):
        continue
    src = name[4:]
    bs = int(src.split("bs")[1].split("_")[0])
    if src.startswith("rep_"):
        parent, label = ics_refs.get(src), f"oracle staircase q 0.75, BS {bs}"
    elif src.startswith("avg_"):
        parent, label = avg_runs.get(src, (None,))[0], f"averaged {src.split('_')[1]}, BS {bs}"
    else:
        version, seed = src.split("_")[0], int(src.split("_s")[-1])
        parent = ics.load(version).get(("a", bs, ""), {}).get(seed, {}).get("heldout")
        label = f"{version} seed {seed}, BS {bs}"
    tr = src.split(f"bs{bs}")[1].split("_s")[0] if not src[-3:-1] == "_s" else src.split(f"bs{bs}")[1][:-3]
    label += f", {ics.TRANSFORM_LABELS.get(tr, tr.lstrip('_'))}"
    if parent is None:
        continue
    exp_gains.append(mean - parent)
    base = ics.target(ics_refs, bs)
    exp_rows.append([label, f"{parent:.2f}", f"<b>{mean:.2f}</b>", f"{mean - parent:+.2f}", "-" if base is None else f"{base:.2f}"])
story += [
    P("Experiment 18. Expanding 5-interval schedules to 10 intervals", "h1"),
    P("Question", "h2"),
    P("Ten intervals represent the baseline at BS 500 and 125 where five fall short (Experiment 16), but a 10-interval "
      "search makes twice the noisy decisions. Does a 5-interval schedule, expanded to 10 intervals by a fixed rule, "
      "score like a 10-interval one?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Expansion", "Each interval's value v is split into two halves: v x (previous / v)^(1/4) and v x (next / "
                         "v)^(1/4) (the first interval's first half stays v; the last interval's second half "
                         "continues the decay, v x (v / previous)^(1/4)). A quarter of the way, in log, toward each "
                         "neighbour: the staircase becomes twice as fine and follows the trend"],
           ["Schedules", "5-interval oracle staircases (q = 0.75, Experiment 16), searched schedules (ICS v2, v5) "
                         "and averaged ones (Experiment 17); held-out seeds 100-107"]], [3.3 * cm, 13.7 * cm]),
]
if exp_gains:
    story += [P("Held-out TTA", "h2"), table(exp_rows, [5.6 * cm, 2.6 * cm, 2.8 * cm, 2.0 * cm, 2.4 * cm]),
              P(f"Mean gain {sum(exp_gains) / len(exp_gains):+.3f} pp over {len(exp_gains)} schedules "
                f"({sum(g > 0 for g in exp_gains)} positive).", "small")]
story += observations("expand_observations.txt")


def ics_compare_section(versions, number, title, question, setup_rows, obs_file):
    """One table: per cell, each version's held-out mean over search seeds and its cost."""
    loaded = {v: ics.load(v) for v in versions}
    story.extend([P(f"Experiment {number}. {title}", "h1"), P("Question", "h2"), P(question), P("Setup", "h2"),
                  table([["Item", "Value"], *setup_rows], [3.3 * cm, 13.7 * cm])])
    rows = [["Cell (baseline; do-nothing)", *versions]]
    within = {v: [0, 0] for v in versions}
    for bs in ics.BATCH_SIZES:
        base = ics.target(ics_refs, bs)
        for sc in ics.SCENARIOS:
            for tr in ics.TRANSFORMS:
                dn = ics.do_nothing(ics_refs, sc, bs, tr)
                label = (f"BS {bs}, {ics.SCENARIO_LABELS[sc]}, {ics.TRANSFORM_LABELS[tr]} "
                         f"({'-' if base is None else f'{base:.2f}'}" + (f"; {dn:.2f})" if tr and dn else ")"))
                cells = [label]
                for v in versions:
                    seeds = loaded[v].get((sc, bs, tr), {})
                    held = [r["heldout"] for r in seeds.values() if "heldout" in r]
                    if not held:
                        cells.append("-")
                        continue
                    mean = sum(held) / len(held)
                    if base:
                        within[v][1] += 1
                        within[v][0] += mean >= base - 0.1
                    cost = max(r["cost"] for r in seeds.values())
                    hit = "*" if any(r["budget_hit"] for r in seeds.values()) else ""
                    cells.append(f"<b>{mean:.2f}</b> ({'/'.join(f'{h:.2f}' for h in held)}); {cost:.0f}x{hit}")
                rows.append(cells)
    width = 13.0 / len(versions)
    story.extend([P("Held-out TTA (mean over search seeds; per seed); cost in full runs", "h2"),
                  table(rows, [4.0 * cm] + [width * cm] * len(versions)),
                  P("Within 0.1 pp of the baseline or above (finished cells): "
                    + "; ".join(f"{v}: {within[v][0]} of {within[v][1]}" for v in versions)
                    + ". * = the search hit its step budget.", "small")])
    story.extend(observations(obs_file))


ics_compare_section(
    ["v9", "v10", "v11"], 19, "ICS v9-v11: 5-interval search expanded to 10 intervals, with a scaled or fixed plan tail",
    "Searching 5 intervals and expanding the result to 10 (Experiment 18), which second-pass lookahead works: the "
    "previous pass's tail scaled by candidate / plan value (v9), or left as it is so a candidate changes only its own "
    "interval (v10, v11)? And should the budget go to replicated decisions (v10) or wider climbs (v11)?",
    [["Common", "5 constant intervals; pass 1: climb each interval from the previous one's value with a one-interval "
                "cooldown at 0.22 x the candidate; pass 2: from the initial state, keep interval 1, climb intervals "
                "2-5 from pass 1's values with a rest-of-run lookahead on pass 1's schedule; the result expanded to "
                "10 intervals (Experiment 18). 4 LRs from 1.0: a joint x7.7 level scan of head and the biases "
                "(largest within 1 pp after a one-interval cooldown), then each alone at most two x7.7 steps up; "
                "the other groups follow conv's profile. Budget 20 full runs; 2 search seeds; held-out seeds 100-107"],
     ["v9", "Pass-2 tail scaled by candidate / pass-1 value; climbs at most 1 grid step from pass 1; decisions "
            "replicated over 2 continuations (3 in the last interval; 4 LRs: only intervals 4-5)"],
     ["v10", "As v9 with the tail unscaled (rest_fixed)"],
     ["v11", "As v10 with replicates only in intervals 4-5 and climbs of up to 2 grid steps"]],
    "ics_v9_v11_observations.txt")

# Experiment 20: why the 4-LR search falls short (diagnostic held-out runs of parts of its schedules).
diag_labels = {
    "full": "searched schedule, all four groups",
    "conv_only": "its conv schedule; head, norm_bias, whiten_bias on the baseline",
    "base_head": "head back on the baseline (the others as searched)",
    "base_norm_bias": "norm_bias back on the baseline",
    "base_whiten_bias": "whiten_bias back on the baseline",
    "wb_down": "whiten_bias one x7.7 step lower",
    "c4x_others": "conv replaced by C4 expanded; the others as searched",
}
rows = [["Variant", "Search seed 0", "Search seed 1"]]
for key, label in diag_labels.items():
    vals = [avg_runs.get(f"diag_b_s{s}_{key}") for s in (0, 1)]
    if all(v is None for v in vals):
        continue
    rows.append([label, *("-" if v is None else f"{v[0]:.2f}" for v in vals)])
story += [
    P("Experiment 20. Why the 4-LR search falls short", "h1"),
    P("Question", "h2"),
    P("From scratch with all four LRs, ICS v12/v13 score 93.3-93.8 at BS 2000 without a transform, below the "
      "conv-only searches. Is the conv schedule or the other groups' levels at fault?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Schedules", "The two committed 10-interval schedules of ICS v13 with 4 LRs at BS 2000, no transform "
                         "(search seeds 0 and 1, head / norm_bias / whiten_bias levels 460 / 60 / 460 and 460 / 60 / "
                         "3500, each tied to conv's profile), and variants with parts replaced"],
           ["Runs", "Plain training runs on held-out seeds 100-107"]], [3.3 * cm, 13.7 * cm]),
    P("Held-out TTA (baseline 93.92)", "h2"), table(rows, [9.0 * cm, 4.0 * cm, 4.0 * cm]),
]
story += observations("diag_b_observations.txt")
ics_compare_section(
    ["v11", "v14", "v15", "v16", "v17"], 21, "ICS v14-v17: fixing the first pass's bias and the second pass's noise",
    "v11's remaining error has two visible causes: the first pass's short cooldown is biased toward small LRs (one "
    "grid step in the middle intervals, more under the ramp and at BS 125; Experiment 12) and the second pass's "
    "single-seed decisions are too noisy to undo it. Which fix helps?",
    [["v11 (reference)", "Short-cooldown pass 1; rest-of-run pass 2 on pass 1's schedule, tail unscaled, climbs up to 2 "
                         "grid steps, replicates only in intervals 4-5; expanded to 10 intervals"],
     ["v14", "v11 + the second pass carries its correction forward: an interval's climb starts at pass 1's value "
             "times the previous interval's pass-2 / pass-1 ratio"],
     ["v15", "v11 with a two-interval cooldown in pass 1 (a longer horizon, less biased in principle)"],
     ["v16", "v11 + pass 1's schedule shifted one grid step up (all but the last interval) before pass 2"],
     ["v17", "v11 with pass 2 run twice from pass 1's schedule (different data order, no replicates), the two "
             "schedules averaged (geometric mean)"],
     ["4 LRs", "v14-v17 also lower the head/bias levels one x7.7 step after the scan and try each one step lower "
               "again in pass 2 (v13's relevel); v17 runs conv only (two second passes do not fit with the level "
               "scan)"]],
    "ics_v14_v17_observations.txt")
ics_compare_section(
    ["v16", "v18", "v19", "v23", "v24"], 22,
    "ICS v16-v24: the shifted plan, with an ensemble, joint 4-LR levels or a third pass",
    "Starting from v16 (the best of Experiment 21), does averaging two second passes (v18), a simpler and cheaper 4-LR "
    "level rule (v19) or a third rest-of-run pass (v23) close the remaining gap, and does the result hold over four "
    "search seeds?",
    [["v16", "Short-cooldown pass 1, its plan shifted one grid step up (all but the last interval), rest-of-run pass 2 "
             "(tail unscaled, climbs up to 2 grid steps, replicates in intervals 4-5), expanded to 10 intervals. 4 "
             "LRs: joint x7.7 level scan of head and the biases, each alone at most two steps up, one step back, and "
             "one step lower again in pass 2 if that scores higher"],
     ["v18", "v16 with pass 2 run twice (different data order, no replicates) and the two schedules averaged"],
     ["v19", "Conv only: as v16. 4 LRs: the joint level only (no individual climbs, no back-off, no relevel), "
             "then shifted with the plan; 4 search seeds (0-3)"],
     ["v23", "v19 with a third pass repeating pass 2 on pass 2's schedule; passes 2-3 climb one grid step at most, "
             "only pass 3 replicates (its last interval)"],
     ["v24", "v19 whose shift moves only conv (4 LRs: head and the biases keep their scanned levels); conv only it "
             "is v19"]],
    "ics_v16_v23_observations.txt")
ics_compare_section(
    ["v19", "v24", "v25", "v26", "v27"], 23,
    "ICS v25-v27: refining from interval 1, and averaging two independent half-budget searches",
    "v24's misses are spread between search seeds (Experiment 22). Does refining every interval in the second pass "
    "(v25), or running the whole search twice at half the budget from different data orders and averaging the two "
    "schedules (v26; v27 with one shared 4-LR level scan), close the gap within the same 20-run budget?",
    [["v19 / v24", "References (Experiment 22): v19 for conv only, v24 for 4 LRs"],
     ["v25", "v24 with the second pass refining every interval from interval 1 (only interval 0 kept), replicates "
             "only in the last interval"],
     ["v26", "Two whole v24-style searches (short-cooldown pass 1, shifted plan, rest-of-run pass 2 climbing one grid "
             "step, replicates only in the last interval), each with half the budget and its own data order; the two "
             "5-interval schedules averaged (geometric mean) and expanded to 10 intervals. Steps are kept back so the "
             "search always finishes within budget"],
     ["v27", "v26 whose second search reuses the first one's 4-LR level scan (one scan, the rest of the budget split "
             "evenly); conv only it is v26"]],
    "ics_v25_v27_observations.txt")

# Experiment 24: held-out probes of the searched schedules' scale and of the committed schedule's shape.
probe_runs = ics.averaged()


def probe_block(version, scenario, columns):
    """Per cell, each column's mean held-out TTA over the search seeds. A column is (label, kind, name):
    kind "own" = the searches' own held-out scores, "cell" = one run per cell named name(cell), "seed" = one
    run per search seed named name(cell, seed) (all seeds must exist), with paired differences to "own"."""
    data = ics.load(version)
    rows = [["Cell (baseline)", *(label for label, _, _ in columns)]]
    paired = {label: [] for label, kind, _ in columns if kind == "seed"}
    for bs in ics.BATCH_SIZES:
        for tr in ics.TRANSFORMS:
            cell = f"{scenario}_bs{bs}{tr}"
            seeds = {s: r for s, r in data.get((scenario, bs, tr), {}).items()
                     if "heldout" in r and r["cost"] <= 20.0001}
            cells = [f"BS {bs}, {ics.TRANSFORM_LABELS[tr]} ({ics.target(ics_refs, bs):.2f})"]
            for label, kind, name in columns:
                mean = None
                if kind == "own" and seeds:
                    mean = sum(r["heldout"] for r in seeds.values()) / len(seeds)
                elif kind == "cell":
                    mean = (probe_runs.get(name(cell)) or [None])[0]
                elif kind == "seed" and seeds:
                    vals = {s: probe_runs.get(name(cell, s)) for s in seeds}
                    if all(v is not None for v in vals.values()):
                        mean = sum(v[0] for v in vals.values()) / len(vals)
                        paired[label] += [v[0] - seeds[s]["heldout"] for s, v in vals.items()]
                cells.append("-" if mean is None else f"<b>{mean:.2f}</b>" if mean >= ics.target(ics_refs, bs) - 0.1
                             else f"{mean:.2f}")
            rows.append(cells)
    summary = []
    for label, x in paired.items():
        if len(x) > 1:
            m = sum(x) / len(x)
            se = (sum((a - m) ** 2 for a in x) / (len(x) - 1) / len(x)) ** 0.5
            summary.append(f"{label}: {m:+.3f} +- {se:.3f} pp ({sum(a > 0 for a in x)} of {len(x)} higher)")
    return rows, summary


def width(rows):
    return [4.4 * cm] + [12.6 / (len(rows[0]) - 1) * cm] * (len(rows[0]) - 1)


scale_rows, scale_sum = probe_block("v26", "a", [
    ("v26 steps", "own", None),
    ("x1.67", "seed", lambda c, s: f"bias_v26_{c}_s{s}_up1"),
    ("x2.78", "seed", lambda c, s: f"bias_v26_{c}_s{s}_up2"),
    ("graded", "seed", lambda c, s: f"bias_v26_{c}_s{s}_grad"),
    ("oracle steps", "cell", lambda c: f"bias_oracle_{c}")])
shape_a_rows, shape_a_sum = probe_block("v26", "a", [
    ("oracle steps", "cell", lambda c: f"bias_oracle_{c}"),
    ("oracle linear", "cell", lambda c: f"lin_oracle_{c}"),
    ("v26 steps", "own", None),
    ("v26 linear (10)", "seed", lambda c, s: f"lin_v26_{c}_s{s}"),
    ("v26 linear (5)", "seed", lambda c, s: f"lin5_v26_{c}_s{s}")])
shape_b_rows, shape_b_sum = probe_block("v27", "b", [
    ("v27 steps", "own", None),
    ("v27 linear (10)", "seed", lambda c, s: f"lin_v27_{c}_s{s}"),
    ("v27 linear (5)", "seed", lambda c, s: f"lin5_v27_{c}_s{s}")])
story += [
    P("Experiment 24. Is the searched schedule too small, or the staircase the wrong shape?", "h1"),
    P("Question", "h2"),
    P("From the middle of the run on, the searched conv LRs lie 1.3-4.6x below the oracle (Experiment 23). Are they "
      "too small, so that scaling them up scores higher? And how much does committing a staircase cost, compared "
      "with a smooth schedule through the same values?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Scaled", "v26's conv-only searched schedules (10 constant intervals) x1/0.6 everywhere, x1/0.36 "
                      "everywhere, and x(1/0.6)^(2i/9) in interval i (x1 rising to x2.8)"],
           ["Oracle", "The baseline conv schedule divided by the transform: its 10 interval means as constant "
                      "steps, and joined linearly"],
           ["Linear", "A schedule's values joined piecewise linearly through the interval midpoints, the end "
                      "half-intervals continuing the neighbouring slope (clipped at 0); a linear schedule's interval "
                      "means give it back exactly. Linear (10): from the 10-interval expanded schedule; linear (5): "
                      "from the two searches' averaged 5-interval schedule, without the expansion. 4 LRs: all four "
                      "groups"],
           ["Runs", "research/eval_schedule.py on held-out seeds 100-107 (the seeds of the searches' own held-out "
                    "scores, so differences are paired); means over search seeds 0-1; bold = within 0.1 pp of the "
                    "baseline or above"]], [3.3 * cm, 13.7 * cm]),
    P("Scaled conv-only schedules: held-out TTA", "h2"), table(scale_rows, width(scale_rows)),
    P("Paired difference to the searched schedule: " + "; ".join(scale_sum) + ".", "small"),
    P("Committed shape, conv only: held-out TTA", "h2"), table(shape_a_rows, width(shape_a_rows)),
    P("Paired difference to the searched steps: " + "; ".join(shape_a_sum) + ".", "small"),
    P("Committed shape, 4 LRs: held-out TTA", "h2"), table(shape_b_rows, width(shape_b_rows)),
    P("Paired difference to the searched steps: " + "; ".join(shape_b_sum) + ".", "small"),
]
story += observations("shape_probe_observations.txt")

ics_compare_section(
    ["v28", "v29", "v30"], 25, "ICS v28-v30: the linear commit, and refining the first interval",
    "v28 commits the averaged schedule linearly (Experiment 24) but keeps pass 1's shifted first interval, which "
    "overshoots under the reverse ramp at BS 2000 (0.13 in all four search seeds; the oracle's first fifth averages "
    "0.065). Does refining the first interval in pass 2 too fix BS 2000, with two half-budget searches (v29) or one "
    "full-budget search (v30)?",
    [["v28", "v27 (two half-budget searches, shared 4-LR levels) committing the averaged 5 values linearly; its "
             "search is v27's, so seeds 0-1 are Experiment 24's linear (5) column; this table shows search seeds 2-3"],
     ["v29", "v28 whose pass 2 also refines interval 1 (one grid step at most, rest-of-run lookahead)"],
     ["v30", "One full-budget search (v25: interval 1 refined, climbs up to 2 grid steps, the last interval "
             "averaged over 2 continuations) committed linearly"]],
    "ics_v28_v30_observations.txt")


# Experiment 26: the committed representation's ceiling (oracle schedules as v29 commits them) and the grid.
rows = [["Cell (baseline)", "Oracle linear (10)", "Oracle linear (5)", "Oracle linear (5), x0.6 grid",
         f"ICS {FINAL_VERSION}"]]
grid_cost = []
v29_cells = ics.load(FINAL_VERSION)
for bs in ics.BATCH_SIZES:
    for tr in ics.TRANSFORMS:
        cell = f"a_bs{bs}{tr}"
        vals = [(probe_runs.get(f"{prefix}_oracle_{cell}") or [None])[0] for prefix in ("lin", "lin5", "lin5g")]
        held = [r["heldout"] for r in v29_cells.get(("a", bs, tr), {}).values() if "heldout" in r]
        vals.append(sum(held) / len(held) if held else None)
        if vals[1] is not None and vals[2] is not None:
            grid_cost.append(vals[1] - vals[2])
        rows.append([f"BS {bs}, {ics.TRANSFORM_LABELS[tr]} ({ics.target(ics_refs, bs):.2f})",
                     *("-" if v is None else f"{v:.2f}" for v in vals)])
story += [
    P("Experiment 26. What does the committed representation allow, and what does the grid cost?", "h1"),
    P("Question", "h2"),
    P("With the linear commit, ICS v29 matches the baseline in most cells but stays about 0.1 pp short at BS 2000. "
      "Is that the search, or the representation it commits: 5 values on the x0.6 grid, joined linearly?"),
    P("Setup", "h2"),
    table([["Item", "Value"],
           ["Schedules", "Conv only: the oracle (baseline conv schedule divided by the transform) reduced to its 10 or "
                         "5 interval means (2 significant figures), and the 5 means snapped to the x0.6 grid (what a "
                         "single search can commit), each joined linearly through the interval midpoints"],
           ["Runs", "research/eval_schedule.py on held-out seeds 100-107; ICS v29 = mean over its search seeds"]],
          [3.3 * cm, 13.7 * cm]),
    P("Held-out TTA, conv only", "h2"), table(rows, [4.4 * cm] + [3.15 * cm] * 4),
    P(f"Snapping the 5 interval means to the x0.6 grid costs {sum(grid_cost) / max(len(grid_cost), 1):.3f} pp on "
      f"average over the {len(grid_cost)} cells.", "small"),
]
story += observations("grid_observations.txt")
ics_compare_section(
    ["v29", "v32", "v33"], 27, "ICS v32-v33: a third search instead of the second pass, and a finer second-pass grid",
    "With 4 LRs v29's second passes get almost no budget; would a third independent first pass use it better (v32)? "
    "And does a second pass climbing on a grid twice as fine recover what the x0.6 grid costs (v33, Experiment 26)?",
    [["v29", "Reference (campaign summary)"],
     ["v32", "Three independent short-cooldown searches (one pass each, a third of the budget after the shared 4-LR "
             "level scan), each shifted one grid step up at the end, averaged and committed linearly"],
     ["v33", "v29 whose second passes climb on a x0.6^0.5 = x0.775 grid, up to 2 steps (v29's range)"]],
    "ics_v32_v33_observations.txt")


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
    canvas.drawString(2 * cm, 1.2 * cm, "Interval search experiments")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Cooldown study").build(story, onFirstPage=footer,
                                                                       onLaterPages=footer)
print(OUT)


"""Build cooldown_study.pdf: which 40-step lookahead cooldown ranks interval candidates like the
final result? Data: research/cooldown_search_probe.py logs via research/cooldown_study.py.
python research/make_cooldown_study.py"""
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
import cooldown_study as cs  # noqa: E402
import importance_study as imp  # noqa: E402

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
          KeepTogether([img, P("Figure 1. Each method's score per candidate, relative to its own best "
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
    story += [KeepTogether([P(f"Interval {k + 1}: score per candidate conv LR (bold: each row's best){partial}",
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
story += [KeepTogether([imp_img, P("Figure 2. One-at-a-time sweeps around each L's joint-search center.",
                                   "small")])]
for length, r in runs.items():
    if len(r["sweep"]) < 4:
        continue
    rows = [["LR"] + [f"{o:+d}" for o in range(-5, 6)]]
    for g in imp.GROUPS:
        sw = r["sweep"][g]
        rows.append([f"{g} ({sw[0][0]:g})"] + [f"{sw[o][1] - sw[0][1]:+.2f}" if o in sw else "-"
                                              for o in range(-5, 6)])
    story += [KeepTogether([P(f"L = {length}: TTA minus the center's (pp) at each grid offset (center LR in "
                              f"brackets; center TTA {100 * r['center']['tta']:.2f})", "h2"),
                            table(rows, [3.4 * cm] + [1.23 * cm] * 11)])]


def attach_headings(flowables):
    """Keep each subheading on the same page as the table / figure that follows it."""
    out = []
    for item in flowables:
        if (isinstance(item, KeepTogether) and out and isinstance(out[-1], Paragraph)
                and out[-1].style.name == "h2"):
            out[-1] = KeepTogether([out[-1], item])
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

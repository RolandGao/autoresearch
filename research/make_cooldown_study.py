"""Build cooldown_study.pdf: which 40-step lookahead cooldown ranks interval candidates like the
final result? Data: research/cooldown_search_probe.py logs via research/cooldown_study.py.
python research/make_cooldown_study.py"""
import os
import sys

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
              "shared": ("#eb6834", "-", "^"), "per_group": ("#eb6834", "--", "v")}
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
    P("Cooldown study: which 40-step lookahead ranks interval candidates like the final result?",
      "title"),
    P("2026-10-16. Experiment design by Roland Gao; runs and analysis by Claude. Code: "
      "research/cooldown_search_probe.py (runs), research/cooldown_study.py (analysis), "
      "research/make_cooldown_study.py (this PDF). Logs: logs/research_cooldown_search_probe_*.", "sub"),
    P("Question", "h1"),
    P("An interval search picks each interval's LR by training the interval with a candidate LR, then "
      "a short lookahead (the cooldown), and scoring the result. Which cooldown ranks the candidates the "
      "way the final accuracy does? Compared, all with a 40-step lookahead (one interval at batch size "
      "2000): no lookahead, linear decay to 0 (conv only, or all four groups), one searched constant "
      "cooldown shared by all groups, and four searched constant cooldowns (one per group)."),
    P("Setup", "h1"),
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
        ["1 searched cooldown", "Constant LRs for 40 steps: conv at m x candidate, each other group at m x its "
                                "mean LR over the interval; m chosen from {1, 0.6, 0.36, 0.22, 0.13, 0.078, "
                                "0.047, 0} by the best score on that seed"],
        ["4 searched cooldowns", "As above but one m per group; coordinate search from the best shared m, one "
                                 "group at a time over the same 8 values, until no group changes (at most 3 "
                                 "passes)"],
        ["Seeds", "4 training seeds (0-3); every method uses the same prefix and data order per seed"],
        ["Score", "Test-set TTA accuracy (%)"],
    ], [3.3 * cm, 13.7 * cm]),
    P("Results", "h1"),
    P("Picked LR = the candidate with the highest seed-mean score. Regret = ground truth at its best "
      "candidate minus ground truth at the picked one (pp). Per-seed regret = the same when each seed picks "
      "from its own scores (what a single-seed search sees), averaged over seeds.", "small"),
]
rows = [["Method"] + [f"Interval {k + 1}" for k in range(4)]]
for m in cs.METHODS:
    cells = [cs.LABELS[m]]
    for s in summaries:
        cells.append("running" if s is None else
                     f"{s['picks'][m]:g} ({s['regret'][m]:.2f}; per seed {s['seed_regret'][m]:.2f})")
    rows.append(cells)
story += [P("Picked LR (regret; per-seed regret)", "h2"),
          table(rows, [5.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]),
          KeepTogether([img, P("Figure 1. Each method's seed-mean score per candidate, relative to its own best "
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
    story += [P(f"Interval {k + 1}: seed-mean score per candidate conv LR (bold: each row's best)", "h2"),
              table(rows, [5.2 * cm] + [width * cm] * len(s["cands"]))]
    shared = s["chosen"]["shared"]
    per = s["chosen"]["per_group"]
    rows = [["Candidate", "1 searched cooldown: chosen m (seeds)",
             "4 searched cooldowns: chosen (conv, head, norm_bias, whiten_bias) (seeds)"]]
    for c in s["cands"]:
        rows.append([f"{c:g}", ", ".join(f"{m:g} x{n}" for m, n in sorted(shared[c].items())),
                     "; ".join(f"({', '.join(f'{v:g}' for v in ms)}) x{n}" for ms, n in sorted(per[c].items()))])
    story += [table(rows, [2.0 * cm, 5.0 * cm, 10.0 * cm])]
if os.path.exists(OBSERVATIONS):
    story += [P("Observations", "h1")] + [P(line) for line in open(OBSERVATIONS).read().split("\n\n") if line.strip()]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Cooldown study (batch size 2000)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Cooldown study").build(story, onFirstPage=footer,
                                                                       onLaterPages=footer)
print(OUT)

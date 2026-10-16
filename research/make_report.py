"""Build interval_search_report.pdf (the readable report; interval_search_writeup.txt is the
run-by-run log it cites as W-sections). python research/make_report.py"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "research"))
import report_data  # noqa: E402
import report_findings  # noqa: E402

NEW = report_data.search_runs(since=report_findings.CUT)


def new_row(name, label):
    """Scenario-A row from runs on the updated baselines (filled in as they finish)."""
    cells = []
    for suffix in ("", "_ramp", "_wave"):
        r = NEW.get(name + suffix)
        cells.append(f"{r['heldout']:.2f} ({r['seconds']:.0f}, n={r['n']})" if r else "running")
    return ["", label] + cells
OUT = os.path.join(REPO, "interval_search_report.pdf")
INK, MUTED, RULE, HEAD_BG = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d9d8d3", "#f1f0ec"))

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontSize=17, leading=21, alignment=TA_LEFT,
                            textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=9, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=base["Heading2"], fontSize=12.5, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=base["Heading3"], fontSize=10.5, leading=13, textColor=INK,
                         spaceBefore=6, spaceAfter=2, keepWithNext=1),
    "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.3, leading=12.6, textColor=INK,
                           spaceAfter=4),
    "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8, leading=10.5, textColor=MUTED,
                            spaceAfter=4),
    "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.3, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8.3, leading=10.4, textColor=INK,
                            fontName="Helvetica-Bold"),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    return ListFlowable([ListItem(P(i, style), leftIndent=10) for i in items],
                        bulletType="bullet", start="-", leftIndent=10, bulletFontSize=8)


def steps(items):
    return ListFlowable([ListItem(P(i), leftIndent=14) for i in items], bulletType="1",
                        leftIndent=14, bulletFontSize=9)


def table(rows, widths, header_rows=1, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if r < header_rows or r in bold_rows else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=header_rows)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, RULE),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(path, width_cm, caption):
    img = Image(os.path.join(REPO, path))
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * img.imageHeight / img.imageWidth
    return KeepTogether([img, P(caption, "small")])


story = [
    P("LR-schedule search for CIFAR-10 airbench", "title"),
    P("Report, 2026-10-16. Code: train_cifar.py (algorithm=\"auto\") and research/. Every run's log "
      "is in logs/. \"W-n\" points to section n of interval_search_writeup.txt, the run-by-run log.",
      "sub"),

    P("Summary", "h1"),
    bullets([
        "<b>Algorithm:</b> a multi-pass knot search. Each searched LR is piecewise linear through knots "
        "every 20% of training. Pass 1 picks the knots interval by interval, scoring candidates with a "
        "lookahead to the end of training. Passes 2-3 redo the sweep using the previous pass's schedule "
        "as the lookahead and score candidates on 2 fresh seeds.",
        "<b>Batch size 2000:</b> matches the baseline (93.94) with no transform and under both "
        "transforms: 93.88-93.96 when searching one LR (~600 s), 93.81-93.96 when searching all four "
        "from scratch (~1800 s).",
        "<b>Batch size 500 and 125:</b> not yet. 0.1-0.25 pp short of the new baselines (94.01 / 93.88). "
        "Those runs used the older, weaker baselines; reruns on the new ones are in progress.",
        "<b>Main findings:</b> short lookaheads pick LRs that are too small; the final interval must "
        "anneal to 0; single-seed searches suffer a winner's curse; a fixed lookahead shape breaks under "
        "transforms, and the previous pass's schedule fixes it.",
    ]),

    P("1. Question", "h1"),
    P("Find one LR-schedule search algorithm that (a) matches the baseline's held-out accuracy, "
      "(b) uses as little search time as possible, (c) still works when an unknown transformation is "
      "applied on top of the schedule, and (d) still works at batch sizes 500 and 125 (development was "
      "at 2000)."),
    table([
        ["Scenario", "What is searched", "Starting point"],
        ["A: good baseline", "One LR schedule (conv). Also tested: head only; conv + head.",
         "The baseline's own LR; other groups stay on the baseline"],
        ["B: from scratch", "All four trained LRs: conv, head, norm_bias, whiten_bias",
         "Every LR at 1.0; no knowledge of scale or shape"],
    ], [3.2 * cm, 7.3 * cm, 6.5 * cm]),

    P("2. Definitions", "h1"),
    table([
        ["Term", "Meaning"],
        ["Spec schedule", "The LR schedule the search writes (piecewise linear; see section 3)."],
        ["Transform", "A fixed multiplier applied to the spec LR before training uses it, unknown to the "
                      "search (x = fraction of training). <b>ramp</b>: x 4^(2x-1), from x0.25 to x4. "
                      "<b>wave</b>: x 2^sin(2 pi x), x1 -> x2 -> x0.5 -> x1. With a transform the best spec "
                      "schedule is no longer linear decay (figure 1)."],
        ["Held-out TTA", "Quality of a found schedule: retrained from scratch on seeds 100-107; mean "
                         "test-set TTA accuracy. Search decisions also use the test set (intentional)."],
        ["Search seeds", "Independent runs of a search; tables give the mean held-out TTA over n of them."],
        ["Cost", "Wall-clock seconds of the search on one A100, held-out retraining excluded. One full "
                 "training run at batch size 2000 takes ~10 s."],
        ["Lookahead", "To score a candidate LR for the current interval, train the interval, then keep "
                      "training with an assumed future schedule; the TTA at the end is the score."],
        ["Do-nothing", "The baseline schedule kept unchanged under the transform (no search): what a "
                       "search must recover."],
    ], [3.0 * cm, 14.0 * cm]),
    Spacer(1, 6),
    figure("plots/lr_transforms.png", 17,
           "Figure 1. The two transforms (left), the applied conv LR if the baseline's linear decay is "
           "kept (middle), and the spec schedule that gives the baseline back with the grid-snapped "
           "5-knot oracle (right). Batch size 2000 conv baseline: 0.22 decaying linearly to 0."),

    P("3. Recommended algorithm: multi-pass knot search", "h1"),
    P("Configs: <font face='Courier'>auto_a_v1x3m2t_*</font> (scenario A) and "
      "<font face='Courier'>auto_b_v1x2m3fas_*</font> (scenario B) in train_cifar.py."),
    P("Representation", "h2"),
    P("Each searched LR is piecewise linear through 6 knots, one at every 20% of training (5 intervals). "
      "Knot values lie on the grid 0.6^k rounded to 2 significant figures."),
    P("Pass 1: one forward sweep over the 5 intervals", "h2"),
    steps([
        "For each searched LR, coordinate-climb the knot at the end of the current interval on the grid, "
        "starting from the previous interval's value: try one step up and one step down, keep moving while "
        "the score improves.",
        "Score a candidate by training the interval from the committed state, then a lookahead to the "
        "<b>end of training</b> in which the searched LRs decay linearly from the candidate value to 0; the "
        "other groups follow their own schedules.",
        "Commit the interval with the best knot and continue from the trained state.",
    ]),
    P("Passes 2 and 3: the same sweep, restarted from the initial weights", "h2"),
    bullets([
        "Each interval starts its climb at the previous pass's knot.",
        "The lookahead is the previous pass's schedule for the rest of training, rescaled by candidate / "
        "previous knot, so one decision shifts the whole remaining schedule.",
        "Every candidate is scored as the mean over 2 fresh training seeds.",
    ]),
    P("Additions for scenario B", "h2"),
    bullets([
        "First interval: before climbing, scan each LR over 7 values from x10^-3 to x10^3 around its start "
        "(with a 20% lookahead) and keep the best.",
        "Last interval: also try annealing every searched LR linearly to 0; keep it if it scores higher.",
        "Two passes instead of three; pass 2 scores on 3 seeds and does not rescale the tail.",
    ]),

    P("4. Results", "h1"),
    P("Held-out TTA (%), mean over n search seeds, with search cost in seconds in brackets.", "small"),
    P("4.1 Scenario A: conv searched, other groups on the baseline", "h2"),
    table([
        ["Batch size", "Method", "No transform", "Ramp", "Wave"],
        ["2000", "Baseline", "93.94", "", ""],
        ["", "Do-nothing", "", "93.42", "93.72"],
        ["", "<b>Recommended (3 passes), n=4</b>", "<b>93.96</b> (588)", "<b>93.93</b> (611)",
         "<b>93.88</b> (572)"],
        ["", "2 passes, n=4", "93.98 (359)", "93.78 (387)", "93.91 (362)"],
        ["", "Pass 1 only, n=2", "93.94 (132)", "93.57 (150)", "93.59 (128)"],
        ["500", "Baseline: old / new", "93.83 / 94.01", "", ""],
        ["", "Do-nothing (old baseline)", "", "93.35", "93.68"],
        ["", "Recommended, n=2", "93.79 (725)", "93.75 (774)", "93.84 (736)"],
        new_row("auto_a_v1x3m2t_bs500", "Recommended, new baseline"),
        ["125", "Baseline: old / new", "93.85 / 93.88", "", ""],
        ["", "Do-nothing (old baseline)", "", "93.31", "93.68"],
        ["", "Recommended, n=2 (ramp n=1)", "93.64 (1159)", "93.88 (1109)", "93.72 (1164)"],
        new_row("auto_a_v1x3m2t_bs125", "Recommended, new baseline"),
    ], [2.0 * cm, 5.8 * cm, 3.1 * cm, 3.0 * cm, 3.0 * cm]),
    Spacer(1, 4),
    table([
        ["Other searched groups (batch size 2000, n=2)", "No transform", "Ramp", "Do-nothing ramp"],
        ["Head only", "93.91", "93.90", "93.82"],
        ["Conv + head", "93.99", "93.73", "93.30"],
    ], [7.2 * cm, 3.2 * cm, 3.2 * cm, 3.3 * cm]),
    P("4.2 Scenario B: all four LRs from 1.0", "h2"),
    table([
        ["Batch size", "Baseline", "No transform", "Ramp", "Wave", "n"],
        ["2000", "93.94", "93.96 (1690)", "93.81 (1867)", "93.91 (1814)", "2"],
        ["500", "94.01", "93.89 (2471)", "93.90 (2698)", "-", "1"],
        ["125", "93.88", "not run yet", "", "", ""],
    ], [2.0 * cm, 2.2 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 1.0 * cm]),
    P("At batch size 2000, do-nothing under ramp scores 93.25 and one-pass variants 93.05-93.36.", "small"),
    P("4.3 Verdict", "h2"),
    bullets([
        "<b>Batch size 2000: goal met</b> in both scenarios and all three conditions (within 0.13 pp, mostly "
        "within 0.05 pp), at ~600 s per searched LR (~60 training runs) or ~1800 s from scratch. The "
        "earlier 8-seed global search cost 3780 s for four groups and cannot change a schedule's shape.",
        "<b>Batch size 500 and 125: not met.</b> Against the new baselines, scenario A is 0.1-0.25 pp short "
        "at 500 and 0.0-0.25 pp short at 125.",
    ]),
    P("4.4 Baseline update", "h2"),
    P("After the campaign, baseline_bs500 and baseline_bs125 were replaced by the best SGD configs of the "
      "earlier seed-averaged searches (same optimizers: Muon2 conv, SGD head and biases): batch size 500 "
      "94.01 +- 0.12 (seeds 100-115; the previous baseline scored 93.83 on seeds 100-107), batch size 125 "
      "93.88 +- 0.11 (previous 93.85, a tie). All batch size 500 / 125 numbers above used the previous "
      "baselines, as the target and as the fixed groups of scenario A. Reruns on the new baselines are in "
      "progress; do-nothing on them: batch size 500 ramp 93.37, wave 93.77; batch size 125 ramp 93.48, "
      "wave 93.69."),

    *report_findings.build(P, table, figure, KeepTogether),

    P("6. What did not work", "h1"),
    table([
        ["Idea", "Result", "Log"],
        ["Searched cooldown multiplier, 20% lookahead", "best 93.78 at batch size 2000", "W-5"],
        ["Prefer a larger LR within x pp of the best", "93.54-93.65", "W-5"],
        ["Search the lookahead's level / a geometric shape", "93.82 / 93.74 without a transform", "W-8"],
        ["Max of linear and geometric lookahead", "93.88 / 93.71 / 93.76", "W-11"],
        ["Full-run knot polish on a single seed", "fixes ramp, not wave", "W-9"],
        ["Sloped first segment (first_knot)", "wave 93.45-93.61", "W-12"],
        ["Cheap pass 1 (20% lookahead), then pass 2", "93.64", "W-17"],
        ["Freeze LRs that look flat in the first scan", "scenario B ramp 93.13", "W-16"],
    ], [8.0 * cm, 6.5 * cm, 2.5 * cm]),

    P("7. Open problems", "h1"),
    bullets([
        "Batch sizes 500 and 125 fall short of the new baselines. The gap is in the search, not the "
        "representation (F3). At batch size 125 the search's own score exceeds its held-out score by about "
        "0.2 pp, a sign of remaining winner's curse.",
        "The spread between search seeds is 0.1-0.4 pp, about the size of the differences between the best "
        "variants; 2-4 search seeds rank them only coarsely.",
        "Scenario B costs about 3x scenario A, and cost roughly doubles from batch size 2000 to 125.",
        "Only multiplicative transforms were tested.",
    ]),

    P("8. Reproduce", "h1"),
    table([
        ["Task", "Command"],
        ["Run a search", "CIFAR_EXPERIMENTS=auto_a_v1x3m2t_bs2000_ramp CIFAR_SEEDS=0 "
                         "/venv/main/bin/python train_cifar.py (the log ends with auto_result and "
                         "heldout_result lines)"],
        ["Summarize runs", "python research/summarize_auto.py [--since YYYYMMDD_HHMMSS] [name ...]"],
        ["Comparison table", "python research/final_table.py"],
        ["Probes", "research/cooldown_fidelity.py (F1, F2), research/overfit_screen.py (F7)"],
        ["Figures", "python research/plot_transforms.py; python research/plot_lookahead_bias.py"],
        ["This report", "python research/make_report.py"],
    ], [3.2 * cm, 13.8 * cm]),
    PageBreak(),
    *report_findings.appendices(P, table),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "LR-schedule search for CIFAR-10 airbench")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="LR-schedule search for CIFAR-10 airbench",
                        author="Roland Gao")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)

"""Build lookahead_joint_study.pdf: live results of the two studies the user asked for on 2026-10-23.

A. research/hparam_lookahead.py (for paper Section 3): one-interval lookaheads for each of the four LRs.
B. research/joint_search.py: joint 3^5 - 1 neighbourhood search of the 5-constant-interval conv schedule.
B2. research/plan_refine_search.py: the same schedule class refined interval by interval (ICS pass 2).
Every number is read from the logs; a lookahead cell appears once all its candidates are logged, a joint
search is marked "running" until it converges.

python research/make_lookahead_joint_report.py
"""
import glob
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
import hparam_lookahead_study as hl  # noqa: E402

OUT = os.path.join(REPO, "lookahead_joint_study.pdf")
PLOTS = os.path.join(REPO, "plots", "lookahead_joint")
os.makedirs(PLOTS, exist_ok=True)

INK, INK2, MUTED, GRID, SURFACE = "#1f1f1e", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"  # slots 1-5 (validated)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

TRANSFORMS = ["none", "ramp", "rampdown", "wave", "waveinv", "ramp8"]
TLABEL = {"none": "no transform", "ramp": "ramp x0.25->x4", "rampdown": "reverse ramp x4->x0.25",
          "wave": "wave x2^sin", "waveinv": "inverted wave", "ramp8": "steep ramp x1/8->x8"}
GROUPS = ["conv", "head", "norm_bias", "whiten_bias"]
GLABEL = {"conv": "conv", "head": "head", "norm_bias": "norm bias", "whiten_bias": "whitening bias"}
LOOKS = {  # variant: (color, linestyle, marker, label)
    "truth": (INK, "-", "o", "truth (rest of run on the plan)"),
    "lin_main": (BLUE, "--", "v", "linear to 0, main LR only"),
    "lin_all": (ORANGE, "--", "^", "linear to 0, all LRs"),
    "s_main": (AQUA, "-", "s", "searched m, main LR only"),
    "s_tied": (YELLOW, "-", "D", "searched m main + tied m others"),
    "s_each": (MAGENTA, "-", "P", "searched m for each LR"),
}
PT = timezone(timedelta(hours=-7))


# ------------------------------------------------------------------ data
def queue_state():
    q = os.path.join(REPO, "queue")
    count = {d: [f for f in os.listdir(os.path.join(q, d)) if f[:4] in ("020_", "021_", "022_", "023_", "024_")] if os.path.isdir(os.path.join(q, d))
             else [] for d in ("pending", "running", "done", "failed")}
    return count


def load_joint():
    runs = {}
    for path in sorted(glob.glob(os.path.join(REPO, "logs", "research_joint_search_*_bs2000_s0_*.log"))):
        name = os.path.basename(path)[len("research_joint_search_"):].split("_bs2000")[0]
        r = dict(evals=[], moves=[], done=None, setup=None)
        for line in open(path):
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            if rec["kind"] == "setup":
                r["setup"] = rec
            elif rec["kind"] == "eval":
                r["evals"].append(rec)
            elif rec["kind"] == "move":
                r["moves"].append(rec)
            elif rec["kind"] == "done":
                r["done"] = rec
        if r["setup"] is not None and r["moves"]:
            runs[name] = r  # the newest log per transform wins (sorted by timestamp)
    return runs


def load_shapes():
    cells = {}  # (length, offset) -> eval record (newest log wins)
    for path in sorted(glob.glob(os.path.join(REPO, "logs", "research_cooldown_shapes_conv_k1_L*_bs2000_s0_*.log"))):
        for line in open(path):
            if line.startswith("{"):
                rec = json.loads(line)
                if rec["kind"] == "eval":
                    cells[(rec["length"], rec["offset"])] = rec
    return cells


def load_refine(kind):
    """research/plan_refine_search.py logs per transform (newest wins): setup, evals, commits, pass_dones, done.
    kind: plan_refine (one step, smaller first), plan_climb (--climb: both sides, then unbounded), plan2_* (2 passes)."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(REPO, "logs", f"research_{kind}_*_bs2000_s0_*.log"))):
        name = os.path.basename(path)[len(f"research_{kind}_"):].split("_bs2000")[0]
        r = dict(evals=[], commits=[], pass_dones=[], done=None, setup=None)
        for line in open(path):
            if line.startswith("{"):
                rec = json.loads(line)
                if rec["kind"] in ("setup", "done"):
                    r[rec["kind"]] = rec
                else:
                    r[rec["kind"] + "s"].append(rec)
        if r["setup"] is not None:
            runs[name] = r
    return runs


def box_best(r, plan):
    """The best schedule the joint search evaluated within one grid step of plan in every interval."""
    def index(v):
        return round(math.log(v) / math.log(0.6))
    box = [e for e in r["evals"] if all(abs(index(a) - index(b)) <= 1 for a, b in zip(e["values"], plan))]
    return max(box, key=lambda e: e["score"]), len(box)


shapes = load_shapes()
VARIANTS = (("plan_refine", "one step, 1 pass"), ("plan2_refine", "one step, 2 passes"),
            ("plan_climb", "both sides + climb, 1 pass"), ("plan2_climb", "both sides + climb, 2 passes"))
variants = {kind: load_refine(kind) for kind, _ in VARIANTS}
refine = variants["plan_refine"]
look = {tr: {cell: r for cell, r in hl.summary(2000, tr).items() if len(r["offsets"]) == 5}  # complete cells only
        for tr in TRANSFORMS}
joint = load_joint()
baseline_s0 = joint["none"]["setup"]["do_nothing"] * 100 if "none" in joint else None


# ------------------------------------------------------------------ figures
def fig_lookahead(tr, path):
    out = look[tr]
    fig, axes = plt.subplots(4, 4, figsize=(8.0, 7.4), layout="constrained")
    for i, g in enumerate(GROUPS):
        for k in range(4):
            ax = axes[i, k]
            r = out.get((g, k))
            if r is None:
                ax.text(0.5, 0.5, "not finished", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                xs = [r["c"][o] for o in r["offsets"]]
                for v, (color, ls, marker, label) in LOOKS.items():
                    row = r["rows"].get(v)
                    if row is None:
                        continue
                    top = max(row["means"].values())
                    ax.plot(xs, [max(-1.0, row["means"][o] - top) for o in r["offsets"]], color=color, ls=ls,
                            lw=1.3 if v != "truth" else 1.8, marker=marker, markersize=3.2, label=label)
                ax.set_xscale("log")
                ax.minorticks_off()
                ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=6)
                ax.set_ylim(-1.05, 0.08)
            if i == 0:
                ax.set_title(f"interval {k + 1}")
            if k == 0:
                ax.set_ylabel(f"{GLABEL[g]}\nscore - its best (pp)")
            if i == 3:
                ax.set_xlabel("candidate LR (specified)")
    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, lab
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


SHAPES = {  # variant: (color, linestyle, marker, label)
    "truth": (INK, "-", "o", "truth"),
    "lin_main": (BLUE, "--", "v", "linear to 0, conv only (old)"),
    "lin_all": (ORANGE, "--", "^", "linear to 0, all LRs (old)"),
    "s_main": (AQUA, "--", "s", "searched m, conv only (old)"),
    "s_tied": (YELLOW, "--", "D", "searched m + tied m others (old)"),
    "s_each": (MAGENTA, "--", "P", "searched m each LR (old)"),
    "pow": (BLUE, "-", "o", "power (1-t)^p, p searched"),
    "exp": (ORANGE, "-", "o", "exponential q^t, q searched"),
    "powc": (AQUA, "-", "o", "power to end q, p and q searched"),
}
LENGTHS = (20, 40, 60, 80)
OFFSETS = (-2, -1, 0, 1, 2)


def shape_cell(length):
    """{offset: record} if all candidates finished at this cooldown length, else None."""
    cell = {o: shapes.get((length, o)) for o in OFFSETS}
    return None if None in cell.values() else cell


def fig_shapes(path):
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 3.1), sharey=True, layout="constrained")
    for ax, length in zip(axes, LENGTHS):
        ax.set_title(f"cooldown {length} steps")
        cell = shape_cell(length)
        if cell is None:
            ax.text(0.5, 0.5, "not finished", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_xticks([])
            continue
        order = sorted(OFFSETS, key=lambda o: cell[o]["c"])
        xs = [cell[o]["c"] for o in order]
        for v, (color, ls, marker, label) in SHAPES.items():
            top = max(cell[o][v] for o in order)
            ax.plot(xs, [max(-1.0, 100 * (cell[o][v] - top)) for o in order], color=color, ls=ls, marker=marker,
                    lw=1.8 if v == "truth" else 1.2, markersize=3.2, label=label)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=6.5)
        ax.set_xlabel("candidate conv LR")
        ax.set_ylim(-1.05, 0.08)
    axes[0].set_ylabel("score - its best (pp)")
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes), key=lambda hl_: len(hl_[0]))
    fig.legend(handles, labels, frameon=False, fontsize=6.8, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def best_eval(r):
    return max(r["evals"], key=lambda e: e["score"])


def fig_joint(path):
    sys.path.insert(0, REPO)
    import train_cifar as t  # noqa: E402 (slow import, only for the transform factors)
    names = [tr for tr in TRANSFORMS if tr in joint]
    fig, axes = plt.subplots(2, 6, figsize=(8.0, 3.6), layout="constrained", squeeze=False)
    for j, tr in enumerate(TRANSFORMS):
        ax_s, ax_p = axes[0, j], axes[1, j]
        ax_s.set_title(tr, fontsize=7.5)
        if tr not in names:
            for ax in (ax_s, ax_p):
                ax.text(0.5, 0.5, "not started", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        r = joint[tr]
        spec = r["setup"]["transform"]
        f = [1.0 if spec is None else t.lr_transform_factor(spec, s + 0.5, 200) for s in range(200)]
        ax_s.plot(range(200), [0.22 * (1 - s / 200) for s in range(200)], color=INK, lw=1.4, label="baseline")
        for values, color, ls, label in ((r["moves"][0]["values"], MUTED, "--", "start (grid oracle)"),
                                         (best_eval(r)["values"], BLUE, "-", "best evaluated so far")):
            ax_s.plot(range(200), [values[min(s // 40, 4)] * f[s] for s in range(200)], color=color, ls=ls, lw=1.2,
                      label=label)
        ax_s.set_xlabel("step", fontsize=6.5)
        best = []
        for e in r["evals"]:
            best.append(max(best[-1] if best else -1, 100 * e["score"]))
        ax_p.plot(range(1, len(best) + 1), best, color=BLUE, lw=1.4)
        if baseline_s0 is not None:
            ax_p.axhline(baseline_s0, color=INK, lw=1, ls=":")
        ax_p.set_xlabel("full runs", fontsize=6.5)
    axes[0, 0].set_ylabel("applied conv LR")
    axes[1, 0].set_ylabel("best score so far (%)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_refine(path):
    sys.path.insert(0, REPO)
    import train_cifar as t  # noqa: E402
    fig, axes = plt.subplots(1, 6, figsize=(8.0, 2.2), layout="constrained", squeeze=False)
    for j, tr in enumerate(TRANSFORMS):
        ax = axes[0, j]
        ax.set_title(tr, fontsize=7.5)
        if tr not in refine or refine[tr]["done"] is None or tr not in joint:
            ax.text(0.5, 0.5, "not done", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        spec = refine[tr]["setup"]["transform"]
        f = [1.0 if spec is None else t.lr_transform_factor(spec, s + 0.5, 200) for s in range(200)]
        ax.plot(range(200), [0.22 * (1 - s / 200) for s in range(200)], color=INK, lw=1.4, label="baseline")
        lines = [(refine[tr]["setup"]["plan"], MUTED, "--", "plan (start)")]
        for kinds, color, label in ((("plan2_refine", "plan_refine"), ORANGE, "one step"),
                                    (("plan2_climb", "plan_climb"), AQUA, "both sides + climb")):
            for kind in kinds:  # the most passes finished
                r = variants[kind].get(tr)
                if r is not None and r["done"] is not None:
                    lines.append((r["done"]["values"], color, "-", f"{label} (best finished)"))
                    break
        for values, color, ls, label in lines + [
                (best_eval(joint[tr])["values"], BLUE, "-", "joint search")]:
            ax.plot(range(200), [values[min(s // 40, 4)] * f[s] for s in range(200)], color=color, ls=ls, lw=1.2,
                    label=label)
        ax.set_xlabel("step", fontsize=6.5)
    axes[0, 0].set_ylabel("applied conv LR")
    handles = {}
    for ax in axes[0]:  # every label any panel has (variants finish at different times)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=7, loc="outside lower center", ncol=5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ document
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.6, leading=12, alignment=TA_JUSTIFY, spaceAfter=4)
S = {
    "body": BODY,
    "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=4),
    "note": ParagraphStyle("note", fontName="Times-Italic", fontSize=8.5, leading=10.5, alignment=TA_CENTER,
                           spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=6, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=10.2, leading=13, spaceBefore=4, spaceAfter=2),
    "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=8.4, leading=10.2, alignment=TA_JUSTIFY,
                              spaceBefore=2, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.8, leading=9.2),
    "cellb": ParagraphStyle("cellb", fontName="Times-Bold", fontSize=7.8, leading=9.2),
}
W = 17.0 * cm


def P(text, style="body"):
    return Paragraph(text, S[style])


def figure(path, caption, width=W):
    img = Image(path)
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return KeepTogether([img, P(caption, "caption")])


def table(rows, widths, caption):
    data = [[Paragraph(str(c), S["cellb" if i == 0 else "cell"]) for c in row] for i, row in enumerate(rows)]
    tb = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    tb.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]))
    return [KeepTogether([P(caption, "caption"), tb]), Spacer(1, 6)]


def sched(values):
    return ", ".join(f"{v:g}" for v in values)


now = datetime.now(timezone.utc)
q = queue_state()
header = [
    P("Interval-2 cooldowns, joint 5-interval search and per-LR lookaheads: live results", "title"),
    P(f"Updated {now.astimezone(PT):%Y-%m-%d %H:%M} PDT ({now:%H:%M} UTC). BS 2000, CIFAR-10 airbench, score = TTA "
      f"test accuracy, one training seed (0). Jobs: {len(q['done'])} done, {len(q['running'])} running, "
      f"{len(q['pending'])} pending, {len(q['failed'])} failed.", "note"),
]
story_b = [
    P("3. Which one-interval lookahead ranks interval candidates like the truth?", "h1"),
    P("<b>Setup.</b> The plan is the baseline run itself, split into 5 intervals of 40 steps. Under a transform, "
      "the multiplier applies to all four LR groups, and the plan is the oracle (each group's baseline divided by "
      "the multiplier), so the plan's applied LRs are the baseline's. For a main LR and interval k, the plan is "
      "followed up to k; there the main LR is replaced by a constant candidate c on the x0.6 grid (the grid value "
      "nearest the plan's mean over the interval, and 1 and 2 grid steps either side; whitening bias, whose plan is "
      "0 from step 75, uses its step-0 plan value), trained for one interval from the same state, and scored by "
      "one more interval of: <b>linear to 0, main LR only</b> (c to 0, the others on the plan); <b>linear to 0, all "
      "LRs</b>; <b>searched m, main LR only</b> (constant m x c); <b>searched m main + tied m others</b> (the other "
      "three at m' x their plan mean over interval k, one m' for all three); <b>searched m for each LR</b>. The "
      "<b>truth</b> is the final score when the run continues on the plan. Each m is searched on the 0.6^k grid "
      "without bounds, from 0.22, one multiplier at a time, moving while a neighbour at k&plusmn;1 or k&plusmn;2 is "
      "better; the result beats all four neighbours. A lookahead decides like the truth when its curve peaks at the "
      "same candidate. One seed: a single-run score has an SD of about 0.12 pp, so differences below ~0.15 pp "
      "between candidates are within noise."),
]
waiting = [TLABEL[tr] for tr in TRANSFORMS if not look[tr]]
for tr in TRANSFORMS:
    out = look[tr]
    if not out:
        continue
    story = story_b
    story.append(P(f"3.{TRANSFORMS.index(tr) + 1} {TLABEL[tr]}", "h2"))
    path = os.path.join(PLOTS, f"lookahead_{tr}.png")
    fig_lookahead(tr, path)
    story.append(figure(path, f"<b>Figure 3.{TRANSFORMS.index(tr) + 1}.</b> {TLABEL[tr]}: each score per candidate "
                              "minus its own best (clipped at -1 pp); rows are the main LR, columns the interval."))
    rows = [["LR, interval", "truth best"] + [LOOKS[v][3] for v in LOOKS if v != "truth"]]
    for (g, k), r in sorted(out.items(), key=lambda kv: (GROUPS.index(kv[0][0]), kv[0][1])):
        if k > 3:
            continue
        cells = [f"{GLABEL[g]}, {k + 1}", f"{r['c'][r['best']]:g}"]
        for v in LOOKS:
            if v == "truth":
                continue
            row = r["rows"].get(v)
            if row is None:
                cells.append("&ndash;")
                continue
            pick = r["c"][row["mean_pick"]]
            txt = f"{pick:g} ({row['regret']:.2f})"
            cells.append(f"<b>{txt}</b>" if row["mean_pick"] == r["best"] else txt)
        rows.append(cells)
    story += table(rows, [2.5 * cm, 1.4 * cm] + [2.62 * cm] * 5,
                   f"<b>Table 3.{TRANSFORMS.index(tr) + 1}.</b> Candidate each lookahead picks and, in brackets, the "
                   "truth it loses against the truth's best (pp); bold = the truth's pick.")
    data = hl.load(2000, tr)
    counts = [["Main LR", "searched m, main only", "main + tied: main m", "main + tied: others' m",
               "each: main m"]]
    for g in GROUPS:
        cs = {key: Counter() for key in ("sm", "tm", "to", "em")}
        for (gg, k, o), seeds in data.items():
            if gg != g or k > 3 or (g, k) not in out:
                continue
            for rec in seeds.values():
                m = rec["searched"]
                if "s_main" in m:
                    cs["sm"][m["s_main"][g]] += 1
                    cs["tm"][m["s_tied"][g]] += 1
                    if "others" in m["s_tied"]:
                        cs["to"][m["s_tied"]["others"]] += 1
                    cs["em"][m["s_each"][g]] += 1
        cells = [GLABEL[g]]
        for key in ("sm", "tm", "to", "em"):
            c = cs[key]
            n = sum(c.values())
            cells.append("&ndash;" if not n else f"0.22 in {c[0.22]}/{n}; else " +
                         ", ".join(f"{m:g}: {x}" for m, x in c.most_common(4) if m != 0.22))
        counts.append(cells)
    story += table(counts, [2.3 * cm] + [3.67 * cm] * 4,
                   f"<b>Table 3.{TRANSFORMS.index(tr) + 1}m.</b> Searched cooldown multipliers over all candidates of "
                   "intervals 1-4: how often 0.22 is chosen, and the most common other values.")
if waiting:
    story_b.append(P("Not finished yet (no complete interval cell): " + "; ".join(waiting) + "."))

story_c = [
    P("1. Conv interval 2: more cooldown shapes and lengths", "h1"),
    P("<b>Setup.</b> The one cell of Section 3 where every cooldown failed: no transform, conv, interval 2 (steps "
      "40-80), candidates 0.047-0.36, where the truth peaks at 0.13 and every one-interval cooldown picked 0.078. "
      "Each candidate is trained for the interval from the plan's state and scored after a cooldown of 20, 40, 60 "
      "or 80 steps (t = i / length, i = 0 .. length-1), with the other LRs on the plan unless stated. Old shapes as "
      "in Section 3; new, conv only: <b>power</b> c (1-t)^p (p = 1 is linear to 0), <b>exponential</b> c q^t (from "
      "c to q c), <b>power to end q</b> c (q + (1-q)(1-t)^p). p is searched on the 0.6^k grid from 1, q and m from "
      "0.22, each unbounded, moving while a k&plusmn;1 or k&plusmn;2 neighbour is better. The truth is recomputed in "
      "every job. One seed (0)."),
]
done_lengths = [length for length in LENGTHS if shape_cell(length)]
if shapes:
    path = os.path.join(PLOTS, "shapes_conv_k1.png")
    fig_shapes(path)
    story_c.append(figure(path, "<b>Figure 1.</b> Score per candidate minus its own best (clipped at -1 pp), per "
                                "cooldown length; a cooldown decides like the truth when it peaks where the truth "
                                "does. Dashed: the Section 3 shapes; solid: the new ones."))
if done_lengths:
    rows = [["Cooldown"] + [f"{length} steps" for length in done_lengths]]
    for v, (_, _, _, label) in SHAPES.items():
        cells = [label]
        for length in done_lengths:
            cell = shape_cell(length)
            truth_best = max(OFFSETS, key=lambda o: cell[o]["truth"])
            pick = max(OFFSETS, key=lambda o: cell[o][v])
            txt = f"{cell[pick]['c']:g} ({100 * (cell[truth_best]['truth'] - cell[pick]['truth']):.2f})"
            cells.append(f"<b>{txt}</b>" if pick == truth_best else txt)
        rows.append(cells)
    truths = "; ".join(f"{length} steps: " + ", ".join(
        f"{shape_cell(length)[o]['c']:g} {100 * shape_cell(length)[o]['truth']:.2f}"
        for o in sorted(OFFSETS, key=lambda o: shape_cell(length)[o]["c"])) for length in done_lengths)
    story_c += table(rows, [5.0 * cm] + [2.6 * cm] * len(done_lengths),
                     "<b>Table 1.</b> Candidate each cooldown picks and, in brackets, the truth it loses against the "
                     f"truth's best (pp); bold = the truth's pick. Truth per candidate (recomputed per job): {truths}.")
    rows = [["Shape, length"] + [f"c = {shape_cell(done_lengths[0])[o]['c']:g}"
                                 for o in sorted(OFFSETS, key=lambda o: shape_cell(done_lengths[0])[o]["c"])]]
    for v in ("s_main", "pow", "exp", "powc"):
        for length in done_lengths:
            cell = shape_cell(length)
            rows.append([f"{SHAPES[v][3]}, {length}"] + [
                ", ".join(f"{name}={val:g}" for name, val in cell[o]["searched"][v].items())
                for o in sorted(OFFSETS, key=lambda o: cell[o]["c"])])
    story_c += table(rows, [5.0 * cm] + [2.4 * cm] * 5, "<b>Table 1p.</b> Searched cooldown parameters per candidate.")
running_lengths = [length for length in LENGTHS if length not in done_lengths]
if running_lengths:
    story_c.append(P("Not finished yet: cooldowns of " + ", ".join(map(str, running_lengths)) + " steps ("
                     f"{sum(1 for k in shapes if k[0] in running_lengths)} of {5 * len(running_lengths)} candidates "
                     "done)."))
story = story_a = [
    P("2. Capacity of the 5-constant-interval conv schedule: joint neighbourhood search", "h1"),
    P("<b>Setup.</b> Conv LR only (the other groups on the baseline); conv's specified schedule is 5 constant "
      "intervals on the x0.6 grid, and the transform multiplies conv's applied LR. One search seed (0). Each "
      "iteration trains all 3^5 - 1 = 242 joint neighbours (every interval -1, 0 or +1 grid step) as full runs, "
      "moves to the best if it beats the current schedule, and repeats until none does. Start: the oracle "
      "(baseline / multiplier) reduced to its 5 interval means, snapped to the grid. The result is the winner's "
      "search score (no replay); baseline and do-nothing are single runs on the same seed."),
]
rows = [["Transform", "do nothing", "start (score)", "best evaluated (score)", "&Delta; vs baseline", "iterations",
         "runs", "status"]]
for tr in TRANSFORMS:
    r = joint.get(tr)
    if r is None:
        rows.append([TLABEL[tr], "&ndash;", "&ndash;", "&ndash;", "&ndash;", "&ndash;", "&ndash;", "queued"])
        continue
    start, best = r["moves"][0], best_eval(r)
    delta = "&ndash;" if baseline_s0 is None else f"{100 * best['score'] - baseline_s0:+.2f}"
    rows.append([TLABEL[tr], f"{100 * r['setup']['do_nothing']:.2f}",
                 f"{sched(start['values'])} ({100 * start['score']:.2f})",
                 f"{sched(best['values'])} ({100 * best['score']:.2f})", delta, str(len(r["moves"]) - 1),
                 str(len(r["evals"])), "converged" if r["done"] else "running"])
story += table(rows, [2.5 * cm, 1.4 * cm, 3.6 * cm, 3.6 * cm, 1.5 * cm, 1.3 * cm, 1.1 * cm, 1.5 * cm],
               "<b>Table 2.</b> Joint search per transform. Baseline (no transform, seed 0): " +
               ("&ndash;" if baseline_s0 is None else f"{baseline_s0:.2f}") +
               ". Iterations = moves made so far; best evaluated = the best full run so far (the schedule the "
               "search moves to next); a running search's best is final only once it converges.")
if joint:
    path = os.path.join(PLOTS, "joint.png")
    fig_joint(path)
    story.append(figure(path, "<b>Figure 2.</b> Top: applied conv LR of the start and the best evaluated schedule "
                              "against the baseline. Bottom: best score among the full runs so far (dotted: "
                              "baseline, seed 0)."))

story += [
    P("<b>Interval-by-interval refinement (ICS pass 2) on the same plan.</b> Same schedule class, transforms and "
      "seed; the plan is the joint search's start (no shift). For each interval in order, the first included, from "
      "the state after the intervals already committed: score the plan's value, then one grid step smaller and, only "
      "if that is not better, one grid step larger; a candidate's score is the interval at the candidate plus the "
      "rest of the run on the plan (a full run's TTA). The best is committed. No two-continuation average for the "
      "last interval; the result is the last committed score (no replay). The joint search's first iteration "
      "trained every schedule within one grid step of the same plan, so its best there (\"best in the box\") "
      "is what a perfect one-step search could reach (research/plan_refine_search.py). <b>Both sides + climb</b> "
      "(--climb): score the plan's value and both one-step neighbours, then keep stepping toward the better "
      "neighbour while the score improves, with no step limit; otherwise identical. <b>2 passes</b> (--passes 2): "
      "a second pass from initialisation whose plan, both the candidates' centre and the lookahead tail, is the "
      "first pass's committed schedule (it starts from the first pass's result, so it cannot end lower)."),
]
def refine_cell(r, with_schedule):
    if r is None:
        return "queued"
    if r["done"] is None:
        return f"running (pass {len(r['pass_dones']) + 1}, interval {len(r['commits']) % 5 + 1})"
    d = r["done"]
    score = f"<b>{100 * d['score']:.2f}</b> ({d['runs']:.1f} runs)"
    return f"{sched(d['values'])}: {score}" if with_schedule else score


done_tr = [tr for tr in TRANSFORMS if tr in joint and tr in refine and refine[tr]["done"] is not None]
rows = [["Transform", "do nothing", "plan"] + [label for _, label in VARIANTS] + ["best in &plusmn;1 box",
                                                                                 "joint search"]]
for tr in done_tr:
    jr = joint[tr]
    bb, _ = box_best(jr, refine[tr]["done"]["plan"])
    rows.append([TLABEL[tr], f"{100 * jr['setup']['do_nothing']:.2f}", f"{100 * jr['moves'][0]['score']:.2f}"]
                + [refine_cell(variants[kind].get(tr), False) for kind, _ in VARIANTS]
                + [f"{100 * bb['score']:.2f}", f"{100 * best_eval(jr)['score']:.2f} ({len(jr['evals'])} runs)"])
story += table(rows, [2.0 * cm, 1.2 * cm, 1.0 * cm] + [2.0 * cm] * 4 + [1.5 * cm, 2.0 * cm],
               "<b>Table 2b.</b> Interval refinement against the joint search (TTA %, seed 0). Best in box = the best "
               "of the 243 schedules within one grid step of the plan (all trained by the joint search). Runs = "
               "training steps / 200 (a 2-pass search's runs include its first pass); the refinement re-scores the "
               "committed value at each interval.")
rows = [["Transform"] + [label for _, label in VARIANTS] + ["joint search"]]
for tr in done_tr:
    rows.append([TLABEL[tr]] + [refine_cell(variants[kind].get(tr), True).split(":")[0]
                                for kind, _ in VARIANTS] + [sched(best_eval(joint[tr])["values"])])
story += table(rows, [2.0 * cm] + [3.0 * cm] * 5, "<b>Table 2c.</b> The committed schedules (conv LR per interval).")
if refine:
    path = os.path.join(PLOTS, "refine.png")
    fig_refine(path)
    story.append(figure(path, "<b>Figure 2b.</b> Applied conv LR: baseline, plan, the interval refinement's schedule "
                              "and the joint search's winner."))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="Per-LR lookaheads and joint 5-interval search",
                        author="Roland Gao")
doc.build(header + story_c + [PageBreak()] + story_a + [PageBreak()] + story_b, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}: {doc.page} pages")

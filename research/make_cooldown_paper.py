"""Build cooldown_study_paper.pdf: an 8-page technical report of the interval-cooldown-search study.
The full record (34 experiments) is cooldown_study.pdf; "Exp. N" below refers to its experiments.
Every table and figure is computed from the logs, except the paired probe effects quoted from the
full study (marked with their experiment number).

python research/make_cooldown_paper.py
"""
import glob
import math
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
import ics_study as ics  # noqa: E402
import hparam_lookahead_study as hls  # noqa: E402
import lookahead_study as las  # noqa: E402

OUT = os.path.join(REPO, "cooldown_study_paper.pdf")
PLOTS = os.path.join(REPO, "plots")
os.makedirs(PLOTS, exist_ok=True)

INK, INK2, MUTED, GRID, SURFACE = "#1f1f1e", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4 (validated)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

# ------------------------------------------------------------------ data
STEPS = {2000: 200, 500: 800, 125: 3200}
BASE_CONV = {2000: 0.22, 500: 0.13, 125: 0.047}
refs = ics.references()
TARGET = {bs: ics.target(refs, bs) for bs in ics.BATCH_SIZES}
v29 = ics.load("v29")
probes = ics.averaged()

# held-out results of every ICS version, from one pass over the logs
HELD = re.compile(r"^heldout_result run=ics_(v\d+)_([ab])_bs(\d+)(\S*?)_seed(\d+) .*?tta_val_acc_mean=(\S+)", re.M)
held_by_version = defaultdict(lambda: defaultdict(dict))  # version -> (sc, bs, tr) -> seed -> TTA
fresh_base = {}
for path in sorted(glob.glob(os.path.join(REPO, "logs", "cifar_baseline_*.log"))):
    text = open(path, errors="replace").read()
    if os.path.basename(path)[len("cifar_baseline_"):][:15] >= "20261017_000000":
        for m in HELD.finditer(text):
            held_by_version[m.group(1)][(m.group(2), int(m.group(3)), m.group(4))][int(m.group(5))] = \
                100 * float(m.group(6))
    for m in re.finditer(r"^seed_summary run=baseline_bs(\d+) seeds=\[200, 201, 202, 203, 204, 205, 206, 207\] "
                         r".*?tta_val_acc_mean=(\S+)", text, re.M):
        fresh_base[int(m.group(1))] = 100 * float(m.group(2))


def summary(version, sc, transforms=ics.TRANSFORMS):
    """(mean delta vs the baseline over cells, cells within 0.1 pp, number of cells, seeds per cell)"""
    deltas, seeds = [], set()
    for bs in ics.BATCH_SIZES:
        for tr in transforms:
            held = held_by_version[version].get((sc, bs, tr), {})
            if held:
                deltas.append(statistics.mean(held.values()) - TARGET[bs])
                seeds.add(len(held))
    if not deltas:
        return None
    return statistics.mean(deltas), sum(d >= -0.1 for d in deltas), len(deltas), max(seeds)


def cell(sc, bs, tr):
    held = [r["heldout"] for r in v29.get((sc, bs, tr), {}).values() if "heldout" in r]
    fresh = [probes[k][0] for k in (f"fresh_v29_{sc}_bs{bs}{tr}_s{s}" for s in range(12)) if k in probes]
    return held, fresh


# ------------------------------------------------------------------ figures
def linear_through(values, total):
    """the committed schedule: piecewise linear through the interval midpoints, ends extrapolated (clipped at 0)"""
    n, length = len(values), total / len(values)
    xs = [0.0] + [(i + 0.5) * length for i in range(n)] + [float(total)]
    ys = ([max(0.0, values[0] + (values[0] - values[1]) / 2)] + list(values)
          + [max(0.0, values[-1] - (values[-2] - values[-1]) / 2)])
    out = []
    for s in range(total + 1):
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            if x0 <= s <= x1:
                out.append(y0 + (y1 - y0) * (s - x0) / (x1 - x0))
                break
    return out


def factor(tr, s, total):
    x = s / total
    return {"": 1.0, "_ramp": 4 ** (2 * x - 1), "_rampdown": 0.25 ** (2 * x - 1),
            "_wave": 2 ** math.sin(2 * math.pi * x), "_waveinv": 2 ** math.sin(2 * math.pi * (x + 0.5))}[tr]


def fig_schedules(path):
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4), sharey=True, layout="constrained")
    bs, total = 2000, STEPS[2000]
    steps = list(range(total + 1))
    base = [BASE_CONV[bs] * (1 - s / total) for s in steps]
    for ax, tr in zip(axes, ics.TRANSFORMS):
        ax.plot(steps, [b * factor(tr, s, total) for s, b in zip(steps, base)], color=MUTED, lw=1.2, ls="--",
                label="do nothing (baseline spec, transformed)")
        for i, (_, r) in enumerate(sorted(v29[("a", bs, tr)].items())):
            spec = linear_through(r["schedules"]["conv"], total)
            ax.plot(steps, [v * factor(tr, s, total) for s, v in zip(steps, spec)], color=BLUE, lw=0.8, alpha=0.45,
                    label="ICS, 12 searches" if i == 0 else None)
        ax.plot(steps, base, color=INK, lw=1.6, label="baseline (target)")
        ax.set_title(ics.TRANSFORM_LABELS[tr])
        ax.set_xlabel("step")
    axes[0].set_ylabel("applied conv LR")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_lookahead(path):
    look = hls.summary(2000, "none")
    styles = {"truth": (INK, "-", "o", "truth (rest of run on the plan)"),
              "lin_main": (BLUE, "--", "v", "linear to 0, conv only"), "lin_all": (ORANGE, "--", "^", "linear to 0, all LRs"),
              "s_main": (AQUA, "-", "s", "searched m, conv only"),
              "s_tied": (YELLOW, "-", "D", "searched m conv + tied m others"),
              "s_each": ("#e87ba4", "-", "P", "searched m each LR")}
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.65), sharey=True, layout="constrained")
    for k, ax in enumerate(axes):
        r = look[("conv", k)]
        xs = [r["c"][o] for o in r["offsets"]]
        for v, (color, ls, marker, label) in styles.items():
            top = max(r["rows"][v]["means"].values())
            ax.plot(xs, [max(-1.0, r["rows"][v]["means"][o] - top) for o in r["offsets"]], color=color, ls=ls,
                    lw=1.8 if v == "truth" else 1.2, marker=marker, markersize=3, label=label)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=6.5)
        ax.set_title(f"interval {k + 1}")
        ax.set_xlabel("candidate conv LR")
    axes[0].set_ylabel("score - its best (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_results(path):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65), sharey=True, layout="constrained")
    for ax, bs in zip(axes, ics.BATCH_SIZES):
        ax.axhspan(-0.1, 0.1, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        for j, (sc, color) in enumerate((("a", BLUE), ("b", ORANGE))):
            for i, tr in enumerate(ics.TRANSFORMS):
                held, fresh = cell(sc, bs, tr)
                x = i + (j - 0.5) * 0.3
                d = [h - TARGET[bs] for h in held]
                m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
                first = i == 0 and bs == 2000
                ax.errorbar(x, m, yerr=1.96 * se, fmt="o", color=color, ms=4, capsize=2, lw=1,
                            label=f"{ics.SCENARIO_LABELS[sc]}, seeds 100-107" if first else None)
                if fresh and bs in fresh_base:
                    ax.plot(x + 0.1, statistics.mean(fresh) - fresh_base[bs], marker="D", ms=3.2, color=color,
                            mfc=SURFACE, ls="none", label=f"{ics.SCENARIO_LABELS[sc]}, fresh seeds 200-207" if first else None)
        ax.set_xticks(range(4), [ics.TRANSFORM_LABELS[t] for t in ics.TRANSFORMS], fontsize=7)
        ax.set_title(f"batch size {bs} (baseline {TARGET[bs]:.2f})")
    axes[0].set_ylabel("held-out TTA - baseline (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


MAIN_LINE = [  # (label, conv-only version, 4-LR version)
    ("v1: one pass, rest-of-run linear lookahead", "v1", "v1"),
    ("v2: short-cooldown pass 1 + rest-of-plan pass 2", "v2", None),
    ("v10: + 10-interval expansion, unscaled plan tail", "v10", "v10"),
    ("v16: + pass-1 plan shifted one grid step up", "v16", "v16"),
    ("v19 / v24: 4 LRs: joint level, shift conv only", "v19", "v24"),
    ("v26 / v27: + two half-budget searches averaged", "v26", "v27"),
    ("v28: + linear commit (no expansion)", "v28", "v28"),
    ("v29: + pass 2 refines interval 1 (final)", "v29", "v29"),
]


def fig_trajectory(path):
    fig, ax = plt.subplots(figsize=(8.0, 2.3), layout="constrained")
    xs = range(len(MAIN_LINE))
    ax.axhspan(-0.1, 0.0, color=GRID, alpha=0.7, lw=0)
    series = {}
    for k, sc in enumerate(("a", "b")):
        series[sc] = {}
        for x, row in zip(xs, MAIN_LINE):
            s = summary(row[1 + k], sc) if row[1 + k] else None
            if s:
                series[sc][x] = (s[0], s[1], s[2])
    for sc, color, label in (("a", BLUE, "conv only"), ("b", ORANGE, "4 LRs")):
        other = series["b" if sc == "a" else "a"]
        pts = sorted(series[sc].items())
        ax.plot([x for x, _ in pts], [max(-0.45, v[0]) for _, v in pts], color=color, lw=1.6, marker="o", ms=4,
                label=label)
        for x, (m, within, n) in pts:
            above = x not in other or m >= other[x][0]
            txt = f"{within}/{n}" + ("" if m >= -0.45 else f" ({m:+.2f})")
            ax.annotate(txt, (x, max(-0.45, m)), textcoords="offset points", xytext=(0, 6 if above else -12),
                        ha="center", fontsize=6.3, color=color)
    ax.set_ylim(-0.5, 0.05)
    ax.set_xticks(list(xs), [r[0].split(":")[0] for r in MAIN_LINE], fontsize=7)
    ax.set_ylabel("mean held-out TTA - baseline (pp)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_ceiling(path):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.45), sharey=True, layout="constrained")
    marks = (("lin5_oracle", "oracle, 5 interval means joined linearly", INK, "s"),
             ("lin5g_oracle", "the same, snapped to the x0.6 grid", MUTED, "^"),
             (None, "ICS v29 (mean of 12 searches)", BLUE, "o"))
    for ax, bs in zip(axes, ics.BATCH_SIZES):
        ax.axhspan(-0.1, 0.1, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        for j, (key, label, color, marker) in enumerate(marks):
            for i, tr in enumerate(ics.TRANSFORMS):
                if key:
                    r = probes.get(f"{key}_a_bs{bs}{tr}")
                    y = None if r is None else r[0] - TARGET[bs]
                else:
                    y = statistics.mean(cell("a", bs, tr)[0]) - TARGET[bs]
                if y is not None:
                    ax.plot(i + (j - 1) * 0.22, y, marker=marker, color=color, ms=4.5, ls="none",
                            label=label if (i == 0 and bs == 2000) else None)
        ax.set_xticks(range(4), [ics.TRANSFORM_LABELS[t] for t in ics.TRANSFORMS], fontsize=7)
        ax.set_title(f"batch size {bs}, conv only")
    axes[0].set_ylabel("held-out TTA - baseline (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_waveinv(path):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.4), sharey=False, layout="constrained")
    for ax, bs in zip(axes, ics.BATCH_SIZES):
        total = STEPS[bs]
        steps = list(range(total + 1))
        base = [BASE_CONV[bs] * (1 - s / total) for s in steps]
        ax.plot(steps, [b * factor("_waveinv", s, total) for s, b in zip(steps, base)], color=MUTED, lw=1.2,
                ls="--", label="do nothing (baseline spec, transformed)")
        for i, (_, r) in enumerate(sorted(v29[("a", bs, "_waveinv")].items())):
            spec = linear_through(r["schedules"]["conv"], total)
            ax.plot(steps, [v * factor("_waveinv", s, total) for s, v in zip(steps, spec)], color=BLUE, lw=0.8,
                    alpha=0.5, label="ICS v29, 8 searches" if i == 0 else None)
        ax.plot(steps, base, color=INK, lw=1.6, label="baseline (target)")
        ax.axvspan(0.2 * total, 0.4 * total, color=GRID, alpha=0.8, lw=0, zorder=0)
        ax.set_title(f"inverted wave, batch size {bs}")
        ax.set_xlabel("step")
    axes[0].set_ylabel("applied conv LR")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


paths = {k: os.path.join(PLOTS, f"paper_{k}.png")
         for k in ("schedules", "lookahead", "results", "trajectory", "ceiling", "waveinv")}
fig_ceiling(paths["ceiling"])
fig_waveinv(paths["waveinv"])
fig_schedules(paths["schedules"])
fig_lookahead(paths["lookahead"])
fig_results(paths["results"])
fig_trajectory(paths["trajectory"])

# ------------------------------------------------------------------ document
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=4)
S = {
    "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=16.5, leading=20, alignment=TA_CENTER, spaceAfter=6),
    "author": ParagraphStyle("author", fontName="Times-Roman", fontSize=10.5, leading=13, alignment=TA_CENTER),
    "note": ParagraphStyle("note", fontName="Times-Italic", fontSize=8.5, leading=10.5, alignment=TA_CENTER,
                           spaceAfter=10),
    "abstract": ParagraphStyle("abstract", parent=BODY, fontSize=9.5, leading=11.8, leftIndent=1.2 * cm,
                               rightIndent=1.2 * cm),
    "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=7, spaceAfter=3),
    "body": BODY,
    "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=8.6, leading=10.4, alignment=TA_JUSTIFY,
                              spaceBefore=2, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.3, leading=9.8),
    "cellb": ParagraphStyle("cellb", fontName="Times-Bold", fontSize=8.3, leading=9.8),
    "algo": ParagraphStyle("algo", fontName="Times-Roman", fontSize=9.2, leading=11.4, leftIndent=0.4 * cm,
                           firstLineIndent=-0.4 * cm, spaceAfter=1.5),
    "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=8.6, leading=10.4, leftIndent=0.5 * cm,
                          firstLineIndent=-0.5 * cm, spaceAfter=1),
}
W = 17.0 * cm


def P(text, style="body"):
    return Paragraph(text, S[style])


def figure(path, caption, width=W):
    img = Image(path)
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return KeepTogether([img, P(caption, "caption")])


def table(rows, widths, caption, rules=()):
    data = [[Paragraph(c, S["cellb" if i == 0 else "cell"]) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    style = [("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
             ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    style += [("LINEBELOW", (0, r), (-1, r), 0.3, colors.HexColor(MUTED)) for r in rules]
    t.setStyle(TableStyle(style))
    return KeepTogether([P(caption, "caption"), t, Spacer(1, 6)])


def fmt(d):
    return f"{d:+.2f}".replace("-", "&minus;")


# Table 2: main results
t1 = [["", *[ics.TRANSFORM_LABELS[t] for t in ics.TRANSFORMS]]]
for bs in ics.BATCH_SIZES:
    for sc in ics.SCENARIOS:
        row = [f"BS {bs}, {ics.SCENARIO_LABELS[sc]}"]
        for tr in ics.TRANSFORMS:
            held, fresh = cell(sc, bs, tr)
            m = statistics.mean(held)
            txt = f"{m:.2f} ({fmt(m - TARGET[bs])}"
            if fresh and bs in fresh_base:
                txt += f" / {fmt(statistics.mean(fresh) - fresh_base[bs])}"
            txt += ")"
            row.append(f"<b>{txt}</b>" if m - TARGET[bs] >= -0.1 else txt)
        t1.append(row)
sa, sb = summary("v29", "a"), summary("v29", "b")
fresh_d = {sc: [statistics.mean(cell(sc, bs, tr)[1]) - fresh_base[bs] for bs in ics.BATCH_SIZES
                for tr in ics.TRANSFORMS if cell(sc, bs, tr)[1]] for sc in ics.SCENARIOS}
conv_none = [h for h in cell("a", 2000, "")[0]]

# Table 3: algorithm variants (each a change to v29), from the logs
VARIANTS = [
    ("ICS v29 (final; 12 searches per setting)", "v29"),
    ("one full-budget search instead of two halves averaged (v30)", "v30"),
    ("budget 15 runs (v37)", "v37"),
    ("budget 15 runs, pass-1 climbs start on the trend (v39)", "v39"),
    ("budget 10 runs (v36)", "v36"),
    ("4 intervals (v40)", "v40"),
    ("8 intervals (v41)", "v41"),
    ("pass 2 on a finer x0.775 grid (v33)", "v33"),
    ("a third pass-1 search instead of pass 2 (v32)", "v32"),
    ("milder pass-1 cooldown, 0.36 x (v45)", "v45"),
    ("unbiased (rest-of-run) pass 1, no shift (v46)", "v46"),
    ("4 LRs: head level scanned on its own (v35)", "v35"),
    ("4 LRs: level scan without the cooldown (v34)", "v34"),
    ("4 LRs: common level one grid step lower (v42)", "v42"),
]
t2 = [["Change to ICS v29", "conv only: mean &Delta; (within 0.1 pp)", "4 LRs: mean &Delta; (within 0.1 pp)"]]
for label, v in VARIANTS:
    cells = [label]
    for sc in ics.SCENARIOS:
        s = summary(v, sc)
        cells.append("&ndash;" if s is None else f"{fmt(s[0])} ({s[1]}/{s[2]})")
    t2.append(cells)

# Table 4: transforms held back during development
t3 = [["Setting", "do nothing", "oracle: 5 means linear (on grid)", "ICS v29", "&Delta; vs baseline"]]
for tr, trl in (("_waveinv", "inverted wave"), ("_ramp8", "ramp x1/8 to x8")):
    for bs in ics.BATCH_SIZES:
        for sc in ics.SCENARIOS:
            held = [r["heldout"] for r in v29.get((sc, bs, tr), {}).values() if "heldout" in r]
            dn = refs.get(f"transformed_{sc}{tr}_bs{bs}")
            orc = probes.get(f"lin5_oracle_{sc}_bs{bs}{tr}")
            orcg = probes.get(f"lin5g_oracle_{sc}_bs{bs}{tr}")
            m = statistics.mean(held)
            t3.append([f"{trl}, BS {bs}, {ics.SCENARIO_LABELS[sc]}", "&ndash;" if dn is None else f"{dn:.2f}",
                       "&ndash;" if orc is None else f"{orc[0]:.2f} ({orcg[0]:.2f})", f"{m:.2f}",
                       f"<b>{fmt(m - TARGET[bs])}</b>" if m - TARGET[bs] >= -0.1 else fmt(m - TARGET[bs])])
held_back = {sc: summary("v29", sc, ("_waveinv", "_ramp8")) for sc in ics.SCENARIOS}

# Table 1: the baselines' schedules (from the configs)
from common import base_config  # noqa: E402
GROUPS = (("conv", "conv (Muon-style)"), ("head", "head (SGD)"), ("norm_bias", "norm biases (SGD)"),
          ("whiten_bias", "whitening bias (SGD)"))
t0 = [["Batch size (steps)", *[label for _, label in GROUPS]]]
for bs in ics.BATCH_SIZES:
    cfg = base_config(bs)["param_groups"]
    row = [f"{bs} ({STEPS[bs]})"]
    for g, _ in GROUPS:
        n, v0 = cfg[g]["lr_scheduler"][0][0], cfg[g]["lr_scheduler"][0][1]
        row.append(f"{v0:g} &rarr; 0" + ("" if n == STEPS[bs] else f" by step {n}"))
    t0.append(row)
ceiling = {key: statistics.mean(probes[f"{key}_a_bs{bs}{tr}"][0] - TARGET[bs] for bs in ics.BATCH_SIZES
                                for tr in ics.TRANSFORMS) for key in ("lin5_oracle", "lin5g_oracle")}

story = [
    P("Interval Cooldown Search: Recovering Tuned Learning-Rate Schedules within Twenty Training Runs", "title"),
    P("Roland Gao", "author"),
    P("Technical report, October 2026. Experiments designed by the author; runs and analysis carried out with "
      "Claude (Anthropic). Full experimental record: cooldown_study.pdf (34 experiments, cited as Exp. N).", "note"),
    P("<b>Abstract.</b> We ask how to search a learning-rate (LR) schedule online, within the training cost of 20 "
      "ordinary runs, such that the result matches a hand-tuned schedule. The test bed is airbench on CIFAR-10 at "
      "three batch sizes (2000, 500, 125), searching either the convolutional LR or all four LR groups, under "
      "multiplicative transforms of the specified schedule (ramp, reverse ramp, wave) that the search has to undo. "
      "Three measurements shape the method: a short lookahead after each candidate interval systematically prefers "
      "small LRs, while a rest-of-run lookahead ranks candidates like the final result but is expensive; a "
      "single seed's comparison of two candidates is as noisy as two independent runs even when they share a trained "
      "state; and committing constant intervals as a staircase costs accuracy that the same values joined linearly "
      "do not. Interval Cooldown Search (ICS) combines a cheap short-cooldown pass, a one-grid-step bias "
      "correction, a rest-of-run refinement pass, averaging of two half-budget searches, and a linear commit. On "
      f"held-out seeds it matches the baseline within 0.1 pp in {sa[1]} of 12 conv-only settings and {sb[1]} of 12 "
      f"four-LR settings (mean shortfall {-sa[0]:.3f} and {-sb[0]:.3f} pp over 12 search seeds), and this holds on "
      "fresh seeds. Under two transforms withheld during development it matches in only "
      f"{held_back['a'][1] + held_back['b'][1]} of 12 settings, which we analyse as its main failure mode.", "abstract"),

    P("1&nbsp;&nbsp;Introduction", "h1"),
    P("The learning-rate schedule is the hyperparameter a training recipe is most sensitive to, and finding it is "
      "usually the most expensive part of tuning: a grid over schedule shapes and peak values easily costs dozens of "
      "full runs per setting. Online schedule search avoids that by deciding the LR interval by interval during "
      "training: at each interval, train several candidate LRs from the same state, score each, keep the best and "
      "continue. Its known weakness is the horizon of the score. A candidate judged right after its interval, or "
      "after a short lookahead, is judged on short-term progress, which favours small LRs (the short-horizon bias "
      "of Wu et al., 2018); judging it on the rest of the run is faithful but multiplies the cost."),
    P("We study this trade-off under a concrete mandate: find a schedule whose held-out accuracy matches a tuned "
      "baseline, spending at most 20 runs' worth of training steps (every step the search trains counts, including "
      "lookaheads), at three batch sizes, with either one or four LR groups searched, and under <i>transforms</i>. "
      "A transform multiplies whatever schedule the search specifies by a fixed factor that varies over training "
      "(for example x0.25 rising to x4). A search that merely proposes a familiar shape, such as linear decay, fails "
      "under a transform; to succeed it has to discover the specified schedule that undoes it. Transforms thus test "
      "whether the search finds schedules rather than recalling them."),
    P("Our contributions are: (i) measurements of how well cheap lookaheads rank interval candidates, and of how "
      "noisy those decisions are (Section 3); (ii) ICS, a two-pass search with a bias correction, search averaging "
      "and a linear commit (Section 4); (iii) an evaluation over 24 settings with 12 independent searches each and a "
      "fresh-seed check (Section 5); and (iv) ablations of every component and an analysis of where the method "
      "fails (Sections 6-7)."),

    P("2&nbsp;&nbsp;Setup", "h1"),
    P("<b>Task.</b> airbench (Jordan, 2024) on CIFAR-10 for 8 epochs: 200, 800 and 3200 steps at batch sizes 2000, "
      "500 and 125. The network has four trained LR groups: the convolution weights, optimised by a Muon-style "
      "orthogonalised update (Jordan et al., 2024), and the classifier head, the norm biases and the whitening bias, "
      "optimised by SGD with Nesterov momentum. Each batch size has a tuned baseline in which every group decays "
      "linearly to zero (conv from 0.22, 0.13 and 0.047 at BS 2000, 500 and 125). The score is test accuracy with "
      "flip test-time augmentation (TTA)."),
    table(t0, [3.2 * cm, 3.45 * cm, 3.45 * cm, 3.45 * cm, 3.45 * cm],
          "<b>Table 1.</b> The tuned baselines: each group's LR decays linearly to zero over the run (whitening bias: "
          "by the given step, then 0). The whitening and norm weights are frozen."),
    P("<b>Noise and what counts as a match.</b> Over 136 seeds the baseline at BS 2000 has a single-run SD of 0.119 "
      "pp (Exp. 6). Every schedule is evaluated as the mean over training seeds 100-107, which the searches never "
      "use; such a mean has an SE of about 0.045 pp. A setting <i>matches</i> when the held-out mean is at most 0.1 "
      f"pp below the baseline's on the same seeds ({TARGET[2000]:.2f}, {TARGET[500]:.2f}, {TARGET[125]:.2f} at BS "
      "2000, 500, 125). Searches score their candidates with the same test-set metric, by design: the study isolates "
      "the efficiency of the search, and \"held-out\" refers to training seeds (initialisation, data order, "
      "augmentation), not to data."),
    P("<b>Settings.</b> Scenario (a), conv only: the conv LR is searched starting from the baseline's initial value, "
      "the other groups stay on the baseline. Scenario (b), 4 LRs: all four are searched from 1.0 with no prior. "
      "Transforms, with x the fraction of training: none; ramp x4<super>2x-1</super>; reverse ramp "
      "x4<super>1-2x</super>; wave x2<super>sin 2&pi;x</super>. Keeping the baseline's specified schedule under a "
      "transform (\"do nothing\") costs up to 0.5 pp (BS 2000: 93.42, 93.57, 93.72). Three batch sizes, two "
      "scenarios and four transforms give 24 settings. Only conv has a narrow optimum: around the baseline, full "
      "runs tolerate a x165 range of the conv LR before losing 10%, while head and biases only have an upper "
      "blow-up edge roughly 35x above their baseline values (Exp. 7). ICS therefore searches conv's shape and gives "
      "the other groups a level and conv's profile."),

    P("3&nbsp;&nbsp;What decides an interval", "h1"),
    P("We take the baseline run itself as the plan (BS 2000, 5 intervals of 40 steps) and, for each LR and each "
      "interval, replace that LR by a constant candidate on the x0.6 grid (the grid value nearest the plan's mean "
      "over the interval, and two grid steps either side), trained for the interval from the plan's state. The "
      "<i>truth</i> of a candidate is the final score when the run continues on the plan. A lookahead scores the "
      "candidate after one more interval of cooldown: linear to zero, for the searched LR alone or for all four; or "
      "constant at m times the candidate, with m searched on the 0.6<super>k</super> grid, for the searched LR "
      "alone, with one tied m for the other three LRs (at m times their plan mean), or with an m per LR. We run every "
      "transform, on one seed (Exp. 32)."),
    figure(paths["lookahead"],
           "<b>Figure 1.</b> Conv, no transform: score per candidate (BS 2000, seed 0) relative to each score's own "
           "best; a lookahead decides like the truth when its curve peaks where the truth's does. From interval 2 "
           "on, the lookaheads mostly peak one or two grid steps below the truth."),
    P("<b>Short lookaheads are biased toward small LRs, but not everywhere.</b> Without a transform, from interval 2 "
      "on the lookaheads pick one or two grid steps below the truth: in interval 2 all five pick 0.078 where the "
      "truth peaks at 0.13 (0.26 pp lost). Searching the cooldown's level does not remove this, and neither do other "
      "shapes or lengths: of eight cooldown shapes (linear, constant, power, exponential, power to a searched end) "
      "at 20 to 80 steps, 29 of 32 still pick 0.047 or 0.078, and the three that pick 0.13 win by at most 0.03 pp "
      "(Exp. 34). Longer cooldowns shrink the gap: at 80 steps the best shapes rate 0.078 and 0.13 equal, but still "
      "rate 0.22 about 0.3 pp lower, where the truth rates it equal to 0.13. Over six transforms (Section 2's four "
      "and the two of Section 7) and four intervals, "
      "the conv picks match the truth in 8-11 of 24 cells and lose 0.08-0.11 pp on average, lying 0.2-0.3 grid "
      "steps low. The error depends on the transform and the interval far more than on the cooldown: in 16 of 24 "
      "cells all five lookaheads pick the same candidate. So no fixed cooldown factor calibrates it."),
    P("<b>Head and biases barely matter per interval.</b> Across candidates spanning x7.7, the truth varies by 0.21-"
      "0.26 pp on average for head and both biases (1.04 pp for conv), close to one run's noise. Their lookahead "
      "picks scatter by up to four grid steps and lose 0.06-0.13 pp. Searched cooldown multipliers are spread over "
      "0.047-1, with 0.22 chosen for 16-28% of candidates."),
    P("<b>Shared state does not cancel noise.</b> One seed's difference between two candidates' truths, trained from "
      "the same state, has an SD of 0.12-0.27 pp in the first two intervals, as large as two independent full runs "
      "(0.17 pp). Single decisions are therefore often wrong by a grid step, but cheap: the truth is flat near its "
      "optimum. Errors of independent searches are partly independent, and averaging two searches' schedules gains "
      "+0.066 pp over the mean of the two (26 settings, 23 positive; Exp. 17)."),
    P("<b>The representation costs more than expected.</b> The oracle (the baseline's applied schedule divided by "
      "the transform) reduced to 10 interval means scores 93.86 at BS 500 when committed as steps, 0.14 pp below the "
      "baseline; the same means joined linearly through the interval midpoints match the baseline in every setting "
      "(Exp. 24). Five means joined linearly are enough (93.84-94.01 across the 12 conv-only settings), and snapping "
      "them to the x0.6 grid costs 0.088 pp on average (Exp. 26). The grid itself does not prevent a match: searched "
      "jointly with full runs (all 3<super>5</super> - 1 neighbours per move), 5 constant intervals of conv on the "
      "grid reach 93.84-94.17 at BS 2000 under all six transforms, against 93.92 for the baseline on the same seed "
      "(one seed, best of 414-744 runs each; Exp. 33), far beyond the 20-run budget."),
    P("4&nbsp;&nbsp;Interval Cooldown Search", "h1"),
    P("Each searched LR is represented during the search by 5 constant intervals of its specified value on the x0.6 "
      "grid; transforms apply on top. The budget is B = 20 runs of training steps."),
    P("<b>1. Two searches.</b> Run the search below twice from initialisation, each with its own data order and B/2 "
      "steps (steps are reserved so each finishes within its share). Average the two 5-value schedules interval by "
      "interval (geometric mean).", "algo"),
    P("<b>2. Levels (4 LRs only, once).</b> In interval 1, from 1.0: scan conv on a x7.7 grid (best score), then head "
      "and both biases together on a x7.7 grid, keeping the largest level within 1 pp of the best; each candidate "
      "is scored after the interval plus one interval at 0.22 x its value. Afterwards head and the biases follow "
      "conv's interval-to-interval profile.", "algo"),
    P("<b>3. Pass 1 (shape).</b> For each interval in order, climb conv from the previous interval's value (smaller "
      "values first, moving while the score improves). Score: the interval plus one interval at 0.22 x the "
      "candidate. Cheap and local, so a transform cannot distort it, but biased toward small LRs (Section 3).", "algo"),
    P("<b>4. Shift.</b> Move pass 1's conv values, all but the last, one grid step up (x1/0.6): the bias correction.",
      "algo"),
    P("<b>5. Pass 2 (refine).</b> Restart from initialisation. For each interval, the first included, try at most one "
      "grid step either side of the shifted plan; score a candidate by the interval plus the rest of the run on the "
      "plan. Score the last interval by the mean of two continuations with fresh data order when the budget allows.",
      "algo"),
    P("<b>6. Commit.</b> Join the averaged 5 values piecewise linearly through the interval midpoints, the end "
      "half-intervals continuing the neighbouring slope.", "algo"),
    Spacer(1, 3),
    P("The design follows Section 3. Pass 1 finds the schedule's shape, including the transform's, at the cost of a "
      "systematic bias of about one grid step, which step 4 removes. Pass 2 uses the faithful rest-of-run lookahead, "
      "and because its tail is the shifted plan, the tail already has the shape the transform needs. Averaging two "
      "half-budget searches cancels independent decision errors, and the linear commit removes the staircase's cost. "
      "Figure 2 shows that the committed schedules reproduce the baseline's applied LR under every transform."),
    figure(paths["schedules"],
           "<b>Figure 2.</b> Applied conv LR (specified schedule x transform) at BS 2000, conv only: the baseline that "
           "the search should reproduce (black), the baseline's specification kept under the transform (dashed), and "
           "the committed schedules of 12 independent ICS searches (blue)."),

    P("5&nbsp;&nbsp;Results", "h1"),
    table(t1, [3.6 * cm] + [3.35 * cm] * 4,
          "<b>Table 2.</b> Held-out TTA of ICS v29 (mean over 12 searches; seeds 100-107) and, in brackets, the "
          "difference to the baseline on seeds 100-107 / on fresh seeds 200-207 (8 searches). Bold: within 0.1 pp of "
          "the baseline. Every search spends its full 20-run budget.", rules=(2, 4)),
    P(f"Conv only, ICS matches the baseline in all 12 settings, {-sa[0]:.3f} pp short on average (Table 2). With all "
      f"four LRs searched from 1.0 it matches in {sb[1]} of 12, {-sb[0]:.3f} pp short on average; it misses at BS 2000 "
      "without a transform (0.18 pp short) and under the reverse ramp (0.11 pp), and at BS 500 under the reverse "
      "ramp (0.10 pp). The method was selected among over 40 versions on seeds 100-107, so we re-evaluated every "
      "committed schedule and the baselines on fresh seeds 200-207: conv only it is "
      f"{-statistics.mean(fresh_d['a']):.3f} pp short ({sum(d >= -0.1 for d in fresh_d['a'])}/12 within 0.1 pp), "
      f"with 4 LRs {-statistics.mean(fresh_d['b']):.3f} pp ({sum(d >= -0.1 for d in fresh_d['b'])}/12). Selection "
      "inflated the scores by only 0.01-0.02 pp; BS 2000 with 4 LRs is the one place it is reliably short, by "
      "0.1-0.15 pp."),
    figure(paths["results"],
           "<b>Figure 3.</b> Difference to the baseline per setting: circles, seeds 100-107, with 95% intervals over "
           "the 12 searches; diamonds, fresh seeds 200-207. The grey band is the 0.1 pp match criterion."),
    P(f"Individual searches vary: at BS 2000 without a transform, the 12 conv-only searches range from "
      f"{min(conv_none):.2f} to {max(conv_none):.2f}. The averaged result is what the table reports; a user running "
      "ICS once gets one draw from this distribution, which is also why the method averages two half-budget searches "
      "rather than spending the whole budget on one (Table 3)."),
    P("<b>How close is ICS to what it can commit?</b> ICS commits 5 values joined linearly, each on the x0.6 grid "
      "or, where its two searches disagree, between grid points. Conv only, the oracle in this form, its 5 interval "
      "means joined linearly, scores " + fmt(ceiling['lin5_oracle']) + " pp against the baseline on average over the "
      "12 settings; snapped to the grid, as a single search must commit it, " + fmt(ceiling['lin5g_oracle']) + " pp. "
      "ICS, at " + fmt(sa[0]) + " pp, lies between the two (Figure 4): averaging its two searches recovers most of "
      "what the grid costs, and at most a few hundredths of a pp are left to gain conv only (Exp. 26)."),
    figure(paths["ceiling"],
           "<b>Figure 4.</b> Conv only: the oracle schedule represented as ICS commits it (5 interval means joined "
           "linearly), exact and snapped to the x0.6 grid, against ICS. The grey band is the 0.1 pp match criterion."),

    P("6&nbsp;&nbsp;What matters", "h1"),
    figure(paths["trajectory"],
           "<b>Figure 5.</b> Development of the method: mean difference to the baseline over the 12 settings of each "
           "scenario (labels: settings within 0.1 pp / settings run; 2 searches per setting, v29 12). Values below "
           "-0.45 pp are drawn at the edge with their value. Each step adds one change to the previous one."),
    P("Figure 5 traces the main line of development. The two-pass structure with a short-cooldown first pass removed "
      "the transform failures of v1. The largest single gain, conv only from 0.25 to 0.11 pp short, came from "
      "committing 10 intervals (each searched interval split in two along the trend) and keeping the plan's tail "
      "unscaled in pass 2; the plan shift, averaging two half-budget searches, the linear commit (which replaced the "
      "expansion) and refining the first interval in pass 2 each closed part of the remaining gap. Table 3 changes one component of v29 at a time."),
    table(t2, [8.3 * cm, 4.35 * cm, 4.35 * cm],
          "<b>Table 3.</b> Single changes to ICS v29: mean held-out difference to the baseline over the 12 settings "
          "(settings within 0.1 pp / settings run; 2 searches per setting, v29 12). &ndash;: not run.", rules=(1,)),
    P("<b>Budget.</b> ICS needs its whole budget. At 15 runs the transformed settings fail first, and with 4 LRs the "
      "ramps collapse (BS 2000: 79.90): when a search runs out of budget in pass 1, the remaining intervals keep the "
      "previous value, several grid steps too high under a ramp. Starting each pass-1 climb on the previous "
      "intervals' trend (v39) removes the collapse at 15 runs and is neutral at 20 (v38), so it is the change to make "
      "if the budget must shrink."),
    P("<b>Structure.</b> Two half-budget searches beat one full-budget search (v30). Four intervals do as well as "
      "five; eight leave too little budget per decision. Replacing pass 2 by a third independent pass-1 search (v32) "
      "fails under the ramps: the refinement pass is needed even when it can only afford its first intervals. A "
      "finer pass-2 grid does not recover the grid's cost (v33; paired with v29 &minus;0.011 &plusmn; 0.007 pp), "
      "because averaging already puts values between grid points where the two searches disagree."),
    P("<b>The bias correction is a pair.</b> A milder pass-1 cooldown (0.36 x instead of 0.22 x, v45) is worse "
      "everywhere, and an unbiased rest-of-run first pass without the shift (v46) is worse still: it can afford fewer, "
      "coarser decisions. The cheap biased pass plus a fixed correction is more accurate per unit of budget than an "
      "unbiased expensive one."),
    P("<b>Levels for the other groups.</b> Tying head and biases to conv's profile is right: decaying them linearly "
      "from their level instead costs 0.154 &plusmn; 0.045 pp (Exp. 27). Their level has to be set jointly: scanning "
      "head on its own (v35) fixes BS 2000 but lets head run to levels that blow up late under the ramps, and the "
      "level scan needs its one-interval cooldown (v34). Probes on committed schedules confirm that the searched "
      "values are not merely too small: scaling them x1.67 costs 0.143 &plusmn; 0.050 pp and x2.78 costs 0.69 pp "
      "(Exp. 24)."),

    P("7&nbsp;&nbsp;Limitations and failure modes", "h1"),
    P("<b>Transforms not seen during development.</b> After v29 was final we ran it under two transforms that had been "
      "held back: the inverted wave (x0.52 early, rising to x1.93 at 70-80% of training) and a steeper ramp (x1/8 to "
      f"x8). It matches in {held_back['a'][1]} of 6 conv-only and {held_back['b'][1]} of 6 four-LR settings (Table 4), "
      f"against {sa[1] + sb[1]} of 24 under the development transforms. The steep ramp is mostly recovered conv only. "
      "The inverted wave is not, and the representation is not the reason: the oracle's 5 means joined linearly "
      "score 93.87-93.88. The failure is the first pass's short cooldown in the wave's trough, where the small-LR "
      "bias is larger than one grid step: at BS 2000 pass 1 drops interval 2 to 0.13-0.17 where the oracle holds "
      "0.29, and pass 2, scoring a single raised interval against the plan's low tail, never raises it. Allowing "
      "two-step climbs in pass 2 (v43), a tail that moves with the candidate (v44), a milder cooldown (v45) and an "
      "unbiased first pass (v46) all fail to fix it (Exp. 31)."),
    table(t3, [5.3 * cm, 1.9 * cm, 4.2 * cm, 2.4 * cm, 2.6 * cm],
          "<b>Table 4.</b> ICS v29 under the two transforms withheld during development (held-out TTA, mean over 8 "
          "searches). Oracle: the baseline's applied schedule divided by the transform, as 5 interval means joined "
          "linearly (in brackets: snapped to the x0.6 grid); conv only. Bold: within 0.1 pp.", rules=(6,)),
    figure(paths["waveinv"],
           "<b>Figure 6.</b> The failure under the inverted wave, conv only: applied conv LR of the baseline (black), "
           "of the baseline's specification under the transform (dashed) and of 8 ICS searches (blue). Shaded: "
           "interval 2, in the wave's trough, where the searches commit too low a value at BS 2000 and 500."),
    P("<b>Four LRs at BS 2000.</b> One common level for head and the biases is right on average, but at BS 2000 head's "
      "optimum lies far above the biases' (baseline 1300 against 99 and 36). A short horizon cannot set head apart "
      "safely, since under a ramp head's LR grows x4 late in the run; a rest-of-run relevel would need budget that "
      "the two half searches do not have."),
    P("<b>Scope.</b> All results are on one task, a fast 8-epoch CIFAR-10 recipe. The searches select on the test "
      "set, and over 40 versions were developed on the same 24 settings and held-out seeds; the fresh seeds bound "
      "the resulting optimism at 0.01-0.02 pp, but Table 4 shows that the larger risk is overfitting the development "
      "transforms. On larger models the budget of 20 runs, not the search's bookkeeping, is the binding cost."),

    P("8&nbsp;&nbsp;Related work", "h1"),
    P("Schedule families such as step decay, cosine with restarts (Loshchilov &amp; Hutter, 2017) and "
      "warmup-stable-decay schedules, whose cooldown is studied by Hagele et al. (2024), fix a shape and tune a few "
      "scalars; schedule-free methods (Defazio et al., 2024) remove the schedule. Online adaptation, by hypergradients "
      "(Baydin et al., 2018) or by population-based training (Jaderberg et al., 2017), adjusts LRs during training "
      "from short-horizon signals, whose bias toward small step sizes was analysed by Wu et al. (2018). Bandit "
      "methods such as Hyperband (Li et al., 2018) allocate budget across configurations by early stopping. ICS is "
      "closest to online adaptation, but it measures the short-horizon bias, corrects it explicitly, and verifies "
      "each decision with a rest-of-run lookahead under a fixed budget of full runs."),

    P("9&nbsp;&nbsp;Conclusion", "h1"),
    P("A schedule search that decides interval by interval can match tuned schedules within 20 runs, if it treats "
      "its lookahead's horizon and noise as quantities to measure and correct rather than to assume away. In our "
      "experiments the cheap short lookahead was usable once its bias was corrected by a fixed step, the faithful "
      "rest-of-run lookahead was affordable only as a refinement, averaging two searches was worth more than spending "
      "the budget on one, and the committed representation mattered as much as the search. The remaining gaps are "
      "specific: the level of groups whose optimum differs from conv's, and a first pass whose bias grows where the "
      "transform pushes the LR far down early in training."),

    P("References", "h1"),
    P("Baydin, A. G., Cornish, R., Martinez Rubio, D., Schmidt, M., Wood, F. (2018). Online learning rate adaptation "
      "with hypergradient descent. ICLR.", "ref"),
    P("Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A., Cutkosky, A. (2024). The road less scheduled. "
      "NeurIPS.", "ref"),
    P("Hagele, A., Bakouch, E., Kosson, A., Ben Allal, L., Von Werra, L., Jaggi, M. (2024). Scaling laws and "
      "compute-optimal training beyond fixed training durations. NeurIPS.", "ref"),
    P("Jaderberg, M., et al. (2017). Population based training of neural networks. arXiv:1711.09846.", "ref"),
    P("Jordan, K. (2024). 94% on CIFAR-10 in 3.29 seconds on a single GPU. arXiv:2404.00498.", "ref"),
    P("Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., Bernstein, J. (2024). Muon: an optimizer "
      "for hidden layers in neural networks. Blog post.", "ref"),
    P("Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2018). Hyperband: a novel bandit-based "
      "approach to hyperparameter optimization. JMLR 18(185).", "ref"),
    P("Loshchilov, I., Hutter, F. (2017). SGDR: stochastic gradient descent with warm restarts. ICLR.", "ref"),
    P("Wu, Y., Ren, M., Liao, R., Grosse, R. (2018). Understanding short-horizon bias in stochastic "
      "meta-optimization. ICLR.", "ref"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Interval Cooldown Search", author="Roland Gao")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}: {doc.page} pages")

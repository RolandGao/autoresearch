"""Findings F1-F8 and the data appendices of the PDF report, each finding with its data table
built from logs/ at build time (research/report_data.py). Used by research/make_report.py."""
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether as _Keep
from reportlab.platypus import Spacer

import report_data as d

CUT = "20261016_183000"  # BS 500 / 125 baselines replaced at this time


def build(P, raw_table, figure, KeepTogether):
    def table(*args):  # Keep each data table on one page, with a gap after it.
        return _Keep([raw_table(*args), Spacer(1, 6)])

    fid1, fid1_80, fid1_rest = d.fidelity(1, ""), d.fidelity(1, "cool2"), d.fidelity(1, "coolrest")
    fixed = d.fixed_runs()
    old = d.search_runs(until=CUT)
    new = d.search_runs(since=CUT)
    lrs = sorted(fid1["truth"])

    def fx(name, n=8, first=100, era="old"):
        r = fixed.get((name, n, first, era))
        return f"{r[1]:.2f} +- {r[2]:.2f}" if r else "-"

    def sr(name, rows=old):
        r = rows.get(name)
        return f"{r['heldout']:.2f}" if r else "-"

    def curve_row(label, values):
        best = max(values[c] for c in lrs)
        return [label] + [f"<b>{values[c]:.2f}</b>" if values[c] == best else f"{values[c]:.2f}"
                          for c in lrs]

    lr_widths = [4.6 * cm] + [1.77 * cm] * len(lrs)
    out = [P("5. Findings, each with its data", "h1"),
           P("Bold marks each row's best value. Held-out TTA = mean +- sd over seeds 100-107 unless "
             "stated; search rows give the mean over n search seeds (appendix A).", "small")]

    # F1
    out += [
        P("F1. Short lookaheads choose LRs that are too small, whatever the cooldown (W-1, W-4)", "h2"),
        P("Probe at batch size 2000, interval 2 (steps 40-80): train with a constant conv LR during the "
          "interval, then either resume the baseline to the end (truth) or run a lookahead and score it. "
          "Seed means over 4 seeds. All 72 cooldown designs with a 40-step lookahead (and no lookahead) "
          "peak at 0.078 or lower (45 at 0.078, 17 at 0.047, 11 at 0.028); the truth peaks at 0.22; a "
          "lookahead to the end of training peaks at 0.13."),
        table([["Interval-2 conv LR"] + [f"{c:g}" for c in lrs],
               curve_row("Truth (final TTA)", fid1["truth"]),
               curve_row("No lookahead", fid1["none"]),
               curve_row("40-step lookahead, 0.22 x LR", fid1["s0.22|ocont"]),
               curve_row("40-step lookahead, linear to 0", fid1["slin0|ocont"]),
               curve_row("80-step lookahead, 0.13 x LR", fid1_80["s0.13|ocont"]),
               curve_row("To the end, 0.13 x LR", fid1_rest["s0.13|ocont"]),
               curve_row("To the end, linear to 0", fid1_rest["slin0|ocont"])], lr_widths),
        figure("plots/lookahead_bias.png", 12.5,
               "Figure 2. The same data, each curve relative to its own best; labels give each curve's "
               "chosen LR."),
    ]
    # F2
    out += [
        P("F2. The non-searched groups' cooldown does not matter (W-1, W-5)", "h2"),
        P("Same probe, 40-step lookahead with the searched LR at 0.22 x candidate, varying what the other "
          "groups (head, biases) do during the lookahead. The curves differ by at most 0.1 pp and peak at the "
          "same LR. Full searches with the three designs end within 0.05 pp of each other."),
        table([["Other groups in the lookahead"] + [f"{c:g}" for c in lrs],
               curve_row("Continue their schedule", fid1["s0.22|ocont"]),
               curve_row("Decay linearly to 0", fid1["s0.22|olin0"]),
               curve_row("Constant 1 x their LR", fid1["s0.22|o1.0"]),
               curve_row("Constant 0.36 x their LR", fid1["s0.22|o0.36"]),
               curve_row("Frozen (LR 0)", fid1["s0.22|o0.0"])], lr_widths),
        table([["Full search (20% intervals, searched multiplier)", "Held-out TTA", "n"],
               ["Other groups continue", sr("auto_a_ssearch_cont_bs2000"), "2"],
               ["Other groups share the searched multiplier", sr("auto_a_ssearch_shared_bs2000"), "2"],
               ["Other groups get their own searched multiplier", sr("auto_a_ssearch_osearch_bs2000"), "2"]],
              [9.5 * cm, 3.0 * cm, 1.5 * cm]),
    ]
    # F3
    out += [
        P("F3. The final interval must anneal to 0; constant intervals are otherwise fine (W-6)", "h2"),
        P("Fixed schedules on held-out seeds (no search). The conv schedule of baseline_bs2000 (0.22 decaying "
          "linearly to 0) is replaced by n constant steps at the mean of each step, optionally snapped to the "
          "0.6^k grid; the last two rows keep 5 constant steps but let the last one decay to 0."),
        table([["Conv schedule (batch size 2000)", "Held-out TTA"],
               ["Baseline: linear decay 0.22 -> 0", fx("baseline_bs2000")],
               ["5 constant steps", fx("staircase_a_n5_bs2000")],
               ["5 constant steps, grid-snapped", fx("staircase_a_n5_grid_bs2000")],
               ["10 constant steps", fx("staircase_a_n10_bs2000")],
               ["20 constant steps", fx("staircase_a_n20_bs2000")],
               ["5 steps, last decays linearly (same mean)", fx("staircase_a_lastlin_bs2000")],
               ["5 steps, last is the baseline's own (0.044 -> 0)", fx("staircase_a_lastexact_bs2000")],
               ["All four groups as 5 constant steps", fx("staircase_b_n5_bs2000")]],
              [9.5 * cm, 4.0 * cm]),
        table([["Grid-snapped 5-knot piecewise-linear baseline", "Held-out TTA", "That batch size's baseline"],
               ["Batch size 2000", fx("staircase_a_knots_grid_bs2000"), fx("baseline_bs2000")],
               ["Batch size 500", fx("staircase_a_knots_grid_bs500"), fx("baseline_bs500") + " (old)"],
               ["Batch size 125", fx("staircase_a_knots_grid_bs125"),
                "93.85 (old; seeds 100-107)"]],
              [7.0 * cm, 3.5 * cm, 4.5 * cm]),
    ]
    # F4
    oracle_ramp = [1.0, 0.36, 0.22, 0.078, 0.017, 0.0]
    oracle_wave = [0.22, 0.078, 0.078, 0.13, 0.078, 0.0]

    def knots(run):
        k = d.final_schedule(run, "conv.initial_lr")
        return ", ".join(f"{v:g}" for v in k) if k else "-"

    out += [
        P("F4. A fixed lookahead shape breaks under transforms (W-7, W-8, W-11)", "h2"),
        P("Pass 1 alone (linear-to-0 lookahead) recovers little under the transforms, and under wave ends "
          "below doing nothing. Its committed knots under ramp stay far below the oracle. Alternative "
          "lookahead shapes trade one condition for another."),
        table([["Batch size 2000, scenario A", "None", "Ramp", "Wave"],
               ["Baseline / do-nothing", "93.94", fx("transformed_a_ramp_bs2000"), fx("transformed_a_wave_bs2000")],
               ["Pass 1 only (linear lookahead)", sr("auto_a_lin_rest_lin_bs2000"), sr("auto_a_v1_bs2000_ramp"),
                sr("auto_a_v1_bs2000_wave")],
               ["Searched lookahead level (plan)", sr("auto_a_v2plan_bs2000"), sr("auto_a_v2plan_bs2000_ramp"),
                sr("auto_a_v2plan_bs2000_wave")],
               ["Geometric lookahead, searched ratio", sr("auto_a_v3_bs2000"), sr("auto_a_v3_bs2000_ramp"),
                sr("auto_a_v3_bs2000_wave")],
               ["Max of linear and geometric", sr("auto_a_v4_bs2000"), sr("auto_a_v4_bs2000_ramp"),
                sr("auto_a_v4_bs2000_wave")]],
              [6.5 * cm, 2.6 * cm, 2.9 * cm, 2.9 * cm]),
        table([["Conv knots under ramp (start, 20%, ..., end)", "Knots"],
               ["Oracle (grid-snapped baseline / ramp)", ", ".join(f"{v:g}" for v in oracle_ramp)],
               ["Pass 1 only, search seed 0", knots("auto_a_v1_bs2000_ramp_seed0")]],
              [7.0 * cm, 8.0 * cm]),
    ]
    # F5
    out += [
        P("F5. The previous pass's schedule is a self-correcting lookahead (W-12, W-19)", "h2"),
        P("Under ramp, each extra pass raises the early knots toward the oracle and the held-out TTA toward "
          "the baseline."),
        table([["Batch size 2000, scenario A", "None", "Ramp", "Wave", "n"],
               ["1 pass", sr("auto_a_lin_rest_lin_bs2000"), sr("auto_a_v1_bs2000_ramp"), sr("auto_a_v1_bs2000_wave"), "2"],
               ["2 passes (pass 2 on the same seed)", sr("auto_a_v1x2_bs2000"), sr("auto_a_v1x2_bs2000_ramp"),
                sr("auto_a_v1x2_bs2000_wave"), "2"],
               ["2 passes, 2 fresh seeds, tail rescaled", sr("auto_a_v1x2m2t_bs2000"), sr("auto_a_v1x2m2t_bs2000_ramp"),
                sr("auto_a_v1x2m2t_bs2000_wave"), "4"],
               ["3 passes, 2 fresh seeds, tail rescaled", sr("auto_a_v1x3m2t_bs2000"), sr("auto_a_v1x3m2t_bs2000_ramp"),
                sr("auto_a_v1x3m2t_bs2000_wave"), "4"]],
              [6.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.0 * cm]),
        table([["Conv knots under ramp", "Knots"],
               ["Oracle", ", ".join(f"{v:g}" for v in oracle_ramp)],
               ["1 pass (seed 0)", knots("auto_a_v1_bs2000_ramp_seed0")],
               ["2 passes (seed 0)", knots("auto_a_v1x2_bs2000_ramp_seed0")],
               ["3 passes (seed 1)", knots("auto_a_v1x3m2t_bs2000_ramp_seed1")]],
              [5.0 * cm, 10.0 * cm]),
    ]
    # F6
    replay = d.interval_scores("auto_a_v1x2_bs2000_wave_seed1")
    rows = [["Interval", "Pass 1: score of the chosen knot", "Pass 2: score of the chosen knot"]]
    for i in range(5):
        p1 = next((s for s in replay if s[0] == i), None)
        p2 = next((s for s in replay if s[0] == i + 5), None)
        rows.append([str(i + 1), f"{p1[2]:.2f} (knot {p1[3]:g})" if p1 else "-",
                     f"{p2[2]:.2f} (knot {p2[3]:g})" if p2 else "-"])

    def gap(name):
        r = old.get(name)
        return f"{r['own']:.2f}", f"{r['heldout']:.2f}", f"{r['own'] - r['heldout']:+.2f}"

    out += [
        P("F6. Single-seed searches suffer a winner's curse (W-12, W-13, W-16)", "h2"),
        P("In a two-pass search whose second pass reuses the first pass's seed, re-scoring the incumbent "
          "replays pass 1 exactly, so pass 2 inherits a lucky score and never moves (wave, search seed 1: the "
          "same 94.09 in every pass-2 interval; held out, this schedule scores about 93.4). Scoring pass 2 on "
          "fresh seeds removes the lock-in."),
        table(rows, [2.0 * cm, 6.5 * cm, 6.5 * cm]),
        table([["Wave, batch size 2000", "Own score", "Held-out", "Optimism"],
               ["2 passes, pass 2 on the same seed", *gap("auto_a_v1x2_bs2000_wave")],
               ["2 passes, pass 2 on 1 fresh seed", *gap("auto_a_v1x2r_bs2000_wave")],
               ["2 passes, pass 2 on 3 fresh seeds", *gap("auto_a_v1x2m3_bs2000_wave")],
               ["... plus tail rescaling", *gap("auto_a_v1x2m3t_bs2000_wave")]],
              [6.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]),
    ]
    # F7
    screen = d.screen()
    full = {2000: dict(conv=0.22, head=1300, norm_bias=99, whiten_bias=36),
            500: dict(conv=0.13, head=280, norm_bias=21, whiten_bias=36),
            125: dict(conv=0.047, head=170, norm_bias=4.6, whiten_bias=7.7)}
    rows = [["Group", "Batch size", "Best LR after 16 steps", "Loss rise one x3.2 step below / above",
             "Full-run LR (old baseline)"]]
    for group in ("conv", "head", "norm_bias", "whiten_bias"):
        for bs in (2000, 500, 125):
            best, below, above = screen[(bs, group)]
            rows.append([group, str(bs), f"{best:g}", f"{below:.3f} / {above:.3f}", f"{full[bs][group]:g}"])
    out += [
        P("F7. A few training steps reveal the conv LR scale, but not the bias LRs (W-2, W-10, W-16)", "h2"),
        P("Each group's constant LR swept over a x sqrt(10) grid (others at their baseline LR), loss on 2000 "
          "test images after 16 training steps, mean of 2 seeds. Conv has a sharp minimum at 0.1 for every "
          "batch size; the bias curves are nearly flat. Over a full run, though, norm_bias matters and "
          "whiten_bias does not."),
        table(rows, [2.3 * cm, 1.9 * cm, 3.2 * cm, 5.0 * cm, 3.5 * cm]),
        table([["baseline_bs2000 with ...", "Held-out TTA"],
               ["nothing changed", fx("baseline_bs2000")],
               ["whiten_bias LR 36 -> 1", fx("baseline_bs2000_wb1")],
               ["norm_bias LR 99 -> 1", fx("baseline_bs2000_nb1")],
               ["both", fx("baseline_bs2000_nb1wb1")]], [7.0 * cm, 4.0 * cm]),
    ]
    # F8
    out += [
        P("F8. The representation is not the limit under transforms (W-11)", "h2"),
        P("Conv knots set to baseline / transform at each 20% boundary (exact, or snapped to the grid), linear "
          "in between, trained under the transform."),
        table([["Batch size 2000", "Exact knots", "Grid-snapped knots", "Do-nothing", "Baseline"],
               ["Ramp", fx("oracle_knots_ramp_bs2000"), fx("oracle_knots_ramp_grid_bs2000"),
                fx("transformed_a_ramp_bs2000"), "93.94"],
               ["Wave", fx("oracle_knots_wave_bs2000"), fx("oracle_knots_wave_grid_bs2000"),
                fx("transformed_a_wave_bs2000"), "93.94"]],
              [2.5 * cm, 3.3 * cm, 3.6 * cm, 3.3 * cm, 2.3 * cm]),
    ]
    return out


def appendices(P, table):
    old = d.search_runs(until=CUT)
    new = d.search_runs(since=CUT)
    rows = [["Search config", "n", "Cost (s)", "Evals", "Own", "Held-out", "Range"]]
    for name, r in sorted(old.items()):
        rows.append([name, str(r["n"]), f"{r['seconds']:.0f}", f"{r['evals']:.0f}", f"{r['own']:.2f}",
                     f"{r['heldout']:.2f}", f"{r['spread']:.2f}"])
    rows_new = [rows[0]] + [[name, str(r["n"]), f"{r['seconds']:.0f}", f"{r['evals']:.0f}",
                             f"{r['own']:.2f}", f"{r['heldout']:.2f}", f"{r['spread']:.2f}"]
                            for name, r in sorted(new.items())]
    widths = [7.6 * cm, 0.8 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.8 * cm, 1.5 * cm]
    fixed_rows = [["Fixed schedule (no search)", "Baselines", "Seeds", "Held-out TTA"]]
    for (name, n, first, era), (_, mean, sd) in sorted(d.fixed_runs().items()):
        depends = ("bs500" in name or "bs125" in name) and not name.startswith("seeded_best")
        fixed_rows.append([name, era if depends else "-",
                           f"{first}-{first + n - 1}", f"{mean:.2f} +- {sd:.2f}"])
    return [
        P("Appendix A. Every search run (from logs/)", "h1"),
        P("One row per search config: n search seeds, mean search cost, mean evaluations, the search's own "
          "final TTA, the held-out TTA (seeds 100-107) and its range over search seeds. A.1: runs with the "
          "previous batch size 500 / 125 baselines (before 2026-10-16 18:30 UTC). A.2: runs after the "
          "baseline update.", "small"),
        P("<b>A.1 Before the baseline update</b>"), table(rows, widths),
        Spacer(1, 6),
        P("<b>A.2 After the baseline update</b>"), table(rows_new, widths),
        P("Appendix B. Every fixed-schedule run on held-out seeds (from logs/)", "h1"),
        P("\"old\" / \"new\" marks batch size 500 / 125 runs made with the previous / updated baselines.",
          "small"),
        table(fixed_rows, [8.4 * cm, 2.0 * cm, 2.6 * cm, 3.6 * cm]),
    ]

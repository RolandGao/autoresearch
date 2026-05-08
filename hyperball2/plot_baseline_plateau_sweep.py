#!/usr/bin/env python3
"""Plot loss and LR curves from the baseline ReduceLROnPlateau sweep log."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("baseline_plateau_sweep.log")
DEFAULT_OUTPUT = Path(__file__).with_name("baseline_plateau_loss_lr_curves.png")
DEFAULT_LINEAR_LR_OUTPUT = Path(__file__).with_name(
    "baseline_plateau_lr_curves_linear.png"
)
DEFAULT_GRID_OUTPUT = Path(__file__).with_name("baseline_plateau_loss_lr_grid.png")
BASELINE_LR_MULT = 1.5
BASELINE_WARMUP_RATIO = 0.0
BASELINE_WARMDOWN_RATIO = 0.7
BASELINE_FINAL_LR_FRAC = 0.05

START_RE = re.compile(r"=== PLATEAU_EXPERIMENT_START (?P<name>.+?) ===")
END_RE = re.compile(
    r"=== PLATEAU_EXPERIMENT_END (?P<name>.+?) exit_code=(?P<exit_code>-?\d+) ==="
)
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+).*?"
    r"\|\s+loss:\s+(?P<loss>[-+0-9.eE]+)\s+"
    r"\|\s+lrm:\s+(?P<lrm>[-+0-9.eE]+)"
)
KEY_VALUE_FLOAT_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+):\s+(?P<value>[-+0-9.eE]+)\s*$")


@dataclass
class RunRecord:
    name: str
    description: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    lrms: list[float] = field(default_factory=list)
    reductions: list[dict] = field(default_factory=list)
    val_bpb: float | None = None
    final_lrm: float | None = None
    plateau_lrm: float | None = None
    exit_code: int | None = None

    @property
    def initial_lr_mult(self) -> float:
        return parse_float(self.env.get("LR_MULT")) or 1.0

    @property
    def effective_lr_mults(self) -> list[float]:
        lr_mult = self.initial_lr_mult
        return [lr_mult * lrm for lrm in self.lrms]

    @property
    def complete(self) -> bool:
        return self.exit_code == 0


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_json_after_marker(line: str, marker: str) -> dict | None:
    idx = line.find(marker)
    if idx < 0:
        return None
    payload = line[idx + len(marker) :].strip()
    if not payload.startswith("{"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def parse_log(path: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    current: RunRecord | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            for line in raw_line.split("\r"):
                line = line.strip()
                if not line:
                    continue

                start_match = START_RE.search(line)
                if start_match:
                    current = RunRecord(name=start_match.group("name"))
                    runs.append(current)
                    continue

                end_match = END_RE.search(line)
                if end_match:
                    name = end_match.group("name")
                    target = current if current and current.name == name else None
                    if target is None:
                        target = next((run for run in reversed(runs) if run.name == name), None)
                    if target is not None:
                        target.exit_code = int(end_match.group("exit_code"))
                    current = None
                    continue

                if current is None:
                    continue

                experiment = parse_json_after_marker(line, "PLATEAU_EXPERIMENT ")
                if experiment:
                    current.description = experiment.get("description")
                    current.env = {
                        str(key): str(value)
                        for key, value in (experiment.get("env") or {}).items()
                    }
                    continue

                scheduler_event = parse_json_after_marker(line, "LR_SCHEDULER ")
                if scheduler_event:
                    current.reductions.append(scheduler_event)
                    continue

                step_match = STEP_RE.search(line)
                if step_match:
                    current.steps.append(int(step_match.group("step")))
                    current.losses.append(float(step_match.group("loss")))
                    current.lrms.append(float(step_match.group("lrm")))
                    continue

                key_value_match = KEY_VALUE_FLOAT_RE.match(line)
                if key_value_match:
                    key = key_value_match.group("key")
                    value = float(key_value_match.group("value"))
                    if key == "val_bpb":
                        current.val_bpb = value
                    elif key == "final_lrm":
                        current.final_lrm = value
                    elif key == "plateau_lrm":
                        current.plateau_lrm = value

    return runs


def run_label(run: RunRecord) -> str:
    suffix = ""
    if run.val_bpb is not None:
        suffix = f" val={run.val_bpb:.4g}"
    elif not run.complete:
        suffix = " incomplete"
    return f"{run.name}{suffix}"


def color_map(runs: list[RunRecord]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#7f7f7f",
        "#003f5c",
        "#ffa600",
    ]
    return {run.name: palette[index % len(palette)] for index, run in enumerate(runs)}


def draw_validation_table(
    ax,
    runs: list[RunRecord],
    colors: dict[str, str],
) -> None:
    ax.axis("off")
    ax.set_title("Final validation\nval_bpb", fontsize=11)
    sorted_runs = sorted(
        runs,
        key=lambda run: (
            run.val_bpb is None,
            float("inf") if run.val_bpb is None else run.val_bpb,
            run.name,
        ),
    )
    y = 0.96
    line_gap = 0.074
    has_incomplete = False
    for run in sorted_runs:
        color = colors[run.name]
        val_text = "n/a" if run.val_bpb is None else f"{run.val_bpb:.5g}"
        suffix = "" if run.complete else " *"
        has_incomplete = has_incomplete or not run.complete
        ax.text(
            0.0,
            y,
            f"{run.name}{suffix}",
            color=color,
            fontsize=8.2,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            1.0,
            y,
            val_text,
            color=color,
            fontsize=8.2,
            transform=ax.transAxes,
            va="top",
            ha="right",
        )
        y -= line_gap
    if has_incomplete:
        ax.text(
            0.0,
            0.02,
            "* incomplete run",
            color="0.35",
            fontsize=8,
            transform=ax.transAxes,
            va="bottom",
        )


def baseline_lrm(progress: float) -> float:
    if progress < BASELINE_WARMUP_RATIO:
        return progress / BASELINE_WARMUP_RATIO if BASELINE_WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - BASELINE_WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / BASELINE_WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * BASELINE_FINAL_LR_FRAC


def baseline_schedule(max_step: int) -> tuple[list[int], list[float]]:
    steps = list(range(max_step + 1))
    denom = max(1, max_step)
    lr_mults = [
        BASELINE_LR_MULT * baseline_lrm(min(step / denom, 1.0)) for step in steps
    ]
    return steps, lr_mults


def plot_lr_comparison(
    runs: list[RunRecord],
    output_path: Path,
    dpi: int,
    *,
    yscale: str,
) -> None:
    if not any(run.steps for run in runs):
        raise ValueError("No training step records found.")

    colors = color_map(runs)
    fig, (lr_ax, val_ax) = plt.subplots(
        1,
        2,
        figsize=(15, 7.5),
        gridspec_kw={"width_ratios": [4.4, 1.45]},
        constrained_layout=True,
    )

    for run in runs:
        if not run.steps:
            continue
        color = colors[run.name]
        alpha = 1.0 if run.complete else 0.55
        lr_ax.plot(
            run.steps,
            run.effective_lr_mults,
            label=run.name,
            color=color,
            linewidth=1.6,
            alpha=alpha,
        )
        for reduction in run.reductions:
            step = reduction.get("step")
            next_lrm = parse_float(reduction.get("next_lrm"))
            if step is None or next_lrm is None:
                continue
            lr_ax.scatter(
                [int(step)],
                [run.initial_lr_mult * next_lrm],
                color=color,
                edgecolors="black",
                linewidths=0.35,
                s=22,
                zorder=3,
            )

    max_step = max(max(run.steps) for run in runs if run.steps)
    baseline_steps, baseline_lr_mults = baseline_schedule(max_step)
    baseline_line = lr_ax.plot(
        baseline_steps,
        baseline_lr_mults,
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="fixed baseline schedule",
        zorder=4,
    )[0]

    scale_title = "Linear" if yscale == "linear" else "Log"
    lr_ax.set_title(f"ReduceLROnPlateau Sweep: Effective LR_MULT ({scale_title} Y)")
    lr_ax.set_xlabel("optimizer step")
    lr_ax.set_ylabel("LR_MULT * lrm")
    lr_ax.set_yscale(yscale)
    lr_ax.grid(True, which="both", alpha=0.25)
    lr_ax.legend(handles=[baseline_line], loc="upper right", fontsize="small")
    draw_validation_table(val_ax, runs, colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_comparison(runs: list[RunRecord], output_path: Path, dpi: int) -> None:
    plot_lr_comparison(runs, output_path, dpi, yscale="log")


def plot_grid(runs: list[RunRecord], output_path: Path, dpi: int) -> None:
    plotted_runs = [run for run in runs if run.steps]
    if not plotted_runs:
        raise ValueError("No training step records found.")

    cols = 3
    rows = math.ceil(len(plotted_runs) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * cols, 3.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, run in zip(axes.flat, plotted_runs):
        lr_ax = ax.twinx()
        ax.plot(run.steps, run.losses, color="tab:blue", linewidth=1.4, label="loss")
        lr_ax.plot(
            run.steps,
            run.effective_lr_mults,
            color="tab:orange",
            linewidth=1.2,
            label="effective LR_MULT",
        )
        ax.set_title(run_label(run), fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("loss", color="tab:blue")
        lr_ax.set_ylabel("LR_MULT * lrm", color="tab:orange")
        lr_ax.set_yscale("log")
        ax.grid(True, alpha=0.22)

    for ax in axes.flat[len(plotted_runs) :]:
        ax.axis("off")

    fig.suptitle("ReduceLROnPlateau Sweep: Per-run Loss and LR")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_summary(runs: list[RunRecord], output_paths: list[Path]) -> None:
    print(f"parsed {len(runs)} runs")
    for run in runs:
        status = "ok" if run.complete else f"exit={run.exit_code}"
        if run.exit_code is None:
            status = "incomplete"
        last_step = run.steps[-1] if run.steps else None
        val = "n/a" if run.val_bpb is None else f"{run.val_bpb:.5g}"
        final_lrm = "n/a" if run.final_lrm is None else f"{run.final_lrm:.5g}"
        print(
            f"{run.name}: {status}, steps={len(run.steps)}"
            f" last_step={last_step}, reductions={len(run.reductions)},"
            f" val_bpb={val}, final_lrm={final_lrm}"
        )
    for output_path in output_paths:
        print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize loss and LR curves from baseline_plateau_sweep.log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Combined plateau sweep log to parse (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Comparison plot output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=DEFAULT_GRID_OUTPUT,
        help=f"Per-run grid output path (default: {DEFAULT_GRID_OUTPUT}).",
    )
    parser.add_argument(
        "--linear-lr-output",
        type=Path,
        default=DEFAULT_LINEAR_LR_OUTPUT,
        help=f"Linear-y aggregate LR output path (default: {DEFAULT_LINEAR_LR_OUTPUT}).",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Only write the main comparison plot.",
    )
    parser.add_argument("--dpi", type=int, default=170, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = parse_log(args.log)
    written = [args.output, args.linear_lr_output]
    plot_comparison(runs, args.output, args.dpi)
    plot_lr_comparison(runs, args.linear_lr_output, args.dpi, yscale="linear")
    if not args.no_grid:
        plot_grid(runs, args.grid_output, args.dpi)
        written.append(args.grid_output)
    print_summary(runs, written)


if __name__ == "__main__":
    main()

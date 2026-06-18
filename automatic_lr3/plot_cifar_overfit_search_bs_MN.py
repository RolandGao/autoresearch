#!/usr/bin/env python3
"""Plot the batch-size/N/M CIFAR overfit LR-search log."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_cifar_overfit_search_momentum import Run, format_number, parse_log, style_axes


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "cifar_overfit_search_bs_MN.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_bs_MN"
RUNTIME_TRADEOFF_PLOT = "first_step_below_0p88_vs_run_seconds.png"
PARETO_CURVES_PLOT = "pareto_first_step_below_0p87_curves.png"
EXPECTED_OUTPUTS = {
    "metric_grids.png",
    "selected_hparams.png",
    "summary.txt",
    RUNTIME_TRADEOFF_PLOT,
    PARETO_CURVES_PLOT,
}


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    csv_name: str
    value_kind: str
    lower_is_better: bool = True

    def value(self, run: Run) -> float | None:
        if self.key.startswith("loss_step_"):
            step = int(self.key.rsplit("_", 1)[1])
            return loss_at_step(run, step)
        if self.key.startswith("first_below_"):
            threshold = float(self.key.rsplit("_", 1)[1].replace("p", "."))
            point = first_step_below(run, threshold)
            return float(point.step) if point is not None else None
        raise ValueError(f"Unknown metric: {self.key}")

    def sort_value(self, run: Run) -> float:
        value = self.value(run)
        if value is None:
            return math.inf if self.lower_is_better else -math.inf
        return value

    def format_value(self, value: float | None) -> str:
        if value is None:
            return "NA"
        if self.value_kind == "step":
            return f"{int(value)}"
        return f"{value:.4f}"


METRICS = [
    Metric("loss_step_30", "Loss after 30 steps", "loss_after_30_steps", "loss"),
    Metric("loss_step_10", "Loss after 10 steps", "loss_after_10_steps", "loss"),
    Metric("loss_step_20", "Loss after 20 steps", "loss_after_20_steps", "loss"),
    Metric("first_below_0p9", "First step below 0.9", "first_step_below_0p9", "step"),
    Metric("first_below_0p88", "First step below 0.88", "first_step_below_0p88", "step"),
    Metric("first_below_0p87", "First step below 0.87", "first_step_below_0p87", "step"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot cifar_overfit_search_bs_MN.log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file to plot. Defaults to {DEFAULT_LOG.name}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG/CSV outputs. Defaults to {DEFAULT_OUTPUT_DIR.name}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_label(run: Run) -> str:
    return f"bs={run.batch_size} N={run.N} M={run.M} run={run.run}"


def sorted_runs(runs: list[Run]) -> list[Run]:
    return sorted(runs, key=lambda run: (run.batch_size, run.N, run.M, run.run))


def ranked_runs(runs: list[Run]) -> list[Run]:
    return sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: (
            run.final_loss if run.final_loss is not None else math.inf,
            run.batch_size,
            run.N,
            run.M,
            run.run,
        ),
    )


def ranked_by_metric(runs: list[Run], metric: Metric) -> list[Run]:
    return sorted(
        runs,
        key=lambda run: (
            metric.sort_value(run)
            if metric.lower_is_better
            else -metric.sort_value(run),
            run.batch_size,
            run.N,
            run.M,
            run.run,
        ),
    )


def parse_summary_float(run: Run, key: str) -> float | None:
    value = run.summary.get(key)
    if value in {None, "none", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def total_evaluated_configs(run: Run) -> int:
    return sum(choice.evaluated_configs for choice in run.interval_choices)


def mean_loss(run: Run) -> float | None:
    if not run.train:
        return None
    return sum(point.loss for point in run.train) / len(run.train)


def min_loss(run: Run) -> tuple[float | None, int | None]:
    if not run.train:
        return None, None
    point = min(run.train, key=lambda item: (item.loss, item.step))
    return point.loss, point.step


def loss_at_step(run: Run, step: int) -> float | None:
    for point in run.train:
        if point.step == step:
            return point.loss
    return run.final_loss if step == max_total_steps([run]) else None


def first_step_below(run: Run, threshold: float):
    for point in sorted(run.train, key=lambda item: item.step):
        if point.loss < threshold:
            return point
    return None


def max_total_steps(runs: list[Run]) -> int:
    return max((point.total_steps for run in runs for point in run.train), default=0)


def pareto_frontier_runs(runs: list[Run], threshold: float = 0.87) -> list[Run]:
    candidates = [
        run
        for run in runs
        if parse_summary_float(run, "Run seconds") is not None
        and first_step_below(run, threshold) is not None
    ]
    frontier = []
    for run in candidates:
        run_seconds = parse_summary_float(run, "Run seconds")
        run_step = first_step_below(run, threshold).step
        dominated = False
        for other in candidates:
            if other is run:
                continue
            other_seconds = parse_summary_float(other, "Run seconds")
            other_step = first_step_below(other, threshold).step
            if (
                other_seconds <= run_seconds
                and other_step <= run_step
                and (other_seconds < run_seconds or other_step < run_step)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(run)
    return sorted(
        frontier,
        key=lambda item: (
            parse_summary_float(item, "Run seconds"),
            first_step_below(item, threshold).step,
            item.N,
            item.M,
            item.run,
        ),
    )


def write_csvs(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "N",
                "M",
                "initial_loss",
                "final_loss",
                "mean_loss",
                "min_loss",
                "min_loss_step",
                "final_muon_lr",
                "final_muon_momentum",
                "final_cooldown_lr",
                "final_cooldown_momentum",
                "interval_searches",
                "evaluated_configs",
                "run_seconds",
                *[metric.csv_name for metric in METRICS],
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            best_loss, best_step = min_loss(run)
            writer.writerow(
                {
                    "run": run.run,
                    "batch_size": run.batch_size,
                    "N": run.N,
                    "M": run.M,
                    "initial_loss": run.initial_loss,
                    "final_loss": run.final_loss,
                    "mean_loss": mean_loss(run),
                    "min_loss": best_loss,
                    "min_loss_step": best_step,
                    "final_muon_lr": run.final_muon_lr,
                    "final_muon_momentum": run.final_muon_momentum,
                    "final_cooldown_lr": parse_summary_float(run, "Final cooldown lr"),
                    "final_cooldown_momentum": parse_summary_float(
                        run, "Final cooldown mom"
                    ),
                    "interval_searches": len(run.interval_choices),
                    "evaluated_configs": total_evaluated_configs(run),
                    "run_seconds": parse_summary_float(run, "Run seconds"),
                    **{
                        metric.csv_name: metric.value(run)
                        for metric in METRICS
                    },
                }
            )

    with (output_dir / "train_losses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "N",
                "M",
                "step",
                "total_steps",
                "loss",
                "head_lr",
                "muon_lr",
                "muon_momentum",
                "muon_nesterov",
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            for point in run.train:
                writer.writerow(
                    {
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "N": run.N,
                        "M": run.M,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                        "muon_momentum": point.muon_momentum,
                        "muon_nesterov": point.muon_nesterov,
                    }
                )

    with (output_dir / "search_choices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "N",
                "M",
                "interval_index",
                "start_step",
                "muon_lr",
                "muon_momentum",
                "cooldown_muon_lr",
                "cooldown_muon_momentum",
                "interval_loss",
                "final_loss",
                "evaluated_interval_configs",
                "evaluated_configs",
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            for choice in run.interval_choices:
                writer.writerow(
                    {
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "N": run.N,
                        "M": run.M,
                        "interval_index": choice.interval_index,
                        "start_step": choice.start_step,
                        "muon_lr": choice.muon_lr,
                        "muon_momentum": choice.muon_momentum,
                        "cooldown_muon_lr": choice.cooldown_muon_lr,
                        "cooldown_muon_momentum": choice.cooldown_muon_momentum,
                        "interval_loss": choice.interval_loss,
                        "final_loss": choice.final_loss,
                        "evaluated_interval_configs": (
                            choice.evaluated_interval_configs
                        ),
                        "evaluated_configs": choice.evaluated_configs,
                    }
                )

    with (output_dir / "metric_rankings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric",
                "rank",
                "value",
                "run",
                "batch_size",
                "N",
                "M",
                "final_loss",
                "final_muon_lr",
                "final_muon_momentum",
            ],
        )
        writer.writeheader()
        for metric in METRICS:
            rank = 0
            for run in ranked_by_metric(runs, metric):
                value = metric.value(run)
                if value is None:
                    continue
                rank += 1
                writer.writerow(
                    {
                        "metric": metric.csv_name,
                        "rank": rank,
                        "value": value,
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "N": run.N,
                        "M": run.M,
                        "final_loss": run.final_loss,
                        "final_muon_lr": run.final_muon_lr,
                        "final_muon_momentum": run.final_muon_momentum,
                    }
                )


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    lines = [
        "CIFAR overfit batch-size/N/M search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        "Metrics are ranked lower-is-better.",
        "",
    ]

    for batch_size, batch_runs in group_by_batch(sorted_runs(runs)).items():
        lines.extend([f"Batch size: {batch_size}", ""])
        frontier = pareto_frontier_runs(batch_runs, threshold=0.87)
        lines.append("Pareto frontier: first step below 0.87 vs run seconds")
        for run in frontier:
            point = first_step_below(run, 0.87)
            run_seconds = parse_summary_float(run, "Run seconds")
            lines.append(
                f"  run={run.run} N={run.N} M={run.M} "
                f"first_step_below_0p87={point.step} "
                f"run_seconds={format_number(run_seconds)} "
                f"final_lr={format_number(run.final_muon_lr)} "
                f"final_momentum={format_number(run.final_muon_momentum)}"
            )
        if not frontier:
            lines.append("  No runs reached 0.87.")
        lines.append("")

        for metric in METRICS:
            metric_ranked = [
                run
                for run in ranked_by_metric(batch_runs, metric)
                if metric.value(run) is not None
            ]
            lines.append(metric.title)
            for rank, run in enumerate(metric_ranked, start=1):
                value = metric.value(run)
                run_seconds = parse_summary_float(run, "Run seconds")
                lines.append(
                    f"  {rank:2d}. run={run.run} N={run.N} M={run.M} "
                    f"{metric.csv_name}={metric.format_value(value)} "
                    f"run_seconds={format_number(run_seconds)} "
                    f"final_lr={format_number(run.final_muon_lr)} "
                    f"final_momentum={format_number(run.final_muon_momentum)}"
                )
            missing = len(batch_runs) - len(metric_ranked)
            if missing:
                lines.append(f"  {missing} runs missing this metric")
            lines.append("")

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def median(values: list[float]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def group_by_batch(runs: list[Run]) -> dict[int, list[Run]]:
    groups: dict[int, list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.batch_size].append(run)
    return dict(sorted(groups.items()))


def group_by_nm(runs: list[Run]) -> dict[tuple[int, int], list[Run]]:
    groups: dict[tuple[int, int], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.N, run.M)].append(run)
    return dict(sorted(groups.items()))


def metric_matrix(
    runs: list[Run],
    metric: Metric,
    batch_size: int,
    n_values: list[int],
    m_values: list[int],
) -> list[list[float]]:
    batch_runs = [run for run in runs if run.batch_size == batch_size]
    return [
        [
            next(
                (
                    value
                    for run in batch_runs
                    if run.N == n_steps
                    and run.M == m_steps
                    and (value := metric.value(run)) is not None
                ),
                math.nan,
            )
            for m_steps in m_values
        ]
        for n_steps in n_values
    ]


def annotate_metric_grid(
    ax,
    matrix: list[list[float]],
    metric: Metric,
    vmin: float,
    vmax: float,
) -> None:
    threshold = vmin + 0.55 * (vmax - vmin) if vmax > vmin else vmax
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            if math.isnan(value):
                continue
            color = "white" if value > threshold else "black"
            ax.text(
                col,
                row,
                metric.format_value(value),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )


def plot_metric_grids(runs: list[Run], output_dir: Path) -> None:
    batch_sizes = sorted({run.batch_size for run in runs})
    n_values = sorted({run.N for run in runs})
    m_values = sorted({run.M for run in runs})
    fig, axes = plt.subplots(
        len(METRICS),
        len(batch_sizes),
        figsize=(4.8 * len(batch_sizes), 3.25 * len(METRICS)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colormaps = {}
    for metric in METRICS:
        cmap = plt.get_cmap("viridis_r" if metric.lower_is_better else "viridis").copy()
        cmap.set_bad("#eeeeee")
        colormaps[metric.key] = cmap

    for row, metric in enumerate(METRICS):
        values = [metric.value(run) for run in runs if metric.value(run) is not None]
        if not values:
            continue
        vmin, vmax = min(values), max(values)
        image = None
        for col, batch_size in enumerate(batch_sizes):
            ax = axes[row][col]
            matrix = metric_matrix(runs, metric, batch_size, n_values, m_values)
            image = ax.imshow(
                matrix,
                cmap=colormaps[metric.key],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{metric.title}\nbs={batch_size}", fontsize=10)
            ax.set_xticks(range(len(m_values)), labels=m_values)
            ax.set_yticks(range(len(n_values)), labels=n_values)
            if row == len(METRICS) - 1:
                ax.set_xlabel("M")
            if col == 0:
                ax.set_ylabel("N")

            batch_runs = [
                run
                for run in runs
                if run.batch_size == batch_size and metric.value(run) is not None
            ]
            if batch_runs:
                best = ranked_by_metric(batch_runs, metric)[0]
                ax.scatter(
                    [m_values.index(best.M)],
                    [n_values.index(best.N)],
                    s=160,
                    facecolors="none",
                    edgecolors="#d62728",
                    linewidths=2,
                )
            annotate_metric_grid(ax, matrix, metric, vmin, vmax)

        if image is not None:
            fig.colorbar(
                image,
                ax=axes[row, :].ravel().tolist(),
                fraction=0.025,
                pad=0.01,
            )

    fig.suptitle("Metric rankings by batch size, N, and M")
    fig.savefig(output_dir / "metric_grids.png", dpi=180)
    plt.close(fig)


def plot_final_loss_by_config(runs: list[Run], output_dir: Path) -> None:
    batch_groups = group_by_batch(sorted_runs(runs))
    n_values = sorted({run.N for run in runs})
    m_values = sorted({run.M for run in runs})
    fig, axes = plt.subplots(
        1,
        len(batch_groups),
        figsize=(5.2 * len(batch_groups), 4.7),
        squeeze=False,
        sharey=True,
    )
    losses = [run.final_loss for run in runs if run.final_loss is not None]
    vmin, vmax = min(losses), max(losses)

    for ax, (batch_size, batch_runs) in zip(axes.flat, batch_groups.items()):
        matrix = [
            [
                next(
                    (
                        run.final_loss
                        for run in batch_runs
                        if run.N == n_steps and run.M == m_steps
                    ),
                    math.nan,
                )
                for m_steps in m_values
            ]
            for n_steps in n_values
        ]
        image = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("M cooldown steps")
        ax.set_xticks(range(len(m_values)), labels=m_values)
        ax.set_yticks(range(len(n_values)), labels=n_values)
        ax.set_ylabel("N search interval")

        best_run = min(
            [run for run in batch_runs if run.final_loss is not None],
            key=lambda run: (run.final_loss, run.N, run.M),
        )
        ax.scatter(
            [m_values.index(best_run.M)],
            [n_values.index(best_run.N)],
            s=180,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2,
        )

        for row, n_steps in enumerate(n_values):
            for col, m_steps in enumerate(m_values):
                value = matrix[row][col]
                if math.isnan(value):
                    continue
                ax.text(
                    col,
                    row,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > (vmin + vmax) / 2 else "black",
                )

    fig.colorbar(image, ax=axes.ravel().tolist(), label="Final train loss")
    fig.suptitle("Final loss by batch size, N, and M")
    fig.savefig(output_dir / "final_loss_by_config.png", dpi=180)
    plt.close(fig)


def plot_loss_curves(runs: list[Run], output_dir: Path) -> None:
    batch_groups = group_by_batch(sorted_runs(runs))
    fig, axes = plt.subplots(
        len(batch_groups),
        1,
        figsize=(11, 3.5 * len(batch_groups)),
        sharex=True,
        sharey=True,
    )
    if len(batch_groups) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10")

    for ax, (batch_size, batch_runs) in zip(axes, batch_groups.items()):
        for run in batch_runs:
            if not run.train:
                continue
            ax.plot(
                [point.step for point in run.train],
                [point.loss for point in run.train],
                color=colors((run.N - 1) % 10),
                linestyle=["-", "--", ":", "-."][run.M % 4],
                linewidth=1.5,
                alpha=0.78,
                label=f"N={run.N} M={run.M}",
            )
        ax.set_title(f"Train loss curves, batch_size={batch_size}")
        ax.set_ylabel("Train loss")
        style_axes(ax)

    axes[-1].set_xlabel("Step")
    handles, labels = axes[0].get_legend_handles_labels()
    keep = {}
    for handle, label in zip(handles, labels):
        keep.setdefault(label, handle)
    fig.legend(
        keep.values(),
        keep.keys(),
        loc="lower center",
        ncol=10,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_selected_hparams(runs: list[Run], output_dir: Path) -> None:
    batch_groups = group_by_batch(sorted_runs(runs))
    fig, axes = plt.subplots(
        len(batch_groups),
        3,
        figsize=(14, 3.4 * len(batch_groups)),
        sharex="col",
    )
    if len(batch_groups) == 1:
        axes = [axes]

    for row_axes, (batch_size, batch_runs) in zip(axes, batch_groups.items()):
        ax_lr, ax_momentum, ax_loss = row_axes
        for run in batch_runs:
            choices = run.interval_choices
            if not choices:
                continue
            steps = [choice.start_step for choice in choices]
            label = f"N={run.N} M={run.M}"
            ax_lr.plot(
                steps,
                [choice.muon_lr for choice in choices],
                linewidth=1.2,
                alpha=0.75,
                label=label,
            )
            ax_momentum.plot(
                steps,
                [choice.muon_momentum for choice in choices],
                linewidth=1.2,
                alpha=0.75,
                label=label,
            )
            ax_loss.plot(
                steps,
                [choice.final_loss for choice in choices],
                linewidth=1.2,
                alpha=0.75,
                label=label,
            )

        ax_lr.set_title(f"batch_size={batch_size}: selected LR")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("Muon LR")
        style_axes(ax_lr)

        ax_momentum.set_title("selected momentum")
        ax_momentum.set_ylim(-0.03, 0.93)
        ax_momentum.set_ylabel("Momentum")
        style_axes(ax_momentum)

        ax_loss.set_title("search final loss")
        ax_loss.set_ylabel("Loss")
        style_axes(ax_loss)

    for ax in axes[-1]:
        ax.set_xlabel("Interval start step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    keep = {}
    for handle, label in zip(handles, labels):
        keep.setdefault(label, handle)
    fig.legend(
        keep.values(),
        keep.keys(),
        loc="lower center",
        ncol=10,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "selected_hparams.png", dpi=180)
    plt.close(fig)


def plot_threshold_runtime_tradeoff(runs: list[Run], output_dir: Path) -> None:
    thresholds = [0.88, 0.87]
    batch_groups = group_by_batch(sorted_runs(runs))
    n_values = sorted({run.N for run in runs})
    m_values = sorted({run.M for run in runs})
    markers = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(
        len(thresholds),
        len(batch_groups),
        figsize=(5.4 * len(batch_groups), 4.2 * len(thresholds)),
        sharex="col",
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    for row, threshold in enumerate(thresholds):
        all_steps = [
            point.step
            for run in runs
            if (point := first_step_below(run, threshold)) is not None
        ]
        y_max = max(all_steps, default=30) + 1

        for col, (batch_size, batch_runs) in enumerate(batch_groups.items()):
            ax = axes[row][col]
            missing = 0
            for run in batch_runs:
                run_seconds = parse_summary_float(run, "Run seconds")
                point = first_step_below(run, threshold)
                if run_seconds is None or point is None:
                    missing += 1
                    continue
                ax.scatter(
                    [run_seconds],
                    [point.step],
                    s=75,
                    color=cmap((run.N - 1) % 10),
                    marker=markers[run.M % len(markers)],
                    edgecolors="black",
                    linewidths=0.5,
                )

            suffix = f" ({missing} missing)" if missing else ""
            ax.set_title(f"batch_size={batch_size}{suffix}")
            if row == len(thresholds) - 1:
                ax.set_xlabel("Run seconds")
            if col == 0:
                ax.set_ylabel(f"First step below {threshold:g}")
            ax.set_ylim(0, y_max)
            style_axes(ax)

    n_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=f"N={n_value}",
            markerfacecolor=cmap((n_value - 1) % 10),
            markeredgecolor="black",
            markersize=7,
        )
        for n_value in n_values
    ]
    m_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[m_value % len(markers)],
            linestyle="None",
            label=f"M={m_value}",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            markersize=7,
        )
        for m_value in m_values
    ]
    fig.legend(
        handles=n_handles + m_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(n_handles) + len(m_handles),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Speed vs. first step below threshold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output_dir / RUNTIME_TRADEOFF_PLOT, dpi=180)
    plt.close(fig)


def plot_pareto_frontier_curves(runs: list[Run], output_dir: Path) -> None:
    threshold = 0.87
    batch_groups = group_by_batch(sorted_runs(runs))
    fig, axes = plt.subplots(
        len(batch_groups),
        3,
        figsize=(15, 3.7 * len(batch_groups)),
        sharex="col",
    )
    if len(batch_groups) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for row_axes, (batch_size, batch_runs) in zip(axes, batch_groups.items()):
        ax_lr, ax_momentum, ax_loss = row_axes
        frontier = pareto_frontier_runs(batch_runs, threshold=threshold)
        for index, run in enumerate(frontier):
            points = sorted(run.train, key=lambda item: item.step)
            if not points:
                continue
            color = cmap(index % 10)
            point = first_step_below(run, threshold)
            run_seconds = parse_summary_float(run, "Run seconds")
            label = (
                f"run={run.run} N={run.N} M={run.M}, "
                f"step={point.step}, sec={format_number(run_seconds)}"
            )
            steps = [item.step for item in points]
            ax_lr.plot(
                steps,
                [item.muon_lr for item in points],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                color=color,
                label=label,
            )
            ax_momentum.plot(
                steps,
                [item.muon_momentum for item in points],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                color=color,
                label=label,
            )
            ax_loss.plot(
                steps,
                [item.loss for item in points],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                color=color,
                label=label,
            )

        ax_lr.set_title(f"batch_size={batch_size}: LR")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("Muon LR")
        style_axes(ax_lr)

        ax_momentum.set_title("Momentum")
        ax_momentum.set_ylim(-0.03, 1.03)
        ax_momentum.set_ylabel("Muon momentum")
        style_axes(ax_momentum)

        ax_loss.set_title("Train loss")
        ax_loss.set_ylabel("Loss")
        style_axes(ax_loss)

        ax_loss.legend(fontsize=7, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("Step")
    fig.suptitle("Pareto frontier curves: first step below 0.87 vs run seconds")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / PARETO_CURVES_PLOT, dpi=180)
    plt.close(fig)


def plot_top_runs(runs: list[Run], output_dir: Path, top_k: int) -> None:
    top_runs = ranked_runs(runs)[:top_k]
    if not top_runs:
        return

    fig, (ax_loss, ax_lr, ax_momentum) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=False
    )
    for run in top_runs:
        label = run_label(run)
        ax_loss.plot(
            [point.step for point in run.train],
            [point.loss for point in run.train],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=label,
        )
        if run.interval_choices:
            steps = [choice.start_step for choice in run.interval_choices]
            ax_lr.plot(
                steps,
                [choice.muon_lr for choice in run.interval_choices],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=label,
            )
            ax_momentum.plot(
                steps,
                [choice.muon_momentum for choice in run.interval_choices],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=label,
            )

    ax_loss.set_title(f"Top {len(top_runs)} runs by final loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train loss")
    style_axes(ax_loss)

    ax_lr.set_title("Selected interval LR")
    ax_lr.set_xlabel("Interval start step")
    ax_lr.set_ylabel("Muon LR")
    ax_lr.set_yscale("log")
    style_axes(ax_lr)

    ax_momentum.set_title("Selected interval momentum")
    ax_momentum.set_xlabel("Interval start step")
    ax_momentum.set_ylabel("Momentum")
    ax_momentum.set_ylim(-0.03, 0.93)
    style_axes(ax_momentum)

    ax_loss.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "top_runs.png", dpi=180)
    plt.close(fig)


def plot_search_counts(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ordered = sorted_runs(runs)
    x_positions = range(len(ordered))
    bars = ax.bar(
        x_positions,
        [total_evaluated_configs(run) for run in ordered],
        color=[plt.get_cmap("tab10")((run.batch_size // 500) % 10) for run in ordered],
    )
    ax.set_title("Search cost by run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Evaluated configs")
    ax.set_xticks(
        list(x_positions),
        labels=[f"{run.run}\n{run.batch_size}\nN{run.N}M{run.M}" for run in ordered],
        fontsize=7,
    )
    style_axes(ax)

    best = ranked_runs(runs)[0]
    best_index = ordered.index(best)
    bars[best_index].set_edgecolor("#d62728")
    bars[best_index].set_linewidth(2.0)
    fig.tight_layout()
    fig.savefig(output_dir / "search_counts.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], output_dir: Path) -> None:
    plot_metric_grids(runs, output_dir)
    plot_selected_hparams(runs, output_dir)
    plot_threshold_runtime_tradeoff(runs, output_dir)
    plot_pareto_frontier_curves(runs, output_dir)


def cleanup_outputs(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and path.name not in EXPECTED_OUTPUTS:
            path.unlink()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")

    write_summary(runs, args.log, output_dir)
    plot_all(runs, output_dir)
    cleanup_outputs(output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

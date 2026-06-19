#!/usr/bin/env python3
"""Plot the linear-scheduler CIFAR overfit LR-search log."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "cifar_overfit_search_linear.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_linear_plots"

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 ]+):\s+(?P<value>\S+)")


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    csv_name: str
    value_kind: str
    lower_is_better: bool = True

    def value(self, run: "Run") -> float | None:
        if self.key.startswith("loss_step_"):
            step = int(self.key.rsplit("_", 1)[1])
            return loss_at_step(run, step)
        if self.key.startswith("first_below_"):
            threshold = float(self.key.rsplit("_", 1)[1].replace("p", "."))
            point = first_step_below(run, threshold)
            return float(point) if point is not None else None
        raise ValueError(f"Unknown metric: {self.key}")

    def sort_value(self, run: "Run") -> float:
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
    Metric(
        "first_below_0p88", "First step below 0.88", "first_step_below_0p88", "step"
    ),
    Metric(
        "first_below_0p87", "First step below 0.87", "first_step_below_0p87", "step"
    ),
]


@dataclass
class TrainPoint:
    step: int
    total_steps: int
    loss: float
    head_lr: float
    muon_lr: float
    muon_momentum: float
    muon_nesterov: str


@dataclass
class SearchChoice:
    start_step: int
    start_muon_lr: float | None
    end_muon_lr: float | None
    muon_lr: float | None
    muon_momentum: float | None
    interval_loss: float | None
    final_loss: float | None
    evaluated_interval_configs: int
    evaluated_configs: int


@dataclass
class Run:
    run: int
    batch_size: int | None = None
    n_steps: int | None = None
    m_steps: int | None = None
    interval_scheduler: str | None = None
    initial_muon_lr: float | None = None
    initial_muon_momentum: float | None = None
    train: list[TrainPoint] = field(default_factory=list)
    choices: list[SearchChoice] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def final_loss(self) -> float | None:
        value = parse_optional_float(self.summary.get("Final train loss"))
        if value is not None:
            return value
        return self.train[-1].loss if self.train else None

    @property
    def initial_loss(self) -> float | None:
        value = parse_optional_float(self.summary.get("Initial train loss"))
        if value is not None:
            return value
        return self.train[0].loss if self.train else None

    @property
    def run_seconds(self) -> float | None:
        return parse_optional_float(self.summary.get("Run seconds"))

    @property
    def completed(self) -> bool:
        return "Final train loss" in self.summary


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"none", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_kv_line(line: str) -> dict[str, str]:
    return {match["key"]: match["value"] for match in KV_RE.finditer(line)}


def style_axes(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{value:g}"


def parse_log(log_path: Path) -> list[Run]:
    runs: dict[int, Run] = {}
    current: Run | None = None

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("cifar_baseline2_overfit_n_search "):
                fields = parse_kv_line(line)
                run_id = int(fields["run"])
                current = Run(
                    run=run_id,
                    batch_size=parse_optional_int(fields.get("batch_size")),
                    n_steps=parse_optional_int(fields.get("N")),
                    m_steps=parse_optional_int(fields.get("M")),
                    interval_scheduler=fields.get("interval_scheduler"),
                    initial_muon_lr=parse_optional_float(fields.get("initial_muon_lr")),
                    initial_muon_momentum=parse_optional_float(
                        fields.get("initial_muon_momentum")
                    ),
                )
                runs[run_id] = current
                continue

            if line.startswith("train_loss "):
                fields = parse_kv_line(line)
                run_id = int(fields["run"])
                run = runs.setdefault(run_id, Run(run=run_id))
                current = run
                step_text = fields["step"]
                step, total_steps = [int(part) for part in step_text.split("/", 1)]
                run.train.append(
                    TrainPoint(
                        step=step,
                        total_steps=total_steps,
                        loss=float(fields["loss"]),
                        head_lr=float(fields["head_lr"]),
                        muon_lr=float(fields["muon_lr"]),
                        muon_momentum=float(fields["muon_momentum"]),
                        muon_nesterov=fields["muon_nesterov"],
                    )
                )
                continue

            if line.startswith("best_interval_") and current is not None:
                fields = parse_kv_line(line)
                last_step = current.train[-1].step if current.train else -1
                current.choices.append(
                    SearchChoice(
                        start_step=last_step + 1,
                        start_muon_lr=parse_optional_float(
                            fields.get("best_interval_start_muon_lr")
                        ),
                        end_muon_lr=parse_optional_float(
                            fields.get("best_interval_end_muon_lr")
                        ),
                        muon_lr=parse_optional_float(
                            fields.get("best_interval_muon_lr")
                        ),
                        muon_momentum=parse_optional_float(
                            fields.get("best_interval_muon_momentum")
                        ),
                        interval_loss=parse_optional_float(fields.get("interval_loss")),
                        final_loss=parse_optional_float(fields.get("final_loss")),
                        evaluated_interval_configs=int(
                            fields.get("evaluated_interval_configs", "0")
                        ),
                        evaluated_configs=int(fields.get("evaluated_configs", "0")),
                    )
                )
                continue

            summary_match = SUMMARY_RE.match(line)
            if summary_match and current is not None:
                current.summary[summary_match["key"].strip()] = summary_match[
                    "value"
                ].strip()

    return sorted(runs.values(), key=lambda item: item.run)


def sorted_runs(runs: list[Run]) -> list[Run]:
    return sorted(
        runs,
        key=lambda run: (
            run.batch_size if run.batch_size is not None else -1,
            run.n_steps if run.n_steps is not None else -1,
            run.run,
        ),
    )


def group_by_batch_size(runs: list[Run]) -> dict[int | None, list[Run]]:
    groups: dict[int | None, list[Run]] = defaultdict(list)
    for run in sorted_runs(runs):
        groups[run.batch_size].append(run)
    return dict(groups)


def first_step_below(run: Run, threshold: float) -> int | None:
    for point in sorted(run.train, key=lambda item: item.step):
        if point.loss < threshold:
            return point.step
    return None


def loss_at_step(run: Run, step: int) -> float | None:
    for point in run.train:
        if point.step == step:
            return point.loss
    if run.train and step == max(point.step for point in run.train):
        return run.final_loss
    return None


def ranked_by_metric(runs: list[Run], metric: Metric) -> list[Run]:
    return sorted(
        runs,
        key=lambda run: (
            metric.sort_value(run)
            if metric.lower_is_better
            else -metric.sort_value(run),
            run.batch_size if run.batch_size is not None else math.inf,
            run.n_steps if run.n_steps is not None else math.inf,
            run.m_steps if run.m_steps is not None else math.inf,
            run.run,
        ),
    )


def pareto_frontier_runs(runs: list[Run], threshold: float = 0.87) -> list[Run]:
    candidates = [
        run
        for run in runs
        if run.run_seconds is not None and first_step_below(run, threshold) is not None
    ]
    frontier = []
    for run in candidates:
        run_seconds = run.run_seconds
        run_step = first_step_below(run, threshold)
        dominated = False
        for other in candidates:
            if other is run:
                continue
            other_seconds = other.run_seconds
            other_step = first_step_below(other, threshold)
            if (
                other_seconds is not None
                and other_step is not None
                and run_seconds is not None
                and run_step is not None
                and other_seconds <= run_seconds
                and other_step <= run_step
                and (other_seconds < run_seconds or other_step < run_step)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(run)
    return sorted(
        frontier,
        key=lambda run: (
            run.run_seconds if run.run_seconds is not None else math.inf,
            first_step_below(run, threshold) or math.inf,
            run.n_steps if run.n_steps is not None else math.inf,
            run.m_steps if run.m_steps is not None else math.inf,
            run.run,
        ),
    )


def total_evaluated_configs(run: Run) -> int:
    return sum(choice.evaluated_configs for choice in run.choices)


def selected_lr_curve(run: Run) -> tuple[list[int], list[float]]:
    if not run.choices:
        points = sorted(run.train, key=lambda item: item.step)
        return [point.step for point in points], [point.muon_lr for point in points]

    x_values: list[int] = []
    y_values: list[float] = []
    n_steps = run.n_steps or 1
    for choice in run.choices:
        if choice.start_muon_lr is None or choice.end_muon_lr is None:
            continue
        x_values.extend([choice.start_step, choice.start_step + n_steps])
        y_values.extend([choice.start_muon_lr, choice.end_muon_lr])
    return x_values, y_values


def best_completed_run(runs: list[Run]) -> Run | None:
    completed = [run for run in runs if run.final_loss is not None and run.completed]
    if not completed:
        return None
    return min(completed, key=lambda run: (run.final_loss, run.batch_size or 0, run.run))


def write_csvs(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "N",
                "M",
                "completed",
                "initial_loss",
                "final_loss",
                "run_seconds",
                "final_muon_lr",
                "final_muon_momentum",
                "first_step_below_0p90",
                "first_step_below_0p88",
                "first_step_below_0p87",
                "search_choices",
                "evaluated_configs",
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            writer.writerow(
                {
                    "run": run.run,
                    "batch_size": run.batch_size,
                    "N": run.n_steps,
                    "M": run.m_steps,
                    "completed": run.completed,
                    "initial_loss": run.initial_loss,
                    "final_loss": run.final_loss,
                    "run_seconds": run.run_seconds,
                    "final_muon_lr": parse_optional_float(
                        run.summary.get("Final Muon lr")
                    ),
                    "final_muon_momentum": parse_optional_float(
                        run.summary.get("Final Muon momentum")
                    ),
                    "first_step_below_0p90": first_step_below(run, 0.90),
                    "first_step_below_0p88": first_step_below(run, 0.88),
                    "first_step_below_0p87": first_step_below(run, 0.87),
                    "search_choices": len(run.choices),
                    "evaluated_configs": total_evaluated_configs(run),
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
                        "N": run.n_steps,
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
                "start_step",
                "start_muon_lr",
                "end_muon_lr",
                "selected_muon_lr",
                "selected_muon_momentum",
                "interval_loss",
                "final_loss",
                "evaluated_interval_configs",
                "evaluated_configs",
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            for choice in run.choices:
                writer.writerow(
                    {
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "N": run.n_steps,
                        "start_step": choice.start_step,
                        "start_muon_lr": choice.start_muon_lr,
                        "end_muon_lr": choice.end_muon_lr,
                        "selected_muon_lr": choice.muon_lr,
                        "selected_muon_momentum": choice.muon_momentum,
                        "interval_loss": choice.interval_loss,
                        "final_loss": choice.final_loss,
                        "evaluated_interval_configs": choice.evaluated_interval_configs,
                        "evaluated_configs": choice.evaluated_configs,
                    }
                )


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    best = best_completed_run(runs)
    lines = [
        "CIFAR overfit linear scheduler search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        f"Completed runs: {sum(run.completed for run in runs)}",
        "Metrics are ranked lower-is-better.",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "Best completed run by final loss:",
                (
                    f"  run={best.run} batch_size={best.batch_size} N={best.n_steps} "
                    f"final_loss={best.final_loss:.6f} "
                    f"run_seconds={format_number(best.run_seconds)}"
                ),
                "",
            ]
        )

    for batch_size, batch_runs in group_by_batch_size(runs).items():
        lines.extend([f"Batch size: {batch_size}", ""])

        lines.append("Final loss")
        ranked = sorted(
            batch_runs,
            key=lambda run: (
                run.final_loss if run.final_loss is not None else math.inf,
                run.n_steps if run.n_steps is not None else math.inf,
            ),
        )
        for run in ranked:
            suffix = "" if run.completed else " partial"
            lines.append(
                f"  run={run.run} N={run.n_steps} final_loss="
                f"{format_number(run.final_loss)} first_below_0p87="
                f"{format_number(first_step_below(run, 0.87))} "
                f"evaluated_configs={total_evaluated_configs(run)}"
                f"{suffix}"
            )
        lines.append("")

        frontier = pareto_frontier_runs(batch_runs, threshold=0.87)
        lines.append("Pareto frontier: first step below 0.87 vs run seconds")
        for run in frontier:
            lines.append(
                f"  run={run.run} N={run.n_steps} M={run.m_steps} "
                f"first_step_below_0p87={first_step_below(run, 0.87)} "
                f"run_seconds={format_number(run.run_seconds)} "
                f"final_lr={format_number(parse_optional_float(run.summary.get('Final Muon lr')))} "
                f"final_momentum="
                f"{format_number(parse_optional_float(run.summary.get('Final Muon momentum')))}"
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
                lines.append(
                    f"  {rank:2d}. run={run.run} N={run.n_steps} M={run.m_steps} "
                    f"{metric.csv_name}={metric.format_value(value)} "
                    f"run_seconds={format_number(run.run_seconds)} "
                    f"final_lr={format_number(parse_optional_float(run.summary.get('Final Muon lr')))} "
                    f"final_momentum="
                    f"{format_number(parse_optional_float(run.summary.get('Final Muon momentum')))}"
                )
            missing = len(batch_runs) - len(metric_ranked)
            if missing:
                lines.append(f"  {missing} runs missing this metric")
            lines.append("")

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_loss_curves(runs: list[Run], output_dir: Path) -> None:
    groups = group_by_batch_size(runs)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(11, 3.5 * len(groups)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, (batch_size, batch_runs) in zip(axes.flat, groups.items()):
        for run in batch_runs:
            points = sorted(run.train, key=lambda item: item.step)
            if not points:
                continue
            label = f"N={run.n_steps} run={run.run}"
            linestyle = "-" if run.completed else "--"
            ax.plot(
                [point.step for point in points],
                [point.loss for point in points],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                linestyle=linestyle,
                label=label,
            )
        ax.set_title(f"Train loss, batch_size={batch_size}")
        ax.set_ylabel("Loss")
        ax.axhline(0.87, color="#888888", linewidth=1.0, linestyle=":", label="0.87")
        style_axes(ax)
        ax.legend(fontsize=8, ncol=3)

    axes[-1, 0].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_curves.png", dpi=180)
    plt.close(fig)


def plot_final_loss_heatmap(runs: list[Run], output_dir: Path) -> None:
    batch_sizes = sorted(
        batch_size for batch_size in {run.batch_size for run in runs} if batch_size
    )
    n_values = sorted(n_steps for n_steps in {run.n_steps for run in runs} if n_steps)
    matrix = []
    for batch_size in batch_sizes:
        row = []
        for n_steps in n_values:
            run = next(
                (
                    item
                    for item in runs
                    if item.batch_size == batch_size and item.n_steps == n_steps
                ),
                None,
            )
            row.append(run.final_loss if run is not None and run.completed else math.nan)
        matrix.append(row)

    if not matrix:
        return

    fig, ax = plt.subplots(figsize=(1.05 * len(n_values) + 4, 4.8))
    image = ax.imshow(matrix, cmap="viridis_r")
    ax.set_title("Completed final loss by batch size and N")
    ax.set_xlabel("N search interval")
    ax.set_ylabel("Batch size")
    ax.set_xticks(range(len(n_values)), labels=n_values)
    ax.set_yticks(range(len(batch_sizes)), labels=batch_sizes)

    values = [value for row in matrix for value in row if not math.isnan(value)]
    midpoint = (min(values) + max(values)) / 2 if values else 0.0
    for row, batch_size in enumerate(batch_sizes):
        for col, n_steps in enumerate(n_values):
            value = matrix[row][col]
            if math.isnan(value):
                label = "partial" if any(
                    run.batch_size == batch_size and run.n_steps == n_steps for run in runs
                ) else ""
                if label:
                    ax.text(col, row, label, ha="center", va="center", fontsize=8)
                continue
            ax.text(
                col,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > midpoint else "black",
            )

    fig.colorbar(image, ax=ax, label="Final train loss")
    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_heatmap.png", dpi=180)
    plt.close(fig)


def plot_selected_hparams(runs: list[Run], output_dir: Path) -> None:
    groups = group_by_batch_size(runs)
    fig, axes = plt.subplots(
        len(groups),
        3,
        figsize=(14, 3.4 * len(groups)),
        sharex="col",
        squeeze=False,
    )

    for row_axes, (batch_size, batch_runs) in zip(axes, groups.items()):
        ax_lr, ax_momentum, ax_loss = row_axes
        for run in batch_runs:
            points = sorted(run.train, key=lambda item: item.step)
            if not points:
                continue
            label = f"N={run.n_steps}"
            lr_steps, lr_values = selected_lr_curve(run)
            ax_lr.plot(
                lr_steps,
                lr_values,
                marker="o",
                markersize=2.2,
                linewidth=1.2,
                label=label,
            )
            ax_momentum.plot(
                [point.step for point in points],
                [point.muon_momentum for point in points],
                marker="o",
                markersize=2.2,
                linewidth=1.2,
                label=label,
            )
            if run.choices:
                ax_loss.plot(
                    [choice.start_step for choice in run.choices],
                    [choice.final_loss for choice in run.choices],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.2,
                    label=label,
                )

        ax_lr.set_title(f"batch_size={batch_size}: selected LR")
        ax_lr.set_ylabel("Muon LR")
        style_axes(ax_lr)

        ax_momentum.set_title("selected momentum")
        ax_momentum.set_ylim(-0.03, 1.03)
        ax_momentum.set_ylabel("Momentum")
        style_axes(ax_momentum)

        ax_loss.set_title("selected search final loss")
        ax_loss.set_ylabel("Loss")
        style_axes(ax_loss)
        ax_loss.legend(fontsize=8, ncol=2)

    for ax in axes[-1]:
        ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_hparams.png", dpi=180)
    plt.close(fig)


def plot_selected_lr_subplots(runs: list[Run], output_dir: Path) -> None:
    ordered = sorted_runs(runs)
    if not ordered:
        return

    ncols = 4
    nrows = math.ceil(len(ordered) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
        squeeze=False,
    )

    for ax, run in zip(axes.flat, ordered):
        lr_steps, lr_values = selected_lr_curve(run)
        if lr_steps:
            ax.plot(
                lr_steps,
                lr_values,
                marker="o",
                markersize=2.4,
                linewidth=1.35,
                linestyle="-" if run.completed else "--",
            )
        ax.set_title(
            f"run={run.run} bs={run.batch_size} N={run.n_steps}",
            fontsize=9,
        )
        style_axes(ax)

    for ax in axes.flat[len(ordered) :]:
        ax.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Muon LR")
    for ax in axes[-1, :]:
        ax.set_xlabel("Step")

    fig.suptitle("Selected LR curves by run", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_lr_curves_by_run.png", dpi=180)
    plt.close(fig)


def plot_runtime_tradeoff(runs: list[Run], output_dir: Path) -> None:
    completed = [run for run in runs if run.completed and run.run_seconds is not None]
    if not completed:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    cmap = plt.get_cmap("tab10")
    for run in completed:
        ax.scatter(
            [run.run_seconds],
            [run.final_loss],
            s=85,
            color=cmap(((run.batch_size or 0) // 500) % 10),
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            f"bs={run.batch_size}\nN={run.n_steps}",
            (run.run_seconds, run.final_loss),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title("Runtime vs final train loss")
    ax.set_xlabel("Run seconds")
    ax.set_ylabel("Final train loss")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_final_loss.png", dpi=180)
    plt.close(fig)


def plot_search_costs(runs: list[Run], output_dir: Path) -> None:
    ordered = sorted_runs(runs)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_positions = list(range(len(ordered)))
    ax.bar(
        x_positions,
        [total_evaluated_configs(run) for run in ordered],
        color="#4c78a8",
    )
    ax.set_title("Evaluated configs by run")
    ax.set_xlabel("Run / batch size / N")
    ax.set_ylabel("Evaluated configs")
    ax.set_xticks(
        x_positions,
        labels=[f"{run.run}\n{run.batch_size}\nN{run.n_steps}" for run in ordered],
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "search_costs.png", dpi=180)
    plt.close(fig)


def plot_top_runs(runs: list[Run], output_dir: Path, top_k: int) -> None:
    ranked = sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: (run.final_loss, 0 if run.completed else 1),
    )[:top_k]
    if not ranked:
        return

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for run in ranked:
        points = sorted(run.train, key=lambda item: item.step)
        ax.plot(
            [point.step for point in points],
            [point.loss for point in points],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            linestyle="-" if run.completed else "--",
            label=(
                f"run={run.run} bs={run.batch_size} N={run.n_steps} "
                f"loss={format_number(run.final_loss)}"
            ),
        )
    ax.set_title(f"Top {len(ranked)} runs by observed final loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    style_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "top_runs.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], log_path: Path, output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(runs, output_dir)
    write_summary(runs, log_path, output_dir)
    plot_loss_curves(runs, output_dir)
    plot_final_loss_heatmap(runs, output_dir)
    plot_selected_hparams(runs, output_dir)
    plot_selected_lr_subplots(runs, output_dir)
    plot_runtime_tradeoff(runs, output_dir)
    plot_search_costs(runs, output_dir)
    plot_top_runs(runs, output_dir, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot cifar_overfit_search_linear.log."
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
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR.name}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of best observed runs to include in top_runs.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = parse_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")

    plot_all(runs, args.log, args.output_dir, args.top_k)

    best = best_completed_run(runs)
    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    if best is not None:
        print(
            f"Best completed run: run={best.run} batch_size={best.batch_size} "
            f"N={best.n_steps} final_loss={best.final_loss:.6f}"
        )
    partial = [run.run for run in runs if not run.completed]
    if partial:
        print(f"Partial runs included: {', '.join(map(str, partial))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot cifar_overfit_global_search.log.

The log is a compact sweep over batch size and interval scheduler. This script
parses the run configs, selected search choices, candidate interval losses, and
per-step train losses, then writes PNG plots plus CSV summaries.
"""

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
DEFAULT_LOG = HERE / "cifar_overfit_global_search.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_global_search_plots"

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 ]+):\s+(?P<value>\S+)")


@dataclass(frozen=True)
class TrainPoint:
    step: int
    total_steps: int
    loss: float
    head_lr: float
    muon_lr: float
    muon_momentum: float
    muon_nesterov: str


@dataclass(frozen=True)
class IntervalCandidate:
    run: int
    search_step: int
    start_muon_lr: float | None
    end_muon_lr: float | None
    muon_lr: float | None
    muon_momentum: float | None
    interval_loss: float | None
    final_loss: float | None = None


@dataclass(frozen=True)
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
    candidates: list[IntervalCandidate] = field(default_factory=list)
    choices: list[SearchChoice] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def initial_loss(self) -> float | None:
        value = parse_optional_float(self.summary.get("Initial train loss"))
        if value is not None:
            return value
        return self.train[0].loss if self.train else None

    @property
    def final_loss(self) -> float | None:
        value = parse_optional_float(self.summary.get("Final train loss"))
        if value is not None:
            return value
        return self.train[-1].loss if self.train else None

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


def format_number(value: float | int | None, precision: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{value:.{precision}g}"


def format_step(value: int | None) -> str:
    return "NA" if value is None else str(value)


def parse_log(log_path: Path) -> list[Run]:
    runs: dict[int, Run] = {}
    current: Run | None = None
    pending_candidate: IntervalCandidate | None = None

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
                pending_candidate = None
                continue

            if line.startswith("train_loss "):
                fields = parse_kv_line(line)
                run_id = int(fields["run"])
                run = runs.setdefault(run_id, Run(run=run_id))
                current = run
                step, total_steps = [
                    int(part) for part in fields["step"].split("/", 1)
                ]
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
                pending_candidate = None
                continue

            if line.startswith("interval_start_muon_lr=") and current is not None:
                fields = parse_kv_line(line)
                pending_candidate = IntervalCandidate(
                    run=current.run,
                    search_step=current.train[-1].step + 1 if current.train else 0,
                    start_muon_lr=parse_optional_float(
                        fields.get("interval_start_muon_lr")
                    ),
                    end_muon_lr=parse_optional_float(fields.get("interval_end_muon_lr")),
                    muon_lr=parse_optional_float(fields.get("interval_muon_lr")),
                    muon_momentum=parse_optional_float(
                        fields.get("interval_muon_momentum")
                    ),
                    interval_loss=parse_optional_float(fields.get("interval_loss")),
                )
                continue

            if (
                line.startswith("cooldown_muon_lr=")
                and current is not None
                and pending_candidate is not None
            ):
                fields = parse_kv_line(line)
                current.candidates.append(
                    IntervalCandidate(
                        run=pending_candidate.run,
                        search_step=pending_candidate.search_step,
                        start_muon_lr=pending_candidate.start_muon_lr,
                        end_muon_lr=pending_candidate.end_muon_lr,
                        muon_lr=pending_candidate.muon_lr,
                        muon_momentum=pending_candidate.muon_momentum,
                        interval_loss=pending_candidate.interval_loss,
                        final_loss=parse_optional_float(fields.get("final_loss")),
                    )
                )
                pending_candidate = None
                continue

            if line.startswith("best_interval_") and current is not None:
                fields = parse_kv_line(line)
                current.choices.append(
                    SearchChoice(
                        start_step=current.train[-1].step + 1 if current.train else 0,
                        start_muon_lr=parse_optional_float(
                            fields.get("best_interval_start_muon_lr")
                        ),
                        end_muon_lr=parse_optional_float(
                            fields.get("best_interval_end_muon_lr")
                        ),
                        muon_lr=parse_optional_float(fields.get("best_interval_muon_lr")),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot cifar_overfit_global_search.log."
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
        help=f"Directory for plots and CSV files. Defaults to {DEFAULT_OUTPUT_DIR.name}.",
    )
    return parser.parse_args()


def sorted_runs(runs: list[Run]) -> list[Run]:
    scheduler_order = {"exp_linear": 0, "linear": 1, "constant": 2}
    return sorted(
        runs,
        key=lambda run: (
            run.batch_size if run.batch_size is not None else -1,
            scheduler_order.get(run.interval_scheduler or "", 99),
            run.run,
        ),
    )


def run_label(run: Run) -> str:
    scheduler = run.interval_scheduler or "unknown"
    return f"bs={run.batch_size} {scheduler}"


def first_step_below(run: Run, threshold: float) -> int | None:
    for point in sorted(run.train, key=lambda item: item.step):
        if point.loss < threshold:
            return point.step
    return None


def total_evaluated_configs(run: Run) -> int:
    return sum(choice.evaluated_configs for choice in run.choices)


def final_summary_float(run: Run, key: str) -> float | None:
    return parse_optional_float(run.summary.get(key))


def best_run(runs: list[Run]) -> Run | None:
    completed = [run for run in runs if run.final_loss is not None]
    if not completed:
        return None
    return min(
        completed,
        key=lambda run: (
            run.final_loss if run.final_loss is not None else math.inf,
            run.batch_size if run.batch_size is not None else math.inf,
            run.run,
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
                "interval_scheduler",
                "completed",
                "initial_loss",
                "final_loss",
                "run_seconds",
                "final_start_lr",
                "final_end_lr",
                "final_muon_lr",
                "final_muon_momentum",
                "first_step_below_0p90",
                "first_step_below_0p88",
                "first_step_below_0p87",
                "candidate_configs",
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
                    "interval_scheduler": run.interval_scheduler,
                    "completed": run.completed,
                    "initial_loss": run.initial_loss,
                    "final_loss": run.final_loss,
                    "run_seconds": run.run_seconds,
                    "final_start_lr": final_summary_float(run, "Final start lr"),
                    "final_end_lr": final_summary_float(run, "Final end lr"),
                    "final_muon_lr": final_summary_float(run, "Final Muon lr"),
                    "final_muon_momentum": final_summary_float(
                        run, "Final Muon momentum"
                    ),
                    "first_step_below_0p90": first_step_below(run, 0.90),
                    "first_step_below_0p88": first_step_below(run, 0.88),
                    "first_step_below_0p87": first_step_below(run, 0.87),
                    "candidate_configs": len(run.candidates),
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
                "interval_scheduler",
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
            for point in sorted(run.train, key=lambda item: item.step):
                writer.writerow(
                    {
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "interval_scheduler": run.interval_scheduler,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                        "muon_momentum": point.muon_momentum,
                        "muon_nesterov": point.muon_nesterov,
                    }
                )

    with (output_dir / "interval_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "interval_scheduler",
                "search_step",
                "start_muon_lr",
                "end_muon_lr",
                "muon_lr",
                "muon_momentum",
                "interval_loss",
                "final_loss",
            ],
        )
        writer.writeheader()
        for run in sorted_runs(runs):
            for candidate in run.candidates:
                writer.writerow(
                    {
                        "run": run.run,
                        "batch_size": run.batch_size,
                        "interval_scheduler": run.interval_scheduler,
                        "search_step": candidate.search_step,
                        "start_muon_lr": candidate.start_muon_lr,
                        "end_muon_lr": candidate.end_muon_lr,
                        "muon_lr": candidate.muon_lr,
                        "muon_momentum": candidate.muon_momentum,
                        "interval_loss": candidate.interval_loss,
                        "final_loss": candidate.final_loss,
                    }
                )


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    best = best_run(runs)
    lines = [
        "CIFAR overfit global search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        f"Completed runs: {sum(run.completed for run in runs)}",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "Best run by final loss:",
                (
                    f"  run={best.run} batch_size={best.batch_size} "
                    f"scheduler={best.interval_scheduler} "
                    f"final_loss={best.final_loss:.6f} "
                    f"run_seconds={format_number(best.run_seconds)}"
                ),
                "",
            ]
        )

    for run in sorted_runs(runs):
        lines.append(
            f"run={run.run:>2} bs={run.batch_size:<5} "
            f"scheduler={run.interval_scheduler:<10} "
            f"final_loss={format_number(run.final_loss)} "
            f"first_below_0p87={format_step(first_step_below(run, 0.87))} "
            f"seconds={format_number(run.run_seconds)} "
            f"final_lr={format_number(final_summary_float(run, 'Final Muon lr'))} "
            f"mom={format_number(final_summary_float(run, 'Final Muon momentum'))}"
        )

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_train_loss_curves(runs: list[Run], output_dir: Path) -> None:
    groups: dict[int | None, list[Run]] = defaultdict(list)
    for run in sorted_runs(runs):
        groups[run.batch_size].append(run)

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(11, 3.3 * len(groups)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (batch_size, batch_runs) in zip(axes.flat, groups.items()):
        for run in batch_runs:
            points = sorted(run.train, key=lambda item: item.step)
            if not points:
                continue
            ax.plot(
                [point.step for point in points],
                [point.loss for point in points],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=f"{run.interval_scheduler} run={run.run}",
            )
        ax.axhline(0.87, color="#777777", linewidth=1.0, linestyle=":", label="0.87")
        ax.set_title(f"Train loss, batch_size={batch_size}")
        ax.set_ylabel("Loss")
        style_axes(ax)
        ax.legend(fontsize=8, ncol=4)

    axes[-1, 0].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_curves.png", dpi=180)
    plt.close(fig)


def plot_final_loss_heatmap(runs: list[Run], output_dir: Path) -> None:
    batch_sizes = sorted({run.batch_size for run in runs if run.batch_size is not None})
    schedulers = sorted(
        {run.interval_scheduler for run in runs if run.interval_scheduler is not None},
        key=lambda item: {"exp_linear": 0, "linear": 1, "constant": 2}.get(item, 99),
    )
    matrix: list[list[float]] = []
    for batch_size in batch_sizes:
        row = []
        for scheduler in schedulers:
            run = next(
                (
                    item
                    for item in runs
                    if item.batch_size == batch_size
                    and item.interval_scheduler == scheduler
                ),
                None,
            )
            row.append(run.final_loss if run is not None and run.final_loss is not None else math.nan)
        matrix.append(row)

    if not matrix:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    image = ax.imshow(matrix, cmap="viridis_r")
    fig.colorbar(image, ax=ax, label="Final train loss")
    ax.set_title("Final loss by batch size and scheduler")
    ax.set_xlabel("Interval scheduler")
    ax.set_ylabel("Batch size")
    ax.set_xticks(range(len(schedulers)), labels=schedulers, rotation=20, ha="right")
    ax.set_yticks(range(len(batch_sizes)), labels=batch_sizes)

    values = [value for row in matrix for value in row if not math.isnan(value)]
    midpoint = (min(values) + max(values)) / 2 if values else 0.0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if math.isnan(value):
                continue
            color = "white" if value < midpoint else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_heatmap.png", dpi=180)
    plt.close(fig)


def plot_final_loss_bars(runs: list[Run], output_dir: Path) -> None:
    ordered = sorted_runs(runs)
    labels = [run_label(run) for run in ordered]
    losses = [run.final_loss if run.final_loss is not None else math.nan for run in ordered]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(range(len(ordered)), losses, color="#4f7cac")
    best = min((loss for loss in losses if not math.isnan(loss)), default=None)
    if best is not None:
        ax.axhline(best, color="#333333", linewidth=1.0, linestyle=":", label="best")
    for bar, loss in zip(bars, losses):
        if math.isnan(loss):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            loss,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Final train loss by run")
    ax.set_ylabel("Final train loss")
    ax.set_xticks(range(len(ordered)), labels=labels, rotation=35, ha="right")
    style_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_by_run.png", dpi=180)
    plt.close(fig)


def plot_runtime_tradeoff(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    markers = {"exp_linear": "o", "linear": "s", "constant": "^"}
    for run in sorted_runs(runs):
        if run.run_seconds is None or run.final_loss is None:
            continue
        ax.scatter(
            run.run_seconds,
            run.final_loss,
            marker=markers.get(run.interval_scheduler or "", "o"),
            s=70,
            label=run.interval_scheduler,
        )
        ax.text(
            run.run_seconds,
            run.final_loss,
            f"  bs={run.batch_size}",
            fontsize=8,
            va="center",
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Scheduler", fontsize=8)
    ax.set_title("Runtime vs final train loss")
    ax.set_xlabel("Run seconds")
    ax.set_ylabel("Final train loss")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_final_loss.png", dpi=180)
    plt.close(fig)


def plot_selected_lr_traces(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for run in sorted_runs(runs):
        points = sorted(run.train, key=lambda item: item.step)
        if not points:
            continue
        ax.plot(
            [point.step for point in points],
            [point.muon_lr for point in points],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=run_label(run),
        )
    ax.set_yscale("log")
    ax.set_title("Applied Muon LR during final run")
    ax.set_xlabel("Step")
    ax.set_ylabel("Muon LR")
    style_axes(ax)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "applied_muon_lr_traces.png", dpi=180)
    plt.close(fig)


def plot_selected_hparams(runs: list[Run], output_dir: Path) -> None:
    ordered = sorted_runs(runs)
    labels = [run_label(run) for run in ordered]
    lrs = [final_summary_float(run, "Final Muon lr") for run in ordered]
    momentums = [final_summary_float(run, "Final Muon momentum") for run in ordered]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(range(len(ordered)), lrs, marker="o", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Final Muon LR")
    axes[0].set_title("Selected final hyperparameters")
    style_axes(axes[0])

    axes[1].plot(range(len(ordered)), momentums, marker="o", color="#b45f06", linewidth=1.6)
    axes[1].set_ylabel("Final momentum")
    axes[1].set_xticks(range(len(ordered)), labels=labels, rotation=35, ha="right")
    style_axes(axes[1])

    fig.tight_layout()
    fig.savefig(output_dir / "selected_hparams.png", dpi=180)
    plt.close(fig)


def plot_interval_candidates(runs: list[Run], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    plotted = False
    for run in sorted_runs(runs):
        candidates = [
            candidate
            for candidate in run.candidates
            if candidate.muon_lr is not None and candidate.final_loss is not None
        ]
        if not candidates:
            continue
        plotted = True
        axes[0].scatter(
            [candidate.muon_lr for candidate in candidates],
            [candidate.final_loss for candidate in candidates],
            s=18,
            alpha=0.65,
            label=run_label(run),
        )
        axes[1].scatter(
            [
                candidate.muon_momentum
                for candidate in candidates
                if candidate.muon_momentum is not None
            ],
            [
                candidate.final_loss
                for candidate in candidates
                if candidate.muon_momentum is not None
            ],
            s=18,
            alpha=0.65,
            label=run_label(run),
        )

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_xscale("log")
    axes[0].set_title("Candidate final loss vs Muon LR")
    axes[0].set_xlabel("Candidate Muon LR")
    axes[0].set_ylabel("Candidate final loss")
    axes[1].set_title("Candidate final loss vs momentum")
    axes[1].set_xlabel("Candidate momentum")
    for ax in axes:
        style_axes(ax)
    axes[1].legend(fontsize=7, ncol=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "interval_candidate_scatter.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], output_dir: Path) -> None:
    plot_train_loss_curves(runs, output_dir)
    plot_final_loss_heatmap(runs, output_dir)
    plot_final_loss_bars(runs, output_dir)
    plot_runtime_tradeoff(runs, output_dir)
    plot_selected_lr_traces(runs, output_dir)
    plot_selected_hparams(runs, output_dir)
    plot_interval_candidates(runs, output_dir)


def main() -> None:
    args = parse_args()
    runs = parse_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(runs, args.output_dir)
    write_summary(runs, args.log, args.output_dir)
    plot_all(runs, args.output_dir)

    print(f"Parsed {len(runs)} runs")
    print(f"Wrote plots and CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()

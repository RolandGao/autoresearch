#!/usr/bin/env python3
"""Plot the CIFAR overfit global scheduler search log."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "cifar_overfit_search_global2.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_global2_plots"
OUTPUT_SUMMARY = "summary.txt"
OUTPUT_PLOT = "curves.png"

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
    muon_grad_momentum_norm_ratio: float | None


@dataclass
class SearchChoice:
    start_step: int
    start_muon_lr: float | None
    end_muon_lr: float | None
    muon_lr: float | None
    muon_momentum: float | None
    cooldown_muon_lr: float | None
    cooldown_muon_momentum: float | None
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
    lr_connectedness: str | None = None
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
    if value is None or value.lower() in {"none", "nan"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_kv_line(line: str) -> dict[str, str]:
    return {match["key"]: match["value"] for match in KV_RE.finditer(line)}


def parse_float_list(value: str | None) -> list[float]:
    if value is None:
        return []
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    return [
        parsed
        for item in value[1:-1].split(",")
        if (parsed := parse_optional_float(item.strip())) is not None
    ]


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
                    lr_connectedness=fields.get("lr_connectedness"),
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
                        muon_grad_momentum_norm_ratio=parse_optional_float(
                            fields.get("muon_grad_momentum_norm_ratio")
                        ),
                    )
                )
                continue

            if line.startswith("best_muon_lr=") and current is not None:
                fields = parse_kv_line(line)
                best_muon_lrs = parse_float_list(fields.get("best_muon_lr"))
                last_step = current.train[-1].step if current.train else -1
                current.choices.append(
                    SearchChoice(
                        start_step=last_step + 1,
                        start_muon_lr=best_muon_lrs[0] if best_muon_lrs else None,
                        end_muon_lr=best_muon_lrs[1]
                        if len(best_muon_lrs) > 1
                        else None,
                        muon_lr=best_muon_lrs[1] if len(best_muon_lrs) > 1 else None,
                        muon_momentum=parse_optional_float(
                            fields.get("best_muon_momentum")
                        ),
                        cooldown_muon_lr=best_muon_lrs[-1]
                        if len(best_muon_lrs) > 2
                        else None,
                        cooldown_muon_momentum=parse_optional_float(
                            fields.get("best_muon_momentum")
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
            run.m_steps if run.m_steps is not None else -1,
            run.interval_scheduler or "",
            run.lr_connectedness or "",
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


def best_completed_run(runs: list[Run]) -> Run | None:
    completed = [run for run in runs if run.final_loss is not None and run.completed]
    if not completed:
        return None
    return min(
        completed,
        key=lambda run: (
            run.final_loss if run.final_loss is not None else math.inf,
            run.batch_size or 0,
            run.run,
        ),
    )


def final_muon_lr(run: Run) -> float | None:
    return parse_optional_float(run.summary.get("Final Muon lr"))


def final_muon_momentum(run: Run) -> float | None:
    return parse_optional_float(run.summary.get("Final Muon momentum"))


def plot_row_label(run: Run) -> str:
    scheduler = run.interval_scheduler or "NA"
    loss = format_number(run.final_loss)
    return f"run {run.run}\n{scheduler}\nloss {loss}"


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    best = best_completed_run(runs)
    lines = [
        "CIFAR overfit global scheduler search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Plot file: {output_dir / OUTPUT_PLOT}",
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
                    f"  run={best.run} batch_size={best.batch_size} "
                    f"N={best.n_steps} M={best.m_steps} "
                    f"interval_scheduler={best.interval_scheduler or 'NA'} "
                    f"lr_connectedness={best.lr_connectedness or 'NA'} "
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
                run.m_steps if run.m_steps is not None else math.inf,
            ),
        )
        for run in ranked:
            suffix = "" if run.completed else " partial"
            lines.append(
                f"  run={run.run} N={run.n_steps} M={run.m_steps} "
                f"interval_scheduler={run.interval_scheduler or 'NA'} "
                f"lr_connectedness={run.lr_connectedness or 'NA'} "
                f"final_loss={format_number(run.final_loss)} "
                f"first_below_0p87={format_number(first_step_below(run, 0.87))} "
                f"evaluated_configs={total_evaluated_configs(run)}"
                f"{suffix}"
            )
        lines.append("")

        frontier = pareto_frontier_runs(batch_runs, threshold=0.87)
        lines.append("Pareto frontier: first step below 0.87 vs run seconds")
        for run in frontier:
            lines.append(
                f"  run={run.run} N={run.n_steps} M={run.m_steps} "
                f"interval_scheduler={run.interval_scheduler or 'NA'} "
                f"lr_connectedness={run.lr_connectedness or 'NA'} "
                f"first_step_below_0p87={first_step_below(run, 0.87)} "
                f"run_seconds={format_number(run.run_seconds)} "
                f"final_lr={format_number(final_muon_lr(run))} "
                f"final_momentum={format_number(final_muon_momentum(run))}"
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
                    f"interval_scheduler={run.interval_scheduler or 'NA'} "
                    f"lr_connectedness={run.lr_connectedness or 'NA'} "
                    f"{metric.csv_name}={metric.format_value(value)} "
                    f"run_seconds={format_number(run.run_seconds)} "
                    f"final_lr={format_number(final_muon_lr(run))} "
                    f"final_momentum={format_number(final_muon_momentum(run))}"
                )
            missing = len(batch_runs) - len(metric_ranked)
            if missing:
                lines.append(f"  {missing} runs missing this metric")
            lines.append("")

    (output_dir / OUTPUT_SUMMARY).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_curves(runs: list[Run], output_dir: Path) -> None:
    ordered = sorted_runs(runs)
    if not ordered:
        return

    fig, axes = plt.subplots(
        len(ordered),
        4,
        figsize=(18, 3.0 * len(ordered)),
        sharex="col",
        squeeze=False,
    )
    columns = [
        ("Muon LR", "muon_lr"),
        ("Muon momentum", "muon_momentum"),
        ("Grad/momentum norm ratio", "muon_grad_momentum_norm_ratio"),
        ("Train loss", "loss"),
    ]

    for row, run in enumerate(ordered):
        points = sorted(run.train, key=lambda item: item.step)
        row_label = plot_row_label(run)

        for col, (title, attr) in enumerate(columns):
            ax = axes[row, col]
            filtered = [
                (point.step, value)
                for point in points
                if (value := getattr(point, attr)) is not None
                and (attr == "loss" or point.step > 0)
            ]
            if filtered:
                plot_steps, plot_values = zip(*filtered)
                ax.plot(
                    plot_steps,
                    plot_values,
                    marker="o",
                    markersize=2.4,
                    linewidth=1.35,
                    linestyle="-" if run.completed else "--",
                    color=f"C{row % 10}",
                )
            if col == 0:
                ax.set_yscale("linear")
            if attr == "muon_momentum":
                ax.set_ylim(-0.03, 1.03)
            if attr == "loss":
                ax.axhline(0.87, color="#888888", linewidth=1.0, linestyle=":")

            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=28,
                )
            style_axes(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel("Step")

    fig.suptitle("CIFAR global scheduler search curves", y=0.998)
    fig.tight_layout(rect=(0.03, 0, 1, 0.985))
    fig.savefig(output_dir / OUTPUT_PLOT, dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(runs, log_path, output_dir)
    plot_curves(runs, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot cifar_overfit_search_global2.log."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = parse_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")

    plot_all(runs, args.log, args.output_dir)

    best = best_completed_run(runs)
    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Wrote {args.output_dir / OUTPUT_SUMMARY}")
    print(f"Wrote {args.output_dir / OUTPUT_PLOT}")
    if best is not None:
        print(
            f"Best completed run: run={best.run} batch_size={best.batch_size} "
            f"N={best.n_steps} M={best.m_steps} "
            f"interval_scheduler={best.interval_scheduler or 'NA'} "
            f"lr_connectedness={best.lr_connectedness or 'NA'} "
            f"final_loss={best.final_loss:.6f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot CIFAR overfit momentum-search runs from multiple log files together."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_cifar_overfit_search_momentum import Run, format_number, parse_log, style_axes


HERE = Path(__file__).resolve().parent
DEFAULT_LOGS = [
    HERE / "cifar_overfit_search_momentum2.log",
    HERE / "cifar_overfit_search_momentum.log",
]
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_momentum_combined"


def log_label(path: Path) -> str:
    stem = path.stem
    prefix = "cifar_overfit_search_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def run_label(source: str, run: Run) -> str:
    parts = [source, f"run {run.run}", run.momentum_config]
    if run.search_nesterov not in {"unknown", "None"}:
        parts.append(f"search_nesterov={run.search_nesterov}")
    elif run.muon_nesterov not in {"unknown", "None"}:
        parts.append(f"nesterov={run.muon_nesterov}")
    return ": ".join(parts[:2]) + " - " + ", ".join(parts[2:])


def nesterov_value(value: str | None) -> float | None:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return None


def load_runs(paths: list[Path]) -> list[tuple[str, Run]]:
    tagged_runs: list[tuple[str, Run]] = []
    for path in paths:
        runs = parse_log(path)
        if not runs:
            raise SystemExit(f"No runs parsed from {path}")
        tagged_runs.extend((log_label(path), run) for run in runs)
    return tagged_runs


def sorted_train_points(run: Run):
    return sorted(run.train, key=lambda point: point.step)


def loss_auc(run: Run) -> float | None:
    points = sorted_train_points(run)
    if len(points) < 2:
        return None
    area = 0.0
    for left, right in zip(points, points[1:]):
        area += 0.5 * (left.loss + right.loss) * (right.step - left.step)
    return area


def mean_loss(run: Run) -> float | None:
    points = sorted_train_points(run)
    if not points:
        return None
    return sum(point.loss for point in points) / len(points)


def min_loss(run: Run) -> tuple[float, int] | tuple[None, None]:
    points = sorted_train_points(run)
    if not points:
        return None, None
    best = min(points, key=lambda point: (point.loss, point.step))
    return best.loss, best.step


def loss_improvement(run: Run) -> float | None:
    if run.initial_loss is None or run.final_loss is None:
        return None
    return run.initial_loss - run.final_loss


def total_evaluated_configs(run: Run) -> int:
    return sum(choice.evaluated_configs for choice in run.interval_choices)


def total_evaluated_interval_configs(run: Run) -> int:
    return sum(choice.evaluated_interval_configs for choice in run.interval_choices)


def final_nesterov(run: Run) -> str:
    summary_value = run.summary.get("Final Muon nesterov")
    if summary_value is not None:
        return summary_value
    if run.train and run.train[-1].muon_nesterov is not None:
        return run.train[-1].muon_nesterov
    return run.muon_nesterov if run.muon_nesterov != "unknown" else "NA"


def final_cooldown_lr(run: Run) -> float | None:
    return parse_summary_float(run, "Final cooldown lr")


def final_cooldown_momentum(run: Run) -> float | None:
    return parse_summary_float(run, "Final cooldown mom")


def parse_summary_float(run: Run, key: str) -> float | None:
    value = run.summary.get(key)
    if value in {None, "none", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ranked_by_metric(
    tagged_runs: list[tuple[str, Run]],
    metric,
    reverse: bool = False,
) -> list[tuple[str, Run, float]]:
    rows = []
    for source, run in tagged_runs:
        value = metric(run)
        if value is not None:
            rows.append((source, run, value))
    rows.sort(
        key=lambda item: (
            -item[2] if reverse else item[2],
            item[0],
            item[1].run,
        )
    )
    return rows


def write_train_csv(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    with (output_dir / "combined_train_losses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source",
                "run",
                "label",
                "momentum_config",
                "search_momentum",
                "muon_nesterov",
                "search_nesterov",
                "step",
                "total_steps",
                "loss",
                "head_lr",
                "muon_lr",
                "muon_momentum",
                "selected_muon_nesterov",
            ],
        )
        writer.writeheader()
        for source, run in tagged_runs:
            label = run_label(source, run)
            for point in run.train:
                writer.writerow(
                    {
                        "source": source,
                        "run": run.run,
                        "label": label,
                        "momentum_config": run.momentum_config,
                        "search_momentum": run.search_momentum,
                        "muon_nesterov": run.muon_nesterov,
                        "search_nesterov": run.search_nesterov,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                        "muon_momentum": point.muon_momentum,
                        "selected_muon_nesterov": point.muon_nesterov,
                    }
                )


def write_run_metrics_csv(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    with (output_dir / "combined_run_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source",
                "run",
                "label",
                "momentum_config",
                "initial_loss",
                "final_loss",
                "loss_improvement",
                "mean_loss",
                "loss_auc",
                "min_loss",
                "min_loss_step",
                "step5_loss",
                "step10_loss",
                "step15_loss",
                "step20_loss",
                "step30_loss",
                "step40_loss",
                "step50_loss",
                "first_step_below_1p0",
                "first_step_below_0p95",
                "first_step_below_0p9",
                "first_step_below_0p88",
                "first_step_below_0p87",
                "final_muon_lr",
                "final_muon_momentum",
                "final_muon_nesterov",
                "final_cooldown_lr",
                "final_cooldown_momentum",
                "total_evaluated_interval_configs",
                "total_evaluated_configs",
            ],
        )
        writer.writeheader()
        for source, run in tagged_runs:
            best_loss, best_step = min_loss(run)
            writer.writerow(
                {
                    "source": source,
                    "run": run.run,
                    "label": run_label(source, run),
                    "momentum_config": run.momentum_config,
                    "initial_loss": run.initial_loss,
                    "final_loss": run.final_loss,
                    "loss_improvement": loss_improvement(run),
                    "mean_loss": mean_loss(run),
                    "loss_auc": loss_auc(run),
                    "min_loss": best_loss,
                    "min_loss_step": best_step,
                    "step5_loss": run.loss_at_step(5),
                    "step10_loss": run.loss_at_step(10),
                    "step15_loss": run.loss_at_step(15),
                    "step20_loss": run.loss_at_step(20),
                    "step30_loss": run.loss_at_step(30),
                    "step40_loss": run.loss_at_step(40),
                    "step50_loss": run.loss_at_step(50),
                    "first_step_below_1p0": first_step_below(run, 1.0),
                    "first_step_below_0p95": first_step_below(run, 0.95),
                    "first_step_below_0p9": first_step_below(run, 0.9),
                    "first_step_below_0p88": first_step_below(run, 0.88),
                    "first_step_below_0p87": first_step_below(run, 0.87),
                    "final_muon_lr": run.final_muon_lr,
                    "final_muon_momentum": run.final_muon_momentum,
                    "final_muon_nesterov": final_nesterov(run),
                    "final_cooldown_lr": final_cooldown_lr(run),
                    "final_cooldown_momentum": final_cooldown_momentum(run),
                    "total_evaluated_interval_configs": (
                        total_evaluated_interval_configs(run)
                    ),
                    "total_evaluated_configs": total_evaluated_configs(run),
                }
            )


def first_step_below(run: Run, threshold: float) -> int | None:
    point = run.first_step_below(threshold)
    return point.step if point is not None else None


def write_choice_csv(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    with (output_dir / "combined_interval_choices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source",
                "run",
                "label",
                "momentum_config",
                "search_momentum",
                "muon_nesterov",
                "search_nesterov",
                "interval_index",
                "start_step",
                "muon_lr",
                "muon_momentum",
                "selected_muon_nesterov",
                "cooldown_muon_lr",
                "cooldown_muon_momentum",
                "cooldown_muon_nesterov",
                "interval_loss",
                "final_loss",
                "evaluated_interval_configs",
                "evaluated_configs",
            ],
        )
        writer.writeheader()
        for source, run in tagged_runs:
            label = run_label(source, run)
            for choice in run.interval_choices:
                writer.writerow(
                    {
                        "source": source,
                        "run": run.run,
                        "label": label,
                        "momentum_config": run.momentum_config,
                        "search_momentum": run.search_momentum,
                        "muon_nesterov": run.muon_nesterov,
                        "search_nesterov": run.search_nesterov,
                        "interval_index": choice.interval_index,
                        "start_step": choice.start_step,
                        "muon_lr": choice.muon_lr,
                        "muon_momentum": choice.muon_momentum,
                        "selected_muon_nesterov": choice.muon_nesterov,
                        "cooldown_muon_lr": choice.cooldown_muon_lr,
                        "cooldown_muon_momentum": choice.cooldown_muon_momentum,
                        "cooldown_muon_nesterov": choice.cooldown_muon_nesterov,
                        "interval_loss": choice.interval_loss,
                        "final_loss": choice.final_loss,
                        "evaluated_interval_configs": (
                            choice.evaluated_interval_configs
                        ),
                        "evaluated_configs": choice.evaluated_configs,
                    }
                )


def write_summary(
    tagged_runs: list[tuple[str, Run]], input_logs: list[Path], output_dir: Path
) -> None:
    lines = [
        "Combined CIFAR overfit LR/momentum search summary",
        "Input logs:",
        *[f"  {path}" for path in input_logs],
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(tagged_runs)}",
        "",
    ]

    add_metric_ranking(
        lines,
        "Final-loss ranking",
        tagged_runs,
        lambda run: run.final_loss,
        "final_loss",
    )

    lines.append("Step-loss rankings")
    for step in [5, 10, 15, 20, 30, 40, 50]:
        add_metric_ranking(
            lines,
            f"  Loss at step {step}",
            tagged_runs,
            lambda run, step=step: run.loss_at_step(step),
            f"step{step}",
            indent="    ",
        )

    lines.append("Aggregate rankings")
    add_metric_ranking(
        lines,
        "  Loss AUC, lower is better",
        tagged_runs,
        loss_auc,
        "auc",
        indent="    ",
    )
    add_metric_ranking(
        lines,
        "  Mean loss, lower is better",
        tagged_runs,
        mean_loss,
        "mean_loss",
        indent="    ",
    )
    add_metric_ranking(
        lines,
        "  Best observed train loss",
        tagged_runs,
        lambda run: min_loss(run)[0],
        "min_loss",
        indent="    ",
    )
    add_metric_ranking(
        lines,
        "  Loss improvement, higher is better",
        tagged_runs,
        loss_improvement,
        "improvement",
        reverse=True,
        indent="    ",
    )

    lines.append("Search-cost rankings")
    add_metric_ranking(
        lines,
        "  Evaluated final configs, lower is cheaper",
        tagged_runs,
        lambda run: float(total_evaluated_configs(run)),
        "evaluated_configs",
        indent="    ",
    )
    add_metric_ranking(
        lines,
        "  Evaluated interval configs, lower is cheaper",
        tagged_runs,
        lambda run: float(total_evaluated_interval_configs(run)),
        "evaluated_interval_configs",
        indent="    ",
    )

    lines.append("Threshold steps")
    for threshold in [1.0, 0.95, 0.9, 0.88, 0.87]:
        reached = [
            (source, run, point)
            for source, run in tagged_runs
            if (point := run.first_step_below(threshold)) is not None
        ]
        reached.sort(key=lambda item: (item[2].step, item[2].loss, item[0], item[1].run))
        lines.append(f"  First step below {threshold:g}")
        for source, run, point in reached:
            lines.append(
                f"    step={point.step:2d} loss={format_number(point.loss)} "
                f"{run_label(source, run)}"
            )
        missing = len(tagged_runs) - len(reached)
        if missing:
            lines.append(f"    {missing} runs never reached this threshold")

    lines.extend(["", "Final hyperparameters"])
    ranked = ranked_by_metric(
        tagged_runs,
        lambda run: run.final_loss,
    )
    for source, run, _ in ranked:
        best_loss, best_step = min_loss(run)
        lines.append(
            f"  {run_label(source, run)}: "
            f"final_loss={format_number(run.final_loss)} "
            f"min_loss={format_number(best_loss)}@step{best_step} "
            f"final_lr={format_number(run.final_muon_lr)} "
            f"final_momentum={format_number(run.final_muon_momentum)} "
            f"final_nesterov={final_nesterov(run)} "
            f"cooldown_lr={format_number(final_cooldown_lr(run))} "
            f"cooldown_momentum={format_number(final_cooldown_momentum(run))} "
            f"evaluated_configs={total_evaluated_configs(run)}"
        )

    (output_dir / "combined_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def add_metric_ranking(
    lines: list[str],
    title: str,
    tagged_runs: list[tuple[str, Run]],
    metric,
    metric_name: str,
    reverse: bool = False,
    indent: str = "  ",
) -> None:
    ranked = ranked_by_metric(tagged_runs, metric, reverse=reverse)
    lines.append(title)
    for rank, (source, run, value) in enumerate(ranked, start=1):
        final_suffix = (
            "" if metric_name == "final_loss" else f" final_loss={format_number(run.final_loss)}"
        )
        lines.append(
            f"{indent}{rank:2d}. {metric_name}={format_number(value)} "
            f"{run_label(source, run)}{final_suffix}"
        )
    missing = len(tagged_runs) - len(ranked)
    if missing:
        lines.append(f"{indent}{missing} runs missing this metric")
    lines.append("")


def plot_train_loss(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for source, run in tagged_runs:
        if not run.train:
            continue
        ax.plot(
            [point.step for point in run.train],
            [point.loss for point in run.train],
            marker="o",
            markersize=2.8,
            linewidth=1.7,
            label=run_label(source, run),
        )
    ax.set_title("Combined train loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.legend(fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_train_loss.png", dpi=180)
    plt.close(fig)


def plot_selected_hparams(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    has_nesterov = any(
        any(choice.muon_nesterov is not None for choice in run.interval_choices)
        for _, run in tagged_runs
    )
    row_count = 4 if has_nesterov else 3
    fig, axes = plt.subplots(row_count, 1, figsize=(11, 2.55 * row_count), sharex=True)
    ax_lr, ax_momentum, ax_loss = axes[:3]

    for source, run in tagged_runs:
        if not run.interval_choices:
            continue
        label = run_label(source, run)
        steps = [choice.start_step for choice in run.interval_choices]
        ax_lr.plot(
            steps,
            [choice.muon_lr for choice in run.interval_choices],
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            label=label,
        )
        ax_momentum.plot(
            steps,
            [choice.muon_momentum for choice in run.interval_choices],
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            label=label,
        )
        ax_loss.plot(
            steps,
            [choice.final_loss for choice in run.interval_choices],
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            label=label,
        )
        if has_nesterov:
            nest_steps = [
                choice.start_step
                for choice in run.interval_choices
                if nesterov_value(choice.muon_nesterov) is not None
            ]
            nest_values = [
                nesterov_value(choice.muon_nesterov)
                for choice in run.interval_choices
                if nesterov_value(choice.muon_nesterov) is not None
            ]
            if nest_steps:
                axes[3].plot(
                    nest_steps,
                    nest_values,
                    marker="o",
                    markersize=2.8,
                    linewidth=1.5,
                    label=label,
                )

    ax_lr.set_title("Selected interval LR")
    ax_lr.set_ylabel("Muon LR")
    ax_lr.set_yscale("log")

    ax_momentum.set_title("Selected interval momentum")
    ax_momentum.set_ylabel("Momentum")
    ax_momentum.set_ylim(-0.03, 0.93)

    ax_loss.set_title("Search-implied final loss")
    ax_loss.set_ylabel("Final loss")

    if has_nesterov:
        axes[3].set_title("Selected nesterov flag")
        axes[3].set_ylabel("Nesterov")
        axes[3].set_yticks([0, 1], labels=["False", "True"])
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].set_xlabel("Interval start step")
    else:
        ax_loss.set_xlabel("Interval start step")

    for ax in axes:
        ax.legend(fontsize=8)
        style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_dir / "combined_selected_hparams.png", dpi=180)
    plt.close(fig)


def plot_final_loss(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    ranked = [
        (source, run)
        for source, run in tagged_runs
        if run.final_loss is not None
    ]
    ranked.sort(key=lambda item: (item[1].final_loss, item[0], item[1].run))
    if not ranked:
        return

    labels = [run_label(source, run) for source, run in ranked]
    losses = [run.final_loss for _, run in ranked]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(ranked)), losses, color="#4c78a8")
    ax.set_xticks(range(len(ranked)), labels=labels, rotation=18, ha="right")
    ax.set_title("Final train loss by run")
    ax.set_ylabel("Final train loss")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_final_loss.png", dpi=180)
    plt.close(fig)


def plot_all(tagged_runs: list[tuple[str, Run]], output_dir: Path) -> None:
    plot_train_loss(tagged_runs, output_dir)
    plot_selected_hparams(tagged_runs, output_dir)
    plot_final_loss(tagged_runs, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multiple CIFAR overfit LR/momentum-search logs together."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=DEFAULT_LOGS,
        help="Log files to combine. Defaults to momentum2 plus momentum.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for combined PNG/CSV/summary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tagged_runs = load_runs(args.logs)
    write_train_csv(tagged_runs, output_dir)
    write_run_metrics_csv(tagged_runs, output_dir)
    write_choice_csv(tagged_runs, output_dir)
    write_summary(tagged_runs, args.logs, output_dir)
    plot_all(tagged_runs, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the CIFAR overfit Nesterov-search log."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_cifar_overfit_search_momentum import Run, format_number, parse_log, style_axes


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "cifar_overfit_search_nesterov.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_nesterov_plots"


def nesterov_value(value: str | None) -> float | None:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return None


def final_nesterov(run: Run) -> str:
    summary_value = run.summary.get("Final Muon nesterov")
    if summary_value is not None:
        return summary_value
    if run.train and run.train[-1].muon_nesterov is not None:
        return run.train[-1].muon_nesterov
    return run.muon_nesterov


def run_label(run: Run) -> str:
    if run.search_nesterov == "True":
        nesterov = "search"
    else:
        nesterov = f"fixed {run.muon_nesterov}"
    return f"run {run.run}: {nesterov}"


def sorted_train(run: Run):
    return sorted(run.train, key=lambda point: point.step)


def target_step(runs: list[Run]) -> int:
    return max((point.total_steps for run in runs for point in run.train), default=0)


def loss_at_step(run: Run, step: int) -> float | None:
    for point in run.train:
        if point.step == step:
            return point.loss
    if step == target_step([run]):
        return run.final_loss
    return None


def first_step_below(run: Run, threshold: float):
    for point in sorted_train(run):
        if point.loss < threshold:
            return point
    return None


def median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def add_loss_step_ranking(lines: list[str], runs: list[Run], step: int) -> None:
    ranked = sorted(
        [run for run in runs if loss_at_step(run, step) is not None],
        key=lambda run: (loss_at_step(run, step), run.run),
    )
    lines.append(f"Ranking: loss after {step} steps")
    for rank, run in enumerate(ranked, start=1):
        loss = loss_at_step(run, step)
        lines.append(
            f"  {rank:2d}. {run_label(run)}: "
            f"step={step} loss={format_number(loss)} "
            f"final_loss={format_number(run.final_loss)} "
            f"final_muon_lr={format_number(run.final_muon_lr)} "
            f"final_muon_momentum={format_number(run.final_muon_momentum)} "
            f"final_nesterov={final_nesterov(run)}"
        )
    missing = len(runs) - len(ranked)
    if missing:
        lines.append(f"  {missing} runs did not have step {step}.")
    lines.append("")


def add_threshold_ranking(lines: list[str], runs: list[Run], threshold: float) -> None:
    reached = [
        (run, point)
        for run in runs
        if (point := first_step_below(run, threshold)) is not None
    ]
    reached.sort(key=lambda item: (item[1].step, item[1].loss, item[0].run))

    lines.append(f"Ranking: first step below {threshold:g}")
    for rank, (run, point) in enumerate(reached, start=1):
        lines.append(
            f"  {rank:2d}. {run_label(run)}: "
            f"first_step={point.step} loss={format_number(point.loss)} "
            f"final_loss={format_number(run.final_loss)} "
            f"final_nesterov={final_nesterov(run)}"
        )
    missing = len(runs) - len(reached)
    if missing:
        lines.append(f"  {missing} runs never went below {threshold:g}.")
    lines.append("")


def write_csvs(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "train_losses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "label",
                "momentum_config",
                "search_nesterov",
                "initial_nesterov",
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
        for run in runs:
            for point in sorted_train(run):
                writer.writerow(
                    {
                        "run": run.run,
                        "label": run_label(run),
                        "momentum_config": run.momentum_config,
                        "search_nesterov": run.search_nesterov,
                        "initial_nesterov": run.muon_nesterov,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                        "muon_momentum": point.muon_momentum,
                        "muon_nesterov": point.muon_nesterov,
                    }
                )

    with (output_dir / "interval_choices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "label",
                "momentum_config",
                "search_nesterov",
                "initial_nesterov",
                "interval_index",
                "start_step",
                "muon_lr",
                "muon_momentum",
                "muon_nesterov",
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
        for run in runs:
            for choice in run.interval_choices:
                writer.writerow(
                    {
                        "run": run.run,
                        "label": run_label(run),
                        "momentum_config": run.momentum_config,
                        "search_nesterov": run.search_nesterov,
                        "initial_nesterov": run.muon_nesterov,
                        "interval_index": choice.interval_index,
                        "start_step": choice.start_step,
                        "muon_lr": choice.muon_lr,
                        "muon_momentum": choice.muon_momentum,
                        "muon_nesterov": choice.muon_nesterov,
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


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    ranked = sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: (run.final_loss, run.run),
    )
    final_step = target_step(runs)
    lines = [
        "CIFAR overfit Nesterov-search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        "",
    ]
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        losses = [run.final_loss for run in ranked if run.final_loss is not None]
        lines.extend(
            [
                "Best run",
                (
                    f"  {run_label(best)}: "
                    f"final_loss={format_number(best.final_loss)} "
                    f"initial_loss={format_number(best.initial_loss)} "
                    f"final_muon_lr={format_number(best.final_muon_lr)} "
                    f"final_muon_momentum={format_number(best.final_muon_momentum)} "
                    f"final_nesterov={final_nesterov(best)}"
                ),
                "",
                "Final-loss distribution",
                f"  min={format_number(min(losses))}",
                f"  median={format_number(median(losses))}",
                f"  max={format_number(max(losses))}",
                f"  spread={format_number(max(losses) - min(losses))}",
                (
                    f"  worst={run_label(worst)}: "
                    f"final_loss={format_number(worst.final_loss)}"
                ),
                "",
            ]
        )

        lines.append("Final-loss ranking")
        for rank, run in enumerate(ranked, start=1):
            lines.append(
                f"  {rank:2d}. {run_label(run)} "
                f"final_loss={format_number(run.final_loss)} "
                f"initial_loss={format_number(run.initial_loss)} "
                f"final_lr={format_number(run.final_muon_lr)} "
                f"final_momentum={format_number(run.final_muon_momentum)} "
                f"final_nesterov={final_nesterov(run)}"
            )
        lines.append("")

        for step in [final_step, 10, 20]:
            if step > 0:
                add_loss_step_ranking(lines, runs, step=step)
        for threshold in [0.9, 0.88, 0.87]:
            add_threshold_ranking(lines, runs, threshold=threshold)

    for run in runs:
        lines.extend(
            [
                run_label(run),
                f"  momentum config: {run.momentum_config}",
                f"  search nesterov: {run.search_nesterov}",
                f"  initial nesterov: {run.muon_nesterov}",
                f"  final nesterov: {final_nesterov(run)}",
                f"  train points: {len(run.train)}",
                f"  interval choices: {len(run.interval_choices)}",
                f"  interval evals: {len(run.interval_evals)}",
                f"  final loss: {format_number(run.final_loss)}",
                f"  final muon lr: {format_number(run.final_muon_lr)}",
                f"  final muon momentum: {format_number(run.final_muon_momentum)}",
                "",
            ]
        )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_train_losses(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in runs:
        points = sorted_train(run)
        ax.plot(
            [point.step for point in points],
            [point.loss for point in points],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=run_label(run),
        )
    ax.set_title("CIFAR overfit train loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_comparison.png", dpi=180)
    plt.close(fig)


def plot_selected_hparams(runs: list[Run], output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax_loss, ax_lr, ax_momentum, ax_nesterov = axes
    offsets = centered_offsets(len(runs), width=0.16)
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":"]

    for run_index, run in enumerate(runs):
        choices = sorted(run.interval_choices, key=lambda choice: choice.start_step)
        if not choices:
            continue
        x_offset = offsets[run_index]
        y_offset = offsets[run_index] * 0.28
        steps = [choice.start_step + x_offset for choice in choices]
        marker = markers[run_index % len(markers)]
        linestyle = linestyles[run_index % len(linestyles)]
        label = run_label(run)
        ax_loss.plot(
            steps,
            [choice.final_loss for choice in choices],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
        ax_lr.plot(
            steps,
            [choice.muon_lr for choice in choices],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
        ax_momentum.plot(
            steps,
            [choice.muon_momentum for choice in choices],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )

        nesterov_points = [
            (choice.start_step + x_offset, nesterov_value(choice.muon_nesterov))
            for choice in choices
            if nesterov_value(choice.muon_nesterov) is not None
        ]
        if nesterov_points:
            ax_nesterov.scatter(
                [point[0] for point in nesterov_points],
                [point[1] + y_offset for point in nesterov_points],
                marker=marker,
                s=38,
                label=label,
            )

    ax_loss.set_title("Selected interval final loss")
    ax_loss.set_ylabel("Loss")
    style_axes(ax_loss)

    ax_lr.set_title("Selected interval LR")
    ax_lr.set_ylabel("Muon LR")
    ax_lr.set_yscale("log")
    style_axes(ax_lr)

    ax_momentum.set_title("Selected interval momentum")
    ax_momentum.set_ylabel("Momentum")
    ax_momentum.set_ylim(-0.03, 1.02)
    style_axes(ax_momentum)

    ax_nesterov.set_title("Selected interval Nesterov")
    ax_nesterov.set_xlabel("Interval start step")
    ax_nesterov.set_ylabel("Nesterov")
    ax_nesterov.set_yticks([0, 1], labels=["False", "True"])
    ax_nesterov.set_ylim(-0.12, 1.12)
    style_axes(ax_nesterov)

    ax_loss.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selected_hparams.png", dpi=180)
    plt.close(fig)


def centered_offsets(count: int, width: float) -> list[float]:
    if count <= 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * width for index in range(count)]


def plot_final_losses(runs: list[Run], output_dir: Path) -> None:
    complete_runs = [run for run in runs if run.final_loss is not None]
    if not complete_runs:
        return
    complete_runs.sort(key=lambda run: run.run)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [run_label(run) for run in complete_runs]
    losses = [run.final_loss for run in complete_runs]
    bars = ax.bar(labels, losses, color=["#4c78a8", "#f58518", "#54a24b"][: len(labels)])
    ax.set_title("Final train loss by Nesterov mode")
    ax.set_ylabel("Final train loss")
    ax.tick_params(axis="x", rotation=20)
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            format_number(loss),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "final_losses.png", dpi=180)
    plt.close(fig)


def plot_interval_eval_cloud(runs: list[Run], output_dir: Path) -> None:
    row_count = len(runs)
    if row_count == 0:
        return
    fig, axes = plt.subplots(row_count, 1, figsize=(10, 3.8 * row_count), squeeze=False)

    for ax, run in zip(axes.flat, runs):
        evals = run.interval_evals
        if not evals:
            ax.axis("off")
            continue
        losses = [item.final_loss for item in evals]
        scatter = ax.scatter(
            [item.muon_lr for item in evals],
            [item.muon_momentum for item in evals],
            c=losses,
            s=35,
            cmap="viridis_r",
            edgecolors="black",
            linewidths=0.25,
        )
        choices = run.interval_choices
        if choices:
            ax.scatter(
                [choice.muon_lr for choice in choices],
                [choice.muon_momentum for choice in choices],
                s=90,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.4,
                label="selected",
            )
            ax.legend(loc="best")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.02)
        ax.set_title(f"{run_label(run)} interval evals")
        ax.set_xlabel("Interval LR")
        ax.set_ylabel("Momentum")
        style_axes(ax)
        fig.colorbar(scatter, ax=ax, label="Final loss")

    fig.tight_layout()
    fig.savefig(output_dir / "interval_eval_cloud.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], output_dir: Path) -> None:
    plot_train_losses(runs, output_dir)
    plot_selected_hparams(runs, output_dir)
    plot_final_losses(runs, output_dir)
    plot_interval_eval_cloud(runs, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cifar_overfit_search_nesterov.log."
    )
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG/CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(log_path)
    if not runs:
        raise SystemExit(f"No runs parsed from {log_path}")

    write_csvs(runs, output_dir)
    write_summary(runs, log_path, output_dir)
    plot_all(runs, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

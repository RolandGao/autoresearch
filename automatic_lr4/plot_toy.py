#!/usr/bin/env python3
"""Plot interval-0 LR heatmaps for the improved CIFAR search log."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np

from plot_cifar_search import (
    HParams,
    MainEval,
    TrainInterval,
    default_output_dir,
    finite_or_neg_inf,
    format_number,
    hparams_match_except,
    is_finite_number,
    parse_log,
)


HERE = Path(__file__).resolve().parent
LOG_FILE_PATH = (
    HERE / "20260706_235846_088906" / "cifar_search_improved.log"
)
COLOR_SCALE_MIN = 0.88
DEFAULT_INTERVALS = [0, 1]


def hparams_sort_key(value: float) -> tuple[int, float]:
    if value > 0:
        return (0, value)
    if value == 0:
        return (1, value)
    return (2, value)


def same_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def best_cooldown_candidate(main_eval: MainEval):
    return max(
        (
            item
            for item in main_eval.candidates
            if is_finite_number(item.tta_val_acc)
        ),
        key=lambda item: finite_or_neg_inf(item.tta_val_acc),
        default=None,
    )


def selected_interval(train_intervals: list[TrainInterval], interval: int) -> TrainInterval:
    for train_interval in train_intervals:
        if train_interval.interval == interval:
            return train_interval
    raise SystemExit(f"No train_hparams entry found for interval={interval}")


def choose_eval_for_x(existing: MainEval | None, candidate: MainEval) -> MainEval:
    if existing is None:
        return candidate
    existing_best = best_cooldown_candidate(existing)
    candidate_best = best_cooldown_candidate(candidate)
    existing_key = (
        finite_or_neg_inf(existing.best_cooldown_acc),
        len(existing.candidates),
        finite_or_neg_inf(existing_best.tta_val_acc if existing_best else None),
        existing.index,
    )
    candidate_key = (
        finite_or_neg_inf(candidate.best_cooldown_acc),
        len(candidate.candidates),
        finite_or_neg_inf(candidate_best.tta_val_acc if candidate_best else None),
        candidate.index,
    )
    return candidate if candidate_key > existing_key else existing


def main_variation_evals(
    main_evals: list[MainEval],
    center: HParams,
    interval: int,
    attr: str,
) -> list[MainEval]:
    by_value: dict[float, MainEval] = {}
    for main_eval in main_evals:
        x_value = getattr(main_eval.hparams, attr)
        if (
            main_eval.interval != interval
            or x_value is None
            or not is_finite_number(x_value)
            or not is_finite_number(main_eval.best_cooldown_acc)
            or not hparams_match_except(main_eval.hparams, center, attr)
        ):
            continue

        sweep = cooldown_sweep(main_eval, attr)
        if len(sweep) < 2:
            continue
        by_value[x_value] = choose_eval_for_x(by_value.get(x_value), main_eval)

    return [by_value[value] for value in sorted(by_value, key=hparams_sort_key)]


def cooldown_sweep(main_eval: MainEval, attr: str) -> dict[float, float]:
    best = best_cooldown_candidate(main_eval)
    if best is None:
        return {}

    sweep: dict[float, float] = {}
    for candidate in main_eval.candidates:
        value = getattr(candidate.hparams, attr)
        score = candidate.tta_val_acc
        if (
            value is None
            or score is None
            or not is_finite_number(value)
            or not is_finite_number(score)
            or not hparams_match_except(candidate.hparams, best.hparams, attr)
        ):
            continue
        sweep[value] = max(score, sweep.get(value, -math.inf))
    return sweep


def build_heatmap(
    main_evals: list[MainEval],
    attr: str,
) -> tuple[list[float], list[float], np.ndarray]:
    x_values = [getattr(main_eval.hparams, attr) for main_eval in main_evals]
    y_values = sorted(
        {
            value
            for main_eval in main_evals
            for value in cooldown_sweep(main_eval, attr)
        },
        key=hparams_sort_key,
    )

    matrix = np.full((len(y_values), len(x_values)), np.nan)
    y_index = {value: index for index, value in enumerate(y_values)}
    for x_index, main_eval in enumerate(main_evals):
        for y_value, score in cooldown_sweep(main_eval, attr).items():
            matrix[y_index[y_value], x_index] = score

    return x_values, y_values, matrix


def annotate_heatmap_cells(ax, image, matrix: np.ndarray) -> None:
    for (y_index, x_index), value in np.ndenumerate(matrix):
        if not is_finite_number(value):
            continue

        rgba = image.cmap(image.norm(float(value)))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text_color = "#222222" if luminance > 0.58 else "white"
        stroke_color = "white" if text_color == "#222222" else "#222222"
        ax.text(
            x_index,
            y_index,
            format_number(float(value), 4),
            ha="center",
            va="center",
            fontsize=5.8,
            color=text_color,
            zorder=3,
            path_effects=[
                path_effects.withStroke(linewidth=0.9, foreground=stroke_color)
            ],
        )


def peak_cells(matrix: np.ndarray) -> list[tuple[int, int]]:
    peaks: list[tuple[int, int]] = []
    row_count, col_count = matrix.shape
    for y_index in range(1, row_count - 1):
        for x_index in range(1, col_count - 1):
            value = matrix[y_index, x_index]
            neighbors = (
                matrix[y_index - 1, x_index],
                matrix[y_index + 1, x_index],
                matrix[y_index, x_index - 1],
                matrix[y_index, x_index + 1],
            )
            if not is_finite_number(value) or any(
                not is_finite_number(neighbor) for neighbor in neighbors
            ):
                continue
            if all(neighbor < value for neighbor in neighbors):
                peaks.append((y_index, x_index))
    return peaks


def mark_peak_cells(ax, matrix: np.ndarray) -> None:
    peaks = peak_cells(matrix)
    if not peaks:
        return
    for y_index, x_index in peaks:
        ax.add_patch(
            Rectangle(
                (x_index - 0.5, y_index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ff4f5e",
                linewidth=1.35,
                zorder=5,
            )
        )


def annotate_best_cell(ax, matrix: np.ndarray) -> None:
    if matrix.size == 0 or np.isnan(matrix).all():
        return
    y_index, x_index = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ax.add_patch(
        Rectangle(
            (x_index - 0.5, y_index - 0.5),
            1,
            1,
            fill=False,
            edgecolor="white",
            linewidth=3.1,
            zorder=6,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_index - 0.5, y_index - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#222222",
            linewidth=1.3,
            zorder=7,
            linestyle="--",
        )
    )


def draw_chosen_column(
    ax,
    x_values: list[float],
    row_count: int,
    chosen_value: float | None,
) -> None:
    if chosen_value is None:
        return
    for index, value in enumerate(x_values):
        if same_float(value, chosen_value):
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -0.5),
                    1,
                    row_count,
                    fill=False,
                    edgecolor="white",
                    linewidth=2.0,
                    linestyle="--",
                )
            )
            return


def plot_lr_heatmap(
    main_evals: list[MainEval],
    selected_hparams: HParams,
    interval: int,
    attr: str,
    output_path: Path,
) -> None:
    if not main_evals:
        raise SystemExit(f"No interval-{interval} main variations found for {attr}")

    x_values, y_values, matrix = build_heatmap(main_evals, attr)
    if matrix.size == 0 or np.isnan(matrix).all():
        raise SystemExit(f"No cooldown sweep values found for {attr}")

    masked = np.ma.masked_invalid(matrix)
    vmax = max(COLOR_SCALE_MIN, float(np.nanmax(matrix)))
    norm = Normalize(vmin=COLOR_SCALE_MIN, vmax=vmax, clip=True)
    fig_width = max(10.5, 0.52 * len(x_values) + 5.0)
    fig_height = max(7.2, 0.34 * len(y_values) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=norm,
    )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([format_number(value, 3) for value in x_values], rotation=60)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([format_number(value, 3) for value in y_values])
    ax.set_xlabel(f"main interval {interval} {attr}")
    ax.set_ylabel(f"cooldown {attr}")

    fixed = ", ".join(
        f"{name}={format_number(getattr(selected_hparams, name), 3)}"
        for name in ("muon_lr", "muon_momentum", "bias_lr", "head_lr")
        if name != attr
    )
    ax.set_title(
        f"Interval {interval} {attr} heatmap around chosen main hparams\n"
        f"fixed main hparams: {fixed}"
    )

    annotate_heatmap_cells(ax, image, matrix)
    mark_peak_cells(ax, matrix)
    draw_chosen_column(ax, x_values, len(y_values), getattr(selected_hparams, attr))
    annotate_best_cell(ax, matrix)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("cooldown tta_val_acc")
    ax.set_facecolor("#eeeeee")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create bias_lr and head_lr heatmaps for the improved "
            "CIFAR search log."
        )
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=LOG_FILE_PATH,
        help=f"Log file to plot. Defaults to {LOG_FILE_PATH}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a sibling directory named <log-stem>_plots.",
    )
    parser.add_argument(
        "--interval",
        nargs="+",
        type=int,
        default=DEFAULT_INTERVALS,
        help="Training interval(s) to plot. Defaults to 0 1.",
    )
    return parser.parse_args()


def heatmap_output_path(output_dir: Path, interval: int, attr: str) -> Path:
    return output_dir / f"interval{interval}_{attr}_heatmap.png"


def main() -> None:
    args = parse_args()
    log_path = args.log
    output_dir = args.output_dir or default_output_dir(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = parse_log(log_path)
    for interval in args.interval:
        train_interval = selected_interval(run.train_intervals, interval)
        selected_hparams = train_interval.hparams

        for attr in ("bias_lr", "head_lr"):
            output_path = heatmap_output_path(output_dir, interval, attr)
            evals = main_variation_evals(
                run.main_evals,
                center=selected_hparams,
                interval=interval,
                attr=attr,
            )
            plot_lr_heatmap(evals, selected_hparams, interval, attr, output_path)
            print(f"Wrote {output_path}")

        print(
            "Selected interval "
            f"{interval}: muon_lr={format_number(selected_hparams.muon_lr, 4)} "
            f"muon_momentum={format_number(selected_hparams.muon_momentum, 4)} "
            f"bias_lr={format_number(selected_hparams.bias_lr, 4)} "
            f"head_lr={format_number(selected_hparams.head_lr, 4)}"
        )


if __name__ == "__main__":
    main()

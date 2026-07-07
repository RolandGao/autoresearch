#!/usr/bin/env python3
"""Plot LR heatmaps for CIFAR search logs."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from plot_cifar_search import (
    default_output_dir,
    finite_or_neg_inf,
    format_number,
    is_finite_number,
    parse_hparam_names,
    parse_kv_line,
    parse_optional_float,
)


HERE = Path(__file__).resolve().parent
LOG_FILE_PATH = (
    HERE / "20260706_235846_088906" / "cifar_search_improved.log"
)
COLOR_SCALE_FLOOR = 0.88
MIN_COLOR_SCALE_SPAN = 1e-4
DEFAULT_COLOR_PERCENTILES = (5.0, 95.0)
DEFAULT_INTERVALS = [0, 1, 2, 3]


@dataclass(frozen=True)
class HParams:
    params: dict[str, float | None]

    @classmethod
    def from_fields(cls, fields: dict[str, str], names: list[str]) -> "HParams":
        return cls({name: parse_optional_float(fields.get(name)) for name in names})

    def __getattr__(self, name: str) -> float | None:
        return self.params.get(name)


@dataclass
class CooldownCandidate:
    interval: int
    eval_index: int
    hparams: HParams
    tta_val_acc: float | None
    blocked: bool


@dataclass
class MainEval:
    interval: int
    index: int
    hparams: HParams
    main_acc: float | None
    best_cooldown_acc: float | None
    blocked: bool
    candidates: list[CooldownCandidate] = field(default_factory=list)


@dataclass
class TrainInterval:
    interval: int
    hparams: HParams


@dataclass
class Run:
    header: dict[str, str] = field(default_factory=dict)
    hparam_names: list[str] = field(default_factory=list)
    train_intervals: list[TrainInterval] = field(default_factory=list)
    main_evals: list[MainEval] = field(default_factory=list)


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


def hparam_names_from_header(header: dict[str, str]) -> list[str]:
    names: list[str] = []
    for key in ("search_hparams", "cooldown_search_hparams"):
        for name in parse_hparam_names(header.get(key)):
            if name not in names:
                names.append(name)
    return names or ["muon_lr", "muon_momentum", "bias_lr", "head_lr"]


def infer_hparam_names(fields: dict[str, str], existing: list[str]) -> list[str]:
    names = list(existing)
    for name in fields:
        if name in names:
            continue
        if name == "muon_momentum" or name.endswith("_lr"):
            names.append(name)
    return names


def assign_interval(
    run: Run,
    interval: int,
    pending_main_evals: list[MainEval],
) -> None:
    for main_eval in pending_main_evals:
        main_eval.interval = interval
        run.main_evals.append(main_eval)


def parse_log(log_path: Path) -> Run:
    run = Run()
    pending_main_evals: list[MainEval] = []
    current_main_eval: MainEval | None = None

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("cifar_search_simple "):
                fields = parse_kv_line(line)
                run.header = fields
                run.hparam_names = hparam_names_from_header(fields)
                continue

            if line.startswith("main hparams: "):
                fields = parse_kv_line(line)
                run.hparam_names = infer_hparam_names(fields, run.hparam_names)
                current_main_eval = MainEval(
                    interval=-1,
                    index=len(run.main_evals) + len(pending_main_evals),
                    hparams=HParams.from_fields(fields, run.hparam_names),
                    main_acc=parse_optional_float(fields.get("main")),
                    best_cooldown_acc=parse_optional_float(fields.get("best_cooldown")),
                    blocked="-> blocked" in line,
                )
                pending_main_evals.append(current_main_eval)
                continue

            if line.startswith("muon_lr=") and " -> " in line:
                fields = parse_kv_line(line)
                run.hparam_names = infer_hparam_names(fields, run.hparam_names)
                candidate = CooldownCandidate(
                    interval=-1,
                    eval_index=current_main_eval.index if current_main_eval else -1,
                    hparams=HParams.from_fields(fields, run.hparam_names),
                    tta_val_acc=parse_optional_float(fields.get("tta_val_acc")),
                    blocked="-> blocked" in line,
                )
                if current_main_eval is not None:
                    current_main_eval.candidates.append(candidate)
                continue

            if line.startswith("train_hparams "):
                fields = parse_kv_line(line)
                interval = int(fields["interval"])
                run.hparam_names = infer_hparam_names(fields, run.hparam_names)
                assign_interval(run, interval, pending_main_evals)
                pending_main_evals = []
                current_main_eval = None
                run.train_intervals.append(
                    TrainInterval(
                        interval=interval,
                        hparams=HParams.from_fields(fields, run.hparam_names),
                    )
                )
                continue

    if pending_main_evals:
        next_interval = max(
            (item.interval for item in run.train_intervals),
            default=-1,
        ) + 1
        assign_interval(run, next_interval, pending_main_evals)

    return run


def hparams_match_except(
    candidate: HParams,
    center: HParams,
    varied_attr: str,
) -> bool:
    for name in center.params:
        if name == varied_attr:
            continue
        if not same_float(getattr(candidate, name), getattr(center, name)):
            return False
    return True


def hparams_key_except(
    hparams: HParams,
    center: HParams,
    varied_attr: str,
) -> tuple[tuple[str, float | None], ...]:
    return tuple(
        (name, getattr(hparams, name))
        for name in center.params
        if name != varied_attr
    )


def hparams_key_match_count(
    key: tuple[tuple[str, float | None], ...],
    center: HParams,
) -> int:
    return sum(1 for name, value in key if same_float(value, getattr(center, name)))


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


def choose_eval_for_x(
    existing: MainEval | None,
    candidate: MainEval,
    attr: str,
) -> MainEval:
    if existing is None:
        return candidate
    existing_best = best_cooldown_candidate(existing)
    candidate_best = best_cooldown_candidate(candidate)
    existing_key = (
        len(cooldown_sweep(existing, attr)),
        finite_or_neg_inf(existing.best_cooldown_acc),
        len(existing.candidates),
        finite_or_neg_inf(existing_best.tta_val_acc if existing_best else None),
        existing.index,
    )
    candidate_key = (
        len(cooldown_sweep(candidate, attr)),
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
    groups: dict[tuple[tuple[str, float | None], ...], dict[float, MainEval]] = {}
    for main_eval in main_evals:
        x_value = getattr(main_eval.hparams, attr)
        if (
            main_eval.interval != interval
            or x_value is None
            or not is_finite_number(x_value)
            or not is_finite_number(main_eval.best_cooldown_acc)
        ):
            continue

        sweep = cooldown_sweep(main_eval, attr)
        if len(sweep) < 2:
            continue

        key = hparams_key_except(main_eval.hparams, center, attr)
        by_value = groups.setdefault(key, {})
        by_value[x_value] = choose_eval_for_x(by_value.get(x_value), main_eval, attr)

    if not groups:
        return []

    center_key = hparams_key_except(center, center, attr)
    exact_group = groups.get(center_key)
    if exact_group is not None and len(exact_group) >= 2:
        selected = exact_group
    else:

        def group_score(
            item: tuple[tuple[tuple[str, float | None], ...], dict[float, MainEval]],
        ) -> tuple[int, int, int, float, int]:
            key, by_value = item
            best_score = max(
                finite_or_neg_inf(main_eval.best_cooldown_acc)
                for main_eval in by_value.values()
            )
            latest_index = max(main_eval.index for main_eval in by_value.values())
            return (
                len(by_value),
                hparams_key_match_count(key, center),
                int(any(same_float(value, getattr(center, attr)) for value in by_value)),
                best_score,
                latest_index,
            )

        _, selected = max(groups.items(), key=group_score)

    return [selected[value] for value in sorted(selected, key=hparams_sort_key)]


def cooldown_sweep(main_eval: MainEval, attr: str) -> dict[float, float]:
    best = best_cooldown_candidate(main_eval)
    if best is None:
        return {}

    groups: dict[tuple[tuple[str, float | None], ...], dict[float, float]] = {}
    for candidate in main_eval.candidates:
        value = getattr(candidate.hparams, attr)
        score = candidate.tta_val_acc
        if (
            value is None
            or score is None
            or not is_finite_number(value)
            or not is_finite_number(score)
        ):
            continue

        key = hparams_key_except(candidate.hparams, main_eval.hparams, attr)
        sweep = groups.setdefault(key, {})
        sweep[value] = max(score, sweep.get(value, -math.inf))

    if not groups:
        return {}

    best_key = hparams_key_except(best.hparams, main_eval.hparams, attr)

    def group_score(
        item: tuple[tuple[tuple[str, float | None], ...], dict[float, float]],
    ) -> tuple[int, float, int]:
        key, sweep = item
        return (
            len(sweep),
            max(sweep.values()),
            int(key == best_key),
        )

    _, selected = max(groups.items(), key=group_score)
    return selected


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


def color_scale_bounds(
    matrix: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    floor: float,
) -> tuple[Normalize, str]:
    values = matrix[np.isfinite(matrix)]
    if values.size == 0:
        return Normalize(vmin=floor, vmax=floor + 1e-6, clip=True), "neither"

    percentile_values = values[values >= floor]
    if percentile_values.size == 0:
        percentile_values = values

    vmin = float(np.percentile(percentile_values, lower_percentile))
    vmax = float(np.percentile(percentile_values, upper_percentile))
    vmin = max(floor, vmin)
    if vmax - vmin < MIN_COLOR_SCALE_SPAN:
        center = max(floor, float(np.median(percentile_values)))
        vmin = max(floor, center - MIN_COLOR_SCALE_SPAN / 2)
        vmax = max(center + MIN_COLOR_SCALE_SPAN / 2, vmin + MIN_COLOR_SCALE_SPAN)

    actual_min = float(np.min(values))
    actual_max = float(np.max(values))
    clipped_low = actual_min < vmin
    clipped_high = actual_max > vmax
    if clipped_low and clipped_high:
        extend = "both"
    elif clipped_low:
        extend = "min"
    elif clipped_high:
        extend = "max"
    else:
        extend = "neither"
    return Normalize(vmin=vmin, vmax=vmax, clip=True), extend


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
    color_percentiles: tuple[float, float],
) -> None:
    if not main_evals:
        raise SystemExit(f"No interval-{interval} main variations found for {attr}")

    x_values, y_values, matrix = build_heatmap(main_evals, attr)
    if matrix.size == 0 or np.isnan(matrix).all():
        raise SystemExit(f"No cooldown sweep values found for {attr}")

    masked = np.ma.masked_invalid(matrix)
    norm, colorbar_extend = color_scale_bounds(
        matrix,
        lower_percentile=color_percentiles[0],
        upper_percentile=color_percentiles[1],
        floor=COLOR_SCALE_FLOOR,
    )
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

    fixed_hparams = main_evals[0].hparams
    fixed = ", ".join(
        f"{name}={format_number(value, 3)}"
        for name, value in fixed_hparams.params.items()
        if name != attr
    )
    ax.set_title(
        f"Interval {interval} {attr} heatmap\n"
        f"fixed main hparams: {fixed}"
    )

    annotate_heatmap_cells(ax, image, matrix)
    mark_peak_cells(ax, matrix)
    draw_chosen_column(ax, x_values, len(y_values), getattr(selected_hparams, attr))
    annotate_best_cell(ax, matrix)

    colorbar = fig.colorbar(image, ax=ax, extend=colorbar_extend)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    colorbar.ax.yaxis.get_offset_text().set_visible(False)
    colorbar.set_label(
        "cooldown tta_val_acc "
        f"({format_number(color_percentiles[0], 3)}-"
        f"{format_number(color_percentiles[1], 3)} pct)"
    )
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
        help="Training interval(s) to plot. Defaults to 0 1 2 3.",
    )
    parser.add_argument(
        "--color-percentiles",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_COLOR_PERCENTILES,
        help=(
            "Per-plot percentiles used for color normalization after applying "
            f"the {COLOR_SCALE_FLOOR:g} floor. Defaults to 5 95."
        ),
    )
    return parser.parse_args()


def heatmap_output_path(output_dir: Path, interval: int, attr: str) -> Path:
    return output_dir / f"interval{interval}_{attr}_heatmap.png"


def summary_output_path(output_dir: Path) -> Path:
    return output_dir / "summary.txt"


def plot_attrs(selected_hparams: HParams) -> list[str]:
    if is_finite_number(selected_hparams.bias_lr):
        attrs = ["bias_lr"]
    else:
        attrs = [
            attr
            for attr in ("whiten_bias_lr", "bn_bias_lr")
            if is_finite_number(getattr(selected_hparams, attr))
        ]
    attrs.append("head_lr")
    return attrs


def format_selected_hparams(hparams: HParams) -> str:
    return " ".join(
        f"{name}={format_number(value, 4)}"
        for name, value in hparams.params.items()
        if is_finite_number(value)
    )


def format_hparams(hparams: HParams, excluded_attr: str | None = None) -> str:
    return " ".join(
        f"{name}={format_number(value, 4)}"
        for name, value in hparams.params.items()
        if name != excluded_attr and is_finite_number(value)
    )


def format_score(value: float | None) -> str:
    if not is_finite_number(value):
        return ""
    return f"{float(value):.4f}"


def heatmap_summary_block(
    main_evals: list[MainEval],
    selected_hparams: HParams,
    interval: int,
    attr: str,
    output_path: Path,
    color_percentiles: tuple[float, float],
) -> str:
    x_values, y_values, matrix = build_heatmap(main_evals, attr)
    norm, colorbar_extend = color_scale_bounds(
        matrix,
        lower_percentile=color_percentiles[0],
        upper_percentile=color_percentiles[1],
        floor=COLOR_SCALE_FLOOR,
    )
    fixed_hparams = main_evals[0].hparams
    peak_items = peak_cells(matrix)
    global_y, global_x = np.unravel_index(np.nanargmax(matrix), matrix.shape)

    lines = [
        f"interval={interval} attr={attr}",
        f"plot={output_path.name}",
        f"selected_train_hparams={format_hparams(selected_hparams)}",
        f"fixed_main_hparams={format_hparams(fixed_hparams, excluded_attr=attr)}",
        f"shape={len(y_values)}x{len(x_values)}",
        (
            "color_scale="
            f"vmin={norm.vmin:.6f} vmax={norm.vmax:.6f} extend={colorbar_extend}"
        ),
        (
            "global_peak="
            f"main_{attr}={format_number(x_values[global_x], 4)} "
            f"cooldown_{attr}={format_number(y_values[global_y], 4)} "
            f"tta_val_acc={format_score(float(matrix[global_y, global_x]))}"
        ),
    ]

    if peak_items:
        lines.append("peaks=")
        for peak_y, peak_x in peak_items:
            lines.append(
                "  "
                f"main_{attr}={format_number(x_values[peak_x], 4)} "
                f"cooldown_{attr}={format_number(y_values[peak_y], 4)} "
                f"tta_val_acc={format_score(float(matrix[peak_y, peak_x]))}"
            )
    else:
        lines.append("peaks=none")

    lines.append("matrix:")
    lines.append(
        "\t".join(
            [f"cooldown_{attr}\\main_{attr}"]
            + [format_number(value, 4) for value in x_values]
        )
    )
    for y_index, y_value in enumerate(y_values):
        lines.append(
            "\t".join(
                [format_number(y_value, 4)]
                + [format_score(float(value)) for value in matrix[y_index]]
            )
        )
    lines.append("main_tta_val_acc_before_cooldown:")
    lines.append(
        "\t".join(
            [
                f"main_{attr}",
                "tta_val_acc_before_cooldown",
            ]
        )
    )
    for main_eval in main_evals:
        lines.append(
            "\t".join(
                [
                    format_number(getattr(main_eval.hparams, attr), 4),
                    format_score(main_eval.main_acc),
                ]
            )
        )
    return "\n".join(lines)


def write_run_summary(
    summary_path: Path,
    log_path: Path,
    intervals: list[int],
    blocks: list[str],
    color_percentiles: tuple[float, float],
) -> None:
    lines = [
        f"log={log_path}",
        f"intervals={' '.join(str(interval) for interval in intervals)}",
        (
            "color_percentiles="
            f"{format_number(color_percentiles[0], 4)} "
            f"{format_number(color_percentiles[1], 4)}"
        ),
        f"color_floor={format_number(COLOR_SCALE_FLOOR, 4)}",
        "",
    ]
    lines.extend("\n\n".join(blocks).splitlines())
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    color_percentiles = tuple(args.color_percentiles)
    if not (0 <= color_percentiles[0] < color_percentiles[1] <= 100):
        raise SystemExit("--color-percentiles must satisfy 0 <= LOW < HIGH <= 100")

    log_path = args.log
    output_dir = args.output_dir or default_output_dir(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = parse_log(log_path)
    summary_blocks: list[str] = []
    for interval in args.interval:
        train_interval = selected_interval(run.train_intervals, interval)
        selected_hparams = train_interval.hparams

        for attr in plot_attrs(selected_hparams):
            output_path = heatmap_output_path(output_dir, interval, attr)
            evals = main_variation_evals(
                run.main_evals,
                center=selected_hparams,
                interval=interval,
                attr=attr,
            )
            plot_lr_heatmap(
                evals,
                selected_hparams,
                interval,
                attr,
                output_path,
                color_percentiles=color_percentiles,
            )
            print(f"Wrote {output_path}")
            summary_blocks.append(
                heatmap_summary_block(
                    evals,
                    selected_hparams,
                    interval,
                    attr,
                    output_path,
                    color_percentiles,
                )
            )

        print(
            f"Selected interval {interval}: "
            f"{format_selected_hparams(selected_hparams)}"
        )

    summary_path = summary_output_path(output_dir)
    write_run_summary(
        summary_path,
        log_path,
        list(args.interval),
        summary_blocks,
        color_percentiles,
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

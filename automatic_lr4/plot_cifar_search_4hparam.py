#!/usr/bin/env python3
"""Plot TTA validation accuracy by searched hyperparameter value."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "20260626_022201_204962" / "cifar_search_4hparam.log"
DEFAULT_OUTPUT = (
    HERE / "20260626_022201_204962" / "search_hparam_tta_val_acc.png"
)
DEFAULT_TRUNCATED_OUTPUT = (
    HERE / "20260626_022201_204962" / "search_hparam_tta_val_acc_max1600.png"
)
DEFAULT_SUMMARY = HERE / "20260626_022201_204962" / "summary.txt"

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")
SUMMARY_SEARCH_HPARAMS_RE = re.compile(r"^Search hparams:\s+(?P<value>.+)$")


@dataclass(frozen=True)
class CalibrationPoint:
    hparam: str
    value: float
    tta_val_acc: float


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().rstrip(",")
    if value.lower() in {"none", "nan", "inf", "-inf"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_kv_line(line: str) -> dict[str, str]:
    return {match["key"]: match["value"].rstrip(",") for match in KV_RE.finditer(line)}


def parse_hparam_names(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_log(log_path: Path) -> tuple[list[str], list[CalibrationPoint]]:
    search_hparams: list[str] = []
    points: list[CalibrationPoint] = []

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("cifar_search_simple "):
                fields = parse_kv_line(line)
                search_hparams = parse_hparam_names(fields.get("search_hparams"))
                continue

            summary_match = SUMMARY_SEARCH_HPARAMS_RE.match(line)
            if summary_match and not search_hparams:
                search_hparams = parse_hparam_names(summary_match["value"])
                continue

            if not line.startswith("calibration_candidate "):
                continue

            fields = parse_kv_line(line)
            hparam = fields.get("varied_hparam")
            if not hparam or hparam == "none":
                continue

            value = parse_optional_float(fields.get(hparam))
            tta_val_acc = parse_optional_float(fields.get("tta_val_acc"))
            if value is None or tta_val_acc is None:
                continue

            points.append(
                CalibrationPoint(
                    hparam=hparam,
                    value=value,
                    tta_val_acc=tta_val_acc,
                )
            )

    if not search_hparams:
        search_hparams = sorted({point.hparam for point in points})

    return search_hparams, points


def style_axes(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_number(value: float) -> str:
    return f"{value:.4g}"


def evenly_spaced_values(
    points: list[CalibrationPoint],
) -> tuple[list[float], list[float], list[str]]:
    values = sorted({point.value for point in points})
    value_to_x = {value: float(index) for index, value in enumerate(values)}
    xs = [value_to_x[point.value] for point in points]
    tick_labels = [format_number(value) for value in values]
    return xs, values, tick_labels


def evenly_spaced_tick_indices(count: int, max_ticks: int = 12) -> list[int]:
    if count <= max_ticks:
        return list(range(count))

    indices = {
        round(index * (count - 1) / (max_ticks - 1))
        for index in range(max_ticks)
    }
    return sorted(indices)


def local_bests(points: list[CalibrationPoint]) -> list[CalibrationPoint]:
    best_by_value: dict[float, CalibrationPoint] = {}
    for point in points:
        current = best_by_value.get(point.value)
        if current is None or point.tta_val_acc > current.tta_val_acc:
            best_by_value[point.value] = point

    ordered = [best_by_value[value] for value in sorted(best_by_value)]
    bests: list[CalibrationPoint] = []
    for index, point in enumerate(ordered):
        left = ordered[index - 1].tta_val_acc if index > 0 else -math.inf
        right = (
            ordered[index + 1].tta_val_acc
            if index < len(ordered) - 1
            else -math.inf
        )
        if point.tta_val_acc >= left and point.tta_val_acc >= right:
            bests.append(point)
    return bests


def local_control_spans(
    points: list[CalibrationPoint],
) -> list[tuple[CalibrationPoint, CalibrationPoint, CalibrationPoint]]:
    best_by_value: dict[float, CalibrationPoint] = {}
    for point in points:
        current = best_by_value.get(point.value)
        if current is None or point.tta_val_acc > current.tta_val_acc:
            best_by_value[point.value] = point

    ordered = [best_by_value[value] for value in sorted(best_by_value)]
    spans: list[tuple[CalibrationPoint, CalibrationPoint, CalibrationPoint]] = []
    for best in local_bests(points):
        index = ordered.index(best)
        left_point = best
        right_point = best

        for left_index in range(index - 1, -1, -1):
            point = ordered[left_index]
            if point.tta_val_acc >= best.tta_val_acc:
                break
            left_point = point

        for right_index in range(index + 1, len(ordered)):
            point = ordered[right_index]
            if point.tta_val_acc >= best.tta_val_acc:
                break
            right_point = point

        spans.append((best, left_point, right_point))

    return spans


def write_summary(
    search_hparams: list[str],
    points: list[CalibrationPoint],
    output_path: Path,
) -> None:
    points_by_hparam: dict[str, list[CalibrationPoint]] = defaultdict(list)
    for point in points:
        points_by_hparam[point.hparam].append(point)

    lines = []
    for hparam in search_hparams:
        hparam_points = sorted(points_by_hparam.get(hparam, []), key=lambda point: point.value)
        if not hparam_points:
            continue
        lines.append(f"{hparam}:")
        for point in hparam_points:
            lines.append(f"{format_number(point.value)} -> {point.tta_val_acc:.4f}")
        lines.append("")

    if not lines:
        raise SystemExit("No hparam rows to write to summary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hparam_points(
    search_hparams: list[str],
    points: list[CalibrationPoint],
    log_path: Path,
    output_path: Path,
    title_suffix: str | None = None,
) -> None:
    points_by_hparam: dict[str, list[CalibrationPoint]] = defaultdict(list)
    for point in points:
        points_by_hparam[point.hparam].append(point)

    hparams = [name for name in search_hparams if points_by_hparam.get(name)]
    if not hparams:
        raise SystemExit(f"No calibration_candidate points parsed from {log_path}")

    cols = min(2, len(hparams))
    rows = math.ceil(len(hparams) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14.0 * cols, 4.2 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, hparam in zip(axes.flat, hparams):
        ax.set_visible(True)
        hparam_points = sorted(points_by_hparam[hparam], key=lambda point: point.value)
        xs, values, _ = evenly_spaced_values(hparam_points)
        value_to_x = {value: float(index) for index, value in enumerate(values)}
        ys = [point.tta_val_acc for point in hparam_points]

        dense_sweep = len(values) > 40
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.2 if dense_sweep else 1.6,
            markersize=2.8 if dense_sweep else 4.5,
            alpha=0.9,
        )
        best = max(hparam_points, key=lambda point: point.tta_val_acc)
        local_best_points = []
        if not dense_sweep:
            local_best_points = [
                point
                for point in local_bests(hparam_points)
                if point.value != best.value
            ]
        if local_best_points:
            ax.scatter(
                [value_to_x[point.value] for point in local_best_points],
                [point.tta_val_acc for point in local_best_points],
                marker="D",
                s=52,
                facecolor="white",
                edgecolor="#6b4c9a",
                linewidth=1.4,
                zorder=4,
                label="local best",
            )
        ax.scatter(
            [value_to_x[best.value]],
            [best.tta_val_acc],
            marker="*",
            s=150,
            color="#c43c39",
            zorder=5,
            label=f"best {format_number(best.value)} -> {best.tta_val_acc:.4f}",
        )

        if not dense_sweep:
            control_spans = local_control_spans(hparam_points)
            lane_start = 0.035
            lane_end = 0.16
            lane_step = (
                (lane_end - lane_start) / (len(control_spans) - 1)
                if len(control_spans) > 1
                else 0.0
            )
            for span_index, (_, left_point, right_point) in enumerate(control_spans):
                control_y = lane_start + lane_step * span_index
                ax.plot(
                    [value_to_x[left_point.value], value_to_x[right_point.value]],
                    [control_y, control_y],
                    marker="|",
                    markersize=12,
                    markeredgewidth=1.8,
                    linewidth=2.4,
                    color="#d89c28",
                    alpha=0.62,
                    solid_capstyle="round",
                    transform=ax.get_xaxis_transform(),
                    zorder=2,
                    label="local control span" if span_index == 0 else None,
                )

        ax.set_title(hparam)
        ax.set_xlabel("value")
        ax.set_ylabel("tta_val_acc")
        tick_indices = evenly_spaced_tick_indices(len(values))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [format_number(values[index]) for index in tick_indices],
            rotation=45,
            ha="right",
        )
        ax.legend(fontsize=8)
        style_axes(ax)

    title = f"TTA validation accuracy by search_hparam\n{log_path.parent.name}/{log_path.name}"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot calibration_candidate value -> tta_val_acc curves with one "
            "subplot per search_hparam."
        )
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file to plot. Defaults to {DEFAULT_LOG}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--truncated-output",
        type=Path,
        default=DEFAULT_TRUNCATED_OUTPUT,
        help=(
            "Output PNG path for the plot with hparam values above "
            f"1600 omitted. Defaults to {DEFAULT_TRUNCATED_OUTPUT}."
        ),
    )
    parser.add_argument(
        "-s",
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Output summary text path. Defaults to {DEFAULT_SUMMARY}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_hparams, points = parse_log(args.log)
    plot_hparam_points(search_hparams, points, args.log, args.output)
    truncated_points = [point for point in points if point.value <= 1600]
    plot_hparam_points(
        search_hparams,
        truncated_points,
        args.log,
        args.truncated_output,
        title_suffix="hparam values > 1600 omitted",
    )
    write_summary(search_hparams, points, args.summary)
    print(f"Parsed {len(points)} calibration points from {args.log}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.truncated_output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()

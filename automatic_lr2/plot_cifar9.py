import argparse
import concurrent.futures
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator


FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\s]+)")
JSON_DECODER = json.JSONDecoder()
LINE_SEARCH_VAL_SETS = [
    dict(
        key="current_batch",
        title="Current Batch",
        losses_field="current_batch_matrix_lr_train_losses",
        fallback_losses_field="matrix_lr_train_losses",
    ),
]
MATRIX_LR_LOSSES_FIELDS = [
    "matrix_lr_train_losses",
    *(spec["losses_field"] for spec in LINE_SEARCH_VAL_SETS),
]
SAMPLE_BEST_LRS_FIELDS = [
    "current_batch_sample_best_matrix_lrs",
]
SAMPLE_MATRIX_LR_LOSSES_FIELDS = [
    "current_batch_sample_matrix_lr_train_losses",
]
SAMPLE_LR_LOSS_CURVES_FIELDS = [
    "current_batch_sample_lr_loss_curves",
]
JSON_FIELDS = [
    *MATRIX_LR_LOSSES_FIELDS,
    *SAMPLE_BEST_LRS_FIELDS,
    *SAMPLE_MATRIX_LR_LOSSES_FIELDS,
    *SAMPLE_LR_LOSS_CURVES_FIELDS,
]
SAMPLE_LR_BASE = 0.24
SAMPLE_LR_FACTOR = 0.8
SAMPLE_LR_LEFT_STEPS = 30
SAMPLE_LR_RIGHT_STEPS = 20
SAMPLE_LR_GRID = [
    0.0,
    *[
        float("%.8g" % (SAMPLE_LR_BASE * (SAMPLE_LR_FACTOR**exponent)))
        for exponent in range(SAMPLE_LR_LEFT_STEPS, -SAMPLE_LR_RIGHT_STEPS - 1, -1)
    ],
]
SAMPLE_LR_POSITION_BY_VALUE = {lr: index for index, lr in enumerate(SAMPLE_LR_GRID)}
SAMPLE_LOSS_CURVE_DETAIL_STEPS = (0, 10, 50, 190)
SAMPLE_LOSS_CURVE_MAX_STEP = 200
SAMPLE_BEST_LR_HISTOGRAM_MAX_SUBPLOTS = 200
SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH = 0.02
SAMPLE_BEST_LR_HISTOGRAM_MAX_LR = 1.0
SAMPLE_BEST_LR_HISTOGRAM_PEAK_RADIUS = 6
PLOT_DPI = 180


def sample_lr_grid_position(lr):
    lr = float("%.8g" % lr)
    if lr in SAMPLE_LR_POSITION_BY_VALUE:
        return SAMPLE_LR_POSITION_BY_VALUE[lr]
    if lr <= 0:
        return 0.0

    positive_grid = SAMPLE_LR_GRID[1:]
    if lr <= positive_grid[0]:
        return 1.0
    if lr >= positive_grid[-1]:
        return float(len(SAMPLE_LR_GRID) - 1)
    for left_pos in range(1, len(SAMPLE_LR_GRID) - 1):
        left_lr = SAMPLE_LR_GRID[left_pos]
        right_lr = SAMPLE_LR_GRID[left_pos + 1]
        if left_lr <= lr <= right_lr:
            weight = (math.log(lr) - math.log(left_lr)) / (
                math.log(right_lr) - math.log(left_lr)
            )
            return left_pos + weight
    return float(len(SAMPLE_LR_GRID) - 1)


def nearest_sample_lr_grid_value(lr):
    lr = float("%.8g" % lr)
    if lr in SAMPLE_LR_POSITION_BY_VALUE:
        return lr
    if lr <= 0:
        return 0.0
    return min(
        SAMPLE_LR_GRID[1:],
        key=lambda grid_lr: abs(math.log(lr) - math.log(grid_lr)),
    )


def sample_lr_tick_positions():
    positions = [0, 1, *range(6, len(SAMPLE_LR_GRID), 5)]
    if positions[-1] != len(SAMPLE_LR_GRID) - 1:
        positions.append(len(SAMPLE_LR_GRID) - 1)
    return positions


def median(values):
    values = sorted(values)
    if not values:
        return float("nan")
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def percentile_value(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return values[index]


def float_or_nan(value):
    return float(value) if isinstance(value, (int, float)) else float("nan")


def normalize_sample_lr_loss_curves(payload):
    if not isinstance(payload, dict):
        return dict(lrs=[], optimal_lr_counts={}, curves=[])
    lrs = [float_or_nan(lr) for lr in payload.get("lrs", [])]
    optimal_lr_counts = {
        float(lr): int(count)
        for lr, count in payload.get("optimal_lr_counts", {}).items()
        if count
    }
    curves = []
    for curve in payload.get("curves", []):
        if not isinstance(curve, dict):
            continue
        losses = curve.get("losses", [])
        if not isinstance(losses, list):
            continue
        sample_index = curve.get("sample_index")
        if not isinstance(sample_index, int):
            sample_index = len(curves)
        curves.append(
            dict(
                sample_index=sample_index,
                optimal_lr=float_or_nan(curve.get("optimal_lr")),
                losses=[float_or_nan(loss) for loss in losses],
            )
        )
    return dict(lrs=lrs, optimal_lr_counts=optimal_lr_counts, curves=curves)


def parse_json_field(row, field, value):
    if field in MATRIX_LR_LOSSES_FIELDS:
        row[field] = sorted(
            (float(matrix_lr), float(loss)) for matrix_lr, loss in value.items()
        )
    elif field in SAMPLE_BEST_LRS_FIELDS:
        row[field] = [float(lr) for lr in value]
    elif field in SAMPLE_MATRIX_LR_LOSSES_FIELDS:
        row[field] = sorted(
            (
                float(matrix_lr),
                [
                    float(loss)
                    for loss in sample_losses
                    if isinstance(loss, (int, float))
                ],
            )
            for matrix_lr, sample_losses in value.items()
            if isinstance(sample_losses, list)
        )
    elif field in SAMPLE_LR_LOSS_CURVES_FIELDS:
        row[field] = normalize_sample_lr_loss_curves(value)


def parse_line_search_row(line):
    line = line.strip()
    if not line.startswith("line_search "):
        return None
    json_field_positions = sorted(
        (line.find(f" {field}="), field, f" {field}=")
        for field in JSON_FIELDS
        if line.find(f" {field}=") >= 0
    )
    first_json_field = json_field_positions[0][0] if json_field_positions else len(line)
    row = {}
    for match in FIELD_RE.finditer(line[:first_json_field]):
        key = match.group("key")
        value = match.group("value")
        row[key] = int(value) if key == "step" else float(value)
    for field_start, field, field_marker in json_field_positions:
        try:
            value, _ = JSON_DECODER.raw_decode(line[field_start + len(field_marker) :])
        except json.JSONDecodeError:
            value = {} if field not in SAMPLE_BEST_LRS_FIELDS else []
        parse_json_field(row, field, value)
    return row if "step" in row else None


def parse_line_search_rows(text):
    rows = []
    for line in text.splitlines():
        row = parse_line_search_row(line)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    return rows


def parse_line_search_file(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            row = parse_line_search_row(line)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["step"])
    return rows


def series(rows, key, fallback=None):
    return [row.get(key, row.get(fallback, float("nan"))) for row in rows]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def has_any(rows, key):
    return any(key in row for row in rows)


def configure_small_multiple_axis(ax, axis_index, ncols, total_axes, labelsize=5.5):
    ax.tick_params(axis="both", labelsize=labelsize)


def save_fast_figure(fig, output_path, show=False, dpi=PLOT_DPI):
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def matrix_lr_losses(row, val_set=None):
    if val_set is None:
        return row.get("matrix_lr_train_losses", [])
    losses = row.get(val_set["losses_field"])
    if losses is None and val_set.get("fallback_losses_field"):
        losses = row.get(val_set["fallback_losses_field"])
    return losses or []


def best_matrix_lr_loss(row, val_set=None):
    losses = [loss for _, loss in matrix_lr_losses(row, val_set) if finite(loss)]
    if not losses:
        return float("nan")
    return min(losses)


def best_matrix_lr(row, val_set=None):
    losses = [
        (matrix_lr, loss)
        for matrix_lr, loss in matrix_lr_losses(row, val_set)
        if finite(matrix_lr) and finite(loss)
    ]
    if not losses:
        return float("nan")
    return min(losses, key=lambda item: item[1])[0]


def zero_matrix_lr_loss(row, val_set=None):
    losses = [
        loss
        for matrix_lr, loss in matrix_lr_losses(row, val_set)
        if matrix_lr == 0.0 and finite(loss)
    ]
    if not losses:
        return float("nan")
    return losses[0]


def largest_matrix_lr_within_loss_gap(row, max_loss_gap, val_set=None):
    best_loss = best_matrix_lr_loss(row, val_set)
    zero_loss = zero_matrix_lr_loss(row, val_set)
    if not finite(best_loss):
        return float("nan")
    candidates = [
        matrix_lr
        for matrix_lr, loss in matrix_lr_losses(row, val_set)
        if (
            finite(matrix_lr)
            and matrix_lr >= 0
            and finite(loss)
            and relative_loss_diff(loss, best_loss, zero_loss) <= max_loss_gap
        )
    ]
    if not candidates:
        return float("nan")
    return max(candidates)


def estimated_loss_at_matrix_lr(row, key, val_set=None):
    target = row.get(key)
    if not finite(target):
        return float("nan")
    losses = [
        (matrix_lr, loss)
        for matrix_lr, loss in matrix_lr_losses(row, val_set)
        if finite(matrix_lr) and matrix_lr >= 0 and finite(loss)
    ]
    if not losses:
        return float("nan")
    losses.sort(key=lambda item: item[0])

    matrix_lr, loss = min(
        losses,
        key=lambda item: abs(item[0] - target) / max(abs(target), 1e-30),
    )
    relative_error = abs(matrix_lr - target) / max(abs(target), 1e-30)
    if relative_error <= 1e-5:
        return loss

    if target < 0:
        return float("nan")

    zero_loss = next((loss for matrix_lr, loss in losses if matrix_lr == 0.0), None)
    positive_losses = [(matrix_lr, loss) for matrix_lr, loss in losses if matrix_lr > 0]
    if not positive_losses:
        return float("nan")

    first_lr, first_loss = positive_losses[0]
    if target < first_lr:
        if zero_loss is None:
            return first_loss
        weight = target / first_lr
        return zero_loss + weight * (first_loss - zero_loss)

    for (left_lr, left_loss), (right_lr, right_loss) in zip(
        positive_losses, positive_losses[1:]
    ):
        if left_lr <= target <= right_lr:
            left_log = math.log(left_lr)
            right_log = math.log(right_lr)
            weight = (math.log(target) - left_log) / (right_log - left_log)
            return left_loss + weight * (right_loss - left_loss)

    return positive_losses[-1][1]


def relative_loss_diff(loss, best_loss, zero_loss):
    if not finite(loss) or not finite(best_loss) or not finite(zero_loss):
        return float("nan")
    denominator = zero_loss - best_loss
    if denominator <= 0:
        return float("nan")
    return (loss - best_loss) / denominator


def ground_truth_relative_loss_diff(row, val_set=None):
    ground_truth_loss = estimated_loss_at_matrix_lr(
        row, "ground_truth_matrix_lr", val_set
    )
    best_loss = best_matrix_lr_loss(row, val_set)
    zero_loss = zero_matrix_lr_loss(row, val_set)
    return relative_loss_diff(ground_truth_loss, best_loss, zero_loss)


def percentile(values, fraction):
    values = sorted(value for value in values if finite(value))
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def positive_linthresh(rows, val_set=None):
    positive_lrs = [
        matrix_lr
        for row in rows
        for matrix_lr, _ in matrix_lr_losses(row, val_set)
        if finite(matrix_lr) and matrix_lr > 0
    ]
    if not positive_lrs:
        return 1e-12
    return max(min(positive_lrs) * 0.5, 1e-12)


def sample_best_matrix_lrs(row):
    return [
        lr
        for lr in row.get("current_batch_sample_best_matrix_lrs", [])
        if finite(lr) and lr >= 0
    ]


def sample_lr_loss_curve_payload(row):
    return row.get(
        "current_batch_sample_lr_loss_curves",
        dict(lrs=[], optimal_lr_counts={}, curves=[]),
    )


def linear_lr_max_bin():
    return int(
        round(SAMPLE_BEST_LR_HISTOGRAM_MAX_LR / SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH)
    )


def linear_lr_bin_value(bin_index):
    max_bin = linear_lr_max_bin()
    bin_index = min(max(bin_index, 0), max_bin)
    return float(f"{bin_index * SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH:.10g}")


def linear_lr_bin_index(lr):
    max_bin = linear_lr_max_bin()
    return min(max(int(round(lr / SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH)), 0), max_bin)


def add_linear_lr_count(counts_by_lr, lr, count):
    if not finite(lr) or lr < 0 or not finite(count) or count <= 0:
        return
    if lr <= 0:
        counts_by_lr[0.0] += count
        return
    if lr >= SAMPLE_BEST_LR_HISTOGRAM_MAX_LR:
        counts_by_lr[SAMPLE_BEST_LR_HISTOGRAM_MAX_LR] += count
        return

    bin_position = lr / SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH
    nearest_bin = round(bin_position)
    if abs(bin_position - nearest_bin) < 1e-9:
        counts_by_lr[linear_lr_bin_value(nearest_bin)] += count
        return

    left_bin = math.floor(bin_position)
    right_bin = left_bin + 1
    right_count = count * (bin_position - left_bin)
    counts_by_lr[linear_lr_bin_value(left_bin)] += count - right_count
    counts_by_lr[linear_lr_bin_value(right_bin)] += right_count


def finalize_linear_lr_counts(counts_by_lr):
    if sum(counts_by_lr.values()) <= 0:
        return Counter()
    for bin_index in range(linear_lr_max_bin() + 1):
        counts_by_lr[linear_lr_bin_value(bin_index)] += 0
    return counts_by_lr


def sample_best_matrix_lr_counts(row):
    payload_counts = sample_lr_loss_curve_payload(row).get("optimal_lr_counts", {})
    if payload_counts:
        counts = Counter()
        for lr, count in payload_counts.items():
            if finite(lr) and lr >= 0 and count > 0:
                counts[nearest_sample_lr_grid_value(lr)] += count
        return counts
    return Counter(nearest_sample_lr_grid_value(lr) for lr in sample_best_matrix_lrs(row))


def sample_best_matrix_lr_linear_counts(row):
    counts = Counter()
    payload_counts = sample_lr_loss_curve_payload(row).get("optimal_lr_counts", {})
    if payload_counts:
        for lr, count in payload_counts.items():
            add_linear_lr_count(counts, lr, count)
        return finalize_linear_lr_counts(counts)

    for lr in sample_best_matrix_lrs(row):
        add_linear_lr_count(counts, lr, 1)
    return finalize_linear_lr_counts(counts)


def weighted_percentile_value(counts_by_value, fraction):
    total = sum(counts_by_value.values())
    if total <= 0:
        return float("nan")
    target = min(total - 1, max(0, math.ceil(total * fraction) - 1))
    seen = 0
    for value, count in sorted(counts_by_value.items()):
        seen += count
        if seen > target:
            return value
    return max(counts_by_value)


def weighted_mean_value(counts_by_value):
    total = sum(counts_by_value.values())
    if total <= 0:
        return float("nan")
    return sum(value * count for value, count in counts_by_value.items()) / total


def triangle_smoothed_peak_value(counts_by_value):
    counts = [0] * len(SAMPLE_LR_GRID)
    for lr, count in counts_by_value.items():
        if count <= 0:
            continue
        lr = nearest_sample_lr_grid_value(lr)
        counts[SAMPLE_LR_POSITION_BY_VALUE[lr]] += count

    if not any(counts):
        return float("nan")

    radius = SAMPLE_BEST_LR_HISTOGRAM_PEAK_RADIUS
    peak_position = max(
        range(len(counts)),
        key=lambda position: sum(
            (radius + 1 - abs(offset)) * counts[position + offset]
            for offset in range(-radius, radius + 1)
            if 0 <= position + offset < len(counts)
        ),
    )
    return SAMPLE_LR_GRID[peak_position]


def triangle_smoothed_linear_peak_value(counts_by_value):
    indexed_counts = Counter()
    for lr, count in counts_by_value.items():
        if not finite(lr) or lr < 0 or not finite(count) or count <= 0:
            continue
        indexed_counts[linear_lr_bin_index(lr)] += count

    if not indexed_counts:
        return float("nan")

    radius = SAMPLE_BEST_LR_HISTOGRAM_PEAK_RADIUS
    candidate_positions = {
        max(0, bin_index + offset)
        for bin_index in indexed_counts
        for offset in range(-radius, radius + 1)
    }
    peak_position = max(
        candidate_positions,
        key=lambda position: sum(
            (radius + 1 - abs(offset)) * indexed_counts[position + offset]
            for offset in range(-radius, radius + 1)
            if position + offset >= 0
        ),
    )
    return linear_lr_bin_value(peak_position)


def sample_matrix_lr_train_losses(row):
    return [
        (matrix_lr, sample_losses)
        for matrix_lr, sample_losses in row.get(
            "current_batch_sample_matrix_lr_train_losses", []
        )
        if finite(matrix_lr)
        and matrix_lr >= 0
        and any(finite(loss) for loss in sample_losses)
    ]


def sample_loss_curve_data(row):
    payload = sample_lr_loss_curve_payload(row)
    lrs = [lr for lr in payload.get("lrs", []) if finite(lr) and lr >= 0]
    curves = [
        curve
        for curve in payload.get("curves", [])
        if len(curve.get("losses", [])) == len(lrs)
    ]
    if lrs and curves:
        return lrs, curves

    sample_losses_by_lr = sample_matrix_lr_train_losses(row)
    if not sample_losses_by_lr:
        return [], []
    sample_losses_by_lr.sort(key=lambda item: item[0])
    lrs = [matrix_lr for matrix_lr, _ in sample_losses_by_lr]
    sample_count = min(len(sample_losses) for _, sample_losses in sample_losses_by_lr)
    best_sample_by_lr = {}
    for sample_index in range(sample_count):
        losses = [
            sample_losses[sample_index]
            if finite(sample_losses[sample_index])
            else float("inf")
            for _, sample_losses in sample_losses_by_lr
        ]
        best_lr = lrs[min(range(len(lrs)), key=lambda index: losses[index])]
        best_sample_by_lr.setdefault(best_lr, sample_index)
    curves = [
        dict(
            sample_index=sample_index,
            optimal_lr=optimal_lr,
            losses=[
                sample_losses[sample_index]
                if finite(sample_losses[sample_index])
                else float("nan")
                for _, sample_losses in sample_losses_by_lr
            ],
        )
        for optimal_lr, sample_index in sorted(best_sample_by_lr.items())
    ]
    return lrs, curves


def plot_landscape(ax, runs, val_set=None):
    all_deltas = []
    scatter_inputs = []
    for label, rows in runs:
        xs = []
        ys = []
        deltas = []
        for row in rows:
            best_loss = best_matrix_lr_loss(row, val_set)
            zero_loss = zero_matrix_lr_loss(row, val_set)
            if not finite(best_loss) or not finite(zero_loss):
                continue
            for matrix_lr, loss in matrix_lr_losses(row, val_set):
                if not finite(matrix_lr) or matrix_lr < 0 or not finite(loss):
                    continue
                delta = relative_loss_diff(loss, best_loss, zero_loss)
                if not finite(delta):
                    continue
                delta = max(0.0, delta)
                xs.append(row["step"])
                ys.append(matrix_lr)
                deltas.append(delta)
                all_deltas.append(delta)
        scatter_inputs.append((Path(label).name, xs, ys, deltas))

    if not all_deltas:
        ax.text(
            0.5,
            0.5,
            "no matrix lr loss data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    vmax = percentile(all_deltas, 0.95)
    if not finite(vmax) or vmax <= 0:
        vmax = max(all_deltas) if all_deltas else 1.0
    if vmax <= 0:
        vmax = 1.0
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

    image = None
    for _, xs, ys, deltas in scatter_inputs:
        if not xs:
            continue
        image = ax.scatter(
            xs,
            ys,
            c=[min(delta, vmax) for delta in deltas],
            s=10,
            cmap="viridis",
            norm=norm,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )

    for label, rows in runs:
        if not rows:
            continue
        steps = series(rows, "step")
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"
        if has_any(rows, "ground_truth_matrix_lr"):
            ax.plot(
                steps,
                series(rows, "ground_truth_matrix_lr"),
                color="tab:orange",
                linewidth=1.8,
                linestyle="--",
                label=f"ground truth lr{suffix}",
            )
        if any(matrix_lr_losses(row, val_set) for row in rows):
            ax.plot(
                steps,
                [best_matrix_lr(row, val_set) for row in rows],
                color="tab:red",
                linewidth=1.8,
                label=f"best loss lr{suffix}",
            )

    ax.set_ylabel("matrix lr")
    ax.grid(True, alpha=0.22)
    ax.legend()
    ax.set_title("Evaluated LR Landscape")
    if image is not None:
        ax.figure.colorbar(
            image,
            ax=ax,
            label="relative candidate loss diff",
        )


def first_loss_landscape_rows(runs, limit, val_set=None):
    rows = []
    for label, run_rows in runs:
        for row in run_rows:
            if matrix_lr_losses(row, val_set):
                rows.append((label, row))
                if len(rows) == limit:
                    return rows
    return rows


def plot_loss_landscape_grid(
    runs, output_path, val_set=None, max_subplots=50, show=False
):
    selected_rows = first_loss_landscape_rows(runs, max_subplots, val_set)
    if not selected_rows:
        return None

    ncols = 5
    nrows = math.ceil(len(selected_rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 24),
    )
    axes = axes.reshape(-1)
    fig.subplots_adjust(
        left=0.045, right=0.99, bottom=0.055, top=0.93, wspace=0.22, hspace=0.45
    )
    linthresh = positive_linthresh([row for _, row in selected_rows], val_set)
    multiple_runs = len(runs) > 1

    for ax, (label, row) in zip(axes, selected_rows):
        losses = [
            (matrix_lr, loss)
            for matrix_lr, loss in matrix_lr_losses(row, val_set)
            if finite(matrix_lr) and matrix_lr >= 0 and finite(loss)
        ]
        losses.sort(key=lambda item: item[0])
        xs = [matrix_lr for matrix_lr, _ in losses]
        ys = [loss for _, loss in losses]
        ax.plot(xs, ys, marker=".", markersize=3.5, linewidth=0.9, color="tab:blue")

        pre_update_loss = row.get("pre_update_train_loss")
        if finite(pre_update_loss):
            ax.scatter(
                [0.0],
                [pre_update_loss],
                marker="o",
                s=18,
                color="red",
                linewidths=0,
                zorder=4,
            )

        ground_truth_lr = row.get("ground_truth_matrix_lr")
        if finite(ground_truth_lr) and ground_truth_lr >= 0:
            ax.axvline(
                ground_truth_lr,
                color="tab:orange",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
            )
        best_lr = best_matrix_lr(row, val_set)
        if finite(best_lr) and best_lr >= 0:
            ax.axvline(
                best_lr,
                color="tab:red",
                linewidth=0.9,
                linestyle=":",
                alpha=0.9,
            )

        title = f"step {row['step']}"
        if multiple_runs:
            title = f"{Path(label).name}\n{title}"
        ax.set_title(title, fontsize=8)
        ax.set_xscale("symlog", linthresh=linthresh)
        if xs:
            ax.set_xlim(left=0.0, right=max(xs))
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(selected_rows) :]:
        ax.set_visible(False)

    fig.supxlabel("matrix lr")
    fig.supylabel("loss")
    title = f"Loss vs LR Landscape for First {len(selected_rows)} Line-Search Substeps"
    if val_set is not None:
        title = f"{val_set['title']} {title}"
    fig.suptitle(title, fontsize=14)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color="tab:blue",
            marker=".",
            linewidth=0.9,
            label="evaluated loss",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=0.9,
            linestyle="--",
            label="ground-truth lr",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=0.9,
            linestyle=":",
            label="best loss lr",
        ),
    ]
    fig.legend(handles=handles, loc="upper right")
    save_fast_figure(fig, output_path, show)
    return output_path


def first_sample_loss_curve_rows(runs, max_step):
    rows = []
    for label, run_rows in runs:
        for row in run_rows:
            if row["step"] >= max_step:
                continue
            _, curves = sample_loss_curve_data(row)
            if curves:
                rows.append((label, row))
    rows.sort(key=lambda item: item[1]["step"])
    return rows


def plot_sample_loss_curve_grid(
    runs,
    output_path,
    max_step=SAMPLE_LOSS_CURVE_MAX_STEP,
    show=False,
):
    selected_rows = first_sample_loss_curve_rows(runs, max_step)
    if not selected_rows:
        return None

    ncols = 5
    nrows = math.ceil(len(selected_rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, max(2.7 * nrows, 6.0)),
    )
    axes = axes.reshape(-1)
    fig.subplots_adjust(
        left=0.04, right=0.995, bottom=0.04, top=0.94, wspace=0.18, hspace=0.42
    )
    linthresh = positive_linthresh(
        [row for _, row in selected_rows], LINE_SEARCH_VAL_SETS[0]
    )
    multiple_runs = len(runs) > 1

    for axis_index, (ax, (label, row)) in enumerate(zip(axes, selected_rows)):
        xs, curves = sample_loss_curve_data(row)
        segments = [
            [
                (x, loss if finite(loss) else float("nan"))
                for x, loss in zip(xs, curve["losses"])
            ]
            for curve in curves
        ]
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors="tab:blue",
                    linewidths=0.45,
                    alpha=0.18,
                )
            )
            ax.autoscale_view()

        mean_losses = []
        for lr_losses in zip(*(curve["losses"] for curve in curves)):
            finite_losses = [loss for loss in lr_losses if finite(loss)]
            mean_losses.append(
                sum(finite_losses) / len(finite_losses)
                if finite_losses
                else float("nan")
            )
        ax.plot(xs, mean_losses, color="black", linewidth=1.0, alpha=0.85)

        ground_truth_lr = row.get("ground_truth_matrix_lr")
        if finite(ground_truth_lr) and ground_truth_lr >= 0:
            ax.axvline(
                ground_truth_lr,
                color="tab:orange",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
            )
        best_lr = best_matrix_lr(row, LINE_SEARCH_VAL_SETS[0])
        if finite(best_lr) and best_lr >= 0:
            ax.axvline(
                best_lr,
                color="tab:red",
                linewidth=0.9,
                linestyle=":",
                alpha=0.9,
            )

        title = f"step {row['step']}"
        if multiple_runs:
            title = f"{Path(label).name}\n{title}"
        ax.set_title(title, fontsize=7)
        ax.set_xscale("symlog", linthresh=linthresh)
        if xs:
            ax.set_xlim(left=0.0, right=max(xs))
        configure_small_multiple_axis(
            ax, axis_index, ncols, len(selected_rows), labelsize=5.5
        )
        ax.grid(True, alpha=0.18)

    for ax in axes[len(selected_rows) :]:
        ax.set_visible(False)

    fig.supxlabel("matrix lr")
    fig.supylabel("sample loss")
    fig.suptitle(
        "Current Batch Sample Loss Curves "
        f"(one curve per optimal lr, steps < {max_step})",
        fontsize=14,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.45,
            label="sample loss curve",
        ),
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="mean sample loss"),
        plt.Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=0.9,
            linestyle="--",
            label="ground-truth lr",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=0.9,
            linestyle=":",
            label="best total loss lr",
        ),
    ]
    fig.legend(handles=handles, loc="upper right")
    save_fast_figure(fig, output_path, show)
    return output_path


def first_row_for_step(runs, target_step):
    for label, run_rows in runs:
        for row in run_rows:
            if row["step"] != target_step:
                continue
            xs, curves = sample_loss_curve_data(row)
            if curves:
                return label, row, xs, curves
    return None


def plot_sample_loss_curve_step_grid(runs, output_path, target_step, show=False):
    selected = first_row_for_step(runs, target_step)
    if selected is None:
        return None

    label, row, xs, curves = selected
    curves = sorted(
        curves,
        key=lambda curve: (
            curve.get("optimal_lr")
            if finite(curve.get("optimal_lr"))
            else float("inf"),
            curve.get("sample_index")
            if isinstance(curve.get("sample_index"), int)
            else float("inf"),
        ),
    )
    ncols = min(5, max(1, math.ceil(math.sqrt(len(curves)))))
    nrows = math.ceil(len(curves) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, max(2.8 * nrows, 4.0)),
    )
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    fig.subplots_adjust(
        left=0.055, right=0.995, bottom=0.04, top=0.94, wspace=0.2, hspace=0.45
    )
    linthresh = positive_linthresh([row], LINE_SEARCH_VAL_SETS[0])
    multiple_runs = len(runs) > 1

    ground_truth_lr = row.get("ground_truth_matrix_lr")
    best_lr = best_matrix_lr(row, LINE_SEARCH_VAL_SETS[0])
    for axis_index, (ax, curve) in enumerate(zip(axes, curves)):
        ys = [loss if finite(loss) else float("nan") for loss in curve["losses"]]
        ax.plot(xs, ys, color="tab:blue", linewidth=0.9, alpha=0.9)

        optimal_lr = curve.get("optimal_lr")
        if finite(optimal_lr) and optimal_lr >= 0:
            ax.axvline(
                optimal_lr,
                color="tab:green",
                linewidth=0.9,
                linestyle="-",
                alpha=0.9,
            )
        if finite(ground_truth_lr) and ground_truth_lr >= 0:
            ax.axvline(
                ground_truth_lr,
                color="tab:orange",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
            )
        if finite(best_lr) and best_lr >= 0:
            ax.axvline(
                best_lr,
                color="tab:red",
                linewidth=0.9,
                linestyle=":",
                alpha=0.9,
            )

        title = f"optimal lr {optimal_lr:.3g}"
        sample_index = curve.get("sample_index")
        if isinstance(sample_index, int):
            title = f"{title}\nsample {sample_index}"
        ax.set_title(title, fontsize=7)
        ax.set_xscale("symlog", linthresh=linthresh)
        if xs:
            ax.set_xlim(left=0.0, right=max(xs))
        configure_small_multiple_axis(ax, axis_index, ncols, len(curves), labelsize=5.5)
        ax.grid(True, alpha=0.18)

    for ax in axes[len(curves) :]:
        ax.set_visible(False)

    title = f"Current Batch Sample Loss Curves at Step {target_step}"
    if multiple_runs:
        title = f"{Path(label).name} {title}"
    fig.supxlabel("matrix lr")
    fig.supylabel("sample loss")
    fig.suptitle(title, fontsize=14)
    handles = [
        plt.Line2D([0], [0], color="tab:blue", linewidth=0.9, label="sample curve"),
        plt.Line2D(
            [0],
            [0],
            color="tab:green",
            linewidth=0.9,
            label="sample optimal lr",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=0.9,
            linestyle="--",
            label="ground-truth lr",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=0.9,
            linestyle=":",
            label="best total loss lr",
        ),
    ]
    fig.legend(handles=handles, loc="upper right")
    save_fast_figure(fig, output_path, show)
    return output_path


def first_sample_best_lr_rows(runs, limit, counts_fn=sample_best_matrix_lr_counts):
    rows = []
    for label, run_rows in runs:
        for row in run_rows:
            if counts_fn(row):
                rows.append((label, row))
                if len(rows) == limit:
                    return rows
    return rows


def plot_sample_best_lr_histogram_grid(
    runs,
    output_path,
    max_subplots=SAMPLE_BEST_LR_HISTOGRAM_MAX_SUBPLOTS,
    show=False,
    linear_lr=False,
):
    if linear_lr:
        counts_fn = sample_best_matrix_lr_linear_counts
    else:
        counts_fn = sample_best_matrix_lr_counts
    peak_fn = (
        triangle_smoothed_linear_peak_value
        if linear_lr
        else triangle_smoothed_peak_value
    )
    selected_rows = first_sample_best_lr_rows(runs, max_subplots, counts_fn)
    if not selected_rows:
        return None

    ncols = 10
    nrows = math.ceil(len(selected_rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(24, max(3.0 * nrows, 6.0)),
    )
    axes = axes.reshape(-1)
    fig.subplots_adjust(
        left=0.035, right=0.995, bottom=0.04, top=0.96, wspace=0.16, hspace=0.48
    )
    multiple_runs = len(runs) > 1

    for axis_index, (ax, (label, row)) in enumerate(zip(axes, selected_rows)):
        counts_by_lr = counts_fn(row)
        if not counts_by_lr:
            ax.set_visible(False)
            continue

        unique_lrs = sorted(counts_by_lr)
        counted_lrs = [lr for lr in unique_lrs if counts_by_lr[lr] > 0]
        counts = [counts_by_lr[lr] for lr in unique_lrs]
        if linear_lr:
            xs = unique_lrs
        else:
            xs = [sample_lr_grid_position(lr) for lr in unique_lrs]
        ax.vlines(xs, 0, counts, color="tab:blue", linewidth=1.2, alpha=0.85)
        ax.scatter(xs, counts, color="tab:blue", s=10, linewidths=0, zorder=3)

        min_lr = min(counted_lrs)
        peak_lr = peak_fn(counts_by_lr)
        median_lr = weighted_percentile_value(counts_by_lr, 0.5)
        p95_lr = weighted_percentile_value(counts_by_lr, 0.95)
        p98_lr = weighted_percentile_value(counts_by_lr, 0.98)
        line_specs = [
            (min_lr, "min", "tab:green", "--"),
            (peak_lr, "peak", "tab:cyan", "-."),
            (p95_lr, "p95", "tab:purple", ":"),
            (p98_lr, "p98", "tab:red", "--"),
            (median_lr, "median", "tab:brown", "-"),
        ]
        ground_truth_lr = row.get("ground_truth_matrix_lr")
        if finite(ground_truth_lr) and ground_truth_lr >= 0:
            line_specs.append((ground_truth_lr, "gt", "tab:orange", "-."))
        if linear_lr:
            ax.set_xlim(0.0, SAMPLE_BEST_LR_HISTOGRAM_MAX_LR)

        ymax = ax.get_ylim()[1]
        for x, name, color, linestyle in line_specs:
            x_pos = x if linear_lr else sample_lr_grid_position(x)
            ax.axvline(
                x_pos, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.9
            )
            ax.annotate(
                f"{name}\n{x:.3g}",
                xy=(x_pos, ymax),
                xytext=(0, -3),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="right",
                fontsize=5.5,
                color=color,
            )

        title = f"step {row['step']}"
        if multiple_runs:
            title = f"{Path(label).name}\n{title}"
        ax.set_title(title, fontsize=7)
        if linear_lr:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.xaxis.set_minor_locator(
                MultipleLocator(SAMPLE_BEST_LR_HISTOGRAM_BIN_WIDTH)
            )
        else:
            tick_positions = sample_lr_tick_positions()
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{SAMPLE_LR_GRID[position]:.3g}" for position in tick_positions],
                rotation=45,
                ha="right",
            )
            ax.set_xticks(range(len(SAMPLE_LR_GRID)), minor=True)
            ax.set_xlim(-1.4, len(SAMPLE_LR_GRID) + 0.4)
        configure_small_multiple_axis(
            ax, axis_index, ncols, len(selected_rows), labelsize=5.5
        )
        ax.tick_params(axis="x", which="minor", length=1.5)
        ax.grid(True, alpha=0.18)

    for ax in axes[len(selected_rows) :]:
        ax.set_visible(False)

    fig.supxlabel("sample best matrix lr")
    fig.supylabel("sample count")
    title_prefix = "Current Batch Per-Sample Best LR"
    if linear_lr:
        title_prefix = "Current Batch Per-Sample Best LR Linear-Bin"
    fig.suptitle(
        f"{title_prefix} Histograms ({len(selected_rows)} steps)",
        fontsize=14,
    )
    save_fast_figure(fig, output_path, show)
    return output_path


def plot_lr_curves(ax, runs, val_set, min_step=None, max_step=None):
    plotted = False
    plotted_steps = []
    for label, rows in runs:
        rows = [
            row
            for row in rows
            if (min_step is None or row["step"] >= min_step)
            and (max_step is None or row["step"] < max_step)
        ]
        has_matrix_lrs = any(matrix_lr_losses(row, val_set) for row in rows)
        has_peak_lrs = any(sample_best_matrix_lr_counts(row) for row in rows)
        if not rows or not (has_matrix_lrs or has_peak_lrs):
            continue

        steps = series(rows, "step")
        plotted_steps.extend(step for step in steps if finite(step))
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"
        if has_any(rows, "ground_truth_matrix_lr"):
            ax.plot(
                steps,
                series(rows, "ground_truth_matrix_lr"),
                label=f"ground truth lr{suffix}",
                linewidth=1.8,
                linestyle="-",
            )
        if has_matrix_lrs:
            ax.plot(
                steps,
                [best_matrix_lr(row, val_set) for row in rows],
                label=f"best loss lr{suffix}",
                linewidth=1.4,
                linestyle="--",
            )
            for max_loss_gap in (0.1, 0.2, 0.4):
                ax.plot(
                    steps,
                    [
                        largest_matrix_lr_within_loss_gap(row, max_loss_gap, val_set)
                        for row in rows
                    ],
                    label=f"largest lr within {max_loss_gap:g} rel loss{suffix}",
                    linewidth=1.3,
                    linestyle=":",
                )
        if has_peak_lrs:
            ax.plot(
                steps,
                [
                    triangle_smoothed_peak_value(sample_best_matrix_lr_counts(row))
                    for row in rows
                ],
                label=f"peak lr{suffix}",
                color="tab:cyan",
                linewidth=1.5,
                linestyle="-.",
            )
        plotted = True

    ax.set_ylabel("matrix lr")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    if plotted_steps:
        left = min(plotted_steps)
        right = max(plotted_steps)
        if left == right:
            left -= 0.5
            right += 0.5
        ax.set_xlim(left, right)
    return plotted


def plot_runs(runs, output_path, val_set, show=False):
    fig, (
        lr_first_ax,
        lr_rest_ax,
        loss_ax,
        gap_ax,
        landscape_ax,
    ) = plt.subplots(
        5,
        1,
        figsize=(11, 15),
        constrained_layout=True,
    )

    plotted = 0
    for label, rows in runs:
        if not rows:
            continue
        steps = series(rows, "step")
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"

        if any(matrix_lr_losses(row, val_set) for row in rows):
            zero_losses = [zero_matrix_lr_loss(row, val_set) for row in rows]
            if any(finite(loss) for loss in zero_losses):
                loss_ax.plot(
                    steps,
                    zero_losses,
                    label=f"lr = 0 loss{suffix}",
                    linewidth=1.4,
                    linestyle="-.",
                )
            loss_ax.plot(
                steps,
                [best_matrix_lr_loss(row, val_set) for row in rows],
                label=f"best evaluated loss{suffix}",
                linewidth=1.4,
                linestyle="--",
            )
            ground_truth_losses = [
                estimated_loss_at_matrix_lr(row, "ground_truth_matrix_lr", val_set)
                for row in rows
            ]
            if any(finite(loss) for loss in ground_truth_losses):
                loss_ax.plot(
                    steps,
                    ground_truth_losses,
                    label=f"estimated ground-truth lr loss{suffix}",
                    linewidth=1.4,
                    linestyle=":",
                )
        gap_ax.plot(
            steps,
            [ground_truth_relative_loss_diff(row, val_set) for row in rows],
            label=f"ground truth - best candidate{suffix}",
            linewidth=1.7,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No line_search rows found in the provided input.")

    lr_first_ax.set_title(f"{val_set['title']} Matrix Learning Rates (Steps < 50)")
    plot_lr_curves(lr_first_ax, runs, val_set, max_step=50)

    lr_rest_ax.set_title(f"{val_set['title']} Matrix Learning Rates (Steps >= 50)")
    plot_lr_curves(lr_rest_ax, runs, val_set, min_step=50)

    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()

    gap_ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    gap_ax.set_ylabel("relative loss diff")
    gap_ax.grid(True, alpha=0.25)
    gap_ax.legend()
    gap_ax.set_title(f"{val_set['title']} Estimated Ground-Truth LR Relative Loss Diff")

    plot_landscape(landscape_ax, runs, val_set)
    landscape_ax.set_xlabel("step")

    save_fast_figure(fig, output_path, show)


def output_dir_for_log(log_path, output_dir=None):
    path = Path(log_path)
    stem_path = path.with_suffix("") if path.suffix else path
    if output_dir is None:
        plot_dir = stem_path.with_name(stem_path.name)
    else:
        plot_dir = Path(output_dir) / stem_path.name

    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_log_job(job):
    label, output_dir, kind, step = job
    output_dir = Path(output_dir)
    rows = parse_line_search_file(label)
    runs = [(label, rows)]
    if kind == "sample_best_lr_histograms":
        output_path = output_dir / "current_batch_sample_best_lr_histograms.png"
        result = plot_sample_best_lr_histogram_grid(runs, output_path)
    elif kind == "sample_best_lr_linear_histograms":
        output_path = output_dir / "current_batch_sample_best_lr_linear_histograms.png"
        result = plot_sample_best_lr_histogram_grid(runs, output_path, linear_lr=True)
    elif kind == "sample_loss_curves":
        output_path = (
            output_dir
            / f"current_batch_sample_loss_curves_first{SAMPLE_LOSS_CURVE_MAX_STEP}.png"
        )
        result = plot_sample_loss_curve_grid(runs, output_path)
    elif kind == "sample_loss_curve_step":
        output_path = output_dir / f"current_batch_sample_loss_curves_step{step:03d}.png"
        result = plot_sample_loss_curve_step_grid(runs, output_path, step)
    elif kind == "lrs":
        val_set = LINE_SEARCH_VAL_SETS[0]
        output_path = output_dir / f"{val_set['key']}_lrs.png"
        plot_runs(runs, output_path, val_set)
        result = output_path
    elif kind == "loss_landscapes":
        val_set = LINE_SEARCH_VAL_SETS[0]
        output_path = output_dir / f"{val_set['key']}_loss_landscapes.png"
        result = plot_loss_landscape_grid(runs, output_path, val_set)
    else:
        raise ValueError(f"unknown plot job kind: {kind}")
    return str(result.resolve()) if result else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot matrix-lr curves from line_search stdout logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Text log files containing line_search prints.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Parent folder for plots. Each log writes into a child folder matching its stem.",
    )
    args = parser.parse_args()

    for label in args.logs:
        output_dir = output_dir_for_log(label, args.output_dir)
        jobs = [
            (label, str(output_dir), "sample_best_lr_histograms", None),
            (label, str(output_dir), "sample_best_lr_linear_histograms", None),
            (label, str(output_dir), "sample_loss_curves", None),
            *[
                (label, str(output_dir), "sample_loss_curve_step", step)
                for step in SAMPLE_LOSS_CURVE_DETAIL_STEPS
            ],
            (label, str(output_dir), "lrs", None),
            (label, str(output_dir), "loss_landscapes", None),
        ]
        max_workers = min(len(jobs), os.cpu_count() or 1, 4)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for output_path in executor.map(plot_log_job, jobs):
                if output_path:
                    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

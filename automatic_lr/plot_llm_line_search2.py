import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\s]+)")
MATRIX_LR_LOSSES_FIELD = " matrix_lr_train_losses="


def parse_line_search_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("line_search "):
            continue
        scalar_text, separator, matrix_loss_text = line.partition(
            MATRIX_LR_LOSSES_FIELD
        )
        row = {}
        for match in FIELD_RE.finditer(scalar_text):
            key = match.group("key")
            value = match.group("value")
            row[key] = int(value) if key == "step" else float(value)
        if separator:
            try:
                matrix_lr_losses, _ = json.JSONDecoder().raw_decode(matrix_loss_text)
            except json.JSONDecodeError:
                matrix_lr_losses = {}
            row["matrix_lr_train_losses"] = sorted(
                (float(matrix_lr), float(loss))
                for matrix_lr, loss in matrix_lr_losses.items()
            )
        if "step" in row:
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    return rows


def read_inputs(paths):
    return [(str(path), Path(path).read_text()) for path in paths]


def series(rows, key, fallback=None):
    return [row.get(key, row.get(fallback, float("nan"))) for row in rows]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def has_any(rows, key):
    return any(key in row for row in rows)


def matrix_lr_losses(row):
    return row.get("matrix_lr_train_losses", [])


def best_matrix_lr_loss(row):
    losses = [loss for _, loss in matrix_lr_losses(row) if finite(loss)]
    if not losses:
        return float("nan")
    return min(losses)


def best_matrix_lr(row):
    losses = [
        (matrix_lr, loss)
        for matrix_lr, loss in matrix_lr_losses(row)
        if finite(matrix_lr) and finite(loss)
    ]
    if not losses:
        return float("nan")
    return min(losses, key=lambda item: item[1])[0]


def largest_matrix_lr_within_loss_gap(row, max_loss_gap):
    best_loss = best_matrix_lr_loss(row)
    if not finite(best_loss):
        return float("nan")
    candidates = [
        matrix_lr
        for matrix_lr, loss in matrix_lr_losses(row)
        if (
            finite(matrix_lr)
            and matrix_lr >= 0
            and finite(loss)
            and relative_loss_diff(loss, best_loss) <= max_loss_gap
        )
    ]
    if not candidates:
        return float("nan")
    return max(candidates)


def estimated_loss_at_matrix_lr(row, key):
    target = row.get(key)
    if not finite(target):
        return float("nan")
    losses = [
        (matrix_lr, loss)
        for matrix_lr, loss in matrix_lr_losses(row)
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


def relative_loss_diff(loss, best_loss):
    if not finite(loss) or not finite(best_loss):
        return float("nan")
    return (loss - best_loss) / max(abs(best_loss), 1e-30)


def ground_truth_relative_loss_diff(row):
    ground_truth_loss = estimated_loss_at_matrix_lr(row, "ground_truth_matrix_lr")
    best_loss = best_matrix_lr_loss(row)
    return relative_loss_diff(ground_truth_loss, best_loss)


def percentile(values, fraction):
    values = sorted(value for value in values if finite(value))
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def positive_linthresh(rows):
    positive_lrs = [
        matrix_lr
        for row in rows
        for matrix_lr, _ in matrix_lr_losses(row)
        if finite(matrix_lr) and matrix_lr > 0
    ]
    if not positive_lrs:
        return 1e-12
    return max(min(positive_lrs) * 0.5, 1e-12)


def plot_landscape(ax, runs):
    all_deltas = []
    scatter_inputs = []
    for label, rows in runs:
        xs = []
        ys = []
        deltas = []
        for row in rows:
            best_loss = best_matrix_lr_loss(row)
            if not finite(best_loss):
                continue
            for matrix_lr, loss in matrix_lr_losses(row):
                if not finite(matrix_lr) or matrix_lr < 0 or not finite(loss):
                    continue
                xs.append(row["step"])
                ys.append(matrix_lr)
                delta = max(0.0, relative_loss_diff(loss, best_loss))
                deltas.append(delta)
                all_deltas.append(delta)
        scatter_inputs.append((Path(label).name, xs, ys, deltas))

    if not all_deltas:
        ax.text(
            0.5,
            0.5,
            "no matrix_lr_train_losses data",
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
        if any(matrix_lr_losses(row) for row in rows):
            ax.plot(
                steps,
                [best_matrix_lr(row) for row in rows],
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


def first_loss_landscape_rows(runs, limit):
    rows = []
    for label, run_rows in runs:
        for row in run_rows:
            if matrix_lr_losses(row):
                rows.append((label, row))
                if len(rows) == limit:
                    return rows
    return rows


def plot_loss_landscape_grid(runs, output_path, max_subplots=50, show=False):
    selected_rows = first_loss_landscape_rows(runs, max_subplots)
    if not selected_rows:
        return None

    ncols = 5
    nrows = math.ceil(len(selected_rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 24),
        constrained_layout=True,
    )
    axes = axes.reshape(-1)
    linthresh = positive_linthresh([row for _, row in selected_rows])
    multiple_runs = len(runs) > 1

    for ax, (label, row) in zip(axes, selected_rows):
        losses = [
            (matrix_lr, loss)
            for matrix_lr, loss in matrix_lr_losses(row)
            if finite(matrix_lr) and matrix_lr >= 0 and finite(loss)
        ]
        losses.sort(key=lambda item: item[0])
        xs = [matrix_lr for matrix_lr, _ in losses]
        ys = [loss for _, loss in losses]
        ax.plot(xs, ys, marker=".", markersize=3.5, linewidth=0.9, color="tab:blue")

        ground_truth_lr = row.get("ground_truth_matrix_lr")
        if finite(ground_truth_lr) and ground_truth_lr >= 0:
            ax.axvline(
                ground_truth_lr,
                color="tab:orange",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
            )
        best_lr = best_matrix_lr(row)
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
    fig.suptitle(
        f"Loss vs LR Landscape for First {len(selected_rows)} Line-Search Substeps",
        fontsize=14,
    )

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
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_lr_curves(ax, runs, max_rows=None):
    plotted = False
    plotted_steps = []
    for label, rows in runs:
        if max_rows is not None:
            rows = rows[:max_rows]
        if not rows or not any(matrix_lr_losses(row) for row in rows):
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
        ax.plot(
            steps,
            [best_matrix_lr(row) for row in rows],
            label=f"best loss lr{suffix}",
            linewidth=1.4,
            linestyle="--",
        )
        for max_loss_gap in (0.01, 0.02, 0.05):
            ax.plot(
                steps,
                [largest_matrix_lr_within_loss_gap(row, max_loss_gap) for row in rows],
                label=f"largest lr within {max_loss_gap:g} rel loss{suffix}",
                linewidth=1.3,
                linestyle=":",
            )
        plotted = True

    ax.set_ylabel("matrix lr")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    if max_rows is not None and plotted_steps:
        left = min(plotted_steps)
        right = max(plotted_steps)
        if left == right:
            left -= 0.5
            right += 0.5
        ax.set_xlim(left, right)


def plot_runs(runs, output_path, show=False):
    fig, (
        lr_ax,
        lr_early_ax,
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

        loss_specs = [
            ("pre_update_train_loss", "pre-update train loss"),
            ("train_loss", "train loss"),
        ]
        for key, name in loss_specs:
            if has_any(rows, key):
                loss_ax.plot(steps, series(rows, key), label=f"{name}{suffix}", linewidth=1.7)
        if any(matrix_lr_losses(row) for row in rows):
            loss_ax.plot(
                steps,
                [best_matrix_lr_loss(row) for row in rows],
                label=f"best evaluated loss{suffix}",
                linewidth=1.4,
                linestyle="--",
            )
            ground_truth_losses = [
                estimated_loss_at_matrix_lr(row, "ground_truth_matrix_lr")
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
            [ground_truth_relative_loss_diff(row) for row in rows],
            label=f"ground truth - best candidate{suffix}",
            linewidth=1.7,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No line_search rows found in the provided input.")

    lr_ax.set_title("Matrix Learning Rates")
    plot_lr_curves(lr_ax, runs)

    lr_early_ax.set_title("Matrix Learning Rates (First 50 Steps)")
    plot_lr_curves(lr_early_ax, runs, max_rows=50)

    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()

    gap_ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    gap_ax.set_ylabel("relative loss diff")
    gap_ax.grid(True, alpha=0.25)
    gap_ax.legend()
    gap_ax.set_title("Estimated Ground-Truth LR Relative Loss Diff")

    plot_landscape(landscape_ax, runs)
    landscape_ax.set_xlabel("step")

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def output_paths_for_log(log_path):
    path = Path(log_path)
    stem_path = path.with_suffix("") if path.suffix else path
    return (
        stem_path.with_name(f"{stem_path.name}_lrs.png"),
        stem_path.with_name(f"{stem_path.name}_loss_landscapes.png"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot matrix-lr curves from line_search stdout logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Text log files containing line_search prints.",
    )
    args = parser.parse_args()

    for label, text in read_inputs(args.logs):
        runs = [(label, parse_line_search_rows(text))]
        output_path, loss_landscape_output = output_paths_for_log(label)
        plot_runs(runs, output_path)
        print(f"wrote {output_path.resolve()}")
        if plot_loss_landscape_grid(runs, loss_landscape_output):
            print(f"wrote {loss_landscape_output.resolve()}")


if __name__ == "__main__":
    main()

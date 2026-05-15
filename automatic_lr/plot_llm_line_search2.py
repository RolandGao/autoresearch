import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    if not paths:
        return [("stdin", sys.stdin.read())]
    return [(str(path), Path(path).read_text()) for path in paths]


def series(rows, key, fallback=None):
    return [row.get(key, row.get(fallback, float("nan"))) for row in rows]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def has_any(rows, key):
    return any(key in row for row in rows)


def matrix_lr_losses(row):
    return row.get("matrix_lr_train_losses", [])


def best_matrix_lr(row):
    losses = [
        (matrix_lr, loss)
        for matrix_lr, loss in matrix_lr_losses(row)
        if finite(matrix_lr) and finite(loss)
    ]
    if not losses:
        return float("nan")
    return min(losses, key=lambda item: item[1])[0]


def best_matrix_lr_loss(row):
    losses = [loss for _, loss in matrix_lr_losses(row) if finite(loss)]
    if not losses:
        return float("nan")
    return min(losses)


def loss_at_matrix_lr(row, key):
    target = row.get(key)
    if not finite(target):
        return float("nan")
    losses = matrix_lr_losses(row)
    if not losses:
        return float("nan")

    matrix_lr, loss = min(
        losses,
        key=lambda item: abs(item[0] - target) / max(abs(target), 1e-30),
    )
    relative_error = abs(matrix_lr - target) / max(abs(target), 1e-30)
    return loss if relative_error <= 1e-5 else float("nan")


def ratio_series(rows, numerator_key, denominator_key):
    ratios = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator in (None, 0):
            ratios.append(float("nan"))
        else:
            ratios.append(numerator / denominator)
    return ratios


def derived_ratio_series(rows, numerator_fn, denominator_key):
    ratios = []
    for row in rows:
        numerator = numerator_fn(row)
        denominator = row.get(denominator_key)
        if not finite(numerator) or not finite(denominator) or denominator == 0:
            ratios.append(float("nan"))
        else:
            ratios.append(numerator / denominator)
    return ratios


def derived_ratio_from_values(rows, numerator_fn, denominator_fn):
    ratios = []
    for row in rows:
        numerator = numerator_fn(row)
        denominator = denominator_fn(row)
        if not finite(numerator) or not finite(denominator) or denominator == 0:
            ratios.append(float("nan"))
        else:
            ratios.append(numerator / denominator)
    return ratios


def percentile(values, fraction):
    values = sorted(value for value in values if finite(value))
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def plot_landscape(ax, runs):
    all_deltas = []
    scatter_inputs = []
    for label, rows in runs:
        xs = []
        ys = []
        deltas = []
        for row in rows:
            ground_truth_lr = row.get("ground_truth_matrix_lr")
            best_loss = best_matrix_lr_loss(row)
            if not finite(ground_truth_lr) or ground_truth_lr <= 0 or not finite(best_loss):
                continue
            for matrix_lr, loss in matrix_lr_losses(row):
                if not finite(matrix_lr) or matrix_lr <= 0 or not finite(loss):
                    continue
                xs.append(row["step"])
                ys.append(matrix_lr / ground_truth_lr)
                delta = max(0.0, loss - best_loss)
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

    image = None
    for _, xs, ys, deltas in scatter_inputs:
        if not xs:
            continue
        image = ax.scatter(
            xs,
            ys,
            c=[min(delta, vmax) for delta in deltas],
            s=8,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )

    for label, rows in runs:
        if not rows:
            continue
        steps = series(rows, "step")
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"
        if has_any(rows, "line_search_matrix_lr") and has_any(
            rows, "ground_truth_matrix_lr"
        ):
            ax.plot(
                steps,
                ratio_series(rows, "line_search_matrix_lr", "ground_truth_matrix_lr"),
                color="white",
                linewidth=2.4,
                alpha=0.9,
                label=f"selected / ground truth{suffix}",
            )
            ax.plot(
                steps,
                ratio_series(rows, "line_search_matrix_lr", "ground_truth_matrix_lr"),
                color="black",
                linewidth=1.2,
                alpha=0.9,
            )
        if any(matrix_lr_losses(row) for row in rows):
            ax.plot(
                steps,
                derived_ratio_series(rows, best_matrix_lr, "ground_truth_matrix_lr"),
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                label=f"best evaluated / ground truth{suffix}",
            )

    ax.axhline(1.0, color="0.85", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("candidate lr / ground truth lr")
    ax.grid(True, alpha=0.18)
    ax.legend()
    ax.set_title("Evaluated LR Landscape")
    if image is not None:
        ax.figure.colorbar(image, ax=ax, label="loss above per-step best")


def plot_runs(runs, output_path, show=False):
    fig, (lr_ax, ratio_ax, loss_ax, landscape_ax) = plt.subplots(
        4,
        1,
        figsize=(11, 13),
        sharex=True,
        constrained_layout=True,
    )

    plotted = 0
    for label, rows in runs:
        if not rows:
            continue
        steps = series(rows, "step")
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"

        lr_specs = [
            ("line_search_matrix_lr", "line search matrix lr"),
            ("ground_truth_matrix_lr", "ground truth matrix lr"),
        ]
        for key, name in lr_specs:
            if has_any(rows, key):
                lr_ax.plot(steps, series(rows, key), label=f"{name}{suffix}", linewidth=1.7)
        if any(matrix_lr_losses(row) for row in rows):
            lr_ax.plot(
                steps,
                [best_matrix_lr(row) for row in rows],
                label=f"best evaluated matrix lr{suffix}",
                linewidth=1.4,
                linestyle="--",
            )

        if has_any(rows, "ground_truth_matrix_lr") and has_any(rows, "line_search_matrix_lr"):
            ratio_ax.plot(
                steps,
                ratio_series(rows, "ground_truth_matrix_lr", "line_search_matrix_lr"),
                label=f"ground truth / line search{suffix}",
                linewidth=1.7,
            )
        if any(matrix_lr_losses(row) for row in rows):
            ratio_ax.plot(
                steps,
                derived_ratio_from_values(
                    rows,
                    lambda row: row.get("ground_truth_matrix_lr"),
                    best_matrix_lr,
                ),
                label=f"ground truth / best evaluated{suffix}",
                linewidth=1.4,
                linestyle="--",
            )

        loss_specs = [
            ("pre_update_train_loss", "pre-update train loss"),
            ("train_loss", "train loss"),
            ("line_search_loss", "line search loss"),
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
                loss_at_matrix_lr(row, "ground_truth_matrix_lr") for row in rows
            ]
            if any(finite(loss) for loss in ground_truth_losses):
                loss_ax.plot(
                    steps,
                    ground_truth_losses,
                    label=f"ground-truth lr loss{suffix}",
                    linewidth=1.4,
                    linestyle=":",
                )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No line_search rows found in the provided input.")

    lr_ax.set_title("LLM line-search matrix learning rate")
    lr_ax.set_ylabel("matrix lr")
    lr_ax.set_yscale("log")
    lr_ax.grid(True, alpha=0.25)
    lr_ax.legend()

    ratio_ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ratio_ax.set_ylabel("lr ratio")
    ratio_ax.set_yscale("log")
    ratio_ax.grid(True, alpha=0.25)
    ratio_ax.legend()

    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()

    plot_landscape(landscape_ax, runs)
    landscape_ax.set_xlabel("step")

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot matrix-lr curves from llm_line_search2.py stdout logs."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Text log files containing line_search prints. Reads stdin when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="llm_line_search2_lrs.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a matplotlib window after saving.",
    )
    args = parser.parse_args()

    runs = [
        (label, parse_line_search_rows(text))
        for label, text in read_inputs(args.logs)
    ]
    plot_runs(runs, args.output, show=args.show)
    print(f"wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

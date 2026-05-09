import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\s]+)")


def parse_line_search_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("line_search "):
            continue
        row = {}
        for match in FIELD_RE.finditer(line):
            key = match.group("key")
            value = match.group("value")
            row[key] = int(value) if key == "step" else float(value)
        if "step" in row:
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    return rows


def read_inputs(paths):
    if not paths:
        return [("stdin", sys.stdin.read())]
    return [(str(path), Path(path).read_text()) for path in paths]


def plot_runs(runs, output_path, show=False):
    fig, (
        whiten_bias_lr_ax,
        bias_lr_ax,
        head_lr_ax,
        optimizer2_lr_ax,
        whiten_bias_ratio_ax,
        bias_ratio_ax,
        head_ratio_ax,
        optimizer2_ratio_ax,
        loss_ax,
    ) = plt.subplots(
        9,
        1,
        figsize=(10, 21),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1, 1, 1]},
        constrained_layout=True,
    )

    plotted = 0

    def series(rows, key, fallback=None):
        return [row.get(key, row.get(fallback, float("nan"))) for row in rows]

    def has_any(rows, key):
        return any(key in row for row in rows)

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

    for label, rows in runs:
        if not rows:
            continue
        steps = series(rows, "step")
        suffix = "" if len(runs) == 1 else f" ({Path(label).name})"

        lr_specs_by_axis = [
            (
                whiten_bias_lr_ax,
                [
                    ("line_search_whiten_bias_lr", "line search whiten bias lr", None),
                    (
                        "second_line_search_whiten_bias_lr",
                        "second line search whiten bias lr",
                        None,
                    ),
                    ("ground_truth_whiten_bias_lr", "ground truth whiten bias lr", None),
                ],
            ),
            (
                bias_lr_ax,
                [
                    ("line_search_bias_lr", "line search bias lr", None),
                    ("second_line_search_bias_lr", "second line search bias lr", None),
                    ("ground_truth_bias_lr", "ground truth bias lr", None),
                ],
            ),
            (
                head_lr_ax,
                [
                    ("line_search_head_lr", "line search head lr", None),
                    ("second_line_search_head_lr", "second line search head lr", None),
                    ("ground_truth_head_lr", "ground truth head lr", None),
                ],
            ),
            (
                optimizer2_lr_ax,
                [
                    (
                        "line_search_optimizer2_lr",
                        "line search optimizer2 lr",
                        "line_search_lr",
                    ),
                    (
                        "second_line_search_optimizer2_lr",
                        "second line search optimizer2 lr",
                        "second_line_search_lr",
                    ),
                    (
                        "ground_truth_optimizer2_lr",
                        "ground truth optimizer2 lr",
                        "ground_truth_lr",
                    ),
                ],
            ),
        ]
        for ax, lr_specs in lr_specs_by_axis:
            for key, name, fallback in lr_specs:
                if has_any(rows, key) or (fallback and has_any(rows, fallback)):
                    ax.plot(
                        steps,
                        series(rows, key, fallback),
                        label=f"{name}{suffix}",
                        linewidth=1.7,
                    )

        ratio_specs = [
            (
                whiten_bias_ratio_ax,
                "ground_truth_whiten_bias_lr",
                "line_search_whiten_bias_lr",
                "whiten bias ground truth / line search",
            ),
            (
                bias_ratio_ax,
                "ground_truth_bias_lr",
                "line_search_bias_lr",
                "bias ground truth / line search",
            ),
            (
                head_ratio_ax,
                "ground_truth_head_lr",
                "line_search_head_lr",
                "head ground truth / line search",
            ),
            (
                optimizer2_ratio_ax,
                "ground_truth_optimizer2_lr",
                "line_search_optimizer2_lr",
                "optimizer2 ground truth / line search",
            ),
        ]
        for ax, numerator_key, denominator_key, name in ratio_specs:
            if has_any(rows, numerator_key) and has_any(rows, denominator_key):
                ax.plot(
                    steps,
                    ratio_series(rows, numerator_key, denominator_key),
                    label=f"{name}{suffix}",
                    linewidth=1.7,
                )
        if has_any(rows, "ground_truth_over_line_search"):
            optimizer2_ratio_ax.plot(
                steps,
                series(rows, "ground_truth_over_line_search"),
                label=f"ground truth / line search{suffix}",
                linewidth=1.7,
            )

        loss_specs = [
            ("train_loss", "train loss"),
            ("line_search_loss", "line search loss"),
            ("second_line_search_loss", "second line search loss"),
        ]
        for key, name in loss_specs:
            if has_any(rows, key):
                loss_ax.plot(steps, series(rows, key), label=f"{name}{suffix}", linewidth=1.7)
        plotted += 1

    if plotted == 0:
        raise SystemExit("No line_search rows found in the provided input.")

    whiten_bias_lr_ax.set_title("CIFAR line-search learning rates")
    for ax, ylabel in [
        (whiten_bias_lr_ax, "whiten bias lr"),
        (bias_lr_ax, "bias lr"),
        (head_lr_ax, "head lr"),
        (optimizer2_lr_ax, "optimizer2 lr"),
    ]:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()

    for ax, ylabel in [
        (whiten_bias_ratio_ax, "whiten bias ratio"),
        (bias_ratio_ax, "bias ratio"),
        (head_ratio_ax, "head ratio"),
        (optimizer2_ratio_ax, "optimizer2 ratio"),
    ]:
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()

    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot lr curves from cifar_line_search.py stdout logs."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Text log files containing line_search prints. Reads stdin when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cifar_line_search_lrs.png",
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

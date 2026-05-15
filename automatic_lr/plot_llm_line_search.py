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
        line = line.strip()
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


def plot_runs(runs, output_path, show=False):
    fig, (lr_ax, ratio_ax, loss_ax) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
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
            ("second_line_search_matrix_lr", "second line search matrix lr"),
            ("ground_truth_matrix_lr", "ground truth matrix lr"),
        ]
        for key, name in lr_specs:
            if has_any(rows, key):
                lr_ax.plot(steps, series(rows, key), label=f"{name}{suffix}", linewidth=1.7)

        if has_any(rows, "ground_truth_matrix_lr") and has_any(rows, "line_search_matrix_lr"):
            ratio_ax.plot(
                steps,
                ratio_series(rows, "ground_truth_matrix_lr", "line_search_matrix_lr"),
                label=f"ground truth / line search{suffix}",
                linewidth=1.7,
            )
        if has_any(rows, "ground_truth_matrix_lr") and has_any(
            rows, "second_line_search_matrix_lr"
        ):
            ratio_ax.plot(
                steps,
                ratio_series(rows, "ground_truth_matrix_lr", "second_line_search_matrix_lr"),
                label=f"ground truth / second line search{suffix}",
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
        description="Plot matrix-lr curves from llm_line_search.py stdout logs."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Text log files containing line_search prints. Reads stdin when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="llm_line_search_lrs.png",
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

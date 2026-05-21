import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline3_bs.log")
DEFAULT_OUTPUT = Path(__file__).with_name("cifar_baseline3_bs.png")

TABLE_LINE_RE = re.compile(r"^\|.*\|$")
RUN_RE = re.compile(
    r"^(?P<index>\d+)\s+"
    r"(?P<schedule>bs(?P<batch_size>\d+)\s+linear\s+"
    r"(?P<start_lr>[0-9.]+)->(?P<end_lr>[0-9.]+))$"
)
SUMMARY_RE = re.compile(r"^(?P<name>[^:]+):\s+(?P<value>[0-9.]+)$")

SUMMARY_KEYS = {
    "Batch size": "batch_size",
    "Train loss": "summary_train_loss",
    "Train acc": "summary_train_acc",
    "25batch train loss": "train25_loss",
    "Val acc": "summary_val_acc",
    "TTA val": "summary_tta_val_acc",
    "BN cal 25batch train loss": "bn_cal_train25_loss",
    "BN cal val acc": "bn_cal_val_acc",
    "BN cal TTA val": "bn_cal_tta_val_acc",
}


@dataclass
class Run:
    index: int
    schedule: str
    batch_size: int
    start_lr: float
    end_lr: float
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def parse_float(text):
    text = text.strip()
    return float(text) if text else float("nan")


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def parse_table_row(line):
    if not TABLE_LINE_RE.match(line):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) != 7:
        return None
    if cells[0] == "run" or cells[1] == "epoch":
        return None
    if set(line.strip()) <= {"-", "|"}:
        return None
    return cells


def parse_log(path):
    runs = []
    current_run = None

    for line in path.read_text().splitlines():
        cells = parse_table_row(line)
        if cells is not None:
            run_name, epoch_text, train_loss, train_acc, val_acc, tta_val_acc, time_seconds = cells
            if run_name:
                match = RUN_RE.match(run_name)
                if match is None:
                    current_run = None
                    continue
                current_run = Run(
                    index=int(match.group("index")),
                    schedule=match.group("schedule"),
                    batch_size=int(match.group("batch_size")),
                    start_lr=float(match.group("start_lr")),
                    end_lr=float(match.group("end_lr")),
                )
                runs.append(current_run)

            if current_run is None:
                continue

            epoch = epoch_text if epoch_text == "eval" else int(epoch_text)
            current_run.rows.append(
                dict(
                    epoch=epoch,
                    train_loss=parse_float(train_loss),
                    train_acc=parse_float(train_acc),
                    val_acc=parse_float(val_acc),
                    tta_val_acc=parse_float(tta_val_acc),
                    time_seconds=parse_float(time_seconds),
                )
            )
            continue

        summary_match = SUMMARY_RE.match(line.strip())
        if current_run is None or summary_match is None:
            continue

        key = SUMMARY_KEYS.get(summary_match.group("name"))
        if key is None:
            continue
        value = float(summary_match.group("value"))
        current_run.summary[key] = int(value) if key == "batch_size" else value

    return runs


def eval_row(run):
    for row in reversed(run.rows):
        if row["epoch"] == "eval":
            return row
    return None


def epoch_rows(run):
    return [row for row in run.rows if isinstance(row["epoch"], int)]


def metric(run, name):
    if name in run.summary:
        return run.summary[name]

    row = eval_row(run)
    if row is not None and name in row:
        return row[name]

    fallback_names = {
        "summary_train_loss": "train_loss",
        "summary_train_acc": "train_acc",
        "summary_val_acc": "val_acc",
        "summary_tta_val_acc": "tta_val_acc",
    }
    fallback_name = fallback_names.get(name)
    if row is not None and fallback_name is not None:
        return row[fallback_name]

    return float("nan")


def by_batch_size(runs):
    groups = {}
    for run in runs:
        groups.setdefault(run.batch_size, []).append(run)
    return {
        batch_size: sorted(batch_runs, key=lambda run: run.start_lr)
        for batch_size, batch_runs in sorted(groups.items())
    }


def best_run(runs, name):
    candidates = [run for run in runs if is_finite(metric(run, name))]
    if not candidates:
        return None
    return max(candidates, key=lambda run: metric(run, name))


def lowest_run(runs, name):
    candidates = [run for run in runs if is_finite(metric(run, name))]
    if not candidates:
        return None
    return min(candidates, key=lambda run: metric(run, name))


def annotate_best(ax, run, value, label, color):
    ax.scatter([run.start_lr], [value], s=75, color=color, zorder=6)
    ax.annotate(
        "%s bs%d lr %.3g %.4f" % (label, run.batch_size, run.start_lr, value),
        xy=(run.start_lr, value),
        xytext=(8, 11),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color=color),
        fontsize=9,
        color=color,
    )


def plot_metric_by_batch(ax, groups, metric_name, ylabel, title=None, linestyle="-"):
    all_lrs = sorted({run.start_lr for batch_runs in groups.values() for run in batch_runs})
    for batch_size, batch_runs in groups.items():
        lrs = [run.start_lr for run in batch_runs]
        values = [metric(run, metric_name) for run in batch_runs]
        ax.plot(lrs, values, marker="o", linewidth=1.5, linestyle=linestyle, label="bs%d" % batch_size)
    ax.set_xscale("log")
    ax.set_xticks(all_lrs)
    ax.set_xticklabels(["%.3g" % lr for lr in all_lrs], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")


def annotate_batch_maxima(ax, groups, metric_name):
    for batch_size, batch_runs in groups.items():
        best = best_run(batch_runs, metric_name)
        if best is None:
            continue
        value = metric(best, metric_name)
        ax.scatter([best.start_lr], [value], s=80, edgecolor="black", facecolor="none", zorder=7)
        ax.annotate(
            "bs%d max\nlr %.3g\n%.4f" % (batch_size, best.start_lr, value),
            xy=(best.start_lr, value),
            xytext=(8, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
            fontsize=8,
        )


def annotate_batch_minima(ax, groups, metric_name):
    for batch_size, batch_runs in groups.items():
        best = lowest_run(batch_runs, metric_name)
        if best is None:
            continue
        value = metric(best, metric_name)
        ax.scatter([best.start_lr], [value], s=80, edgecolor="black", facecolor="none", zorder=7)
        ax.annotate(
            "bs%d min\nlr %.3g\n%.4f" % (batch_size, best.start_lr, value),
            xy=(best.start_lr, value),
            xytext=(8, -28),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
            fontsize=8,
        )


def plot_results(runs, output_path, top_k):
    if not runs:
        raise ValueError("No runs parsed from log")

    groups = by_batch_size(runs)
    best_tta = best_run(runs, "summary_tta_val_acc")
    best_bn_tta = best_run(runs, "bn_cal_tta_val_acc")
    top_by_tta = sorted(
        [run for run in runs if is_finite(metric(run, "summary_tta_val_acc"))],
        key=lambda run: metric(run, "summary_tta_val_acc"),
        reverse=True,
    )[:top_k]
    highlighted = {run.index for run in top_by_tta}

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)
    fig.suptitle("CIFAR baseline3 batch-size Muon LR sweep", fontsize=14)

    ax = axes[0]
    plot_metric_by_batch(ax, groups, "summary_tta_val_acc", "Accuracy", "Final TTA accuracy")
    for batch_size, batch_runs in groups.items():
        lrs = [run.start_lr for run in batch_runs]
        values = [metric(run, "bn_cal_tta_val_acc") for run in batch_runs]
        ax.plot(lrs, values, marker="x", linewidth=1.1, linestyle="--", label="bs%d BN cal" % batch_size)
    if best_tta is not None:
        annotate_best(ax, best_tta, metric(best_tta, "summary_tta_val_acc"), "best TTA", "tab:red")
    if best_bn_tta is not None:
        annotate_best(ax, best_bn_tta, metric(best_bn_tta, "bn_cal_tta_val_acc"), "best BN", "tab:purple")
    annotate_batch_maxima(ax, groups, "summary_tta_val_acc")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    plot_metric_by_batch(ax, groups, "summary_val_acc", "Accuracy", "Final val accuracy")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[2]
    plot_metric_by_batch(ax, groups, "train25_loss", "Loss", "25-batch train loss")
    for batch_size, batch_runs in groups.items():
        lrs = [run.start_lr for run in batch_runs]
        values = [metric(run, "bn_cal_train25_loss") for run in batch_runs]
        ax.plot(lrs, values, marker="x", linewidth=1.1, linestyle="--", label="bs%d BN cal" % batch_size)
    annotate_batch_minima(ax, groups, "train25_loss")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[3]
    for run in runs:
        rows = epoch_rows(run)
        epochs = [row["epoch"] for row in rows]
        vals = [row["val_acc"] for row in rows]
        if run.index in highlighted:
            ax.plot(
                epochs,
                vals,
                linewidth=2.0,
                label="run %03d bs%d lr %.3g tta %.4f"
                % (run.index, run.batch_size, run.start_lr, metric(run, "summary_tta_val_acc")),
            )
        else:
            ax.plot(epochs, vals, color="0.72", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Epoch curves, with top TTA runs highlighted")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return best_tta, best_bn_tta


def print_summary(runs, best_tta, best_bn_tta):
    print("Parsed %d runs" % len(runs))
    for batch_size, batch_runs in by_batch_size(runs).items():
        best = best_run(batch_runs, "summary_tta_val_acc")
        if best is None:
            continue
        print(
            "Best bs%d TTA: run %03d lr=%.6g val=%.4f tta=%.4f bn_tta=%.4f"
            % (
                batch_size,
                best.index,
                best.start_lr,
                metric(best, "summary_val_acc"),
                metric(best, "summary_tta_val_acc"),
                metric(best, "bn_cal_tta_val_acc"),
            )
        )
    if best_tta is not None:
        print(
            "Best overall TTA: run %03d bs%d lr=%.6g val=%.4f tta=%.4f"
            % (
                best_tta.index,
                best_tta.batch_size,
                best_tta.start_lr,
                metric(best_tta, "summary_val_acc"),
                metric(best_tta, "summary_tta_val_acc"),
            )
        )
    if best_bn_tta is not None:
        print(
            "Best overall BN cal TTA: run %03d bs%d lr=%.6g val=%.4f tta=%.4f"
            % (
                best_bn_tta.index,
                best_bn_tta.batch_size,
                best_bn_tta.start_lr,
                metric(best_bn_tta, "bn_cal_val_acc"),
                metric(best_bn_tta, "bn_cal_tta_val_acc"),
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot CIFAR baseline3 batch-size Muon LR sweep results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of best TTA runs to highlight in the epoch plot.",
    )
    args = parser.parse_args()

    runs = parse_log(args.log)
    best_tta, best_bn_tta = plot_results(runs, args.output, args.top_k)
    print_summary(runs, best_tta, best_bn_tta)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

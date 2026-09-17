import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline3_linear2.log")
DEFAULT_OUTPUT = Path(__file__).with_name("cifar_baseline3_linear2.png")

TABLE_LINE_RE = re.compile(r"^\|.*\|$")
RUN_RE = re.compile(
    r"^(?P<index>\d+)\s+(?P<schedule>linear\s+(?P<start_lr>[0-9.]+)->(?P<end_lr>[0-9.]+))$"
)
SUMMARY_RE = re.compile(r"^(?P<name>[^:]+):\s+(?P<value>[0-9.]+)$")

SUMMARY_KEYS = {
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
                    continue
                current_run = Run(
                    index=int(match.group("index")),
                    schedule=match.group("schedule"),
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
        if key is not None:
            current_run.summary[key] = float(summary_match.group("value"))

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


def best_run(runs, name):
    candidates = [run for run in runs if is_finite(metric(run, name))]
    if not candidates:
        return None
    return max(candidates, key=lambda run: metric(run, name))


def annotate_best(ax, run, value, label, color):
    ax.scatter([run.start_lr], [value], s=70, color=color, zorder=5)
    ax.annotate(
        "%s %.4f at lr %.2f" % (label, value, run.start_lr),
        xy=(run.start_lr, value),
        xytext=(10, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color=color),
        fontsize=9,
        color=color,
    )


def plot_results(runs, output_path, top_k):
    if not runs:
        raise ValueError("No runs parsed from log")

    runs = sorted(runs, key=lambda run: run.start_lr)
    lrs = [run.start_lr for run in runs]

    top_by_tta = sorted(
        [run for run in runs if is_finite(metric(run, "summary_tta_val_acc"))],
        key=lambda run: metric(run, "summary_tta_val_acc"),
        reverse=True,
    )[:top_k]
    highlighted = {run.index for run in top_by_tta}

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=False)
    fig.suptitle("CIFAR baseline3 linear2 Muon LR sweep", fontsize=14)

    val_accs = [metric(run, "summary_val_acc") for run in runs]
    tta_accs = [metric(run, "summary_tta_val_acc") for run in runs]
    bn_val_accs = [metric(run, "bn_cal_val_acc") for run in runs]
    bn_tta_accs = [metric(run, "bn_cal_tta_val_acc") for run in runs]

    ax = axes[0]
    ax.plot(lrs, val_accs, marker="o", linewidth=1.6, label="Val")
    ax.plot(lrs, tta_accs, marker="o", linewidth=1.6, label="TTA val")
    ax.plot(lrs, bn_val_accs, marker="o", linewidth=1.2, linestyle="--", label="BN cal val")
    ax.plot(lrs, bn_tta_accs, marker="o", linewidth=1.2, linestyle="--", label="BN cal TTA val")

    best_tta = best_run(runs, "summary_tta_val_acc")
    best_bn_tta = best_run(runs, "bn_cal_tta_val_acc")
    if best_tta is not None:
        annotate_best(ax, best_tta, metric(best_tta, "summary_tta_val_acc"), "best TTA", "tab:red")
    if best_bn_tta is not None:
        annotate_best(
            ax,
            best_bn_tta,
            metric(best_bn_tta, "bn_cal_tta_val_acc"),
            "best BN cal TTA",
            "tab:purple",
        )
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    train25_losses = [metric(run, "train25_loss") for run in runs]
    bn_train25_losses = [metric(run, "bn_cal_train25_loss") for run in runs]

    ax = axes[1]
    ax.plot(lrs, train25_losses, marker="o", linewidth=1.6, label="25-batch train loss")
    ax.plot(
        lrs,
        bn_train25_losses,
        marker="o",
        linewidth=1.4,
        linestyle="--",
        label="BN cal 25-batch train loss",
    )
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2]
    bn_tta_deltas = [
        metric(run, "bn_cal_tta_val_acc") - metric(run, "summary_tta_val_acc")
        for run in runs
    ]
    bn_val_deltas = [
        metric(run, "bn_cal_val_acc") - metric(run, "summary_val_acc")
        for run in runs
    ]
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.plot(lrs, bn_val_deltas, marker="o", linewidth=1.4, label="BN cal val - val")
    ax.plot(lrs, bn_tta_deltas, marker="o", linewidth=1.4, label="BN cal TTA - TTA")
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel("Accuracy delta")
    ax.grid(True, alpha=0.25)
    ax.legend()

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
                label="run %03d lr %.2f tta %.4f"
                % (run.index, run.start_lr, metric(run, "summary_tta_val_acc")),
            )
        else:
            ax.plot(epochs, vals, color="0.72", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Epoch curves, with top TTA runs highlighted")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return best_tta, best_bn_tta


def main():
    parser = argparse.ArgumentParser(
        description="Plot CIFAR baseline3 linear2 Muon LR sweep results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of best TTA schedules to highlight in the epoch plot.",
    )
    args = parser.parse_args()

    runs = parse_log(args.log)
    best_tta, best_bn_tta = plot_results(runs, args.output, args.top_k)

    print("Parsed %d runs from %s" % (len(runs), args.log))
    if best_tta is not None:
        print(
            "Best TTA: run %03d %s val=%.4f tta=%.4f"
            % (
                best_tta.index,
                best_tta.schedule,
                metric(best_tta, "summary_val_acc"),
                metric(best_tta, "summary_tta_val_acc"),
            )
        )
    if best_bn_tta is not None:
        print(
            "Best BN cal TTA: run %03d %s val=%.4f tta=%.4f"
            % (
                best_bn_tta.index,
                best_bn_tta.schedule,
                metric(best_bn_tta, "bn_cal_val_acc"),
                metric(best_bn_tta, "bn_cal_tta_val_acc"),
            )
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

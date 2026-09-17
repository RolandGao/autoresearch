import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline3_linear.log")
DEFAULT_OUTPUT = Path(__file__).with_name("cifar_baseline3_linear.png")
TABLE_LINE_RE = re.compile(r"^\|.*\|$")
RUN_RE = re.compile(
    r"^(?P<index>\d+)\s+(?P<schedule>linear\s+(?P<start_lr>[0-9.]+)->(?P<end_lr>[0-9.]+))$"
)


@dataclass
class Run:
    index: int
    schedule: str
    start_lr: float
    end_lr: float
    rows: list


def parse_float(text):
    text = text.strip()
    return float(text) if text else float("nan")


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
        if cells is None:
            continue

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
                rows=[],
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

    return runs


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def eval_row(run):
    for row in reversed(run.rows):
        if row["epoch"] == "eval":
            return row
    return None


def epoch_rows(run):
    return [row for row in run.rows if isinstance(row["epoch"], int)]


def metric_at_eval(run, metric):
    row = eval_row(run)
    return row[metric] if row is not None else float("nan")


def best_run(runs, metric):
    candidates = [run for run in runs if is_finite(metric_at_eval(run, metric))]
    if not candidates:
        return None
    return max(candidates, key=lambda run: metric_at_eval(run, metric))


def plot_results(runs, output_path, top_k):
    if not runs:
        raise ValueError("No runs parsed from log")

    runs = sorted(runs, key=lambda run: run.start_lr)
    top_by_tta = sorted(
        [run for run in runs if is_finite(metric_at_eval(run, "tta_val_acc"))],
        key=lambda run: metric_at_eval(run, "tta_val_acc"),
        reverse=True,
    )[:top_k]
    highlighted = {run.index for run in top_by_tta}
    best_tta = best_run(runs, "tta_val_acc")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    fig.suptitle("CIFAR baseline3 linear Muon LR sweep", fontsize=14)

    lrs = [run.start_lr for run in runs]
    val_accs = [metric_at_eval(run, "val_acc") for run in runs]
    tta_accs = [metric_at_eval(run, "tta_val_acc") for run in runs]
    train_losses = [metric_at_eval(run, "train_loss") for run in runs]

    ax = axes[0]
    ax.plot(lrs, val_accs, marker="o", linewidth=1.6, label="Val accuracy")
    ax.plot(lrs, tta_accs, marker="o", linewidth=1.6, label="TTA val accuracy")
    if best_tta is not None:
        best_lr = best_tta.start_lr
        best_acc = metric_at_eval(best_tta, "tta_val_acc")
        ax.scatter([best_lr], [best_acc], s=80, color="tab:red", zorder=5)
        ax.annotate(
            "best TTA %.4f at lr %.2f" % (best_acc, best_lr),
            xy=(best_lr, best_acc),
            xytext=(10, 12),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
            fontsize=9,
        )
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(lrs, train_losses, marker="o", linewidth=1.6, color="tab:green")
    ax.set_xlabel("Initial Muon LR")
    ax.set_ylabel("Eval train loss")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
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
                % (run.index, run.start_lr, metric_at_eval(run, "tta_val_acc")),
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

    return best_tta


def main():
    parser = argparse.ArgumentParser(
        description="Plot CIFAR baseline3 linear Muon LR sweep results."
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
    best_tta = plot_results(runs, args.output, args.top_k)

    print("Parsed %d runs from %s" % (len(runs), args.log))
    if best_tta is not None:
        print(
            "Best TTA: run %03d %s val=%.4f tta=%.4f"
            % (
                best_tta.index,
                best_tta.schedule,
                metric_at_eval(best_tta, "val_acc"),
                metric_at_eval(best_tta, "tta_val_acc"),
            )
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

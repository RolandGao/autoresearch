import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("cifar_angular3.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_angular3_plots")
SUMMARY_MAX_TRAIN25_LOSS = 1.2

RUN_RE = re.compile(r"^angular_lr_search run=(?P<run>\d+) angular_lr=(?P<lr>\S+)")
STEP_RE = re.compile(r"^step=(?P<step>\d+)/(?P<total_steps>\d+)\s+")
EVAL_RE = re.compile(r"^eval(?: run=(?P<run>\S+))? epoch=(?P<epoch>\S+)\s+")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_.]+)=(?P<value>\S+)")


@dataclass
class Run:
    index: int
    angular_lr: float
    steps: list[dict[str, float]] = field(default_factory=list)
    evals: list[dict[str, float]] = field(default_factory=list)


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_key_values(line):
    return {
        match.group("key"): parse_float(match.group("value"))
        for match in KEY_VALUE_RE.finditer(line)
    }


def parse_log(path):
    runs = []
    current = None

    for line in path.read_text().splitlines():
        run_match = RUN_RE.match(line)
        if run_match is not None:
            current = Run(
                index=int(run_match.group("run")),
                angular_lr=parse_float(run_match.group("lr")),
            )
            runs.append(current)
            continue

        if current is None:
            continue

        step_match = STEP_RE.match(line)
        if step_match is not None:
            row = parse_key_values(line)
            row["step"] = int(step_match.group("step"))
            row["total_steps"] = int(step_match.group("total_steps"))
            current.steps.append(row)
            continue

        eval_match = EVAL_RE.match(line)
        if eval_match is not None:
            row = parse_key_values(line)
            epoch = eval_match.group("epoch")
            row["epoch"] = epoch if epoch == "final" else int(epoch)
            current.evals.append(row)

    return runs


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def final_eval(run):
    for row in reversed(run.evals):
        if row.get("epoch") == "final":
            return row
    return run.evals[-1] if run.evals else {}


def final_metric(run, name):
    return final_eval(run).get(name, float("nan"))


def best_run(runs, metric):
    candidates = [run for run in runs if finite(final_metric(run, metric))]
    if not candidates:
        return None
    return max(candidates, key=lambda run: final_metric(run, metric))


def min_run(runs, metric):
    candidates = [run for run in runs if finite(final_metric(run, metric))]
    if not candidates:
        return None
    return min(candidates, key=lambda run: final_metric(run, metric))


def top_runs(runs, metric, count):
    candidates = [run for run in runs if finite(final_metric(run, metric))]
    return sorted(candidates, key=lambda run: final_metric(run, metric), reverse=True)[
        :count
    ]


def lowest_runs(runs, metric, count):
    candidates = [run for run in runs if finite(final_metric(run, metric))]
    return sorted(candidates, key=lambda run: final_metric(run, metric))[:count]


def values_for(runs, metric):
    return [final_metric(run, metric) for run in runs]


def annotate_metric(ax, run, metric, label, color="tab:red", xytext=(10, 12)):
    if run is None:
        return
    y = final_metric(run, metric)
    if not finite(y):
        return
    ax.scatter([run.angular_lr], [y], color=color, s=70, zorder=4)
    ax.annotate(
        "%s %.4f\nangular_lr %.6g" % (label, y, run.angular_lr),
        xy=(run.angular_lr, y),
        xytext=xytext,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
        fontsize=9,
    )


def format_lr_tick(value):
    text = "%.3f" % value if value < 0.1 else "%.2f" % value
    return text.rstrip("0").rstrip(".")


def annotate_summary_x_axis(ax, runs):
    ax.set_xticks([run.angular_lr for run in runs])
    ax.set_xticklabels(
        ["%d\n%s" % (run.index, format_lr_tick(run.angular_lr)) for run in runs],
        fontsize=6,
    )
    ax.tick_params(axis="x", which="major", pad=2)
    ax.set_xlabel("Run index / initial angular LR")


def initial_post_warmup_head_lr(run):
    for row in run.steps:
        head_lr = row.get("head_lr", float("nan"))
        if finite(head_lr) and head_lr < 1.0:
            return head_lr
    return float("nan")


def first_nan_step(run):
    for row in run.steps:
        if not finite(row.get("loss", float("nan"))):
            return row["step"]
    return float("nan")


def plot_summary(runs, output_path):
    runs = [
        run
        for run in sorted(runs, key=lambda run: run.angular_lr)
        if finite(final_metric(run, "25batch_train_loss"))
        and final_metric(run, "25batch_train_loss") <= SUMMARY_MAX_TRAIN25_LOSS
    ]
    if not runs:
        raise ValueError(
            "No runs with finite 25batch_train_loss <= %.4f"
            % SUMMARY_MAX_TRAIN25_LOSS
        )

    best_tta = best_run(runs, "tta_val_acc")
    min_train_loss = min_run(runs, "25batch_train_loss")
    lrs = [run.angular_lr for run in runs]

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(
        "CIFAR Angular LR Sweep (25-batch train loss <= %.1f)"
        % SUMMARY_MAX_TRAIN25_LOSS,
        fontsize=14,
    )

    ax = axes[0]
    ax.plot(lrs, values_for(runs, "val_acc"), marker="o", label="Val accuracy")
    ax.plot(lrs, values_for(runs, "tta_val_acc"), marker="o", label="TTA val accuracy")
    annotate_metric(ax, best_tta, "tta_val_acc", "best TTA")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(
        lrs,
        values_for(runs, "25batch_train_loss"),
        marker="o",
        color="tab:green",
        label="25-batch train loss",
    )
    annotate_metric(
        ax,
        min_train_loss,
        "25batch_train_loss",
        "min train loss",
        color="tab:orange",
        xytext=(10, -28),
    )
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2]
    ax.plot(
        lrs,
        [initial_post_warmup_head_lr(run) for run in runs],
        marker="o",
        label="First post-warmup head_lr",
    )
    ax.set_ylabel("Head LR")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[3]
    ax.plot(lrs, values_for(runs, "time_seconds"), marker="o", label="Final seconds")
    ax2 = ax.twinx()
    ax2.plot(
        lrs,
        [first_nan_step(run) for run in runs],
        marker="x",
        linestyle="--",
        color="tab:red",
        label="First NaN step",
    )
    ax.set_xlabel("Initial angular LR")
    ax.set_ylabel("Seconds")
    ax2.set_ylabel("Step")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    for ax in axes:
        ax.set_xscale("log")
    annotate_summary_x_axis(axes[-1], runs)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return best_tta


def plot_epoch_curves(runs, output_path, top_k):
    selected = {run.index for run in top_runs(runs, "tta_val_acc", top_k)}

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Validation Accuracy by Epoch", fontsize=14)

    for run in sorted(runs, key=lambda run: run.angular_lr):
        rows = [row for row in run.evals if isinstance(row.get("epoch"), int)]
        epochs = [row["epoch"] for row in rows]
        vals = [row.get("val_acc", float("nan")) for row in rows]
        if run.index in selected:
            ax.plot(
                epochs,
                vals,
                linewidth=2.0,
                label="run %d lr %.6g tta %.4f"
                % (run.index, run.angular_lr, final_metric(run, "tta_val_acc")),
            )
        else:
            ax.plot(epochs, vals, color="0.72", linewidth=0.8, alpha=0.45)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_training_curves(runs, output_path, top_k):
    selected = lowest_runs(runs, "25batch_train_loss", top_k)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Lowest 25-Batch Train Loss Runs", fontsize=14)

    for run in selected:
        label = "run %d lr %.6g loss %.4f tta %.4f" % (
            run.index,
            run.angular_lr,
            final_metric(run, "25batch_train_loss"),
            final_metric(run, "tta_val_acc"),
        )
        steps = [row["step"] for row in run.steps]
        axes[0].plot(
            steps,
            [row.get("loss", float("nan")) for row in run.steps],
            label=label,
        )
        axes[1].plot(
            steps,
            [row.get("head_lr", float("nan")) for row in run.steps],
            label=label,
        )
        axes[2].plot(
            steps,
            [row.get("head_angle_rad", float("nan")) for row in run.steps],
            label=label,
        )

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Head LR")
    axes[2].set_ylabel("Head angle rad")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_angles(runs, output_path):
    run = best_run(runs, "tta_val_acc")
    if run is None:
        return

    angle_names = sorted(
        {
            key
            for row in run.steps
            for key in row
            if key.endswith("_angle_rad") and key != "head_angle_rad"
        }
    )
    steps = [row["step"] for row in run.steps]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Best Run Weight Angles: run %d, angular_lr %.6g, TTA %.4f"
        % (run.index, run.angular_lr, final_metric(run, "tta_val_acc")),
        fontsize=14,
    )
    ax.plot(
        steps,
        [row.get("head_angle_rad", float("nan")) for row in run.steps],
        linewidth=2.0,
        color="black",
        label="head",
    )
    ax.plot(
        steps,
        [
            row.get("head_lr", float("nan"))
            if row.get("head_lr", float("nan")) < 1.0
            else float("nan")
            for row in run.steps
        ],
        linewidth=1.5,
        color="tab:red",
        linestyle="--",
        label="angular target lr",
    )
    ax.axvline(5, color="0.35", linestyle=":", linewidth=1.0, label="AngularSGD starts")
    for name in angle_names:
        label = name.removesuffix("_angle_rad").replace("muon.", "")
        ax.plot(
            steps,
            [row.get(name, float("nan")) for row in run.steps],
            linewidth=1.2,
            alpha=0.85,
            label=label,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Angle radians")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all(runs, output_dir, top_k):
    if not runs:
        raise ValueError("No runs parsed from log")

    output_dir.mkdir(parents=True, exist_ok=True)
    best = plot_summary(runs, output_dir / "summary.png")
    plot_epoch_curves(runs, output_dir / "epoch_curves.png", top_k)
    plot_top_training_curves(runs, output_dir / "top_training_curves.png", top_k)
    plot_best_angles(runs, output_dir / "best_run_angles.png")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Plot autoresearch/automatic_lr2/cifar_angular3.log."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of runs to highlight in curve plots.",
    )
    args = parser.parse_args()

    runs = parse_log(args.log)
    best = plot_all(runs, args.output_dir, args.top_k)

    print("Parsed %d runs from %s" % (len(runs), args.log))
    if best is not None:
        print(
            "Best TTA: run %d angular_lr=%.6g val=%.4f tta=%.4f"
            % (
                best.index,
                best.angular_lr,
                final_metric(best, "val_acc"),
                final_metric(best, "tta_val_acc"),
            )
        )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

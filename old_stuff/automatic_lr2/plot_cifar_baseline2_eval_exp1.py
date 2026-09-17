import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval_exp1_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+)"
)
SNAPSHOT_RE = re.compile(
    r"^training_batch_losses step=(?P<step>\d+)/(?P<total_steps>\d+) losses=(?P<losses>\[.*\])$"
)
FINAL_RE = re.compile(
    r"^eval epoch=final 25batch_train_loss=(?P<train25_loss>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)


@dataclass
class Snapshot:
    step: int
    total_steps: int
    losses: np.ndarray


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    snapshots: list[Snapshot] = field(default_factory=list)
    train25_loss: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None

    @property
    def label(self):
        return f"run {self.index}: bs={self.batch_size}, muon_lr={self.muon_lr:g}"


def parse_log(path):
    runs = []
    current = None
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current = Run(
                    index=int(match.group("run")),
                    batch_size=int(match.group("batch_size")),
                    muon_lr=float(match.group("muon_lr")),
                )
                runs.append(current)
                continue

            match = SNAPSHOT_RE.match(line)
            if match:
                if current is None:
                    raise ValueError(
                        f"Found snapshot before run header on line {line_number}"
                    )
                losses = np.array(json.loads(match.group("losses")), dtype=np.float64)
                current.snapshots.append(
                    Snapshot(
                        step=int(match.group("step")),
                        total_steps=int(match.group("total_steps")),
                        losses=losses,
                    )
                )
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.train25_loss = float(match.group("train25_loss"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                current.time_seconds = float(match.group("time_seconds"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def snapshot_matrix(run):
    if not run.snapshots:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack([snapshot.losses for snapshot in run.snapshots])


def snapshot_progress(run):
    return np.array(
        [snapshot.step / snapshot.total_steps for snapshot in run.snapshots],
        dtype=np.float64,
    )


def finite_metric(value):
    return value is not None and math.isfinite(value)


def plot_mean_loss_curves(runs, output_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for run in runs:
        losses = snapshot_matrix(run)
        if losses.size == 0:
            continue
        progress = snapshot_progress(run)
        means = losses.mean(axis=1)
        p10 = np.percentile(losses, 10, axis=1)
        p90 = np.percentile(losses, 90, axis=1)
        label = run.label
        if finite_metric(run.tta_val_acc):
            label += f", TTA={run.tta_val_acc:.4f}"
        (line,) = ax.plot(progress, means, linewidth=2.0, label=label)
        ax.fill_between(progress, p10, p90, color=line.get_color(), alpha=0.12)

    ax.set_title("Training-Batch Loss Snapshots")
    ax.set_xlabel("training progress")
    ax.set_ylabel("loss across all batches")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(runs, output_path):
    labels = [f"run {run.index}\nbs={run.batch_size}\nlr={run.muon_lr:g}" for run in runs]
    val_acc = [run.val_acc if finite_metric(run.val_acc) else np.nan for run in runs]
    tta_val_acc = [
        run.tta_val_acc if finite_metric(run.tta_val_acc) else np.nan for run in runs
    ]
    train25_loss = [
        run.train25_loss if finite_metric(run.train25_loss) else np.nan for run in runs
    ]

    fig, (acc_ax, loss_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(runs))
    width = 0.35

    acc_ax.bar(x - width / 2, val_acc, width=width, label="val acc")
    acc_ax.bar(x + width / 2, tta_val_acc, width=width, label="TTA val acc")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_ylim(bottom=max(0.0, np.nanmin(val_acc + tta_val_acc) - 0.02))
    acc_ax.grid(True, axis="y", alpha=0.25)
    acc_ax.legend()

    loss_ax.bar(x, train25_loss, width=0.55, color="tab:green")
    loss_ax.set_ylabel("25-batch train loss")
    loss_ax.set_xticks(x, labels)
    loss_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Final Evaluation Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_heatmaps(runs, output_path):
    plot_runs = sorted(runs, key=lambda run: (run.batch_size != 125, run.index))
    matrices = [snapshot_matrix(run) for run in plot_runs if run.snapshots]
    if not matrices:
        return

    cmap = plt.get_cmap("gray_r", 20)
    ncols = 1
    nrows = math.ceil(len(plot_runs) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, 3.2 * nrows),
        sharey=False,
        squeeze=False,
    )

    image = None
    for ax, run in zip(axes.ravel(), plot_runs):
        losses = snapshot_matrix(run)
        if losses.size == 0:
            ax.axis("off")
            continue
        flat_losses = losses.ravel()
        order = np.argsort(flat_losses, kind="stable")
        flat_bins = np.empty_like(order)
        flat_bins[order] = np.floor(
            np.arange(len(flat_losses)) * cmap.N / len(flat_losses)
        ).astype(np.int64)
        binned_losses = flat_bins.reshape(losses.shape)
        image = ax.imshow(
            binned_losses,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=-0.5,
            vmax=cmap.N - 0.5,
            extent=(0, losses.shape[1], 0, 8),
        )
        epoch_ticks = np.arange(0, 9)
        ax.set_title(run.label)
        ax.set_xlabel("training batch epoch")
        ax.set_ylabel("snapshot epoch")
        ax.set_xticks(epoch_ticks * losses.shape[1] / 8)
        ax.set_xticklabels([str(epoch) for epoch in epoch_ticks])
        ax.set_yticks(epoch_ticks)
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.set_label("equal-count loss bucket")
        cbar.set_ticks([0, cmap.N - 1])
        cbar.set_ticklabels(["low", "high"])

    for ax in axes.ravel()[len(plot_runs) :]:
        ax.axis("off")

    fig.suptitle("Loss for Every Training Batch at Each Snapshot")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_float_for_filename(value):
    return ("%g" % value).replace(".", "p").replace("-", "m")


def plot_batch_loss_profiles(runs, output_dir):
    output_paths = []
    stale_output = output_dir / "batch_loss_profiles.png"
    if stale_output.exists():
        stale_output.unlink()

    for run in runs:
        losses = snapshot_matrix(run)
        if losses.size == 0:
            continue
        ncols = 5
        nrows = math.ceil(len(run.snapshots) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 1.8 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        x = np.linspace(0, 8, losses.shape[1], endpoint=False)
        ymin, ymax = losses.min(), losses.max()
        margin = 0.04 * (ymax - ymin) if ymax > ymin else 0.01

        for ax, snapshot, snapshot_losses in zip(
            axes.ravel(), run.snapshots, losses
        ):
            ax.plot(
                x,
                snapshot_losses,
                color="tab:blue",
                linewidth=0.65,
                alpha=0.85,
            )
            ax.set_title(
                "snapshot epoch %.2f" % (8 * snapshot.step / snapshot.total_steps),
                fontsize=8,
            )
            ax.set_xlim(0, 8)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_xticks(np.arange(0, 9))
            ax.grid(True, alpha=0.2)

        for ax in axes.ravel()[len(run.snapshots) :]:
            ax.axis("off")

        for ax in axes[:, 0]:
            ax.set_ylabel("loss")
        for ax in axes[-1, :]:
            ax.set_xlabel("training batch epoch")

        fig.suptitle(run.label)
        fig.tight_layout()
        output_path = (
            output_dir
            / (
                "batch_loss_profiles_run%02d_bs%d_lr%s.png"
                % (
                    run.index,
                    run.batch_size,
                    safe_float_for_filename(run.muon_lr),
                )
            )
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval training-batch loss snapshots."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_loss_curves(runs, args.output_dir / "mean_loss_curves.png")
    plot_loss_heatmaps(runs, args.output_dir / "loss_heatmaps.png")
    plot_batch_loss_profiles(runs, args.output_dir)
    plot_final_metrics(runs, args.output_dir / "final_metrics.png")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()

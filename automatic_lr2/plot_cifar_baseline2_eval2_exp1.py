import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp1_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+)"
    r"(?: name=(?P<name>\S+))?(?: use_best_lr=(?P<use_best_lr>\S+))?"
)
SNAPSHOT_PREFIX = "training_batch_losses "
BEST_LR_RE = re.compile(
    r"^best_lr step=(?P<step>\d+) "
    r"init_lr=(?P<init_lr>\S+) best_lr=(?P<best_lr>\S+) "
    r"best_lr_ema=(?P<best_lr_ema>\S+) best_loss=(?P<best_loss>\S+) "
    r"losses=(?P<losses>\{.*\})$"
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
    update: str
    losses: np.ndarray


@dataclass
class BestLRLog:
    step: int
    init_lr: float
    best_lr: float
    best_lr_ema: float
    best_loss: float
    losses_by_lr: dict[float, float]


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    name: str | None = None
    use_best_lr: bool = False
    snapshots: list[Snapshot] = field(default_factory=list)
    best_lr_logs: list[BestLRLog] = field(default_factory=list)
    train25_loss: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None

    @property
    def label(self):
        name = self.name or f"run {self.index}"
        return f"{name}: bs={self.batch_size}, muon_lr={self.muon_lr:g}"


def parse_bool(value):
    return str(value).lower() == "true"


def parse_snapshot(line):
    fields, losses_json = line[len(SNAPSHOT_PREFIX) :].split(" losses=", 1)
    values = {}
    for field in fields.split():
        key, value = field.split("=", 1)
        values[key] = value
    step, total_steps = values["step"].split("/", 1)
    return Snapshot(
        step=int(step),
        total_steps=int(total_steps),
        update=values.get("update", "post"),
        losses=np.array(json.loads(losses_json), dtype=np.float64),
    )


def parse_best_lr(match):
    losses_by_lr = {
        float(lr): float(loss)
        for lr, loss in json.loads(match.group("losses")).items()
    }
    return BestLRLog(
        step=int(match.group("step")),
        init_lr=float(match.group("init_lr")),
        best_lr=float(match.group("best_lr")),
        best_lr_ema=float(match.group("best_lr_ema")),
        best_loss=float(match.group("best_loss")),
        losses_by_lr=losses_by_lr,
    )


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
                    name=match.group("name"),
                    use_best_lr=parse_bool(match.group("use_best_lr")),
                )
                runs.append(current)
                continue

            if line.startswith(SNAPSHOT_PREFIX):
                if current is None:
                    raise ValueError(
                        f"Found snapshot before run header on line {line_number}"
                    )
                current.snapshots.append(parse_snapshot(line))
                continue

            match = BEST_LR_RE.match(line)
            if match:
                if current is None:
                    raise ValueError(
                        f"Found best_lr row before run header on line {line_number}"
                    )
                current.best_lr_logs.append(parse_best_lr(match))
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


def snapshots(run, update=None):
    selected = run.snapshots
    if update is not None:
        selected = [snapshot for snapshot in selected if snapshot.update == update]
    return selected


def snapshot_matrix(run, update=None):
    selected = snapshots(run, update)
    if not selected:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack([snapshot.losses for snapshot in selected])


def snapshot_progress(run, update=None):
    selected = snapshots(run, update)
    return np.array(
        [snapshot.step / snapshot.total_steps for snapshot in selected],
        dtype=np.float64,
    )


def finite_metric(value):
    return value is not None and math.isfinite(value)


def paired_snapshots(run):
    by_key = {}
    for snapshot in run.snapshots:
        by_key.setdefault(snapshot.step, {})[snapshot.update] = snapshot
    return [
        (updates["pre"], updates["post"])
        for _, updates in sorted(by_key.items())
        if "pre" in updates and "post" in updates
    ]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def plot_mean_loss_curves(runs, output_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for run in runs:
        post_losses = snapshot_matrix(run, "post")
        if post_losses.size == 0:
            continue
        progress = snapshot_progress(run, "post")
        means = post_losses.mean(axis=1)
        p10 = np.percentile(post_losses, 10, axis=1)
        p90 = np.percentile(post_losses, 90, axis=1)
        label = run.label
        if finite_metric(run.tta_val_acc):
            label += f", TTA={run.tta_val_acc:.4f}"
        (line,) = ax.plot(progress, means, linewidth=2.0, label=label)
        ax.fill_between(progress, p10, p90, color=line.get_color(), alpha=0.12)

        pre_losses = snapshot_matrix(run, "pre")
        if pre_losses.size:
            ax.plot(
                snapshot_progress(run, "pre"),
                pre_losses.mean(axis=1),
                linestyle=":",
                linewidth=1.4,
                color=line.get_color(),
                alpha=0.75,
            )

    ax.set_title("Training-Batch Loss Snapshots")
    ax.set_xlabel("training progress")
    ax.set_ylabel("loss across all batches")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pre_post_delta(runs, output_path):
    fig, ax = plt.subplots(figsize=(11, 5.8))

    for run in runs:
        pairs = paired_snapshots(run)
        if not pairs:
            continue
        progress = np.array([post.step / post.total_steps for _, post in pairs])
        deltas = np.array(
            [(post.losses - pre.losses).mean() for pre, post in pairs],
            dtype=np.float64,
        )
        ax.plot(progress, deltas, marker="o", markersize=3, linewidth=1.8, label=run.label)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_title("Mean Post-Update Loss Minus Pre-Update Loss")
    ax.set_xlabel("training progress")
    ax.set_ylabel("post - pre loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(runs, output_path):
    labels = [run.name or f"run {run.index}" for run in runs]
    val_acc = [run.val_acc if finite_metric(run.val_acc) else np.nan for run in runs]
    tta_val_acc = [
        run.tta_val_acc if finite_metric(run.tta_val_acc) else np.nan for run in runs
    ]
    train25_loss = [
        run.train25_loss if finite_metric(run.train25_loss) else np.nan for run in runs
    ]

    fig, (acc_ax, loss_ax) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    x = np.arange(len(runs))
    width = 0.35

    acc_ax.bar(x - width / 2, val_acc, width=width, label="val acc")
    acc_ax.bar(x + width / 2, tta_val_acc, width=width, label="TTA val acc")
    acc_ax.set_ylabel("accuracy")
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


def plot_loss_heatmaps(runs, output_path, update):
    plot_runs = [run for run in runs if snapshot_matrix(run, update).size]
    if not plot_runs:
        return

    nrows = len(plot_runs)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(13, 3.0 * nrows),
        sharex=False,
        squeeze=False,
    )

    image = None
    for ax, run in zip(axes.ravel(), plot_runs):
        losses = snapshot_matrix(run, update)
        progress = snapshot_progress(run, update)
        extent = (0, losses.shape[1], progress[0], progress[-1])
        image = ax.imshow(
            losses,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="viridis",
            extent=extent,
        )
        ax.set_title(f"{run.label} ({update})")
        ax.set_xlabel("training batch index")
        ax.set_ylabel("training progress")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("loss")
    fig.suptitle(f"{update.capitalize()}-Update Loss for Every Training Batch")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_lr_trace(runs, output_path):
    runs = [run for run in runs if run.best_lr_logs]
    if not runs:
        return

    fig, (lr_ax, loss_ax) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    for run in runs:
        steps = np.array([row.step for row in run.best_lr_logs], dtype=np.float64)
        total_steps = max((snapshot.total_steps for snapshot in run.snapshots), default=steps[-1])
        progress = steps / total_steps
        best_lr = np.array([row.best_lr for row in run.best_lr_logs], dtype=np.float64)
        init_lr = np.array([row.init_lr for row in run.best_lr_logs], dtype=np.float64)
        ema_lr = np.array([row.best_lr_ema for row in run.best_lr_logs], dtype=np.float64)
        best_loss = np.array([row.best_loss for row in run.best_lr_logs], dtype=np.float64)

        lr_ax.plot(progress, best_lr, marker="o", markersize=3, linewidth=1.8, label=f"{run.label} best")
        lr_ax.plot(progress, ema_lr, linestyle="--", linewidth=1.5, label=f"{run.label} ema")
        lr_ax.plot(progress, init_lr, linestyle=":", linewidth=1.2, label=f"{run.label} init")
        loss_ax.plot(progress, best_loss, marker="o", markersize=3, linewidth=1.8, label=run.label)

    lr_ax.set_yscale("log")
    lr_ax.set_ylabel("Muon LR")
    lr_ax.set_title("Best LR Search Trace")
    lr_ax.grid(True, alpha=0.25, which="both")
    lr_ax.legend(fontsize=8)

    loss_ax.set_xlabel("training progress")
    loss_ax.set_ylabel("current-batch selected loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_landscape_snapshots(runs, output_path, max_panels=24):
    rows = [(run, row) for run in runs for row in run.best_lr_logs]
    if not rows:
        return
    if len(rows) > max_panels:
        indices = np.linspace(0, len(rows) - 1, max_panels).round().astype(int)
        rows = [rows[index] for index in indices]

    ncols = 4
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 2.8 * nrows),
        squeeze=False,
    )

    for ax, (run, row) in zip(axes.ravel(), rows):
        lr_losses = sorted(row.losses_by_lr.items())
        lrs = np.array([lr for lr, _ in lr_losses], dtype=np.float64)
        losses = np.array([loss for _, loss in lr_losses], dtype=np.float64)
        ax.plot(lrs, losses, marker="o", linewidth=1.4)
        ax.axvline(row.best_lr, color="tab:red", linewidth=1.0, alpha=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{run.name or run.index} step {row.step}", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")

    fig.suptitle("Best-LR Probe Loss Landscapes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_batch_loss_profiles(runs, output_dir, update="post"):
    output_paths = []
    for run in runs:
        losses = snapshot_matrix(run, update)
        selected = snapshots(run, update)
        if losses.size == 0:
            continue

        ncols = 5
        nrows = math.ceil(len(selected) / ncols)
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

        for ax, snapshot, snapshot_losses in zip(axes.ravel(), selected, losses):
            ax.plot(x, snapshot_losses, color="tab:blue", linewidth=0.65, alpha=0.85)
            ax.set_title(
                "epoch %.2f" % (8 * snapshot.step / snapshot.total_steps),
                fontsize=8,
            )
            ax.set_xlim(0, 8)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_xticks(np.arange(0, 9))
            ax.grid(True, alpha=0.2)

        for ax in axes.ravel()[len(selected) :]:
            ax.axis("off")

        for ax in axes[:, 0]:
            ax.set_ylabel("loss")
        for ax in axes[-1, :]:
            ax.set_xlabel("training batch epoch")

        fig.suptitle(f"{run.label} ({update})")
        fig.tight_layout()
        output_path = output_dir / f"batch_loss_profiles_{safe_name(run.name or run.index)}_{update}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def write_summary(runs, output_path):
    lines = [
        "run,name,batch_size,muon_lr,use_best_lr,train25_loss,val_acc,tta_val_acc,time_seconds,best_lr_logs,final_best_lr_ema"
    ]
    for run in runs:
        final_best_lr_ema = (
            run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else ""
        )
        fields = [
            run.index,
            run.name or "",
            run.batch_size,
            "%.8g" % run.muon_lr,
            run.use_best_lr,
            "" if run.train25_loss is None else "%.8g" % run.train25_loss,
            "" if run.val_acc is None else "%.8g" % run.val_acc,
            "" if run.tta_val_acc is None else "%.8g" % run.tta_val_acc,
            "" if run.time_seconds is None else "%.8g" % run.time_seconds,
            len(run.best_lr_logs),
            "" if final_best_lr_ema == "" else "%.8g" % final_best_lr_ema,
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 pre/post and best_lr results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip large per-run batch-loss profile grids.",
    )
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_loss_curves(runs, args.output_dir / "mean_loss_curves.png")
    plot_pre_post_delta(runs, args.output_dir / "pre_post_delta.png")
    plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    plot_loss_heatmaps(runs, args.output_dir / "loss_heatmap_pre.png", "pre")
    plot_loss_heatmaps(runs, args.output_dir / "loss_heatmap_post.png", "post")
    plot_best_lr_trace(runs, args.output_dir / "best_lr_trace.png")
    plot_lr_landscape_snapshots(runs, args.output_dir / "best_lr_landscapes.png")
    if not args.skip_profiles:
        plot_batch_loss_profiles(runs, args.output_dir, "post")
    write_summary(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in runs:
        print(
            f"run={run.index} name={run.name} snapshots={len(run.snapshots)} "
            f"best_lr_logs={len(run.best_lr_logs)}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

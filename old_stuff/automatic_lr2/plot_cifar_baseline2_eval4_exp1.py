import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval4_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval4_exp1_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
SCHEDULE_RE = re.compile(
    r"^lr_mult_schedule name=(?P<scheduler>\S+) values=(?P<values>\S+)"
)
BEST_LR_RE = re.compile(
    r"^best_lr step=(?P<step>\d+) init_lr=(?P<init_lr>\S+) "
    r"best_lr=(?P<best_lr>\S+) best_lr_ema=(?P<best_lr_ema>\S+) "
    r"best_loss=(?P<best_loss>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
EVAL_RE = re.compile(
    r"^eval(?: run=(?P<run>\S+))? epoch=(?P<epoch>\d+) "
    r"val_acc=(?P<val_acc>\S+) time_seconds=(?P<time_seconds>\S+)"
)
FINAL_RE = re.compile(
    r"^eval epoch=final train_loss=(?P<train_loss>\S+) "
    r"val_loss=(?P<val_loss>\S+) train_acc=(?P<train_acc>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)
RUN_TIME_RE = re.compile(
    r"^run_time run=(?P<run>\d+) name=(?P<name>\S+) "
    r"wall_time_seconds=(?P<wall_time_seconds>\S+) "
    r"cuda_time_seconds=(?P<cuda_time_seconds>\S+)"
)
SCHED_ID_RE = re.compile(r"mult_sched_(?P<scheduler_id>\d+)_")


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    sgd_lr_mult: float
    name: str
    scheduler: str
    scheduler_id: int
    schedule: list[float] = field(default_factory=list)
    best_lr: list[float] = field(default_factory=list)
    best_loss: list[float] = field(default_factory=list)
    applied_lr: list[float] = field(default_factory=list)
    epoch_val_acc: list[float] = field(default_factory=list)
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None


def parse_float_list(text):
    return [float(value) for value in text.split(",") if value]


def scheduler_id(name):
    match = SCHED_ID_RE.search(name)
    if not match:
        return -1
    return int(match.group("scheduler_id"))


def parse_log(path):
    runs = []
    runs_by_name = {}
    current = None
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current = Run(
                    index=int(match.group("run")),
                    batch_size=int(match.group("batch_size")),
                    muon_lr=float(match.group("muon_lr")),
                    sgd_lr_mult=float(match.group("sgd_lr_mult")),
                    name=match.group("name"),
                    scheduler=match.group("best_lr_scheduler"),
                    scheduler_id=scheduler_id(match.group("best_lr_scheduler")),
                )
                runs.append(current)
                runs_by_name[current.name] = current
                continue

            match = SCHEDULE_RE.match(line)
            if match and current is not None:
                current.schedule = parse_float_list(match.group("values"))
                continue

            match = BEST_LR_RE.match(line)
            if match and current is not None:
                current.best_lr.append(float(match.group("best_lr")))
                current.best_loss.append(float(match.group("best_loss")))
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.applied_lr.append(float(match.group("muon_lr")))
                continue

            match = EVAL_RE.match(line)
            if match and current is not None:
                current.epoch_val_acc.append(float(match.group("val_acc")))
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.train_loss = float(match.group("train_loss"))
                current.val_loss = float(match.group("val_loss"))
                current.train_acc = float(match.group("train_acc"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                current.time_seconds = float(match.group("time_seconds"))
                continue

            match = RUN_TIME_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"))
                if run is not None:
                    run.wall_time_seconds = float(match.group("wall_time_seconds"))
                    run.cuda_time_seconds = float(match.group("cuda_time_seconds"))
                continue

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def runs_for_batch(runs, batch_size):
    return sorted(
        [run for run in runs if run.batch_size == batch_size],
        key=lambda run: run.scheduler_id,
    )


def top_runs(runs, batch_size, metric="train_loss", n=8, reverse=False):
    rows = [run for run in runs_for_batch(runs, batch_size) if getattr(run, metric) is not None]
    return sorted(rows, key=lambda run: getattr(run, metric), reverse=reverse)[:n]


def schedule_grid_values(runs, batch_size, metric):
    grid = np.full((11, 9), np.nan)
    for run in runs_for_batch(runs, batch_size):
        if run.scheduler_id <= 0 or getattr(run, metric) is None:
            continue
        index = run.scheduler_id - 1
        start_index = index // 9
        control_index = index % 9
        grid[start_index, control_index] = getattr(run, metric)
    return grid


def plot_metric_curves(runs, output_path):
    sizes = batch_sizes(runs)
    metrics = [
        ("train loss", "train_loss"),
        ("val loss", "val_loss"),
        ("val acc", "val_acc"),
        ("TTA val acc", "tta_val_acc"),
    ]
    fig, axes = plt.subplots(
        len(metrics), len(sizes), figsize=(5.8 * len(sizes), 12), squeeze=False
    )
    for col, batch_size in enumerate(sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        x = [run.scheduler_id for run in batch_runs]
        for row, (label, attr) in enumerate(metrics):
            ax = axes[row, col]
            y = [getattr(run, attr) for run in batch_runs]
            ax.plot(x, y, marker=".", linewidth=1.2)
            best = min(batch_runs, key=lambda run: getattr(run, attr))
            if "acc" in attr:
                best = max(batch_runs, key=lambda run: getattr(run, attr))
            ax.scatter([best.scheduler_id], [getattr(best, attr)], s=80, color="tab:red", zorder=5)
            ax.set_title(f"bs={batch_size}" if row == 0 else "")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
        axes[-1, col].set_xlabel("scheduler id")
    fig.suptitle("Final Metrics Across LR Multiplier Schedules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmaps(runs, output_path):
    sizes = batch_sizes(runs)
    metric_specs = [
        ("train_loss", "Final Train Loss", "magma_r"),
        ("tta_val_acc", "TTA Val Acc", "viridis"),
    ]
    fig, axes = plt.subplots(
        len(metric_specs), len(sizes), figsize=(5.2 * len(sizes), 9), squeeze=False
    )
    for col, batch_size in enumerate(sizes):
        for row, (metric, title, cmap) in enumerate(metric_specs):
            ax = axes[row, col]
            grid = schedule_grid_values(runs, batch_size, metric)
            image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
            for start_i in range(grid.shape[0]):
                for control_i in range(grid.shape[1]):
                    value = grid[start_i, control_i]
                    if np.isfinite(value):
                        ax.text(
                            control_i,
                            start_i,
                            f"{value:.3f}" if "loss" in metric else f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                        )
            ax.set_title(f"bs={batch_size} {title}")
            ax.set_xlabel("control index")
            ax.set_ylabel("start index")
            fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle("Bezier Schedule Grid, Excluding Constant Schedule")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_schedules(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5), squeeze=False)
    steps = np.arange(1, 11)
    for ax, batch_size in zip(axes.ravel(), sizes):
        ranked = top_runs(runs, batch_size, metric="train_loss", n=10)
        for rank, run in enumerate(ranked, start=1):
            width = 2.6 if rank == 1 else 1.2
            alpha = 1.0 if rank == 1 else 0.65
            ax.plot(
                steps,
                run.schedule,
                marker="o" if rank == 1 else None,
                linewidth=width,
                alpha=alpha,
                label=f"#{rank} s{run.scheduler_id} loss={run.train_loss:.4f}",
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("step")
        ax.set_ylabel("LR multiplier")
        ax.set_ylim(0.45, 2.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Top Schedules by Final Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all_schedules(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5), squeeze=False)
    steps = np.arange(1, 11)
    for ax, batch_size in zip(axes.ravel(), sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        values = np.array([run.train_loss for run in batch_runs], dtype=float)
        norm = plt.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        cmap = plt.get_cmap("viridis_r")
        for run in batch_runs:
            color = cmap(norm(run.train_loss))
            linewidth = 2.4 if run == min(batch_runs, key=lambda row: row.train_loss) else 0.9
            alpha = 1.0 if linewidth > 1.0 else 0.45
            ax.plot(
                steps,
                run.schedule,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        best = min(batch_runs, key=lambda row: row.train_loss)
        ax.plot(
            steps,
            best.schedule,
            color="black",
            linewidth=1.0,
            linestyle=":",
            label=f"best s{best.scheduler_id} loss={best.train_loss:.4f}",
        )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("step")
        ax.set_ylabel("LR multiplier")
        ax.set_ylim(0.45, 2.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="final train loss")
    fig.suptitle("All 100 LR Multiplier Schedules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_traces(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(2, len(sizes), figsize=(5.8 * len(sizes), 8), squeeze=False)
    steps = np.arange(1, 11)
    for col, batch_size in enumerate(sizes):
        ranked = top_runs(runs, batch_size, metric="train_loss", n=6)
        for run in ranked:
            label = f"s{run.scheduler_id} loss={run.train_loss:.4f}"
            axes[0, col].plot(steps[: len(run.best_lr)], run.best_lr, marker=".", label=label)
            axes[1, col].plot(steps[: len(run.applied_lr)], run.applied_lr, marker=".", label=label)
        axes[0, col].set_title(f"bs={batch_size} searched LR")
        axes[1, col].set_title(f"bs={batch_size} applied LR")
        for ax in axes[:, col]:
            ax.set_xlabel("step")
            ax.set_ylabel("Muon LR")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
    fig.suptitle("LR Traces for Top Schedules by Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = [
        "CIFAR Baseline Eval4 Exp1 Overfit Sweep",
        "=" * 43,
        "",
        f"Evaluated runs: {len(runs)}",
        f"Batch sizes: {', '.join(str(size) for size in batch_sizes(runs))}",
        "",
    ]
    for batch_size in batch_sizes(runs):
        ranked_loss = top_runs(runs, batch_size, metric="train_loss", n=12)
        ranked_tta = top_runs(runs, batch_size, metric="tta_val_acc", n=8, reverse=True)
        lines.extend([f"batch_size={batch_size} best train loss", "-" * 38])
        for rank, run in enumerate(ranked_loss, start=1):
            lines.append(
                f"{rank}. run={run.index} scheduler={run.scheduler} "
                f"train_loss={fmt(run.train_loss)} train_acc={fmt(run.train_acc)} "
                f"val_acc={fmt(run.val_acc)} tta_val_acc={fmt(run.tta_val_acc)} "
                f"schedule={','.join(fmt(value) for value in run.schedule)}"
            )
        lines.extend(["", f"batch_size={batch_size} best TTA val acc", "-" * 38])
        for rank, run in enumerate(ranked_tta, start=1):
            lines.append(
                f"{rank}. run={run.index} scheduler={run.scheduler} "
                f"tta_val_acc={fmt(run.tta_val_acc)} train_loss={fmt(run.train_loss)} "
                f"val_acc={fmt(run.val_acc)}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    headers = [
        "run",
        "batch_size",
        "scheduler",
        "scheduler_id",
        "muon_lr",
        "sgd_lr_mult",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
        "schedule",
        "best_lr",
        "applied_lr",
    ]
    lines = [",".join(headers)]
    for run in sorted(runs, key=lambda row: (row.batch_size, row.scheduler_id)):
        row = [
            run.index,
            run.batch_size,
            run.scheduler,
            run.scheduler_id,
            fmt(run.muon_lr),
            fmt(run.sgd_lr_mult),
            fmt(run.train_loss),
            fmt(run.val_loss),
            fmt(run.train_acc),
            fmt(run.val_acc),
            fmt(run.tta_val_acc),
            fmt(run.wall_time_seconds),
            fmt(run.cuda_time_seconds),
            ";".join(fmt(value) for value in run.schedule),
            ";".join(fmt(value) for value in run.best_lr),
            ";".join(fmt(value) for value in run.applied_lr),
        ]
        lines.append(",".join(str(value) for value in row))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval4 exp1 overfit-schedule sweep."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_curves(runs, args.output_dir / "final_metrics_by_scheduler.png")
    plot_heatmaps(runs, args.output_dir / "schedule_grid_heatmaps.png")
    plot_all_schedules(runs, args.output_dir / "all_schedule_shapes.png")
    plot_top_schedules(runs, args.output_dir / "top_schedule_shapes.png")
    plot_lr_traces(runs, args.output_dir / "top_lr_traces.png")
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for batch_size in batch_sizes(runs):
        best_loss = top_runs(runs, batch_size, metric="train_loss", n=1)[0]
        best_tta = top_runs(runs, batch_size, metric="tta_val_acc", n=1, reverse=True)[0]
        print(
            f"bs={batch_size} best_loss=s{best_loss.scheduler_id} "
            f"train_loss={best_loss.train_loss:.4f} "
            f"best_tta=s{best_tta.scheduler_id} tta={best_tta.tta_val_acc:.4f}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

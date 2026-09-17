import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval5_no_multiplier.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name(
    "cifar_baseline2_eval5_no_multiplier_plots"
)

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"muon_momentum=(?P<muon_momentum>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
BEST_LR_RE = re.compile(
    r"^best_lr step=(?P<step>\d+) init_lr=(?P<init_lr>\S+) "
    r"best_lr=(?P<best_lr>\S+) best_lr_ema=(?P<best_lr_ema>\S+) "
    r"best_loss=(?P<best_loss>\S+) losses=(?P<losses>\{.*\})"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
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
COMPLETE_RE = re.compile(
    r"^min_loss_run complete train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_momentum=(?P<muon_momentum>\S+) "
    r"multiplier=(?P<multiplier>\S+) train_loss=(?P<train_loss>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+)"
)


@dataclass
class BestLRLog:
    step: int
    init_lr: float
    best_lr: float
    best_lr_ema: float
    best_loss: float
    losses: dict[float, float]


@dataclass
class Run:
    index: int
    train_steps: int
    batch_size: int
    muon_lr: float
    muon_momentum: float
    sgd_lr_mult: float
    name: str
    best_lr_strategy: str
    best_lr_scheduler: str
    applied_lr: list[float] = field(default_factory=list)
    best_lr_logs: list[BestLRLog] = field(default_factory=list)
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None
    multiplier: float = 1.0


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def momentum_label(momentum):
    return "%.8g" % momentum


def parse_losses(text):
    return {float(key): float(value) for key, value in json.loads(text).items()}


def group_key(run):
    return run.batch_size, run.train_steps


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
                    train_steps=int(match.group("train_steps")),
                    batch_size=int(match.group("batch_size")),
                    muon_lr=float(match.group("muon_lr")),
                    muon_momentum=float(match.group("muon_momentum")),
                    sgd_lr_mult=float(match.group("sgd_lr_mult")),
                    name=match.group("name"),
                    best_lr_strategy=match.group("best_lr_strategy"),
                    best_lr_scheduler=match.group("best_lr_scheduler"),
                )
                runs.append(current)
                runs_by_name[current.name] = current
                continue

            match = BEST_LR_RE.match(line)
            if match and current is not None:
                current.best_lr_logs.append(
                    BestLRLog(
                        step=int(match.group("step")),
                        init_lr=float(match.group("init_lr")),
                        best_lr=float(match.group("best_lr")),
                        best_lr_ema=float(match.group("best_lr_ema")),
                        best_loss=float(match.group("best_loss")),
                        losses=parse_losses(match.group("losses")),
                    )
                )
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.applied_lr.append(float(match.group("muon_lr")))
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

            match = COMPLETE_RE.match(line)
            if match and current is not None:
                current.multiplier = float(match.group("multiplier"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def sorted_batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def sorted_train_steps(runs):
    return sorted({run.train_steps for run in runs})


def sorted_momentums(runs):
    return sorted({run.muon_momentum for run in runs})


def runs_for_group(runs, batch_size, train_steps):
    return sorted(
        [
            run
            for run in runs
            if run.batch_size == batch_size and run.train_steps == train_steps
        ],
        key=lambda run: run.muon_momentum,
    )


def best_run(runs, key, metric, reverse=False):
    rows = [run for run in runs if group_key(run) == key and getattr(run, metric) is not None]
    if not rows:
        return None
    return sorted(rows, key=lambda run: getattr(run, metric), reverse=reverse)[0]


def plot_applied_lr_grid(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    train_steps_list = sorted_train_steps(runs)
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(train_steps_list),
        figsize=(4.3 * len(train_steps_list), 3.2 * len(batch_sizes)),
        squeeze=False,
        sharey=True,
    )

    for row, batch_size in enumerate(batch_sizes):
        for col, train_steps in enumerate(train_steps_list):
            ax = axes[row, col]
            group_runs = runs_for_group(runs, batch_size, train_steps)
            if not group_runs:
                ax.set_visible(False)
                continue
            for run in group_runs:
                steps = np.arange(1, len(run.applied_lr) + 1)
                ax.plot(
                    steps,
                    run.applied_lr,
                    marker="o",
                    linewidth=2.0,
                    label=f"m={momentum_label(run.muon_momentum)} "
                    f"loss={run.train_loss:.4f}",
                )
            if row == 0:
                ax.set_title(f"{train_steps} train steps")
            if col == 0:
                ax.set_ylabel(f"bs={batch_size}\nMuon LR")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6.4)

    fig.suptitle("Applied Muon LR From min_loss Search, Multiplier = 1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_candidates(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    train_steps_list = sorted_train_steps(runs)
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(train_steps_list),
        figsize=(4.3 * len(train_steps_list), 3.2 * len(batch_sizes)),
        squeeze=False,
        sharey=True,
    )

    for row, batch_size in enumerate(batch_sizes):
        for col, train_steps in enumerate(train_steps_list):
            ax = axes[row, col]
            group_runs = runs_for_group(runs, batch_size, train_steps)
            if not group_runs:
                ax.set_visible(False)
                continue
            for run in group_runs:
                steps = [entry.step for entry in run.best_lr_logs]
                best_lrs = [entry.best_lr for entry in run.best_lr_logs]
                init_lrs = [entry.init_lr for entry in run.best_lr_logs]
                ax.plot(
                    steps,
                    best_lrs,
                    marker="o",
                    linewidth=2.0,
                    label=f"best m={momentum_label(run.muon_momentum)}",
                )
                ax.plot(
                    steps,
                    init_lrs,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.7,
                    label=f"init m={momentum_label(run.muon_momentum)}",
                )
            if row == 0:
                ax.set_title(f"{train_steps} train steps")
            if col == 0:
                ax.set_ylabel(f"bs={batch_size}\nLR")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=5.8)

    fig.suptitle("min_loss LR Choices and Search Initialization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics_by_batch_size(runs, output_path):
    metrics = [
        ("train_loss", "Train Loss", "lower"),
        ("val_loss", "Val Loss", "lower"),
        ("val_acc", "Val Acc", "higher"),
        ("tta_val_acc", "TTA Val Acc", "higher"),
    ]
    batch_sizes = sorted_batch_sizes(runs)
    momentums = sorted_momentums(runs)
    train_steps_list = sorted_train_steps(runs)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), squeeze=False)

    for ax, (metric, title, direction) in zip(axes.ravel(), metrics):
        for batch_size in batch_sizes:
            for momentum in momentums:
                rows = [
                    run
                    for run in runs
                    if run.batch_size == batch_size
                    and run.muon_momentum == momentum
                    and getattr(run, metric) is not None
                ]
                rows = sorted(rows, key=lambda run: run.train_steps)
                if not rows:
                    continue
                ax.plot(
                    [run.train_steps for run in rows],
                    [getattr(run, metric) for run in rows],
                    marker="o",
                    label=f"bs={batch_size}, m={momentum_label(momentum)}",
                )
        ax.set_title(f"{title} ({direction} is better)")
        ax.set_xticks(train_steps_list)
        ax.set_xlabel("train steps")
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=6.2, ncol=2)
    fig.suptitle("Metrics by Batch Size, Train Steps, and Muon Momentum")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_momentum_comparison(runs, output_path):
    groups = sorted({group_key(run) for run in runs})
    labels = [f"bs={batch_size}\nsteps={train_steps}" for batch_size, train_steps in groups]
    x = np.arange(len(groups))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(groups) * 0.7), 8.0), squeeze=False)
    ax_loss, ax_acc = axes.ravel()

    for offset, momentum in zip((-width / 2, width / 2), sorted_momentums(runs)):
        rows = []
        for key in groups:
            row = next(
                (
                    run
                    for run in runs
                    if group_key(run) == key and run.muon_momentum == momentum
                ),
                None,
            )
            rows.append(row)
        ax_loss.bar(
            x + offset,
            [row.train_loss if row is not None else np.nan for row in rows],
            width,
            label=f"m={momentum_label(momentum)}",
        )
        ax_acc.bar(
            x + offset,
            [row.tta_val_acc if row is not None else np.nan for row in rows],
            width,
            label=f"m={momentum_label(momentum)}",
        )

    ax_loss.set_title("Train Loss")
    ax_loss.set_ylabel("loss")
    ax_acc.set_title("TTA Val Accuracy")
    ax_acc.set_ylabel("accuracy")
    for ax in (ax_loss, ax_acc):
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Muon Momentum Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_train_loss_vs_tta(runs, output_path):
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    momentums = sorted_momentums(runs)
    markers = ["o", "s", "^", "D"]
    for marker, momentum in zip(markers, momentums):
        rows = [run for run in runs if run.muon_momentum == momentum]
        scatter = ax.scatter(
            [run.train_loss for run in rows],
            [run.tta_val_acc for run in rows],
            c=[run.batch_size for run in rows],
            cmap="viridis",
            marker=marker,
            s=70,
            edgecolor="black",
            linewidth=0.35,
            label=f"m={momentum_label(momentum)}",
        )
        for run in rows:
            ax.annotate(
                str(run.train_steps),
                (run.train_loss, run.tta_val_acc),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6.8,
            )
    ax.set_xlabel("train loss")
    ax.set_ylabel("TTA val acc")
    ax.set_title("Train Loss vs TTA Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(title="momentum")
    fig.colorbar(scatter, ax=ax, label="batch size")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = [
        "CIFAR Baseline Eval5 No-Multiplier min_loss Runs",
        "=" * 55,
        "",
        f"Evaluated runs: {len(runs)}",
        "Multiplier scheduler: constant",
        "Multiplier: 1",
        "",
    ]

    groups = sorted({group_key(run) for run in runs})
    for batch_size, train_steps in groups:
        rows = runs_for_group(runs, batch_size, train_steps)
        lines.extend(
            [
                f"batch_size={batch_size}, train_steps={train_steps}",
                "-" * 48,
            ]
        )
        for run in rows:
            lines.append(
                f"m={momentum_label(run.muon_momentum)} r{run.index} "
                f"train_loss={fmt(run.train_loss)} val_loss={fmt(run.val_loss)} "
                f"train_acc={fmt(run.train_acc)} val_acc={fmt(run.val_acc)} "
                f"tta_val_acc={fmt(run.tta_val_acc)} "
                f"applied_lr={','.join(fmt(value) for value in run.applied_lr)}"
            )
        best_loss = best_run(runs, (batch_size, train_steps), "train_loss")
        best_tta = best_run(runs, (batch_size, train_steps), "tta_val_acc", reverse=True)
        if best_loss is not None:
            lines.append(
                f"best_train_loss: m={momentum_label(best_loss.muon_momentum)} "
                f"loss={fmt(best_loss.train_loss)}"
            )
        if best_tta is not None:
            lines.append(
                f"best_tta_val_acc: m={momentum_label(best_tta.muon_momentum)} "
                f"tta={fmt(best_tta.tta_val_acc)}"
            )
        lines.append("")

    global_best_loss = min(
        [run for run in runs if run.train_loss is not None],
        key=lambda run: run.train_loss,
    )
    global_best_tta = max(
        [run for run in runs if run.tta_val_acc is not None],
        key=lambda run: run.tta_val_acc,
    )
    lines.extend(
        [
            "Overall Best",
            "-" * 48,
            (
                "train_loss: "
                f"bs={global_best_loss.batch_size} steps={global_best_loss.train_steps} "
                f"m={momentum_label(global_best_loss.muon_momentum)} "
                f"loss={fmt(global_best_loss.train_loss)}"
            ),
            (
                "tta_val_acc: "
                f"bs={global_best_tta.batch_size} steps={global_best_tta.train_steps} "
                f"m={momentum_label(global_best_tta.muon_momentum)} "
                f"tta={fmt(global_best_tta.tta_val_acc)}"
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    max_steps = max(run.train_steps for run in runs)
    headers = [
        "run",
        "batch_size",
        "train_steps",
        "muon_momentum",
        "muon_lr",
        "sgd_lr_mult",
        "best_lr_strategy",
        "best_lr_scheduler",
        "multiplier",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
    ]
    headers += [f"applied_lr_step{i}" for i in range(1, max_steps + 1)]
    headers += [f"best_loss_step{i}" for i in range(1, max_steps + 1)]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for run in sorted(
            runs, key=lambda row: (row.batch_size, row.train_steps, row.muon_momentum)
        ):
            row = [
                run.index,
                run.batch_size,
                run.train_steps,
                fmt(run.muon_momentum),
                fmt(run.muon_lr),
                fmt(run.sgd_lr_mult),
                run.best_lr_strategy,
                run.best_lr_scheduler,
                fmt(run.multiplier),
                fmt(run.train_loss),
                fmt(run.val_loss),
                fmt(run.train_acc),
                fmt(run.val_acc),
                fmt(run.tta_val_acc),
                fmt(run.wall_time_seconds),
                fmt(run.cuda_time_seconds),
            ]
            row += [fmt(value) for value in run.applied_lr]
            row += [""] * (max_steps - len(run.applied_lr))
            best_losses = [entry.best_loss for entry in run.best_lr_logs]
            row += [fmt(value) for value in best_losses]
            row += [""] * (max_steps - len(best_losses))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval5_no_multiplier min_loss runs."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_applied_lr_grid(runs, args.output_dir / "applied_lr_grid.png")
    plot_best_lr_candidates(runs, args.output_dir / "best_lr_choices.png")
    plot_metrics_by_batch_size(runs, args.output_dir / "metrics_by_batch_size.png")
    plot_momentum_comparison(runs, args.output_dir / "momentum_comparison.png")
    plot_train_loss_vs_tta(runs, args.output_dir / "train_loss_vs_tta.png")
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for batch_size, train_steps in sorted({group_key(run) for run in runs}):
        rows = runs_for_group(runs, batch_size, train_steps)
        print(f"bs={batch_size} steps={train_steps}")
        for run in rows:
            print(
                f"  m={momentum_label(run.muon_momentum)} "
                f"train_loss={run.train_loss:.4f} "
                f"val_acc={run.val_acc:.4f} tta_val_acc={run.tta_val_acc:.4f}"
            )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

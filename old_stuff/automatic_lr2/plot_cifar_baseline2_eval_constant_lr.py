import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval_constant_lr.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name(
    "cifar_baseline2_eval_constant_lr_plots"
)

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"muon_momentum=(?P<muon_momentum>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) search=(?P<search>\S+) "
    r"muon_lr_schedule=(?P<muon_lr_schedule>\S+) "
    r"sgd_lr_schedule=(?P<sgd_lr_schedule>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
STEP_TRAIN_LOSS_RE = re.compile(
    r"^step_train_loss step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) train_loss=(?P<train_loss>\S+)"
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


@dataclass
class Run:
    index: int
    train_steps: int
    batch_size: int
    muon_lr: float
    muon_momentum: float
    sgd_lr_mult: float
    name: str
    muon_lr_schedule: str
    sgd_lr_schedule: str
    applied_lr: list[float] = field(default_factory=list)
    step_train_losses: list[float] = field(default_factory=list)
    eval_train_loss: float | None = None
    train_acc: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None

    @property
    def final_step_train_loss(self):
        if not self.step_train_losses:
            return None
        return self.step_train_losses[-1]


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


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
                    muon_lr_schedule=match.group("muon_lr_schedule"),
                    sgd_lr_schedule=match.group("sgd_lr_schedule"),
                )
                runs.append(current)
                runs_by_name[current.name] = current
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.applied_lr.append(float(match.group("muon_lr")))
                continue

            match = STEP_TRAIN_LOSS_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.step_train_losses.append(float(match.group("train_loss")))
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.eval_train_loss = float(match.group("train_loss"))
                current.train_acc = float(match.group("train_acc"))
                continue

            match = RUN_TIME_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"))
                if run is not None:
                    run.wall_time_seconds = float(match.group("wall_time_seconds"))
                    run.cuda_time_seconds = float(match.group("cuda_time_seconds"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def sorted_batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def sorted_lrs(runs):
    return sorted({run.muon_lr for run in runs}, reverse=True)


def final_loss(run):
    return run.final_step_train_loss if run.final_step_train_loss is not None else run.eval_train_loss


def runs_for_batch_size(runs, batch_size):
    return sorted(
        [run for run in runs if run.batch_size == batch_size],
        key=lambda run: run.muon_lr,
        reverse=True,
    )


def plot_final_loss_by_lr(runs, output_path):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for batch_size in sorted_batch_sizes(runs):
        rows = sorted(runs_for_batch_size(runs, batch_size), key=lambda run: run.muon_lr)
        ax.plot(
            [run.muon_lr for run in rows],
            [final_loss(run) for run in rows],
            marker="o",
            linewidth=2.0,
            label=f"bs={batch_size}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("constant Muon LR")
    ax.set_ylabel("final step train loss")
    ax.set_title("Constant LR Sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_loss_heatmap(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    lrs = sorted_lrs(runs)
    by_key = {(run.batch_size, run.muon_lr): run for run in runs}
    matrix = np.full((len(batch_sizes), len(lrs)), np.nan)
    for row, batch_size in enumerate(batch_sizes):
        for col, lr in enumerate(lrs):
            run = by_key.get((batch_size, lr))
            if run is not None:
                matrix[row, col] = final_loss(run)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(lrs)), [fmt(lr) for lr in lrs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(batch_sizes)), [str(batch_size) for batch_size in batch_sizes])
    ax.set_xlabel("constant Muon LR")
    ax.set_ylabel("batch size")
    ax.set_title("Final Step Train Loss")
    for row in range(len(batch_sizes)):
        for col in range(len(lrs)):
            value = matrix[row, col]
            if math.isfinite(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="train loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_train_losses(runs, output_path, max_step=None, title=None):
    batch_sizes = sorted_batch_sizes(runs)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), squeeze=False, sharex=True)
    cmap = plt.get_cmap("turbo")
    lrs = sorted_lrs(runs)
    color_by_lr = {lr: cmap(index / max(1, len(lrs) - 1)) for index, lr in enumerate(lrs)}

    for ax, batch_size in zip(axes.ravel(), batch_sizes):
        for run in runs_for_batch_size(runs, batch_size):
            if not run.step_train_losses:
                continue
            losses = run.step_train_losses
            if max_step is not None:
                losses = losses[:max_step]
            steps = np.arange(1, len(losses) + 1)
            ax.plot(
                steps,
                losses,
                color=color_by_lr[run.muon_lr],
                linewidth=1.5,
                alpha=0.9,
                label=f"lr={fmt(run.muon_lr)} final={fmt(final_loss(run))}",
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("training step")
        ax.set_ylabel("train loss")
        if max_step is not None:
            ax.set_xlim(1, max_step)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.2, ncol=2)

    fig.suptitle(title or "Per-Step Train Loss by Constant LR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = [
        "CIFAR Baseline2 Constant LR Sweep",
        "=" * 38,
        "",
        f"Runs: {len(runs)}",
        f"Batch sizes: {', '.join(str(value) for value in sorted_batch_sizes(runs))}",
        f"Muon LRs: {', '.join(fmt(value) for value in sorted_lrs(runs))}",
        "",
    ]

    for batch_size in sorted_batch_sizes(runs):
        rows = runs_for_batch_size(runs, batch_size)
        best = min(rows, key=final_loss)
        lines.extend(
            [
                f"batch_size={batch_size}",
                "-" * 40,
                (
                    f"best_lr={fmt(best.muon_lr)} "
                    f"final_step_train_loss={fmt(final_loss(best))} "
                    f"eval_train_loss={fmt(best.eval_train_loss)}"
                ),
            ]
        )
        for run in rows:
            lines.append(
                f"lr={fmt(run.muon_lr)} run={run.index} "
                f"final_step_train_loss={fmt(final_loss(run))} "
                f"eval_train_loss={fmt(run.eval_train_loss)} "
                f"steps_logged={len(run.step_train_losses)}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    max_steps = max(len(run.step_train_losses) for run in runs)
    headers = [
        "run",
        "batch_size",
        "muon_lr",
        "muon_momentum",
        "sgd_lr_mult",
        "final_step_train_loss",
        "eval_train_loss",
        "train_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
    ]
    headers += [f"step_train_loss_{step}" for step in range(1, max_steps + 1)]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for run in sorted(runs, key=lambda row: (row.batch_size, -row.muon_lr)):
            row = [
                run.index,
                run.batch_size,
                fmt(run.muon_lr),
                fmt(run.muon_momentum),
                fmt(run.sgd_lr_mult),
                fmt(final_loss(run)),
                fmt(run.eval_train_loss),
                fmt(run.train_acc),
                fmt(run.wall_time_seconds),
                fmt(run.cuda_time_seconds),
            ]
            row += [fmt(value) for value in run.step_train_losses]
            row += [""] * (max_steps - len(run.step_train_losses))
            writer.writerow(row)


def main(default_log=DEFAULT_LOG, default_output_dir=DEFAULT_OUTPUT_DIR):
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval_constant_lr sweep results."
    )
    parser.add_argument("--log", type=Path, default=default_log)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_loss_by_lr(runs, args.output_dir / "final_train_loss_by_lr.png")
    plot_final_loss_heatmap(runs, args.output_dir / "final_train_loss_heatmap.png")
    plot_step_train_losses(runs, args.output_dir / "step_train_losses.png")
    plot_step_train_losses(
        runs,
        args.output_dir / "step_train_losses_first20.png",
        max_step=20,
        title="Per-Step Train Loss by Constant LR, First 20 Steps",
    )
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for batch_size in sorted_batch_sizes(runs):
        best = min(runs_for_batch_size(runs, batch_size), key=final_loss)
        print(
            f"bs={batch_size} best_lr={fmt(best.muon_lr)} "
            f"final_step_train_loss={fmt(final_loss(best))}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

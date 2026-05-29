import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_n_search.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_n_search_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
STEP_TRAIN_LOSS_RE = re.compile(
    r"^step_train_loss step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) train_loss=(?P<train_loss>\S+)"
)
INTERVAL_SELECTED_RE = re.compile(
    r"^n_search interval_selected run=(?P<run>\d+) interval=(?P<interval>\d+) "
    r"steps=(?P<start_step>\d+)-(?P<end_step>\d+) N=(?P<N>\d+) "
    r"initial_lr=(?P<initial_lr>\S+) selected_k=(?P<selected_k>-?\d+) "
    r"selected_lr=(?P<selected_lr>\S+) next_initial_lr=(?P<next_initial_lr>\S+) "
    r"ema=(?P<ema>\S+) train_loss=(?P<train_loss>\S+) "
    r"evaluated_candidates=(?P<evaluated_candidates>\d+)"
)
EVAL_RE = re.compile(
    r"^eval(?: run=(?P<run>\d+))? epoch=(?P<epoch>\d+) "
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
NAME_N_RE = re.compile(r"_N(?P<N>\d+)_ema")


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    sgd_lr_mult: float
    name: str
    strategy: str | None
    scheduler: str
    applied_lrs: list[tuple[int, int, float]] = field(default_factory=list)
    step_train_losses: list[tuple[int, int, float]] = field(default_factory=list)
    interval_selections: list[dict] = field(default_factory=list)
    eval_logs: list[tuple[int, float, float]] = field(default_factory=list)
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None

    @property
    def kind(self):
        return "baseline" if self.strategy is None else self.strategy

    @property
    def interval_n(self):
        if self.strategy is None:
            return None
        match = NAME_N_RE.search(self.name)
        if match:
            return int(match.group("N"))
        if self.interval_selections:
            return self.interval_selections[0]["N"]
        return None

    @property
    def label(self):
        if self.strategy is None:
            return "baseline"
        if self.interval_n is not None:
            return f"n_search N={self.interval_n}"
        return "n_search"


def parse_strategy(value):
    return None if value == "None" else value


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def fmt3(value):
    if value is None:
        return "NA"
    return f"{value:.3g}"


def parse_log(path):
    runs = []
    by_name = {}
    by_index = {}
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
                    strategy=parse_strategy(match.group("best_lr_strategy")),
                    scheduler=match.group("best_lr_scheduler"),
                )
                runs.append(current)
                by_name[current.name] = current
                by_index[current.index] = current
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = by_name.get(match.group("name"), current)
                if run is not None:
                    run.applied_lrs.append(
                        (
                            int(match.group("step")),
                            int(match.group("total_steps")),
                            float(match.group("muon_lr")),
                        )
                    )
                continue

            match = STEP_TRAIN_LOSS_RE.match(line)
            if match:
                run = by_name.get(match.group("name"), current)
                if run is not None:
                    run.step_train_losses.append(
                        (
                            int(match.group("step")),
                            int(match.group("total_steps")),
                            float(match.group("train_loss")),
                        )
                    )
                continue

            match = INTERVAL_SELECTED_RE.match(line)
            if match:
                run = by_index.get(int(match.group("run")))
                if run is not None:
                    run.interval_selections.append(
                        {
                            "interval": int(match.group("interval")),
                            "start_step": int(match.group("start_step")),
                            "end_step": int(match.group("end_step")),
                            "N": int(match.group("N")),
                            "initial_lr": float(match.group("initial_lr")),
                            "selected_k": int(match.group("selected_k")),
                            "selected_lr": float(match.group("selected_lr")),
                            "next_initial_lr": float(match.group("next_initial_lr")),
                            "ema": float(match.group("ema")),
                            "train_loss": float(match.group("train_loss")),
                            "evaluated_candidates": int(
                                match.group("evaluated_candidates")
                            ),
                        }
                    )
                continue

            match = EVAL_RE.match(line)
            if match and current is not None:
                current.eval_logs.append(
                    (
                        int(match.group("epoch")),
                        float(match.group("val_acc")),
                        float(match.group("time_seconds")),
                    )
                )
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.train_loss = float(match.group("train_loss"))
                current.val_loss = float(match.group("val_loss"))
                current.train_acc = float(match.group("train_acc"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                continue

            match = RUN_TIME_RE.match(line)
            if match:
                run = by_name.get(match.group("name"))
                if run is not None:
                    run.wall_time_seconds = float(match.group("wall_time_seconds"))
                    run.cuda_time_seconds = float(match.group("cuda_time_seconds"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def sorted_batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def sorted_runs(runs):
    order = {"baseline": 0, "n_search": 1}
    return sorted(
        runs,
        key=lambda run: (
            run.batch_size,
            order.get(run.kind, 99),
            run.interval_n if run.interval_n is not None else -1,
        ),
    )


def runs_for_batch(runs, batch_size):
    return sorted_runs([run for run in runs if run.batch_size == batch_size])


def plot_final_metrics(runs, output_path):
    metrics = [
        ("train loss", lambda run: run.train_loss),
        ("val loss", lambda run: run.val_loss),
        ("train acc", lambda run: run.train_acc),
        ("val acc", lambda run: run.val_acc),
        ("TTA val acc", lambda run: run.tta_val_acc),
        ("wall seconds", lambda run: run.wall_time_seconds),
    ]
    batch_sizes = sorted_batch_sizes(runs)
    colors = {
        "baseline": "tab:gray",
        "n_search N=1": "tab:blue",
        "n_search N=5": "tab:orange",
        "n_search N=10": "tab:green",
    }
    fig, axes = plt.subplots(
        len(metrics),
        len(batch_sizes),
        figsize=(4.3 * len(batch_sizes), 2.7 * len(metrics)),
        squeeze=False,
    )
    for col, batch_size in enumerate(batch_sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        labels = [run.label for run in batch_runs]
        x = np.arange(len(batch_runs))
        bar_colors = [colors.get(run.label, "tab:purple") for run in batch_runs]
        for row, (ylabel, getter) in enumerate(metrics):
            ax = axes[row][col]
            ax.bar(x, [getter(run) if getter(run) is not None else np.nan for run in batch_runs], color=bar_colors)
            ax.set_title(f"bs={batch_size}" if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Eval2 Baseline vs N-Search Final Metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_curves(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    fig, axes = plt.subplots(
        1, len(batch_sizes), figsize=(4.8 * len(batch_sizes), 4.0), squeeze=False
    )
    for ax, batch_size in zip(axes.ravel(), batch_sizes):
        for run in runs_for_batch(runs, batch_size):
            if not run.eval_logs:
                continue
            epochs = [row[0] for row in run.eval_logs]
            vals = [row[1] for row in run.eval_logs]
            ax.plot(epochs, vals, marker="o", linewidth=1.4, label=run.label)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Validation Accuracy During Training", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_schedules(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    fig, axes = plt.subplots(
        1, len(batch_sizes), figsize=(4.8 * len(batch_sizes), 4.2), squeeze=False
    )
    for ax, batch_size in zip(axes.ravel(), batch_sizes):
        all_lrs = []
        for run in runs_for_batch(runs, batch_size):
            if not run.applied_lrs:
                continue
            steps = [row[0] for row in run.applied_lrs]
            total = max(row[1] for row in run.applied_lrs)
            progress = [step / total for step in steps]
            lrs = [row[2] for row in run.applied_lrs]
            all_lrs.extend(lrs)
            ax.plot(progress, lrs, linewidth=1.3, label=run.label)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("applied Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Applied Muon LR Schedules", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_n_search_step_train_losses(runs, output_path):
    n_runs = [run for run in runs if run.step_train_losses]
    if not n_runs:
        return
    batch_sizes = sorted_batch_sizes(n_runs)
    fig, axes = plt.subplots(
        1, len(batch_sizes), figsize=(4.8 * len(batch_sizes), 4.2), squeeze=False
    )
    for ax, batch_size in zip(axes.ravel(), batch_sizes):
        for run in runs_for_batch(n_runs, batch_size):
            steps = [row[0] for row in run.step_train_losses]
            total = max(row[1] for row in run.step_train_losses)
            progress = [step / total for step in steps]
            losses = [row[2] for row in run.step_train_losses]
            ax.plot(progress, losses, linewidth=1.4, label=run.label)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("n_search step train loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Committed N-Search Step Train Losses", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_n_search_selected_lrs(runs, output_path):
    n_runs = [run for run in runs if run.interval_selections]
    if not n_runs:
        return
    batch_sizes = sorted_batch_sizes(n_runs)
    fig, axes = plt.subplots(
        1, len(batch_sizes), figsize=(4.8 * len(batch_sizes), 4.2), squeeze=False
    )
    for ax, batch_size in zip(axes.ravel(), batch_sizes):
        for run in runs_for_batch(n_runs, batch_size):
            xs = [row["end_step"] for row in run.interval_selections]
            total = max((row[1] for row in run.applied_lrs), default=max(xs))
            progress = [step / total for step in xs]
            lrs = [row["selected_lr"] for row in run.interval_selections]
            ax.step(progress, lrs, where="post", linewidth=1.5, label=run.label)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("selected interval LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("N-Search Selected Interval LRs", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    rows = [
        (
            "bs",
            "kind",
            "N",
            "muon_lr",
            "sgd_mult",
            "train_loss",
            "val_acc",
            "tta",
            "wall_s",
        )
    ]
    for run in sorted_runs(runs):
        rows.append(
            (
                str(run.batch_size),
                run.label,
                "NA" if run.interval_n is None else str(run.interval_n),
                fmt(run.muon_lr),
                fmt(run.sgd_lr_mult),
                fmt3(run.train_loss),
                fmt3(run.val_acc),
                fmt3(run.tta_val_acc),
                fmt3(run.wall_time_seconds),
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = [
        "  ".join(value.rjust(widths[col]) for col, value in enumerate(row))
        for row in rows
    ]
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    fieldnames = [
        "run",
        "name",
        "batch_size",
        "kind",
        "N",
        "muon_lr",
        "sgd_lr_mult",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
        "applied_lr_steps",
        "step_train_loss_steps",
        "n_search_intervals",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted_runs(runs):
            writer.writerow(
                {
                    "run": run.index,
                    "name": run.name,
                    "batch_size": run.batch_size,
                    "kind": run.label,
                    "N": run.interval_n,
                    "muon_lr": run.muon_lr,
                    "sgd_lr_mult": run.sgd_lr_mult,
                    "train_loss": run.train_loss,
                    "val_loss": run.val_loss,
                    "train_acc": run.train_acc,
                    "val_acc": run.val_acc,
                    "tta_val_acc": run.tta_val_acc,
                    "wall_time_seconds": run.wall_time_seconds,
                    "cuda_time_seconds": run.cuda_time_seconds,
                    "applied_lr_steps": len(run.applied_lrs),
                    "step_train_loss_steps": len(run.step_train_losses),
                    "n_search_intervals": len(run.interval_selections),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 baseline vs n_search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    plot_lr_schedules(runs, args.output_dir / "applied_lr_schedules.png")
    plot_n_search_step_train_losses(runs, args.output_dir / "n_search_step_train_losses.png")
    plot_n_search_selected_lrs(runs, args.output_dir / "n_search_selected_lrs.png")
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    for run in sorted_runs(runs):
        print(
            f"run={run.index} bs={run.batch_size} kind={run.label} "
            f"train_loss={fmt(run.train_loss)} val_acc={fmt(run.val_acc)} "
            f"tta={fmt(run.tta_val_acc)}"
        )


if __name__ == "__main__":
    main()

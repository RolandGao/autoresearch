import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp3.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp3_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
BEST_LR_RE = re.compile(
    r"^best_lr step=(?P<step>\d+) "
    r"init_lr=(?P<init_lr>\S+) best_lr=(?P<best_lr>\S+) "
    r"best_lr_ema=(?P<best_lr_ema>\S+) best_loss=(?P<best_loss>\S+) "
    r"losses=(?P<losses>\{.*\})$"
)
EVAL_RE = re.compile(
    r"^eval(?: run=(?P<run>\S+))? epoch=(?P<epoch>\d+) "
    r"val_acc=(?P<val_acc>\S+) time_seconds=(?P<time_seconds>\S+)"
)
FINAL_RE = re.compile(
    r"^eval epoch=final 25batch_train_loss=(?P<train25_loss>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)
FINAL_FULL_RE = re.compile(
    r"^eval epoch=final train_loss=(?P<train_loss>\S+) "
    r"val_loss=(?P<val_loss>\S+) train_acc=(?P<train_acc>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)


@dataclass
class AppliedLR:
    step: int
    total_steps: int
    lr: float


@dataclass
class BestLRLog:
    step: int
    init_lr: float
    searched_lr: float
    best_lr_ema: float
    best_loss: float
    losses_by_lr: dict[float, float]


@dataclass
class EvalLog:
    epoch: int
    val_acc: float
    time_seconds: float


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    name: str
    best_lr_strategy: str | None
    best_lr_linear_decay: bool
    applied_lrs: list[AppliedLR] = field(default_factory=list)
    best_lr_logs: list[BestLRLog] = field(default_factory=list)
    eval_logs: list[EvalLog] = field(default_factory=list)
    train25_loss: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None

    @property
    def is_best_lr(self):
        return self.best_lr_strategy is not None

    @property
    def version_label(self):
        if not self.is_best_lr:
            return "fixed"
        parts = self.name.split("_bs", 1)[0].split("_")
        return "_".join(parts[:4]) if len(parts) >= 4 else self.best_lr_strategy

    @property
    def label(self):
        return f"{self.name}"


def parse_bool(value):
    return str(value).lower() == "true"


def parse_strategy(value):
    return None if value == "None" else value


def parse_best_lr(match):
    losses_by_lr = {
        float(lr): float(loss)
        for lr, loss in json.loads(match.group("losses")).items()
    }
    return BestLRLog(
        step=int(match.group("step")),
        init_lr=float(match.group("init_lr")),
        searched_lr=float(match.group("best_lr")),
        best_lr_ema=float(match.group("best_lr_ema")),
        best_loss=float(match.group("best_loss")),
        losses_by_lr=losses_by_lr,
    )


def parse_log(path):
    runs = []
    current = None
    by_name = {}
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
                    best_lr_strategy=parse_strategy(match.group("best_lr_strategy")),
                    best_lr_linear_decay=parse_bool(
                        match.group("best_lr_linear_decay")
                    ),
                )
                runs.append(current)
                by_name[current.name] = current
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = by_name.get(match.group("name"), current)
                if run is None:
                    raise ValueError(
                        f"Found applied_lr before run header on line {line_number}"
                    )
                run.applied_lrs.append(
                    AppliedLR(
                        step=int(match.group("step")),
                        total_steps=int(match.group("total_steps")),
                        lr=float(match.group("muon_lr")),
                    )
                )
                continue

            match = BEST_LR_RE.match(line)
            if match:
                if current is None:
                    raise ValueError(
                        f"Found best_lr before run header on line {line_number}"
                    )
                current.best_lr_logs.append(parse_best_lr(match))
                continue

            match = EVAL_RE.match(line)
            if match and current is not None:
                current.eval_logs.append(
                    EvalLog(
                        epoch=int(match.group("epoch")),
                        val_acc=float(match.group("val_acc")),
                        time_seconds=float(match.group("time_seconds")),
                    )
                )
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.train25_loss = float(match.group("train25_loss"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                current.time_seconds = float(match.group("time_seconds"))
                continue

            match = FINAL_FULL_RE.match(line)
            if match and current is not None:
                current.train_loss = float(match.group("train_loss"))
                current.val_loss = float(match.group("val_loss"))
                current.train_acc = float(match.group("train_acc"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                current.time_seconds = float(match.group("time_seconds"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def finite_metric(value):
    return value is not None and math.isfinite(value)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def run_order(runs):
    return sorted(runs, key=lambda run: (run.batch_size, run.index))


def applied_lr_by_step(run):
    return {row.step: row.lr for row in run.applied_lrs}


def best_runs(runs):
    return [run for run in runs if run.best_lr_logs]


def final_train_loss(run):
    return run.train_loss if run.train_loss is not None else run.train25_loss


def plot_final_metrics(runs, output_path):
    runs = run_order(runs)
    labels = [run.name.replace("_", "\n", 2) for run in runs]
    x = np.arange(len(runs))
    val_acc = [run.val_acc if finite_metric(run.val_acc) else np.nan for run in runs]
    tta_val_acc = [
        run.tta_val_acc if finite_metric(run.tta_val_acc) else np.nan for run in runs
    ]
    train_loss = [
        final_train_loss(run) if finite_metric(final_train_loss(run)) else np.nan
        for run in runs
    ]

    fig, (acc_ax, loss_ax) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    width = 0.38
    acc_ax.bar(x - width / 2, val_acc, width=width, label="val acc")
    acc_ax.bar(x + width / 2, tta_val_acc, width=width, label="TTA val acc")
    acc_ax.set_ylabel("accuracy")
    acc_ax.grid(True, axis="y", alpha=0.25)
    acc_ax.legend()

    loss_ax.bar(x, train_loss, width=0.6, color="tab:green")
    loss_ax.set_ylabel("train loss")
    loss_ax.set_xticks(x, labels, rotation=35, ha="right")
    loss_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Final Evaluation Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_curves(runs, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, batch_size in zip(axes, sorted({run.batch_size for run in runs})):
        for run in runs:
            if run.batch_size != batch_size or not run.eval_logs:
                continue
            epochs = [row.epoch for row in run.eval_logs]
            accs = [row.val_acc for row in run.eval_logs]
            ax.plot(epochs, accs, marker="o", markersize=3, linewidth=1.5, label=run.name)
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val acc")
    fig.suptitle("Validation Accuracy During Training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_applied_lr_traces(runs, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
    for ax, batch_size in zip(axes, sorted({run.batch_size for run in runs})):
        for run in runs:
            if run.batch_size != batch_size or not run.applied_lrs:
                continue
            steps = np.array([row.step for row in run.applied_lrs], dtype=np.float64)
            total_steps = max(row.total_steps for row in run.applied_lrs)
            progress = steps / total_steps
            lrs = np.array([row.lr for row in run.applied_lrs], dtype=np.float64)
            ax.plot(progress, lrs, linewidth=1.5, label=run.name)
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("applied Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Applied Muon LR Every Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_traces(runs, output_path):
    runs = best_runs(runs)
    if not runs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
    axes = axes.ravel()

    for ax, batch_size in zip(axes[:2], sorted({run.batch_size for run in runs})):
        for run in runs:
            if run.batch_size != batch_size:
                continue
            applied_by_step = applied_lr_by_step(run)
            total_steps = max((row.total_steps for row in run.applied_lrs), default=1)
            steps = np.array([row.step for row in run.best_lr_logs], dtype=np.float64)
            progress = steps / total_steps
            searched = np.array([row.searched_lr for row in run.best_lr_logs])
            ema = np.array([row.best_lr_ema for row in run.best_lr_logs])
            applied = np.array([applied_by_step.get(row.step, np.nan) for row in run.best_lr_logs])
            ax.plot(progress, searched, linewidth=1.5, label=f"{run.version_label} searched")
            ax.plot(progress, applied, linestyle="--", linewidth=1.2, label=f"{run.version_label} applied")
            ax.plot(progress, ema, linestyle=":", linewidth=1.0, label=f"{run.version_label} ema")
        ax.set_title(f"LR trace, batch_size={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=7)

    for ax, batch_size in zip(axes[2:], sorted({run.batch_size for run in runs})):
        for run in runs:
            if run.batch_size != batch_size:
                continue
            total_steps = max((row.total_steps for row in run.applied_lrs), default=1)
            progress = np.array([row.step / total_steps for row in run.best_lr_logs])
            losses = np.array([row.best_loss for row in run.best_lr_logs])
            ax.plot(progress, losses, linewidth=1.5, label=run.version_label)
        ax.set_title(f"Selected current-batch loss, batch_size={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Best-LR Search Traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_landscape_snapshots(runs, output_path, max_panels_per_run=4):
    rows = []
    for run in best_runs(runs):
        if len(run.best_lr_logs) <= max_panels_per_run:
            selected = run.best_lr_logs
        else:
            indices = np.linspace(
                0, len(run.best_lr_logs) - 1, max_panels_per_run
            ).round().astype(int)
            selected = [run.best_lr_logs[index] for index in indices]
        rows.extend((run, row) for row in selected)
    if not rows:
        return

    ncols = 4
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.8 * nrows), squeeze=False)

    for ax, (run, row) in zip(axes.ravel(), rows):
        lr_losses = sorted(row.losses_by_lr.items())
        lrs = np.array([lr for lr, _ in lr_losses], dtype=np.float64)
        losses = np.array([loss for _, loss in lr_losses], dtype=np.float64)
        ax.plot(lrs, losses, marker="o", linewidth=1.3)
        ax.axvline(row.searched_lr, color="tab:red", linewidth=1.0, alpha=0.85)
        ax.set_xscale("log")
        ax.set_title(f"{run.name}\nstep {row.step}", fontsize=8)
        ax.grid(True, alpha=0.25, which="both")

    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")

    fig.suptitle("Best-LR Probe Loss Landscapes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = [
        "run,name,batch_size,muon_lr,best_lr_strategy,best_lr_linear_decay,"
        "train_loss,train25_loss,val_loss,train_acc,val_acc,tta_val_acc,"
        "time_seconds,steps,final_applied_lr,"
        "final_searched_lr,final_best_lr_ema"
    ]
    for run in runs:
        final_applied_lr = run.applied_lrs[-1].lr if run.applied_lrs else ""
        final_searched_lr = run.best_lr_logs[-1].searched_lr if run.best_lr_logs else ""
        final_best_lr_ema = run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else ""
        fields = [
            run.index,
            run.name,
            run.batch_size,
            "%.8g" % run.muon_lr,
            run.best_lr_strategy or "",
            run.best_lr_linear_decay,
            "" if run.train_loss is None else "%.8g" % run.train_loss,
            "" if run.train25_loss is None else "%.8g" % run.train25_loss,
            "" if run.val_loss is None else "%.8g" % run.val_loss,
            "" if run.train_acc is None else "%.8g" % run.train_acc,
            "" if run.val_acc is None else "%.8g" % run.val_acc,
            "" if run.tta_val_acc is None else "%.8g" % run.tta_val_acc,
            "" if run.time_seconds is None else "%.8g" % run.time_seconds,
            len(run.applied_lrs),
            "" if final_applied_lr == "" else "%.8g" % final_applied_lr,
            "" if final_searched_lr == "" else "%.8g" % final_searched_lr,
            "" if final_best_lr_ema == "" else "%.8g" % final_best_lr_ema,
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 exp3 LR-search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    plot_applied_lr_traces(runs, args.output_dir / "applied_lr_traces.png")
    plot_best_lr_traces(runs, args.output_dir / "best_lr_traces.png")
    plot_lr_landscape_snapshots(runs, args.output_dir / "best_lr_landscapes.png")
    write_summary(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in runs:
        print(
            f"run={run.index} name={run.name} steps={len(run.applied_lrs)} "
            f"best_lr_logs={len(run.best_lr_logs)} tta={run.tta_val_acc}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

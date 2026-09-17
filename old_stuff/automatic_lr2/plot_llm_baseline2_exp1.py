import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("llm_baseline2_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("llm_baseline2_exp1_plots")

RUN_RE = re.compile(
    r"llm_baseline2 run=(?P<run>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
BEST_LR_RE = re.compile(
    r"best_lr step=(?P<step>\d+) init_lr=(?P<init_lr>\S+) "
    r"best_lr=(?P<best_lr>\S+) best_lr_ema=(?P<best_lr_ema>\S+) "
    r"best_loss=(?P<best_loss>\S+) losses=(?P<losses>\{.*?\})(?=\s|$)"
)
PROGRESS_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\((?P<pct_done>\S+)%\)\s+\|\s+"
    r"loss:\s+(?P<loss>\S+)\s+\|\s+lrm:\s+(?P<lr_multiplier>\S+)\s+\|\s+"
    r"dt:\s+(?P<dt_ms>\S+)ms\s+\|\s+tok/sec:\s+(?P<tok_per_sec>[\d,]+)\s+\|\s+"
    r"mfu:\s+(?P<mfu_percent>\S+)%\s+\|\s+epoch:\s+(?P<epoch>\d+)\s+\|\s+"
    r"remaining_steps:\s+(?P<remaining_steps>\d+)"
)
FINAL_RE = re.compile(
    r"eval epoch=final train_loss=(?P<train_loss>\S+) "
    r"val_loss=(?P<val_loss>\S+) val_bpb=(?P<val_bpb>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)
RUN_TIME_RE = re.compile(
    r"run_time run=(?P<run>\d+) name=(?P<name>\S+) "
    r"wall_time_seconds=(?P<wall_time_seconds>\S+) "
    r"cuda_time_seconds=(?P<cuda_time_seconds>\S+)"
)
TRAINING_BATCH_LOSSES_RE = re.compile(
    r"training_batch_losses step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"update=(?P<update>\S+) losses=(?P<losses>\[.*?\])"
)
SUMMARY_RE = re.compile(r"^(?P<key>[a-zA-Z_]+):\s+(?P<value>\S+)", re.MULTILINE)


@dataclass
class AppliedLR:
    step: int
    total_steps: int
    lr: float


@dataclass
class BestLR:
    step: int
    init_lr: float
    searched_lr: float
    best_lr_ema: float
    best_loss: float
    losses_by_lr: dict[float, float]


@dataclass
class Progress:
    step: int
    pct_done: float
    loss: float
    lr_multiplier: float
    dt_ms: float
    tok_per_sec: float
    mfu_percent: float
    epoch: int
    remaining_steps: int


@dataclass
class Run:
    index: int
    muon_lr: float
    name: str
    best_lr_strategy: str | None
    best_lr_scheduler: str
    best_lr_linear_decay: bool
    applied_lrs: list[AppliedLR] = field(default_factory=list)
    best_lr_logs: list[BestLR] = field(default_factory=list)
    progress_logs: list[Progress] = field(default_factory=list)
    final_post_losses: list[float] = field(default_factory=list)
    train_loss: float | None = None
    val_loss: float | None = None
    val_bpb: float | None = None
    time_seconds: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None
    summary: dict[str, float] = field(default_factory=dict)


def parse_bool(value):
    return value == "True"


def parse_strategy(value):
    return None if value == "None" else value


def parse_losses_map(text):
    raw = json.loads(text)
    return {float(key): float(value) for key, value in raw.items()}


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    run_matches = list(RUN_RE.finditer(text))
    if not run_matches:
        raise ValueError(f"No runs found in {path}")

    runs = []
    for index, match in enumerate(run_matches):
        block_start = match.start()
        block_end = (
            run_matches[index + 1].start()
            if index + 1 < len(run_matches)
            else len(text)
        )
        block = text[block_start:block_end]
        run = Run(
            index=int(match.group("run")),
            muon_lr=float(match.group("muon_lr")),
            name=match.group("name"),
            best_lr_strategy=parse_strategy(match.group("best_lr_strategy")),
            best_lr_scheduler=match.group("best_lr_scheduler"),
            best_lr_linear_decay=parse_bool(match.group("best_lr_linear_decay")),
        )

        for lr_match in APPLIED_LR_RE.finditer(block):
            run.applied_lrs.append(
                AppliedLR(
                    step=int(lr_match.group("step")),
                    total_steps=int(lr_match.group("total_steps")),
                    lr=float(lr_match.group("muon_lr")),
                )
            )

        for best_match in BEST_LR_RE.finditer(block):
            run.best_lr_logs.append(
                BestLR(
                    step=int(best_match.group("step")),
                    init_lr=float(best_match.group("init_lr")),
                    searched_lr=float(best_match.group("best_lr")),
                    best_lr_ema=float(best_match.group("best_lr_ema")),
                    best_loss=float(best_match.group("best_loss")),
                    losses_by_lr=parse_losses_map(best_match.group("losses")),
                )
            )

        for progress_match in PROGRESS_RE.finditer(block.replace("\r", "\n")):
            run.progress_logs.append(
                Progress(
                    step=int(progress_match.group("step")) + 1,
                    pct_done=float(progress_match.group("pct_done")),
                    loss=float(progress_match.group("loss")),
                    lr_multiplier=float(progress_match.group("lr_multiplier")),
                    dt_ms=float(progress_match.group("dt_ms")),
                    tok_per_sec=float(progress_match.group("tok_per_sec").replace(",", "")),
                    mfu_percent=float(progress_match.group("mfu_percent")),
                    epoch=int(progress_match.group("epoch")),
                    remaining_steps=int(progress_match.group("remaining_steps")),
                )
            )

        final_match = FINAL_RE.search(block)
        if final_match:
            run.train_loss = float(final_match.group("train_loss"))
            run.val_loss = float(final_match.group("val_loss"))
            run.val_bpb = float(final_match.group("val_bpb"))
            run.time_seconds = float(final_match.group("time_seconds"))

        time_match = RUN_TIME_RE.search(block)
        if time_match:
            run.wall_time_seconds = float(time_match.group("wall_time_seconds"))
            run.cuda_time_seconds = float(time_match.group("cuda_time_seconds"))

        losses_match = TRAINING_BATCH_LOSSES_RE.search(block)
        if losses_match and losses_match.group("update") == "post":
            run.final_post_losses = [
                float(value) for value in json.loads(losses_match.group("losses"))
            ]

        for summary_match in SUMMARY_RE.finditer(block):
            try:
                run.summary[summary_match.group("key")] = float(summary_match.group("value"))
            except ValueError:
                pass

        runs.append(run)
    return runs


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def run_label(run):
    if run.best_lr_strategy is None:
        return "fixed"
    if run.best_lr_scheduler == "constant":
        return "best_lr const"
    if run.best_lr_scheduler == "linear":
        return "best_lr 1->0.1"
    return f"best_lr {run.best_lr_scheduler}"


def total_steps(run):
    return max((row.total_steps for row in run.applied_lrs), default=0)


def progress_x(run, rows):
    total = total_steps(run)
    if total:
        return np.array([row.step / total for row in rows], dtype=float)
    return np.array([row.step for row in rows], dtype=float)


def moving_average(values, window):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def finite_or_nan(value):
    return value if value is not None and math.isfinite(value) else np.nan


def plot_training_curves(runs, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for run in runs:
        if not run.progress_logs:
            continue
        x = progress_x(run, run.progress_logs)
        losses = [row.loss for row in run.progress_logs]
        axes[0].plot(x, losses, linewidth=0.8, alpha=0.35, label=f"{run_label(run)} raw")
        axes[0].plot(
            x,
            moving_average(losses, 25),
            linewidth=1.8,
            label=f"{run_label(run)} 25-step",
        )
        axes[1].plot(
            x,
            [row.tok_per_sec for row in run.progress_logs],
            linewidth=1.2,
            label=run_label(run),
        )
        axes[2].plot(
            x,
            [row.mfu_percent for row in run.progress_logs],
            linewidth=1.2,
            label=run_label(run),
        )
    axes[0].set_ylabel("logged train loss")
    axes[1].set_ylabel("tokens/sec")
    axes[2].set_ylabel("MFU percent")
    axes[2].set_xlabel("training progress")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("LLM Training Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_traces(runs, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for run in runs:
        if run.applied_lrs:
            x = progress_x(run, run.applied_lrs)
            axes[0].plot(x, [row.lr for row in run.applied_lrs], linewidth=1.3, label=run_label(run))
        if run.best_lr_logs:
            total = total_steps(run)
            x = np.array([row.step / total for row in run.best_lr_logs])
            axes[1].plot(x, [row.searched_lr for row in run.best_lr_logs], linewidth=1.2, label=f"{run_label(run)} searched")
            axes[1].plot(x, [row.best_lr_ema for row in run.best_lr_logs], linestyle="--", linewidth=1.0, label=f"{run_label(run)} ema")
    axes[0].set_ylabel("applied Muon LR")
    axes[1].set_ylabel("searched LR / EMA")
    axes[1].set_xlabel("training progress")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Muon LR Traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_loss(runs, output_path):
    runs = [run for run in runs if run.best_lr_logs]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for run in runs:
        total = total_steps(run)
        x = np.array([row.step / total for row in run.best_lr_logs])
        ax.plot(x, [row.best_loss for row in run.best_lr_logs], linewidth=1.2, label=run_label(run))
    ax.set_title("Best-LR Selected Batch Loss")
    ax.set_xlabel("training progress")
    ax.set_ylabel("line-search loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_landscapes(runs, output_path):
    runs = [run for run in runs if run.best_lr_logs]
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4.2 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        snapshots = [
            ("first", run.best_lr_logs[0]),
            ("middle", run.best_lr_logs[len(run.best_lr_logs) // 2]),
            ("last", run.best_lr_logs[-1]),
        ]
        for ax, (label, log) in zip(axes[row], snapshots):
            pairs = sorted(log.losses_by_lr.items())
            ax.plot([lr for lr, _ in pairs], [loss for _, loss in pairs], marker="o")
            ax.axvline(log.searched_lr, color="tab:red", linestyle="--", label="selected")
            ax.set_xscale("log")
            ax.set_title(f"{run_label(run)} {label}, step {log.step}")
            ax.set_xlabel("candidate Muon LR")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Best-LR Landscape Snapshots")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(runs, output_path):
    labels = [run_label(run) for run in runs]
    x = np.arange(len(runs))
    metrics = [
        ("val bpb", lambda run: run.val_bpb),
        ("val loss", lambda run: run.val_loss),
        ("train loss", lambda run: run.train_loss),
        ("wall seconds", lambda run: run.wall_time_seconds),
        ("MFU percent", lambda run: run.summary.get("mfu_percent")),
        ("peak VRAM MB", lambda run: run.summary.get("peak_vram_mb")),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 14), sharex=True)
    for ax, (ylabel, getter) in zip(axes, metrics):
        ax.bar(x, [finite_or_nan(getter(run)) for run in runs])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("Final Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    ranked = sorted(runs, key=lambda run: run.val_bpb if run.val_bpb is not None else float("inf"))
    lines = [
        "LLM Baseline2 Exp1 Results",
        "=" * 27,
        "",
        f"Runs: {len(runs)}",
        "",
        "Ranking by val_bpb",
        "-" * 18,
    ]
    for rank, run in enumerate(ranked, start=1):
        lines.extend(
            [
                f"{rank}. {run.name}",
                f"   label: {run_label(run)}",
                f"   val_bpb: {fmt(run.val_bpb)}",
                f"   val_loss: {fmt(run.val_loss)}",
                f"   train_loss: {fmt(run.train_loss)}",
                f"   wall_time_seconds: {fmt(run.wall_time_seconds)}",
                f"   cuda_time_seconds: {fmt(run.cuda_time_seconds)}",
                f"   final_applied_lr: {fmt(run.applied_lrs[-1].lr if run.applied_lrs else None)}",
                f"   final_searched_lr: {fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else None)}",
                f"   final_best_lr_ema: {fmt(run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else None)}",
                f"   final_post_loss_mean: {fmt(sum(run.final_post_losses) / len(run.final_post_losses) if run.final_post_losses else None)}",
                "",
            ]
        )

    headers = [
        "run",
        "name",
        "label",
        "muon_lr",
        "scheduler",
        "train_loss",
        "val_loss",
        "val_bpb",
        "wall_s",
        "cuda_s",
        "steps",
        "final_applied_lr",
        "final_searched_lr",
    ]
    rows = [
        [
            run.index,
            run.name,
            run_label(run),
            fmt(run.muon_lr),
            run.best_lr_scheduler,
            fmt(run.train_loss),
            fmt(run.val_loss),
            fmt(run.val_bpb),
            fmt(run.wall_time_seconds),
            fmt(run.cuda_time_seconds),
            len(run.applied_lrs),
            fmt(run.applied_lrs[-1].lr if run.applied_lrs else None),
            fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else None),
        ]
        for run in runs
    ]
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *rows)]
    lines.extend(["All Runs", "-" * 8])
    lines.append("  ".join(str(value).ljust(width) for value, width in zip(headers, widths)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    lines = [
        "run,name,label,muon_lr,strategy,scheduler,train_loss,val_loss,val_bpb,"
        "wall_time_seconds,cuda_time_seconds,steps,total_steps,progress_points,"
        "best_lr_points,final_applied_lr,final_searched_lr,final_best_lr_ema,"
        "mfu_percent,peak_vram_mb,total_tokens_M"
    ]
    for run in runs:
        lines.append(
            ",".join(
                str(value)
                for value in [
                    run.index,
                    run.name,
                    run_label(run),
                    fmt(run.muon_lr),
                    run.best_lr_strategy or "",
                    run.best_lr_scheduler,
                    fmt(run.train_loss),
                    fmt(run.val_loss),
                    fmt(run.val_bpb),
                    fmt(run.wall_time_seconds),
                    fmt(run.cuda_time_seconds),
                    len(run.applied_lrs),
                    total_steps(run),
                    len(run.progress_logs),
                    len(run.best_lr_logs),
                    fmt(run.applied_lrs[-1].lr if run.applied_lrs else None),
                    fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else None),
                    fmt(run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else None),
                    fmt(run.summary.get("mfu_percent")),
                    fmt(run.summary.get("peak_vram_mb")),
                    fmt(run.summary.get("total_tokens_M")),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot llm_baseline2 exp1 results.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(runs, args.output_dir / "training_curves.png")
    plot_lr_traces(runs, args.output_dir / "lr_traces.png")
    plot_best_lr_loss(runs, args.output_dir / "best_lr_loss.png")
    plot_best_lr_landscapes(runs, args.output_dir / "best_lr_landscapes.png")
    plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in runs:
        print(
            f"run={run.index} name={run.name} label={run_label(run)} "
            f"steps={len(run.applied_lrs)}/{total_steps(run)} "
            f"progress_logs={len(run.progress_logs)} best_lr_logs={len(run.best_lr_logs)} "
            f"val_bpb={run.val_bpb} wall={run.wall_time_seconds}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

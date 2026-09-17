import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_cifar_baseline2_eval2_exp3 as base


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp8.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp8_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"(?:sgd_lr_mult=(?P<sgd_lr_mult>\S+) )?"
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
RUN_TIME_RE = re.compile(
    r"^run_time run=(?P<run>\d+) name=(?P<name>\S+) "
    r"wall_time_seconds=(?P<wall_time_seconds>\S+) "
    r"cuda_time_seconds=(?P<cuda_time_seconds>\S+)"
)
TRAINING_BATCH_LOSSES_RE = re.compile(
    r"^training_batch_losses step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"update=(?P<update>\S+) losses=(?P<losses>\[.*\])$"
)
SUMMARY_SGD_RE = re.compile(r"^SGD lr mult:\s+(?P<sgd_lr_mult>\S+)")

RUN_KIND_ORDER = {
    "fixed": 0,
    "min_loss_constant": 1,
    "min_loss_linear_2_to_0.01": 2,
}


def parse_exp8_log(path):
    runs = []
    runs_by_name = {}
    current = None

    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current = base.Run(
                    index=int(match.group("run")),
                    batch_size=int(match.group("batch_size")),
                    muon_lr=float(match.group("muon_lr")),
                    name=match.group("name"),
                    best_lr_strategy=base.parse_strategy(
                        match.group("best_lr_strategy")
                    ),
                    best_lr_linear_decay=base.parse_bool(
                        match.group("best_lr_linear_decay")
                    ),
                )
                current.sgd_lr_mult = (
                    float(match.group("sgd_lr_mult"))
                    if match.group("sgd_lr_mult") is not None
                    else None
                )
                current.best_lr_scheduler = match.group("best_lr_scheduler")
                current.wall_time_seconds = None
                current.cuda_time_seconds = None
                current.final_post_losses = []
                current.final_post_loss_mean = None
                current.final_post_loss_sum = None
                runs.append(current)
                runs_by_name[current.name] = current
                continue

            match = base.APPLIED_LR_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is None:
                    raise ValueError(
                        f"Found applied_lr before run header on line {line_number}"
                    )
                run.applied_lrs.append(
                    base.AppliedLR(
                        step=int(match.group("step")),
                        total_steps=int(match.group("total_steps")),
                        lr=float(match.group("muon_lr")),
                    )
                )
                continue

            match = base.BEST_LR_RE.match(line)
            if match:
                if current is None:
                    raise ValueError(
                        f"Found best_lr before run header on line {line_number}"
                    )
                current.best_lr_logs.append(base.parse_best_lr(match))
                continue

            match = base.EVAL_RE.match(line)
            if match and current is not None:
                current.eval_logs.append(
                    base.EvalLog(
                        epoch=int(match.group("epoch")),
                        val_acc=float(match.group("val_acc")),
                        time_seconds=float(match.group("time_seconds")),
                    )
                )
                continue

            match = base.FINAL_RE.match(line)
            if match and current is not None:
                current.train25_loss = float(match.group("train25_loss"))
                current.val_acc = float(match.group("val_acc"))
                current.tta_val_acc = float(match.group("tta_val_acc"))
                current.time_seconds = float(match.group("time_seconds"))
                continue

            match = base.FINAL_FULL_RE.match(line)
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

            match = TRAINING_BATCH_LOSSES_RE.match(line)
            if match and current is not None and match.group("update") == "post":
                losses = [float(value) for value in json.loads(match.group("losses"))]
                if int(match.group("step")) == int(match.group("total_steps")):
                    current.final_post_losses = losses
                    current.final_post_loss_mean = (
                        sum(losses) / len(losses) if losses else None
                    )
                    current.final_post_loss_sum = sum(losses)
                continue

            match = SUMMARY_SGD_RE.match(line)
            if match and current is not None and current.sgd_lr_mult is None:
                current.sgd_lr_mult = float(match.group("sgd_lr_mult"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def fmt(value):
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def run_kind(run):
    if run.best_lr_strategy is None:
        return "fixed"
    scheduler = getattr(run, "best_lr_scheduler", "constant")
    if scheduler == "constant":
        return "min_loss_constant"
    return f"min_loss_{scheduler}"


def run_label(run):
    labels = {
        "fixed": "fixed",
        "min_loss_constant": "min_loss const",
        "min_loss_linear_2_to_0.01": "min_loss 2->0.01",
    }
    return labels.get(run_kind(run), run_kind(run))


def run_sorted(runs):
    return sorted(
        runs,
        key=lambda run: (
            run.batch_size,
            RUN_KIND_ORDER.get(run_kind(run), 99),
            run.index,
        ),
    )


def runs_by_batch(runs):
    return {
        batch_size: [run for run in runs if run.batch_size == batch_size]
        for batch_size in sorted({run.batch_size for run in runs})
    }


def finite_or_nan(value):
    return value if value is not None and math.isfinite(value) else np.nan


def plot_final_metrics(runs, output_path):
    metrics = [
        ("TTA val acc", lambda run: run.tta_val_acc),
        ("val acc", lambda run: run.val_acc),
        ("train acc", lambda run: getattr(run, "train_acc", None)),
        ("train loss", base.final_train_loss),
        ("val loss", lambda run: getattr(run, "val_loss", None)),
        ("final post-loss mean", lambda run: getattr(run, "final_post_loss_mean", None)),
        ("wall time seconds", lambda run: getattr(run, "wall_time_seconds", None)),
    ]
    grouped = runs_by_batch(runs)
    colors = {
        "fixed": "tab:gray",
        "min_loss_constant": "tab:blue",
        "min_loss_linear_2_to_0.01": "tab:orange",
    }

    fig, axes = plt.subplots(
        len(metrics),
        len(grouped),
        figsize=(6.2 * len(grouped), 3.0 * len(metrics)),
        squeeze=False,
    )
    for col, (batch_size, batch_runs) in enumerate(grouped.items()):
        batch_runs = run_sorted(batch_runs)
        labels = [run_label(run) for run in batch_runs]
        x = np.arange(len(batch_runs))
        bar_colors = [colors.get(run_kind(run), "tab:purple") for run in batch_runs]
        for row, (ylabel, getter) in enumerate(metrics):
            ax = axes[row][col]
            ax.bar(x, [finite_or_nan(getter(run)) for run in batch_runs], color=bar_colors)
            ax.set_title(f"batch_size={batch_size}" if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Exp8 Final Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_traces(runs, output_path):
    grouped = runs_by_batch(runs)
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(7.2 * len(grouped), 5.4), squeeze=False
    )
    for ax, (batch_size, batch_runs) in zip(axes.ravel(), grouped.items()):
        for run in run_sorted(batch_runs):
            if not run.applied_lrs:
                continue
            total_steps = max(row.total_steps for row in run.applied_lrs)
            progress = np.array([row.step / total_steps for row in run.applied_lrs])
            lrs = np.array([row.lr for row in run.applied_lrs])
            ax.plot(progress, lrs, linewidth=1.4, label=run_label(run))
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("training progress")
        ax.set_ylabel("applied Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Applied Muon LR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_traces(runs, output_path):
    best_runs = [run for run in runs if run.best_lr_logs]
    if not best_runs:
        return
    grouped = runs_by_batch(best_runs)
    fig, axes = plt.subplots(
        2,
        len(grouped),
        figsize=(7.2 * len(grouped), 9.0),
        squeeze=False,
    )
    for col, (batch_size, batch_runs) in enumerate(grouped.items()):
        for run in run_sorted(batch_runs):
            total_steps = max((row.total_steps for row in run.applied_lrs), default=1)
            applied_by_step = {row.step: row.lr for row in run.applied_lrs}
            steps = np.array([row.step for row in run.best_lr_logs], dtype=float)
            progress = steps / total_steps
            searched = np.array([row.searched_lr for row in run.best_lr_logs])
            ema = np.array([row.best_lr_ema for row in run.best_lr_logs])
            applied = np.array(
                [applied_by_step.get(row.step, np.nan) for row in run.best_lr_logs]
            )
            label = run_label(run)
            axes[0][col].plot(progress, searched, linewidth=1.4, label=f"{label} searched")
            axes[0][col].plot(progress, applied, linestyle="--", linewidth=1.2, label=f"{label} applied")
            axes[0][col].plot(progress, ema, linestyle=":", linewidth=1.0, label=f"{label} ema")
            axes[1][col].plot(
                progress,
                [row.best_loss for row in run.best_lr_logs],
                linewidth=1.3,
                label=label,
            )
        axes[0][col].set_title(f"LR trace, batch_size={batch_size}")
        axes[0][col].set_ylabel("LR")
        axes[0][col].grid(True, alpha=0.25)
        axes[0][col].legend(fontsize=7)
        axes[1][col].set_title(f"selected batch loss, batch_size={batch_size}")
        axes[1][col].set_xlabel("training progress")
        axes[1][col].set_ylabel("loss")
        axes[1][col].grid(True, alpha=0.25)
        axes[1][col].legend(fontsize=7)
    fig.suptitle("Best-LR Search Traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_curves(runs, output_path):
    grouped = runs_by_batch(runs)
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(7.2 * len(grouped), 5.4), squeeze=False
    )
    for ax, (batch_size, batch_runs) in zip(axes.ravel(), grouped.items()):
        for run in run_sorted(batch_runs):
            if not run.eval_logs:
                continue
            ax.plot(
                [row.epoch for row in run.eval_logs],
                [row.val_acc for row in run.eval_logs],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label=run_label(run),
            )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Validation Accuracy During Training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_post_loss_distributions(runs, output_path):
    runs = [run for run in run_sorted(runs) if getattr(run, "final_post_losses", [])]
    if not runs:
        return
    grouped = runs_by_batch(runs)
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(7.2 * len(grouped), 5.4), sharey=True, squeeze=False
    )
    for ax, (batch_size, batch_runs) in zip(axes.ravel(), grouped.items()):
        batch_runs = run_sorted(batch_runs)
        ax.boxplot(
            [run.final_post_losses for run in batch_runs],
            tick_labels=[run_label(run) for run in batch_runs],
            showfliers=False,
        )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("run")
        ax.grid(True, axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("final post-update batch losses")
    fig.suptitle("Final Post-Loss Distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    ordered = run_sorted(runs)
    lines = [
        "CIFAR Baseline Eval2 Exp8 Results",
        "=" * 34,
        "",
        f"Runs: {len(runs)}",
        "",
    ]
    for batch_size, batch_runs in runs_by_batch(runs).items():
        ranked = sorted(
            batch_runs,
            key=lambda run: run.tta_val_acc if run.tta_val_acc is not None else -1,
            reverse=True,
        )
        lines.extend([f"batch_size={batch_size}", "-" * 15])
        for rank, run in enumerate(ranked, start=1):
            lines.extend(
                [
                    f"{rank}. {run.name}",
                    f"   kind: {run_label(run)}",
                    f"   muon_lr: {fmt(run.muon_lr)}",
                    f"   sgd_lr_mult: {fmt(getattr(run, 'sgd_lr_mult', None))}",
                    f"   TTA val acc: {fmt(run.tta_val_acc)}",
                    f"   val acc: {fmt(run.val_acc)}",
                    f"   train acc: {fmt(getattr(run, 'train_acc', None))}",
                    f"   train loss: {fmt(base.final_train_loss(run))}",
                    f"   val loss: {fmt(getattr(run, 'val_loss', None))}",
                    f"   final post-loss mean: {fmt(getattr(run, 'final_post_loss_mean', None))}",
                    f"   wall time seconds: {fmt(getattr(run, 'wall_time_seconds', None))}",
                    f"   final applied LR: {fmt(run.applied_lrs[-1].lr if run.applied_lrs else '')}",
                    f"   final searched LR: {fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else '')}",
                    f"   final best-LR EMA: {fmt(run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else '')}",
                ]
            )
        lines.append("")

    headers = [
        "run",
        "name",
        "batch",
        "kind",
        "muon_lr",
        "sgd_mult",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "post_loss_mean",
        "wall_s",
        "final_applied_lr",
        "final_searched_lr",
    ]
    rows = [
        [
            run.index,
            run.name,
            run.batch_size,
            run_label(run),
            fmt(run.muon_lr),
            fmt(getattr(run, "sgd_lr_mult", None)),
            fmt(base.final_train_loss(run)),
            fmt(getattr(run, "val_loss", None)),
            fmt(getattr(run, "train_acc", None)),
            fmt(run.val_acc),
            fmt(run.tta_val_acc),
            fmt(getattr(run, "final_post_loss_mean", None)),
            fmt(getattr(run, "wall_time_seconds", None)),
            fmt(run.applied_lrs[-1].lr if run.applied_lrs else ""),
            fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else ""),
        ]
        for run in ordered
    ]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    lines.extend(["All Runs", "-" * 8])
    lines.append(
        "  ".join(str(value).ljust(width) for value, width in zip(headers, widths))
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(str(value).ljust(width) for value, width in zip(row, widths))
        )
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    lines = [
        "run,name,batch_size,kind,muon_lr,sgd_lr_mult,train_loss,val_loss,"
        "train_acc,val_acc,tta_val_acc,final_post_loss_mean,wall_time_seconds,"
        "cuda_time_seconds,final_applied_lr,final_searched_lr,final_best_lr_ema"
    ]
    for run in run_sorted(runs):
        lines.append(
            ",".join(
                str(value)
                for value in [
                    run.index,
                    run.name,
                    run.batch_size,
                    run_label(run),
                    fmt(run.muon_lr),
                    fmt(getattr(run, "sgd_lr_mult", None)),
                    fmt(base.final_train_loss(run)),
                    fmt(getattr(run, "val_loss", None)),
                    fmt(getattr(run, "train_acc", None)),
                    fmt(run.val_acc),
                    fmt(run.tta_val_acc),
                    fmt(getattr(run, "final_post_loss_mean", None)),
                    fmt(getattr(run, "wall_time_seconds", None)),
                    fmt(getattr(run, "cuda_time_seconds", None)),
                    fmt(run.applied_lrs[-1].lr if run.applied_lrs else ""),
                    fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else ""),
                    fmt(run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else ""),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 exp8 results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_exp8_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    plot_lr_traces(runs, args.output_dir / "applied_lr_traces.png")
    plot_best_lr_traces(runs, args.output_dir / "best_lr_traces.png")
    plot_final_post_loss_distributions(
        runs, args.output_dir / "final_post_loss_distributions.png"
    )
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in run_sorted(runs):
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        print(
            f"run={run.index} name={run.name} kind={run_label(run)} "
            f"bs={run.batch_size} steps={len(run.applied_lrs)}/{total_steps} "
            f"tta={run.tta_val_acc} wall={getattr(run, 'wall_time_seconds', None)}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

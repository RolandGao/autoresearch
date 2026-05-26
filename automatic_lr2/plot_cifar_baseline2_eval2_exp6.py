import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_cifar_baseline2_eval2_exp3 as base


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp6.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp6_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+)"
    r"(?: best_lr_scheduler=(?P<best_lr_scheduler>\S+))?"
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

SCHEDULER_ORDER = {"constant": 0, "linear": 1, "last2_linear": 2}


def scheduler_from_name(name, fallback):
    if fallback:
        return fallback
    if "last2_decay" in name:
        return "last2_linear"
    if "_decay_" in name:
        return "linear"
    return "constant"


def parse_exp6_log(path):
    runs = base.parse_log(path)
    runs_by_name = {run.name: run for run in runs}
    current = None

    for run in runs:
        run.best_lr_scheduler = scheduler_from_name(run.name, None)
        run.wall_time_seconds = None
        run.cuda_time_seconds = None
        run.final_post_losses = []
        run.final_post_loss_mean = None
        run.final_post_loss_sum = None

    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current = runs_by_name.get(match.group("name"))
                if current is not None:
                    current.best_lr_scheduler = scheduler_from_name(
                        current.name, match.group("best_lr_scheduler")
                    )
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

    return runs


def fmt(value):
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def strategy_label(run):
    return "fixed_muon" if run.best_lr_strategy is None else run.best_lr_strategy


def run_label(run):
    if run.best_lr_strategy is None:
        return f"fixed lr={run.muon_lr:g}"
    return getattr(run, "best_lr_scheduler", "unknown")


def compact_name(run):
    if run.best_lr_strategy is None:
        return f"fixed {run.muon_lr:g}"
    scheduler = getattr(run, "best_lr_scheduler", "unknown")
    return scheduler.replace("_linear", "").replace("_", " ")


def runs_by_batch(runs):
    return {
        batch_size: [run for run in runs if run.batch_size == batch_size]
        for batch_size in sorted({run.batch_size for run in runs})
    }


def run_sorted(runs):
    return sorted(
        runs,
        key=lambda run: (
            0 if run.best_lr_strategy is None else 1,
            run.muon_lr,
            SCHEDULER_ORDER.get(getattr(run, "best_lr_scheduler", ""), 99),
            run.index,
        ),
    )


def finite_or_nan(value):
    return value if value is not None and math.isfinite(value) else np.nan


def plot_run_comparison(runs, output_path):
    metrics = [
        ("TTA val acc", lambda run: run.tta_val_acc),
        ("val acc", lambda run: run.val_acc),
        ("train acc", lambda run: getattr(run, "train_acc", None)),
        ("train loss", base.final_train_loss),
        ("val loss", lambda run: getattr(run, "val_loss", None)),
        (
            "final post-loss mean",
            lambda run: getattr(run, "final_post_loss_mean", None),
        ),
        ("wall time seconds", lambda run: getattr(run, "wall_time_seconds", None)),
        ("CUDA time seconds", lambda run: getattr(run, "cuda_time_seconds", None)),
    ]
    grouped = runs_by_batch(runs)
    fig, axes = plt.subplots(
        len(metrics),
        len(grouped),
        figsize=(6.2 * len(grouped), 3.0 * len(metrics)),
        squeeze=False,
    )

    colors = {
        "fixed_muon": "tab:gray",
        "constant": "tab:blue",
        "linear": "tab:orange",
        "last2_linear": "tab:green",
    }

    for col, (batch_size, batch_runs) in enumerate(grouped.items()):
        batch_runs = run_sorted(batch_runs)
        labels = [compact_name(run) for run in batch_runs]
        x = np.arange(len(batch_runs))
        bar_colors = [
            colors.get(
                "fixed_muon"
                if run.best_lr_strategy is None
                else getattr(run, "best_lr_scheduler", ""),
                "tab:purple",
            )
            for run in batch_runs
        ]
        for row, (ylabel, getter) in enumerate(metrics):
            ax = axes[row][col]
            values = [finite_or_nan(getter(run)) for run in batch_runs]
            ax.bar(x, values, color=bar_colors)
            ax.set_title(f"batch_size={batch_size}" if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Exp6 Run Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_post_loss_distributions(runs, output_path):
    runs = [run for run in base.run_order(runs) if getattr(run, "final_post_losses", [])]
    if not runs:
        return

    grouped = runs_by_batch(runs)
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(7.2 * len(grouped), 5.8), sharey=True, squeeze=False
    )
    for ax, (batch_size, batch_runs) in zip(axes.ravel(), grouped.items()):
        batch_runs = run_sorted(batch_runs)
        data = [run.final_post_losses for run in batch_runs]
        labels = [compact_name(run) for run in batch_runs]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("run")
        ax.grid(True, axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("final post-update batch losses")
    fig.suptitle("Final Post-Loss Distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accuracy_vs_loss(runs, output_path):
    complete = [
        run
        for run in runs
        if run.tta_val_acc is not None and base.final_train_loss(run) is not None
    ]
    if not complete:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, batch_size in zip(axes, sorted({run.batch_size for run in complete})):
        batch_runs = run_sorted([run for run in complete if run.batch_size == batch_size])
        for run in batch_runs:
            marker = "o" if run.best_lr_strategy is None else "s"
            ax.scatter(
                base.final_train_loss(run),
                run.tta_val_acc,
                s=55,
                marker=marker,
                label=compact_name(run),
            )
            ax.annotate(
                compact_name(run),
                (base.final_train_loss(run), run.tta_val_acc),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("train loss")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("TTA val acc")
    fig.suptitle("TTA Accuracy vs Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = [
        "run,name,batch_size,muon_lr,strategy,scheduler,train_loss,train25_loss,"
        "val_loss,train_acc,val_acc,tta_val_acc,final_post_loss_mean,"
        "final_post_loss_sum,wall_time_seconds,cuda_time_seconds,steps,"
        "final_applied_lr,final_searched_lr,final_best_lr_ema"
    ]
    for run in base.run_order(runs):
        final_applied_lr = run.applied_lrs[-1].lr if run.applied_lrs else ""
        final_searched_lr = run.best_lr_logs[-1].searched_lr if run.best_lr_logs else ""
        final_best_lr_ema = run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else ""
        fields = [
            run.index,
            run.name,
            run.batch_size,
            fmt(run.muon_lr),
            strategy_label(run),
            run_label(run),
            fmt(getattr(run, "train_loss", None)),
            fmt(run.train25_loss),
            fmt(getattr(run, "val_loss", None)),
            fmt(getattr(run, "train_acc", None)),
            fmt(run.val_acc),
            fmt(run.tta_val_acc),
            fmt(getattr(run, "final_post_loss_mean", None)),
            fmt(getattr(run, "final_post_loss_sum", None)),
            fmt(getattr(run, "wall_time_seconds", None)),
            fmt(getattr(run, "cuda_time_seconds", None)),
            len(run.applied_lrs),
            fmt(final_applied_lr),
            fmt(final_searched_lr),
            fmt(final_best_lr_ema),
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def write_progress_summary(runs, output_path):
    lines = [
        "run,name,batch_size,muon_lr,strategy,scheduler,steps,total_steps,"
        "progress,best_lr_logs,eval_points,is_complete"
    ]
    for run in base.run_order(runs):
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        steps = len(run.applied_lrs)
        progress = "%.8g" % (steps / total_steps) if total_steps else ""
        fields = [
            run.index,
            run.name,
            run.batch_size,
            fmt(run.muon_lr),
            strategy_label(run),
            run_label(run),
            steps,
            total_steps,
            progress,
            len(run.best_lr_logs),
            len(run.eval_logs),
            run.tta_val_acc is not None,
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def write_important_info(runs, output_path):
    complete_runs = [run for run in runs if run.tta_val_acc is not None]
    lines = [
        "CIFAR Baseline Exp6 Results",
        "=" * 27,
        "",
        f"Runs: {len(runs)}",
        f"Complete runs: {len(complete_runs)}",
        "",
    ]

    metric_specs = [
        ("train loss", lambda run: base.final_train_loss(run)),
        ("val loss", lambda run: getattr(run, "val_loss", None)),
        ("train acc", lambda run: getattr(run, "train_acc", None)),
        ("val acc", lambda run: run.val_acc),
        ("val tta", lambda run: run.tta_val_acc),
    ]
    table_rows = [
        (run.name, *[fmt(getter(run)) for _, getter in metric_specs])
        for run in base.run_order(complete_runs)
    ]
    headers = ("name", *[name for name, _ in metric_specs])
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *table_rows)
    ]
    lines.extend(["All Runs", "-" * 8])
    lines.append(
        "  ".join(str(value).ljust(width) for value, width in zip(headers, widths))
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(
            "  ".join(str(value).ljust(width) for value, width in zip(row, widths))
        )
    lines.append("")

    def add_correlation_section(title, section_runs):
        usable_runs = []
        values = []
        for run in base.run_order(section_runs):
            row = [getter(run) for _, getter in metric_specs]
            if all(value is not None and math.isfinite(value) for value in row):
                usable_runs.append(run)
                values.append(row)

        lines.extend([title, "-" * len(title)])
        if len(values) < 2:
            lines.extend(["Not enough complete rows.", ""])
            return

        matrix = np.corrcoef(np.array(values, dtype=np.float64), rowvar=False)
        corr_headers = ["metric", *[name for name, _ in metric_specs]]
        corr_rows = [
            (name, *["%.4f" % value for value in row])
            for (name, _), row in zip(metric_specs, matrix)
        ]
        corr_widths = [
            max(len(str(value)) for value in column)
            for column in zip(corr_headers, *corr_rows)
        ]
        lines.append(f"Rows: {len(usable_runs)}")
        lines.append(
            "  ".join(
                str(value).ljust(width)
                for value, width in zip(corr_headers, corr_widths)
            )
        )
        lines.append("  ".join("-" * width for width in corr_widths))
        for row in corr_rows:
            lines.append(
                "  ".join(
                    str(value).ljust(width) for value, width in zip(row, corr_widths)
                )
            )
        lines.append("")

    add_correlation_section("Metric Correlations, All Runs", complete_runs)
    for batch_size, batch_runs in runs_by_batch(complete_runs).items():
        add_correlation_section(
            f"Metric Correlations, batch_size={batch_size}", batch_runs
        )

    for batch_size, batch_runs in runs_by_batch(complete_runs).items():
        ranked = sorted(batch_runs, key=lambda run: -run.tta_val_acc)
        lines.extend([f"batch_size={batch_size}", "-" * 15])
        for rank, run in enumerate(ranked, start=1):
            lines.extend(
                [
                    f"{rank}. {run.name}",
                    f"   strategy: {strategy_label(run)}",
                    f"   scheduler: {run_label(run)}",
                    f"   TTA val acc: {fmt(run.tta_val_acc)}",
                    f"   val acc: {fmt(run.val_acc)}",
                    f"   train acc: {fmt(getattr(run, 'train_acc', None))}",
                    f"   train loss: {fmt(base.final_train_loss(run))}",
                    f"   val loss: {fmt(getattr(run, 'val_loss', None))}",
                    f"   final post-loss mean: {fmt(getattr(run, 'final_post_loss_mean', None))}",
                    f"   final post-loss sum: {fmt(getattr(run, 'final_post_loss_sum', None))}",
                    f"   wall time seconds: {fmt(getattr(run, 'wall_time_seconds', None))}",
                    f"   CUDA time seconds: {fmt(getattr(run, 'cuda_time_seconds', None))}",
                    f"   final applied LR: {fmt(run.applied_lrs[-1].lr if run.applied_lrs else '')}",
                    f"   final searched LR: {fmt(run.best_lr_logs[-1].searched_lr if run.best_lr_logs else '')}",
                    f"   final best-LR EMA: {fmt(run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else '')}",
                ]
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 exp6 fixed-Muon and min-loss runs."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_exp6_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base.plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    base.plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    base.plot_applied_lr_traces(runs, args.output_dir / "applied_lr_traces.png")
    base.plot_best_lr_traces(runs, args.output_dir / "best_lr_traces.png")
    base.plot_lr_landscape_snapshots(
        runs, args.output_dir / "best_lr_landscapes.png"
    )
    plot_run_comparison(runs, args.output_dir / "run_comparison.png")
    plot_final_post_loss_distributions(
        runs, args.output_dir / "final_post_loss_distributions.png"
    )
    plot_accuracy_vs_loss(runs, args.output_dir / "accuracy_vs_train_loss.png")
    write_summary(runs, args.output_dir / "summary.csv")
    write_progress_summary(runs, args.output_dir / "progress_summary.csv")
    write_important_info(runs, args.output_dir / "important_info.txt")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in base.run_order(runs):
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        progress = f"{len(run.applied_lrs) / total_steps:.3f}" if total_steps else ""
        print(
            f"run={run.index} name={run.name} label={run_label(run)} "
            f"steps={len(run.applied_lrs)}/{total_steps} progress={progress} "
            f"tta={run.tta_val_acc} wall={getattr(run, 'wall_time_seconds', None)}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

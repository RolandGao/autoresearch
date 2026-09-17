import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_cifar_baseline2_eval5_exp3 import (
    fmt,
    group_key,
    parse_log,
    plot_best_by_train_steps,
    plot_metric_rankings,
    plot_search_progress,
    plot_step_sensitivity,
    point_label,
    run_for_completion,
    runs_for_group,
    sorted_groups,
    top_runs,
    write_csv,
)


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval5_no_momentum.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval5_no_momentum_plots")


def plot_top_schedules(runs, output_path):
    batch_sizes = sorted({run.batch_size for run in runs})
    train_steps_list = sorted({run.train_steps for run in runs})
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(train_steps_list),
        figsize=(4.4 * len(train_steps_list), 3.5 * len(batch_sizes)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    for row, batch_size in enumerate(batch_sizes):
        for col, train_steps in enumerate(train_steps_list):
            ax = axes[row, col]
            key = (train_steps, batch_size)
            group_runs = runs_for_group(runs, key)
            if not group_runs:
                ax.set_visible(False)
                continue
            for rank, run in enumerate(top_runs(runs, key, n=6), start=1):
                steps = np.arange(1, len(run.schedule) + 1)
                ax.plot(
                    steps,
                    run.schedule,
                    marker="o" if rank == 1 else None,
                    linewidth=2.4 if rank == 1 else 1.2,
                    alpha=1.0 if rank == 1 else 0.7,
                    label=f"#{rank} loss={run.train_loss:.4f}",
                )
            if row == 0:
                ax.set_title(f"{train_steps} train steps")
            if col == 0:
                ax.set_ylabel(f"bs={batch_size}\nMuon LR")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=5.8)

    fig.suptitle("Top Applied LR Schedules by Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, search_steps, completions, cache_hits, output_path):
    lines = [
        "CIFAR Baseline Eval5 No-Momentum Applied LR Schedule Search",
        "=" * 64,
        "",
        f"Evaluated runs: {len(runs)}",
        f"Search steps: {len(search_steps)}",
        f"Cache hits: {len(cache_hits)}",
        "",
    ]

    for completion in sorted(
        completions, key=lambda row: (row.batch_size, row.train_steps)
    ):
        selected = run_for_completion(runs, completion)
        lines.extend(
            [
                f"train_steps={completion.train_steps}, batch_size={completion.batch_size} selected best",
                "-" * 56,
                f"point: {point_label(completion.best_point)}",
                f"lrs: {','.join(fmt(value) for value in completion.lrs)}",
                f"train_loss: {fmt(completion.train_loss)}",
                f"evaluated_points: {completion.evaluated_points}",
            ]
        )
        if selected is not None:
            lines.extend(
                [
                    f"val_loss: {fmt(selected.val_loss)}",
                    f"val_acc: {fmt(selected.val_acc)}",
                    f"tta_val_acc: {fmt(selected.tta_val_acc)}",
                ]
            )
        lines.append("")

    for key in sorted(sorted_groups(runs), key=lambda row: (row[1], row[0])):
        train_steps, batch_size = key
        lines.extend([f"Top runs, train_steps={train_steps}, batch_size={batch_size}", "-" * 56])
        for rank, run in enumerate(top_runs(runs, key, n=12), start=1):
            lines.append(
                f"{rank}. r{run.index} point={point_label(run.point)} "
                f"train_loss={fmt(run.train_loss)} val_loss={fmt(run.val_loss)} "
                f"val_acc={fmt(run.val_acc)} tta_val_acc={fmt(run.tta_val_acc)} "
                f"schedule={','.join(fmt(value) for value in run.schedule)}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval5 no-momentum applied LR schedule search."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs, search_steps, completions, cache_hits = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_search_progress(search_steps, completions, args.output_dir / "search_progress.png")
    plot_top_schedules(runs, args.output_dir / "top_schedule_shapes.png")
    plot_metric_rankings(runs, args.output_dir / "metric_rankings.png")
    plot_best_by_train_steps(runs, completions, args.output_dir / "selected_metrics.png")
    plot_step_sensitivity(runs, args.output_dir / "step_sensitivity.png")
    write_summary(runs, search_steps, completions, cache_hits, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Parsed {len(search_steps)} search steps and {len(cache_hits)} cache hits")
    for completion in sorted(completions, key=lambda row: group_key(row)):
        print(
            f"steps={completion.train_steps} bs={completion.batch_size} "
            f"best={point_label(completion.best_point)} "
            f"train_loss={completion.train_loss:.4f} "
            f"evaluated={completion.evaluated_points}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_cifar_baseline2_eval5_exp3 import (
    fmt,
    group_key,
    parse_log,
    point_label,
    runs_for_group,
    top_runs,
)


HERE = Path(__file__).resolve().parent
DEFAULT_LOGS = [
    ("exp2", HERE / "cifar_baseline2_eval5_exp2.log"),
    ("exp3", HERE / "cifar_baseline2_eval5_exp3.log"),
]
DEFAULT_OUTPUT_DIR = HERE / "cifar_baseline2_eval5_exp2_exp3_plots"


@dataclass
class LabeledRun:
    source: str
    run: object


@dataclass
class LabeledCompletion:
    source: str
    completion: object
    run: object | None


def load_logs(log_specs):
    datasets = []
    for source, path in log_specs:
        runs, search_steps, completions, cache_hits = parse_log(path)
        datasets.append(
            dict(
                source=source,
                path=path,
                runs=runs,
                search_steps=search_steps,
                completions=completions,
                cache_hits=cache_hits,
            )
        )
    return datasets


def selected_run_for(runs, completion):
    return next(
        (
            run
            for run in runs
            if group_key(run) == group_key(completion)
            and run.point == completion.best_point
        ),
        None,
    )


def selected_completions(datasets):
    rows = []
    for data in datasets:
        for completion in data["completions"]:
            rows.append(
                LabeledCompletion(
                    source=data["source"],
                    completion=completion,
                    run=selected_run_for(data["runs"], completion),
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            row.completion.batch_size,
            row.completion.train_steps,
            row.source,
        ),
    )


def combined_groups(datasets):
    groups = set()
    for data in datasets:
        groups.update(group_key(run) for run in data["runs"])
    return sorted(groups, key=lambda key: (key[1], key[0]))


def group_source(datasets, key):
    matches = [
        data
        for data in datasets
        if any(group_key(run) == key for run in data["runs"])
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def make_grid(num_items, width=4.4, height=3.3, cols=None):
    cols = cols or min(4, max(1, num_items))
    rows = math.ceil(num_items / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in axes.ravel()[num_items:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_selected_metrics(selected, output_path):
    rows = [row for row in selected if row.run is not None]
    batch_sizes = sorted({row.run.batch_size for row in rows})
    train_steps = sorted({row.run.train_steps for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), squeeze=False)
    ax_loss, ax_acc = axes.ravel()
    for batch_size in batch_sizes:
        batch_rows = [
            row
            for row in rows
            if row.run.batch_size == batch_size
        ]
        batch_rows.sort(key=lambda row: row.run.train_steps)
        label = f"bs={batch_size} ({batch_rows[0].source})"
        xs = [row.run.train_steps for row in batch_rows]
        ax_loss.plot(
            xs,
            [row.run.train_loss for row in batch_rows],
            marker="o",
            label=f"{label} train",
        )
        ax_loss.plot(
            xs,
            [row.run.val_loss for row in batch_rows],
            marker="s",
            linestyle="--",
            label=f"{label} val",
        )
        ax_acc.plot(
            xs,
            [row.run.val_acc for row in batch_rows],
            marker="^",
            label=f"{label} val",
        )
        ax_acc.plot(
            xs,
            [row.run.tta_val_acc for row in batch_rows],
            marker="D",
            linestyle="--",
            label=f"{label} TTA",
        )

    for ax in axes.ravel():
        ax.set_xticks(train_steps)
        ax.set_xlabel("train steps")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5, ncol=2)
    ax_loss.set_title("Selected Loss")
    ax_loss.set_ylabel("loss")
    ax_acc.set_title("Selected Accuracy")
    ax_acc.set_ylabel("accuracy")
    fig.suptitle("Eval5 Exp2 + Exp3 Selected Schedule Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_heatmaps(selected, output_path):
    rows = [row for row in selected if row.run is not None]
    batch_sizes = sorted({row.run.batch_size for row in rows})
    train_steps = sorted({row.run.train_steps for row in rows})
    metrics = [
        ("train loss", "train_loss", "magma_r"),
        ("TTA val acc", "tta_val_acc", "viridis"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4.8), squeeze=False)
    for ax, (title, attr, cmap) in zip(axes.ravel(), metrics):
        matrix = np.full((len(batch_sizes), len(train_steps)), np.nan)
        labels = [["" for _ in train_steps] for _ in batch_sizes]
        for row in rows:
            y = batch_sizes.index(row.run.batch_size)
            x = train_steps.index(row.run.train_steps)
            matrix[y, x] = getattr(row.run, attr)
            labels[y][x] = row.source
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(train_steps)), [str(step) for step in train_steps])
        ax.set_yticks(np.arange(len(batch_sizes)), [str(size) for size in batch_sizes])
        ax.set_xlabel("train steps")
        ax.set_ylabel("batch size")
        for y in range(len(batch_sizes)):
            for x in range(len(train_steps)):
                if np.isfinite(matrix[y, x]):
                    ax.text(
                        x,
                        y,
                        f"{matrix[y, x]:.3f}\n{labels[y][x]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if attr == "train_loss" else "black",
                    )
        fig.colorbar(image, ax=ax, shrink=0.86)
    fig.suptitle("Selected Schedule Metrics Heatmaps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_schedules(datasets, selected, output_path):
    selected_by_group = {
        (row.run.train_steps, row.run.batch_size): row
        for row in selected
        if row.run is not None
    }
    groups = sorted(selected_by_group, key=lambda key: (key[1], key[0]))
    fig, axes = make_grid(len(groups), width=4.3, height=3.35)
    for ax, key in zip(axes, groups):
        row = selected_by_group[key]
        data = next(item for item in datasets if item["source"] == row.source)
        for rank, run in enumerate(top_runs(data["runs"], key, n=4), start=1):
            xs = np.arange(1, len(run.schedule) + 1)
            is_selected = run.point == row.run.point
            ax.plot(
                xs,
                run.schedule,
                marker="o" if is_selected else None,
                linewidth=2.5 if is_selected else 1.1,
                alpha=1.0 if is_selected else 0.6,
                label=(
                    f"selected loss={run.train_loss:.4f}"
                    if is_selected
                    else f"#{rank} loss={run.train_loss:.4f}"
                ),
            )
        train_steps, batch_size = key
        ax.set_title(f"{row.source}: steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("training step")
        ax.set_ylabel("Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("Selected and Top Applied LR Schedules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_search_progress(datasets, output_path):
    panels = []
    for data in datasets:
        for key in sorted({group_key(step) for step in data["search_steps"]}, key=lambda k: (k[1], k[0])):
            panels.append((data, key))

    fig, axes = make_grid(len(panels), width=4.3, height=3.1)
    for ax, (data, key) in zip(axes, panels):
        steps = sorted(
            [step for step in data["search_steps"] if group_key(step) == key],
            key=lambda step: step.move,
        )
        completion = next(
            (row for row in data["completions"] if group_key(row) == key),
            None,
        )
        ax.plot(
            [step.move for step in steps],
            [step.center_train_loss for step in steps],
            marker="o",
            label="center",
        )
        ax.plot(
            [step.move for step in steps],
            [step.best_train_loss for step in steps],
            marker="s",
            label="best",
        )
        if completion is not None:
            ax.axhline(completion.train_loss, color="tab:green", linestyle=":", label="selected")
        train_steps, batch_size = key
        ax.set_title(f"{data['source']}: steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("search move")
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("Search Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_scatter(datasets, output_path):
    groups = combined_groups(datasets)
    fig, axes = make_grid(len(groups), width=4.3, height=3.25)
    for ax, key in zip(axes, groups):
        data = group_source(datasets, key)
        if data is None:
            continue
        runs = runs_for_group(data["runs"], key)
        scatter = ax.scatter(
            [run.train_loss for run in runs],
            [run.tta_val_acc for run in runs],
            c=[run.val_acc for run in runs],
            cmap="viridis",
            s=34,
            edgecolor="black",
            linewidth=0.25,
        )
        best_train = min(runs, key=lambda run: run.train_loss)
        best_tta = max(runs, key=lambda run: run.tta_val_acc)
        ax.scatter(
            [best_train.train_loss],
            [best_train.tta_val_acc],
            marker="*",
            s=150,
            color="gold",
            edgecolor="black",
            label="best train",
        )
        ax.scatter(
            [best_tta.train_loss],
            [best_tta.tta_val_acc],
            marker="X",
            s=72,
            color="tab:red",
            edgecolor="black",
            label="best TTA",
        )
        train_steps, batch_size = key
        ax.set_title(f"{data['source']}: steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("train loss")
        ax.set_ylabel("TTA val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6)
        fig.colorbar(scatter, ax=ax, shrink=0.82, label="val acc")
    fig.suptitle("All Runs: Train Loss vs TTA Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(datasets, selected, output_path):
    lines = [
        "CIFAR Baseline Eval5 Exp2 + Exp3 Combined",
        "=" * 48,
        "",
    ]
    for data in datasets:
        lines.extend(
            [
                f"{data['source']}: {data['path']}",
                f"runs={len(data['runs'])} search_steps={len(data['search_steps'])} "
                f"cache_hits={len(data['cache_hits'])}",
                "",
            ]
        )
    for row in selected:
        completion = row.completion
        lines.extend(
            [
                f"{row.source}: train_steps={completion.train_steps}, "
                f"batch_size={completion.batch_size}",
                "-" * 48,
                f"point: {point_label(completion.best_point)}",
                f"lrs: {','.join(fmt(value) for value in completion.lrs)}",
                f"train_loss: {fmt(completion.train_loss)}",
                f"evaluated_points: {completion.evaluated_points}",
            ]
        )
        if row.run is not None:
            lines.extend(
                [
                    f"val_loss: {fmt(row.run.val_loss)}",
                    f"val_acc: {fmt(row.run.val_acc)}",
                    f"tta_val_acc: {fmt(row.run.tta_val_acc)}",
                ]
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(selected, output_path):
    max_steps = max(row.completion.train_steps for row in selected)
    headers = [
        "source",
        "train_steps",
        "batch_size",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "evaluated_points",
    ]
    headers += [f"k_step{i}" for i in range(1, max_steps + 1)]
    headers += [f"lr_step{i}" for i in range(1, max_steps + 1)]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in selected:
            completion = row.completion
            run = row.run
            point = list(completion.best_point)
            lrs = list(completion.lrs)
            values = [
                row.source,
                completion.train_steps,
                completion.batch_size,
                fmt(completion.train_loss),
                fmt(run.val_loss if run else None),
                fmt(run.train_acc if run else None),
                fmt(run.val_acc if run else None),
                fmt(run.tta_val_acc if run else None),
                completion.evaluated_points,
            ]
            values += point + [""] * (max_steps - len(point))
            values += [fmt(value) for value in lrs] + [""] * (max_steps - len(lrs))
            writer.writerow(values)


def parse_log_specs(items):
    specs = []
    for item in items:
        if "=" in item:
            source, path = item.split("=", 1)
        else:
            path = item
            source = Path(path).stem
        specs.append((source, Path(path)))
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval5 exp2 and exp3 together."
    )
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        help="Log spec as label=path. Defaults to exp2 and exp3 logs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    log_specs = parse_log_specs(args.log) if args.log else DEFAULT_LOGS
    datasets = load_logs(log_specs)
    selected = selected_completions(datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_selected_metrics(selected, args.output_dir / "selected_metrics.png")
    plot_selected_heatmaps(selected, args.output_dir / "selected_metric_heatmaps.png")
    plot_selected_schedules(datasets, selected, args.output_dir / "selected_schedule_shapes.png")
    plot_search_progress(datasets, args.output_dir / "search_progress.png")
    plot_metric_scatter(datasets, args.output_dir / "metric_scatter.png")
    write_summary(datasets, selected, args.output_dir / "summary.txt")
    write_csv(selected, args.output_dir / "selected_summary.csv")

    for data in datasets:
        print(
            f"{data['source']}: parsed {len(data['runs'])} runs, "
            f"{len(data['search_steps'])} search steps, "
            f"{len(data['cache_hits'])} cache hits"
        )
    for row in selected:
        completion = row.completion
        print(
            f"{row.source} steps={completion.train_steps} "
            f"bs={completion.batch_size} best={point_label(completion.best_point)} "
            f"train_loss={completion.train_loss:.4f}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

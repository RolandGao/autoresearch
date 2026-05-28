import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval5_exp3.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval5_exp3_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) muon_lr=(?P<muon_lr>\S+) "
    r"sgd_lr_mult=(?P<sgd_lr_mult>\S+) name=(?P<name>\S+) "
    r"best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
SCHEDULE_RE = re.compile(
    r"^applied_lr_schedule name=(?P<scheduler>\S+) point=(?P<point>\([^)]+\)) "
    r"lrs=(?P<lrs>\S+) values=(?P<values>\S+)"
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
SEARCH_STEP_RE = re.compile(
    r"^schedule_search step train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) move=(?P<move>\d+) "
    r"center=(?P<center>\([^)]+\)) center_train_loss=(?P<center_train_loss>\S+) "
    r"best=(?P<best>\([^)]+\)) best_train_loss=(?P<best_train_loss>\S+) "
    r"neighbors=(?P<neighbors>\d+)"
)
SEARCH_COMPLETE_RE = re.compile(
    r"^schedule_search complete train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) best_point=(?P<best_point>\([^)]+\)) "
    r"lrs=(?P<lrs>\S+) train_loss=(?P<train_loss>\S+) "
    r"evaluated_points=(?P<evaluated_points>\d+)"
)
SEARCH_CACHE_HIT_RE = re.compile(
    r"^schedule_search cache_hit train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) point=(?P<point>\([^)]+\)) "
    r"cache_key=(?P<cache_key>\([^)]+\)) lrs=(?P<lrs>\S+) "
    r"train_loss=(?P<train_loss>\S+)"
)


@dataclass
class Run:
    index: int
    train_steps: int
    batch_size: int
    muon_lr: float
    sgd_lr_mult: float
    name: str
    scheduler: str
    point: tuple[int, ...] | None = None
    lrs: tuple[float, ...] | None = None
    schedule: list[float] = field(default_factory=list)
    applied_lr: list[float] = field(default_factory=list)
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    tta_val_acc: float | None = None
    time_seconds: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None


@dataclass
class SearchStep:
    train_steps: int
    batch_size: int
    move: int
    center: tuple[int, ...]
    center_train_loss: float
    best: tuple[int, ...]
    best_train_loss: float
    neighbors: int


@dataclass
class SearchComplete:
    train_steps: int
    batch_size: int
    best_point: tuple[int, ...]
    lrs: tuple[float, ...]
    train_loss: float
    evaluated_points: int


def parse_tuple(text, cast):
    value = ast.literal_eval(text)
    if not isinstance(value, tuple):
        value = (value,)
    return tuple(cast(item) for item in value)


def parse_float_list(text):
    return [float(value) for value in text.split(",") if value]


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def point_label(point):
    return "(" + ", ".join(str(value) for value in point) + ")"


def group_key(row):
    return row.train_steps, row.batch_size


def parse_log(path):
    runs = []
    runs_by_name = {}
    search_steps = []
    completions = []
    cache_hits = []
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
                    sgd_lr_mult=float(match.group("sgd_lr_mult")),
                    name=match.group("name"),
                    scheduler=match.group("best_lr_scheduler"),
                )
                runs.append(current)
                runs_by_name[current.name] = current
                continue

            match = SCHEDULE_RE.match(line)
            if match and current is not None:
                current.point = parse_tuple(match.group("point"), int)
                current.lrs = tuple(parse_float_list(match.group("lrs")))
                current.schedule = parse_float_list(match.group("values"))
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

            match = SEARCH_STEP_RE.match(line)
            if match:
                search_steps.append(
                    SearchStep(
                        train_steps=int(match.group("train_steps")),
                        batch_size=int(match.group("batch_size")),
                        move=int(match.group("move")),
                        center=parse_tuple(match.group("center"), int),
                        center_train_loss=float(match.group("center_train_loss")),
                        best=parse_tuple(match.group("best"), int),
                        best_train_loss=float(match.group("best_train_loss")),
                        neighbors=int(match.group("neighbors")),
                    )
                )
                continue

            match = SEARCH_COMPLETE_RE.match(line)
            if match:
                completions.append(
                    SearchComplete(
                        train_steps=int(match.group("train_steps")),
                        batch_size=int(match.group("batch_size")),
                        best_point=parse_tuple(match.group("best_point"), int),
                        lrs=tuple(parse_float_list(match.group("lrs"))),
                        train_loss=float(match.group("train_loss")),
                        evaluated_points=int(match.group("evaluated_points")),
                    )
                )
                continue

            match = SEARCH_CACHE_HIT_RE.match(line)
            if match:
                cache_hits.append(
                    dict(
                        train_steps=int(match.group("train_steps")),
                        batch_size=int(match.group("batch_size")),
                        point=parse_tuple(match.group("point"), int),
                        lrs=tuple(parse_float_list(match.group("lrs"))),
                        train_loss=float(match.group("train_loss")),
                    )
                )

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs, search_steps, completions, cache_hits


def sorted_groups(rows):
    return sorted({group_key(row) for row in rows})


def runs_for_group(runs, key):
    return sorted([run for run in runs if group_key(run) == key], key=lambda run: run.index)


def top_runs(runs, key, n=6):
    rows = [run for run in runs_for_group(runs, key) if run.train_loss is not None]
    return sorted(rows, key=lambda run: run.train_loss)[:n]


def completion_for(completions, key):
    return next((row for row in completions if group_key(row) == key), None)


def run_for_completion(runs, completion):
    if completion is None:
        return None
    return next(
        (
            run
            for run in runs
            if group_key(run) == group_key(completion) and run.point == completion.best_point
        ),
        None,
    )


def make_grid(num_items, width=5.0, height=3.6):
    cols = min(3, max(1, num_items))
    rows = math.ceil(num_items / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in axes.ravel()[num_items:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_search_progress(search_steps, completions, output_path):
    groups = sorted_groups(search_steps)
    fig, axes = make_grid(len(groups), width=5.4, height=3.8)
    for ax, key in zip(axes, groups):
        steps = sorted(
            [step for step in search_steps if group_key(step) == key],
            key=lambda step: step.move,
        )
        moves = [step.move for step in steps]
        ax.plot(moves, [step.center_train_loss for step in steps], marker="o", label="center")
        ax.plot(moves, [step.best_train_loss for step in steps], marker="s", label="best")
        completion = completion_for(completions, key)
        if completion is not None:
            ax.axhline(
                completion.train_loss,
                color="tab:green",
                linestyle=":",
                label=f"selected {point_label(completion.best_point)}",
            )
        train_steps, batch_size = key
        ax.set_title(f"steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("search move")
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Schedule Search Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_schedules(runs, output_path):
    groups = sorted_groups(runs)
    fig, axes = make_grid(len(groups), width=5.4, height=3.8)
    for ax, key in zip(axes, groups):
        for rank, run in enumerate(top_runs(runs, key, n=6), start=1):
            steps = np.arange(1, len(run.schedule) + 1)
            ax.plot(
                steps,
                run.schedule,
                marker="o" if rank == 1 else None,
                linewidth=2.4 if rank == 1 else 1.2,
                alpha=1.0 if rank == 1 else 0.7,
                label=f"#{rank} loss={run.train_loss:.4f} {point_label(run.point)}",
            )
        train_steps, batch_size = key
        ax.set_title(f"steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("training step")
        ax.set_ylabel("applied Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.2)
    fig.suptitle("Top Applied LR Schedules by Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_rankings(runs, output_path):
    groups = sorted_groups(runs)
    fig, axes = make_grid(len(groups), width=5.4, height=3.8)
    for ax, key in zip(axes, groups):
        group_runs = runs_for_group(runs, key)
        train_loss = np.array([run.train_loss for run in group_runs], dtype=float)
        val_acc = np.array([run.val_acc for run in group_runs], dtype=float)
        tta_val_acc = np.array([run.tta_val_acc for run in group_runs], dtype=float)
        scatter = ax.scatter(
            train_loss,
            tta_val_acc,
            c=val_acc,
            cmap="viridis",
            s=46,
            edgecolor="black",
            linewidth=0.35,
        )
        best_train = min(group_runs, key=lambda run: run.train_loss)
        best_tta = max(group_runs, key=lambda run: run.tta_val_acc)
        ax.scatter(
            [best_train.train_loss],
            [best_train.tta_val_acc],
            marker="*",
            color="gold",
            edgecolor="black",
            s=190,
            label="best train loss",
        )
        ax.scatter(
            [best_tta.train_loss],
            [best_tta.tta_val_acc],
            marker="X",
            color="tab:red",
            edgecolor="black",
            s=90,
            label="best TTA acc",
        )
        train_steps, batch_size = key
        ax.set_title(f"steps={train_steps}, bs={batch_size}")
        ax.set_xlabel("train loss")
        ax.set_ylabel("TTA val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5)
        fig.colorbar(scatter, ax=ax, label="val acc")
    fig.suptitle("Train Loss vs Validation Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_by_train_steps(runs, completions, output_path):
    selected_by_group = {}
    for completion in sorted(completions, key=lambda row: group_key(row)):
        run = run_for_completion(runs, completion)
        if run is not None:
            selected_by_group[group_key(run)] = run
    if not selected_by_group:
        return

    batch_sizes = sorted({key[1] for key in selected_by_group})
    train_steps = sorted({key[0] for key in selected_by_group})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    ax_loss, ax_acc = axes.ravel()
    for batch_size in batch_sizes:
        rows = [
            selected_by_group[(step, batch_size)]
            for step in train_steps
            if (step, batch_size) in selected_by_group
        ]
        xs = [run.train_steps for run in rows]
        ax_loss.plot(
            xs,
            [run.train_loss for run in rows],
            marker="o",
            label=f"bs={batch_size} train",
        )
        ax_loss.plot(
            xs,
            [run.val_loss for run in rows],
            marker="s",
            linestyle="--",
            label=f"bs={batch_size} val",
        )
        ax_acc.plot(
            xs,
            [run.val_acc for run in rows],
            marker="^",
            label=f"bs={batch_size} val",
        )
        ax_acc.plot(
            xs,
            [run.tta_val_acc for run in rows],
            marker="D",
            linestyle="--",
            label=f"bs={batch_size} TTA",
        )

    ax_loss.set_title("Selected Loss")
    ax_loss.set_xlabel("train steps")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(fontsize=7, ncol=2)
    ax_acc.set_title("Selected Accuracy")
    ax_acc.set_xlabel("train steps")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(fontsize=7, ncol=2)
    fig.suptitle("Selected Schedule Metrics by Batch Size")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_sensitivity(runs, output_path):
    groups = sorted_groups(runs)
    max_steps = max(key[0] for key in groups)
    matrix = np.full((len(groups), max_steps), np.nan, dtype=float)
    ylabels = []

    for row, key in enumerate(groups):
        group_runs = [run for run in runs_for_group(runs, key) if run.point is not None]
        ylabels.append(f"{key[0]} steps, bs={key[1]}")
        for step_idx in range(key[0]):
            buckets = defaultdict(list)
            for run in group_runs:
                buckets[run.point[step_idx]].append(run.train_loss)
            means = sorted((np.mean(values), point) for point, values in buckets.items())
            if len(means) >= 2:
                matrix[row, step_idx] = means[1][0] - means[0][0]
            elif len(means) == 1:
                matrix[row, step_idx] = 0.0

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(max_steps), [str(i) for i in range(1, max_steps + 1)])
    ax.set_yticks(np.arange(len(groups)), ylabels)
    ax.set_xlabel("schedule coordinate")
    ax.set_ylabel("search group")
    ax.set_title("Train-Loss Sensitivity by Schedule Coordinate")
    fig.colorbar(im, ax=ax, label="mean loss gap between best and second-best k")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, search_steps, completions, cache_hits, output_path):
    lines = [
        "CIFAR Baseline Eval5 Exp3 Applied LR Schedule Search",
        "=" * 58,
        "",
        f"Evaluated runs: {len(runs)}",
        f"Search steps: {len(search_steps)}",
        f"Cache hits: {len(cache_hits)}",
        "",
    ]

    for completion in sorted(completions, key=lambda row: group_key(row)):
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

    for key in sorted_groups(runs):
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


def write_csv(runs, output_path):
    max_steps = max(run.train_steps for run in runs)
    headers = [
        "run",
        "train_steps",
        "batch_size",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
    ]
    headers += [f"k_step{i}" for i in range(1, max_steps + 1)]
    headers += [f"lr_step{i}" for i in range(1, max_steps + 1)]
    headers += ["schedule", "applied_lr"]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for run in sorted(runs, key=lambda row: (row.train_steps, row.batch_size, row.index)):
            point = list(run.point or ())
            lrs = list(run.lrs or ())
            row = [
                run.index,
                run.train_steps,
                run.batch_size,
                fmt(run.train_loss),
                fmt(run.val_loss),
                fmt(run.train_acc),
                fmt(run.val_acc),
                fmt(run.tta_val_acc),
                fmt(run.wall_time_seconds),
                fmt(run.cuda_time_seconds),
            ]
            row += point + [""] * (max_steps - len(point))
            row += [fmt(value) for value in lrs] + [""] * (max_steps - len(lrs))
            row += [
                ";".join(fmt(value) for value in run.schedule),
                ";".join(fmt(value) for value in run.applied_lr),
            ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval5_exp3 applied LR schedule search."
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

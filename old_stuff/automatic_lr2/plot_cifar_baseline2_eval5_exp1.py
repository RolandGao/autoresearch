import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval5_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval5_exp1_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
SCHEDULE_RE = re.compile(
    r"^applied_lr_schedule name=(?P<scheduler>\S+) point=(?P<point>\([^)]+\)) "
    r"step1_lr=(?P<step1_lr>\S+) step5_lr=(?P<step5_lr>\S+) "
    r"step10_lr=(?P<step10_lr>\S+) values=(?P<values>\S+)"
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
    r"^schedule_search step batch_size=(?P<batch_size>\d+) move=(?P<move>\d+) "
    r"center=(?P<center>\([^)]+\)) center_train_loss=(?P<center_train_loss>\S+) "
    r"best=(?P<best>\([^)]+\)) best_train_loss=(?P<best_train_loss>\S+) "
    r"neighbors=(?P<neighbors>\d+)"
)
SEARCH_COMPLETE_RE = re.compile(
    r"^schedule_search complete batch_size=(?P<batch_size>\d+) "
    r"best_point=(?P<best_point>\([^)]+\)) "
    r"step1_lr=(?P<step1_lr>\S+) step5_lr=(?P<step5_lr>\S+) "
    r"step10_lr=(?P<step10_lr>\S+) train_loss=(?P<train_loss>\S+) "
    r"evaluated_points=(?P<evaluated_points>\d+)"
)
SEARCH_CACHE_HIT_RE = re.compile(
    r"^schedule_search cache_hit batch_size=(?P<batch_size>\d+) "
    r"point=(?P<point>\([^)]+\)) cache_key=(?P<cache_key>\([^)]+\)) "
    r"step1_lr=(?P<step1_lr>\S+) step5_lr=(?P<step5_lr>\S+) "
    r"step10_lr=(?P<step10_lr>\S+) train_loss=(?P<train_loss>\S+)"
)


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    sgd_lr_mult: float
    name: str
    scheduler: str
    point: tuple[int, int, int] | None = None
    step_lrs: tuple[float, float, float] | None = None
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
    batch_size: int
    move: int
    center: tuple[int, int, int]
    center_train_loss: float
    best: tuple[int, int, int]
    best_train_loss: float
    neighbors: int


@dataclass
class SearchComplete:
    batch_size: int
    best_point: tuple[int, int, int]
    step_lrs: tuple[float, float, float]
    train_loss: float
    evaluated_points: int


def parse_point(text):
    return tuple(int(value) for value in ast.literal_eval(text))


def parse_float_tuple(text):
    return tuple(float(value) for value in ast.literal_eval(text))


def parse_float_list(text):
    return [float(value) for value in text.split(",") if value]


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def point_label(point):
    return f"({point[0]}, {point[1]}, {point[2]})"


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
                current.point = parse_point(match.group("point"))
                current.step_lrs = (
                    float(match.group("step1_lr")),
                    float(match.group("step5_lr")),
                    float(match.group("step10_lr")),
                )
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
                        batch_size=int(match.group("batch_size")),
                        move=int(match.group("move")),
                        center=parse_point(match.group("center")),
                        center_train_loss=float(match.group("center_train_loss")),
                        best=parse_point(match.group("best")),
                        best_train_loss=float(match.group("best_train_loss")),
                        neighbors=int(match.group("neighbors")),
                    )
                )
                continue

            match = SEARCH_COMPLETE_RE.match(line)
            if match:
                completions.append(
                    SearchComplete(
                        batch_size=int(match.group("batch_size")),
                        best_point=parse_point(match.group("best_point")),
                        step_lrs=(
                            float(match.group("step1_lr")),
                            float(match.group("step5_lr")),
                            float(match.group("step10_lr")),
                        ),
                        train_loss=float(match.group("train_loss")),
                        evaluated_points=int(match.group("evaluated_points")),
                    )
                )
                continue

            match = SEARCH_CACHE_HIT_RE.match(line)
            if match:
                cache_hits.append(
                    dict(
                        batch_size=int(match.group("batch_size")),
                        point=parse_point(match.group("point")),
                        cache_key=parse_float_tuple(match.group("cache_key")),
                        train_loss=float(match.group("train_loss")),
                    )
                )
                continue

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs, search_steps, completions, cache_hits


def batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def runs_for_batch(runs, batch_size):
    return sorted([run for run in runs if run.batch_size == batch_size], key=lambda run: run.index)


def top_runs(runs, batch_size, n=8):
    rows = [run for run in runs_for_batch(runs, batch_size) if run.train_loss is not None]
    return sorted(rows, key=lambda run: run.train_loss)[:n]


def completion_for(completions, batch_size):
    return next((row for row in completions if row.batch_size == batch_size), None)


def plot_search_progress(search_steps, completions, output_path):
    sizes = sorted({step.batch_size for step in search_steps})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.7 * len(sizes), 4.5), squeeze=False)
    for ax, batch_size in zip(axes.ravel(), sizes):
        steps = sorted(
            [step for step in search_steps if step.batch_size == batch_size],
            key=lambda step: step.move,
        )
        moves = [step.move for step in steps]
        center = [step.center_train_loss for step in steps]
        best = [step.best_train_loss for step in steps]
        ax.plot(moves, center, marker="o", label="center")
        ax.plot(moves, best, marker="s", label="best candidate")
        for step in steps:
            ax.annotate(
                f"{point_label(step.center)}->{point_label(step.best)}",
                (step.move, step.best_train_loss),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.2,
            )
        completion = completion_for(completions, batch_size)
        if completion is not None:
            ax.axhline(
                completion.train_loss,
                color="tab:green",
                linestyle=":",
                label=f"selected {point_label(completion.best_point)}",
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("search move")
        ax.set_ylabel("final train loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Neighborhood Search Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_search_3d(runs, completions, output_path):
    sizes = batch_sizes(runs)
    fig = plt.figure(figsize=(5.8 * len(sizes), 5.2))
    for col, batch_size in enumerate(sizes, start=1):
        ax = fig.add_subplot(1, len(sizes), col, projection="3d")
        batch_runs = [run for run in runs_for_batch(runs, batch_size) if run.point is not None]
        xs = np.array([run.point[0] for run in batch_runs], dtype=float)
        ys = np.array([run.point[1] for run in batch_runs], dtype=float)
        zs = np.array([run.point[2] for run in batch_runs], dtype=float)
        vals = np.array([run.train_loss for run in batch_runs], dtype=float)
        scatter = ax.scatter(xs, ys, zs, c=vals, cmap="magma_r", s=65, depthshade=False)
        completion = completion_for(completions, batch_size)
        if completion is not None:
            ax.scatter(
                [completion.best_point[0]],
                [completion.best_point[1]],
                [completion.best_point[2]],
                marker="*",
                s=230,
                color="gold",
                edgecolor="black",
                linewidth=0.8,
                label="selected",
            )
            ax.legend(fontsize=7)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("step 1 k")
        ax.set_ylabel("step 5 k")
        ax.set_zlabel("step 10 k")
        fig.colorbar(scatter, ax=ax, shrink=0.65, label="train loss")
    fig.suptitle("Sampled Train Loss in k-Space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pair_projections(runs, completions, output_path):
    sizes = batch_sizes(runs)
    pairs = [
        ("step 1 k", "step 5 k", 0, 1),
        ("step 1 k", "step 10 k", 0, 2),
        ("step 5 k", "step 10 k", 1, 2),
    ]
    fig, axes = plt.subplots(
        len(pairs), len(sizes), figsize=(5.2 * len(sizes), 11), squeeze=False
    )
    for col, batch_size in enumerate(sizes):
        batch_runs = [run for run in runs_for_batch(runs, batch_size) if run.point is not None]
        vals = np.array([run.train_loss for run in batch_runs], dtype=float)
        completion = completion_for(completions, batch_size)
        for row, (xlabel, ylabel, xi, yi) in enumerate(pairs):
            ax = axes[row, col]
            xs = np.array([run.point[xi] for run in batch_runs], dtype=float)
            ys = np.array([run.point[yi] for run in batch_runs], dtype=float)
            scatter = ax.scatter(
                xs, ys, c=vals, cmap="magma_r", s=105, edgecolor="black", linewidth=0.45
            )
            if completion is not None:
                ax.scatter(
                    [completion.best_point[xi]],
                    [completion.best_point[yi]],
                    marker="*",
                    s=320,
                    color="gold",
                    edgecolor="black",
                    linewidth=0.8,
                )
            ax.set_title(f"bs={batch_size}" if row == 0 else "")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            fig.colorbar(scatter, ax=ax, label="train loss")
    fig.suptitle("Pairwise Search Projections")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_schedules(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5), squeeze=False)
    steps = np.arange(1, 11)
    for ax, batch_size in zip(axes.ravel(), sizes):
        for rank, run in enumerate(top_runs(runs, batch_size, n=8), start=1):
            label = (
                f"#{rank} {point_label(run.point)} "
                f"loss={run.train_loss:.4f}"
            )
            ax.plot(
                steps,
                run.schedule,
                marker="o" if rank == 1 else None,
                linewidth=2.5 if rank == 1 else 1.25,
                alpha=1.0 if rank == 1 else 0.72,
                label=label,
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("step")
        ax.set_ylabel("applied Muon LR")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5)
    fig.suptitle("Top Applied LR Schedules by Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_rankings(runs, output_path):
    sizes = batch_sizes(runs)
    metrics = [
        ("train loss", "train_loss", False),
        ("val loss", "val_loss", False),
        ("val acc", "val_acc", True),
        ("TTA val acc", "tta_val_acc", True),
    ]
    fig, axes = plt.subplots(
        len(metrics), len(sizes), figsize=(5.8 * len(sizes), 12), squeeze=False
    )
    for col, batch_size in enumerate(sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        xs = [run.index for run in batch_runs]
        for row, (label, attr, higher_is_better) in enumerate(metrics):
            ax = axes[row, col]
            ys = [getattr(run, attr) for run in batch_runs]
            ax.plot(xs, ys, marker=".", linewidth=1.1)
            valid = [run for run in batch_runs if getattr(run, attr) is not None]
            if valid:
                best = (
                    max(valid, key=lambda run: getattr(run, attr))
                    if higher_is_better
                    else min(valid, key=lambda run: getattr(run, attr))
                )
                ax.scatter([best.index], [getattr(best, attr)], s=80, color="tab:red", zorder=5)
            ax.set_title(f"bs={batch_size}" if row == 0 else "")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
        axes[-1, col].set_xlabel("run index")
    fig.suptitle("Final Metrics by Evaluation Order")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(runs, search_steps, completions, cache_hits, output_path):
    lines = [
        "CIFAR Baseline Eval5 Exp1 Applied LR Schedule Search",
        "=" * 58,
        "",
        f"Evaluated runs: {len(runs)}",
        f"Search steps: {len(search_steps)}",
        f"Cache hits: {len(cache_hits)}",
        "",
    ]

    for completion in sorted(completions, key=lambda row: row.batch_size):
        lines.extend(
            [
                f"batch_size={completion.batch_size} selected best",
                "-" * 40,
                f"point: {point_label(completion.best_point)}",
                f"step1_lr: {fmt(completion.step_lrs[0])}",
                f"step5_lr: {fmt(completion.step_lrs[1])}",
                f"step10_lr: {fmt(completion.step_lrs[2])}",
                f"train_loss: {fmt(completion.train_loss)}",
                f"evaluated_points: {completion.evaluated_points}",
                "",
            ]
        )

    for batch_size in batch_sizes(runs):
        lines.extend([f"Ranking by train loss, batch_size={batch_size}", "-" * 40])
        for rank, run in enumerate(top_runs(runs, batch_size, n=20), start=1):
            lines.append(
                f"{rank}. r{run.index} point={point_label(run.point)} "
                f"step_lrs=({fmt(run.step_lrs[0])}, {fmt(run.step_lrs[1])}, "
                f"{fmt(run.step_lrs[2])}) train_loss={fmt(run.train_loss)} "
                f"train_acc={fmt(run.train_acc)} val_loss={fmt(run.val_loss)} "
                f"val_acc={fmt(run.val_acc)} tta_val_acc={fmt(run.tta_val_acc)} "
                f"schedule={','.join(fmt(value) for value in run.schedule)}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    headers = [
        "run",
        "batch_size",
        "k_step1",
        "k_step5",
        "k_step10",
        "step1_lr",
        "step5_lr",
        "step10_lr",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_time_seconds",
        "cuda_time_seconds",
        "schedule",
        "applied_lr",
    ]
    lines = [",".join(headers)]
    for run in sorted(runs, key=lambda row: (row.batch_size, row.index)):
        row = [
            run.index,
            run.batch_size,
            run.point[0],
            run.point[1],
            run.point[2],
            fmt(run.step_lrs[0]),
            fmt(run.step_lrs[1]),
            fmt(run.step_lrs[2]),
            fmt(run.train_loss),
            fmt(run.val_loss),
            fmt(run.train_acc),
            fmt(run.val_acc),
            fmt(run.tta_val_acc),
            fmt(run.wall_time_seconds),
            fmt(run.cuda_time_seconds),
            ";".join(fmt(value) for value in run.schedule),
            ";".join(fmt(value) for value in run.applied_lr),
        ]
        lines.append(",".join(str(value) for value in row))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval5 exp1 applied LR schedule search."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs, search_steps, completions, cache_hits = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_search_progress(search_steps, completions, args.output_dir / "search_progress.png")
    plot_search_3d(runs, completions, args.output_dir / "search_3d.png")
    plot_pair_projections(runs, completions, args.output_dir / "pair_projections.png")
    plot_top_schedules(runs, args.output_dir / "top_schedule_shapes.png")
    plot_metric_rankings(runs, args.output_dir / "metric_rankings.png")
    write_summary(runs, search_steps, completions, cache_hits, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Parsed {len(search_steps)} search steps and {len(cache_hits)} cache hits")
    for completion in sorted(completions, key=lambda row: row.batch_size):
        print(
            f"bs={completion.batch_size} best={point_label(completion.best_point)} "
            f"step_lrs=({completion.step_lrs[0]:.6g}, "
            f"{completion.step_lrs[1]:.6g}, {completion.step_lrs[2]:.6g}) "
            f"train_loss={completion.train_loss:.4f} "
            f"evaluated={completion.evaluated_points}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

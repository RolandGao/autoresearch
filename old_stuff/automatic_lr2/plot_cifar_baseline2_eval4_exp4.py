import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval4_exp4.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval4_exp4_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
SCHEDULE_RE = re.compile(
    r"^lr_mult_schedule name=(?P<scheduler>\S+) point=(?P<point>\([^)]+\)) "
    r"multipliers=(?P<multipliers>\S+) values=(?P<values>\S+)"
)
BEST_LR_RE = re.compile(
    r"^best_lr step=(?P<step>\d+) init_lr=(?P<init_lr>\S+) "
    r"best_lr=(?P<best_lr>\S+) best_lr_ema=(?P<best_lr_ema>\S+) "
    r"best_loss=(?P<best_loss>\S+)"
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
    r"best_point=(?P<best_point>\([^)]+\)) multipliers=(?P<multipliers>\S+) "
    r"train_loss=(?P<train_loss>\S+) evaluated_points=(?P<evaluated_points>\d+)"
)
SEARCH_CACHE_HIT_RE = re.compile(
    r"^schedule_search cache_hit batch_size=(?P<batch_size>\d+) "
    r"point=(?P<point>\([^)]+\)) cache_key=(?P<cache_key>\([^)]+\)) "
    r"multipliers=(?P<multipliers>\S+) train_loss=(?P<train_loss>\S+)"
)


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    sgd_lr_mult: float
    name: str
    scheduler: str
    point: tuple[int, ...] | None = None
    multipliers: list[float] = field(default_factory=list)
    schedule: list[float] = field(default_factory=list)
    best_lr: list[float] = field(default_factory=list)
    best_loss: list[float] = field(default_factory=list)
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
    center: tuple[int, ...]
    center_train_loss: float
    best: tuple[int, ...]
    best_train_loss: float
    neighbors: int


@dataclass
class SearchComplete:
    batch_size: int
    best_point: tuple[int, ...]
    multipliers: list[float]
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
    return "(" + ", ".join(str(value) for value in point) + ")"


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
                current.multipliers = parse_float_list(match.group("multipliers"))
                current.schedule = parse_float_list(match.group("values"))
                continue

            match = BEST_LR_RE.match(line)
            if match and current is not None:
                current.best_lr.append(float(match.group("best_lr")))
                current.best_loss.append(float(match.group("best_loss")))
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
                        multipliers=parse_float_list(match.group("multipliers")),
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
                        multipliers=parse_float_list(match.group("multipliers")),
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


def point_matrix(runs):
    return np.array([run.point for run in runs if run.point is not None], dtype=float)


def train_loss_vector(runs):
    return np.array([run.train_loss for run in runs if run.point is not None], dtype=float)


def pca2(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def plot_search_progress(search_steps, completions, output_path):
    sizes = sorted({step.batch_size for step in search_steps})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 4.6), squeeze=False)
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
        completion = completion_for(completions, batch_size)
        if completion is not None:
            ax.axhline(
                completion.train_loss,
                color="tab:green",
                linestyle=":",
                label=f"selected move={steps[-1].move if steps else 'NA'}",
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("search move")
        ax.set_ylabel("final train loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Neighborhood Search Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca_projection(runs, completions, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5), squeeze=False)
    for ax, batch_size in zip(axes.ravel(), sizes):
        batch_runs = [run for run in runs_for_batch(runs, batch_size) if run.point is not None]
        points = point_matrix(batch_runs)
        coords = pca2(points)
        vals = train_loss_vector(batch_runs)
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            cmap="magma_r",
            s=80,
            edgecolor="black",
            linewidth=0.4,
        )
        completion = completion_for(completions, batch_size)
        if completion is not None:
            all_points = point_matrix(batch_runs)
            idx = np.where((all_points == np.array(completion.best_point)).all(axis=1))[0]
            if len(idx):
                ax.scatter(
                    [coords[idx[0], 0]],
                    [coords[idx[0], 1]],
                    marker="*",
                    s=340,
                    color="gold",
                    edgecolor="black",
                    linewidth=0.8,
                    label="selected",
                )
                ax.legend(fontsize=8)
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("PC1 of 9-step k vector")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="train loss")
    fig.suptitle("9D Search Samples Projected with PCA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_sensitivity(runs, completions, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(len(sizes), 9, figsize=(22, 3.2 * len(sizes)), squeeze=False)
    for row, batch_size in enumerate(sizes):
        batch_runs = [run for run in runs_for_batch(runs, batch_size) if run.point is not None]
        vals = np.array([run.train_loss for run in batch_runs], dtype=float)
        completion = completion_for(completions, batch_size)
        for step_idx in range(9):
            ax = axes[row, step_idx]
            xs = np.array([run.point[step_idx] for run in batch_runs], dtype=float)
            ax.scatter(xs, vals, s=28, alpha=0.65)
            if completion is not None:
                ax.axvline(completion.best_point[step_idx], color="tab:red", linestyle=":")
                ax.axhline(completion.train_loss, color="tab:green", linestyle=":", alpha=0.7)
            ax.set_title(f"step {step_idx + 1}" if row == 0 else "")
            if step_idx == 0:
                ax.set_ylabel(f"bs={batch_size}\ntrain loss")
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.2)
    fig.suptitle("One-Dimensional Views of 9D Search")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_multiplier_heatmap(completions, output_path):
    rows = sorted(completions, key=lambda row: row.batch_size)
    data = np.array([row.multipliers for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"s{i}" for i in range(1, data.shape[1] + 1)])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"bs={row.batch_size}" for row in rows])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt(data[i, j]), ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, label="selected LR multiplier")
    ax.set_title("Selected Multipliers for Steps 1-9")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_schedules(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.8 * len(sizes), 5), squeeze=False)
    steps = np.arange(1, 11)
    for ax, batch_size in zip(axes.ravel(), sizes):
        for rank, run in enumerate(top_runs(runs, batch_size, n=8), start=1):
            label = f"#{rank} loss={run.train_loss:.4f} {point_label(run.point)}"
            ax.plot(
                steps,
                run.schedule,
                marker="o" if rank == 1 else None,
                linewidth=2.5 if rank == 1 else 1.15,
                alpha=1.0 if rank == 1 else 0.72,
                label=label,
            )
        ax.set_title(f"bs={batch_size}")
        ax.set_xlabel("step")
        ax.set_ylabel("LR multiplier")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=5.8)
    fig.suptitle("Top Multiplier Schedules by Train Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_traces(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(2, len(sizes), figsize=(5.8 * len(sizes), 8), squeeze=False)
    steps = np.arange(1, 11)
    for col, batch_size in enumerate(sizes):
        for run in top_runs(runs, batch_size, n=5):
            label = f"r{run.index} loss={run.train_loss:.4f}"
            axes[0, col].plot(steps[: len(run.best_lr)], run.best_lr, marker=".", label=label)
            axes[1, col].plot(steps[: len(run.applied_lr)], run.applied_lr, marker=".", label=label)
        axes[0, col].set_title(f"bs={batch_size} searched LR")
        axes[1, col].set_title(f"bs={batch_size} applied LR")
        for ax in axes[:, col]:
            ax.set_xlabel("step")
            ax.set_ylabel("Muon LR")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6.2)
    fig.suptitle("LR Traces for Top Schedules")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_schedules_and_lr_traces(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(3, len(sizes), figsize=(5.8 * len(sizes), 11.5), squeeze=False)
    steps = np.arange(1, 11)
    for col, batch_size in enumerate(sizes):
        batch_top_runs = top_runs(runs, batch_size, n=5)
        for rank, run in enumerate(batch_top_runs, start=1):
            label = f"#{rank} r{run.index} loss={run.train_loss:.4f}"
            style = dict(
                marker="o" if rank == 1 else None,
                linewidth=2.5 if rank == 1 else 1.25,
                alpha=1.0 if rank == 1 else 0.72,
                label=label,
            )
            axes[0, col].plot(steps[: len(run.schedule)], run.schedule, **style)
            axes[1, col].plot(steps[: len(run.best_lr)], run.best_lr, **style)
            axes[2, col].plot(steps[: len(run.applied_lr)], run.applied_lr, **style)

        axes[0, col].set_title(f"bs={batch_size} multiplier schedule")
        axes[1, col].set_title(f"bs={batch_size} searched LR")
        axes[2, col].set_title(f"bs={batch_size} applied LR")
        axes[0, col].set_ylabel("LR multiplier")
        axes[1, col].set_ylabel("Muon LR")
        axes[2, col].set_ylabel("Muon LR")
        for ax in axes[:, col]:
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6.0)
    fig.suptitle("Top Schedules and LR Traces by Train Loss")
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
    fig, axes = plt.subplots(len(metrics), len(sizes), figsize=(5.8 * len(sizes), 12), squeeze=False)
    for col, batch_size in enumerate(sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        xs = [run.index for run in batch_runs]
        for row, (label, attr, higher_is_better) in enumerate(metrics):
            ax = axes[row, col]
            ys = [getattr(run, attr) for run in batch_runs]
            ax.plot(xs, ys, marker=".", linewidth=1.0)
            valid = [run for run in batch_runs if getattr(run, attr) is not None]
            if valid:
                best = max(valid, key=lambda run: getattr(run, attr)) if higher_is_better else min(valid, key=lambda run: getattr(run, attr))
                ax.scatter([best.index], [getattr(best, attr)], s=75, color="tab:red", zorder=5)
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
        "CIFAR Baseline Eval4 Exp4 9-Step Multiplier Search",
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
                f"multipliers: {','.join(fmt(value) for value in completion.multipliers)}",
                f"train_loss: {fmt(completion.train_loss)}",
                f"evaluated_points: {completion.evaluated_points}",
                "",
            ]
        )

    for batch_size in batch_sizes(runs):
        lines.extend([f"Ranking by train loss, batch_size={batch_size}", "-" * 42])
        for rank, run in enumerate(top_runs(runs, batch_size, n=25), start=1):
            lines.append(
                f"{rank}. r{run.index} point={point_label(run.point)} "
                f"train_loss={fmt(run.train_loss)} train_acc={fmt(run.train_acc)} "
                f"val_loss={fmt(run.val_loss)} val_acc={fmt(run.val_acc)} "
                f"tta_val_acc={fmt(run.tta_val_acc)} "
                f"multipliers={','.join(fmt(value) for value in run.multipliers)} "
                f"schedule={','.join(fmt(value) for value in run.schedule)}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    headers = (
        ["run", "batch_size"]
        + [f"k_step{i}" for i in range(1, 10)]
        + [f"mult_step{i}" for i in range(1, 10)]
        + [
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "tta_val_acc",
            "wall_time_seconds",
            "cuda_time_seconds",
            "schedule",
            "best_lr",
            "applied_lr",
        ]
    )
    lines = [",".join(headers)]
    for run in sorted(runs, key=lambda row: (row.batch_size, row.index)):
        row = (
            [run.index, run.batch_size]
            + list(run.point)
            + [fmt(value) for value in run.multipliers]
            + [
                fmt(run.train_loss),
                fmt(run.val_loss),
                fmt(run.train_acc),
                fmt(run.val_acc),
                fmt(run.tta_val_acc),
                fmt(run.wall_time_seconds),
                fmt(run.cuda_time_seconds),
                ";".join(fmt(value) for value in run.schedule),
                ";".join(fmt(value) for value in run.best_lr),
                ";".join(fmt(value) for value in run.applied_lr),
            ]
        )
        lines.append(",".join(str(value) for value in row))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval4 exp4 9-step multiplier search."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs, search_steps, completions, cache_hits = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_search_progress(search_steps, completions, args.output_dir / "search_progress.png")
    plot_pca_projection(runs, completions, args.output_dir / "search_pca.png")
    plot_step_sensitivity(runs, completions, args.output_dir / "step_sensitivity.png")
    plot_best_multiplier_heatmap(completions, args.output_dir / "selected_multiplier_heatmap.png")
    plot_top_schedules(runs, args.output_dir / "top_schedule_shapes.png")
    plot_lr_traces(runs, args.output_dir / "top_lr_traces.png")
    plot_top_schedules_and_lr_traces(
        runs, args.output_dir / "top_schedules_and_lr_traces.png"
    )
    plot_metric_rankings(runs, args.output_dir / "metric_rankings.png")
    write_summary(runs, search_steps, completions, cache_hits, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Parsed {len(search_steps)} search steps and {len(cache_hits)} cache hits")
    for completion in sorted(completions, key=lambda row: row.batch_size):
        print(
            f"bs={completion.batch_size} best={point_label(completion.best_point)} "
            f"train_loss={completion.train_loss:.4f} "
            f"evaluated={completion.evaluated_points} "
            f"multipliers={','.join(fmt(value) for value in completion.multipliers)}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

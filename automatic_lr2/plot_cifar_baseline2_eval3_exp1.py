import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval3_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval3_exp1_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) best_lr_strategy=(?P<best_lr_strategy>\S+) "
    r"best_lr_linear_decay=(?P<best_lr_linear_decay>\S+) "
    r"best_lr_scheduler=(?P<best_lr_scheduler>\S+)"
)
NAME_POINT_RE = re.compile(r"_mu(?P<mu_exp>[pm]?\d+)_sgd(?P<sgd_exp>[pm]?\d+)_")
EVAL_RE = re.compile(
    r"^eval(?: run=(?P<run>\S+))? epoch=(?P<epoch>\d+) "
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
SEARCH_STEP_RE = re.compile(
    r"^joint_line_search step batch_size=(?P<batch_size>\d+) move=(?P<move>\d+) "
    r"center=\((?P<center_mu>-?\d+), (?P<center_sgd>-?\d+)\) "
    r"center_tta=(?P<center_tta>\S+) "
    r"best=\((?P<best_mu>-?\d+), (?P<best_sgd>-?\d+)\) "
    r"best_tta=(?P<best_tta>\S+)"
)
SEARCH_COMPLETE_RE = re.compile(
    r"^joint_line_search complete batch_size=(?P<batch_size>\d+) "
    r"best_point=\((?P<best_mu>-?\d+), (?P<best_sgd>-?\d+)\) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"tta_val_acc=(?P<tta_val_acc>\S+) evaluated_points=(?P<evaluated_points>\d+)"
)
CACHE_HIT_RE = re.compile(
    r"^joint_line_search cache_hit batch_size=(?P<batch_size>\d+) "
    r"point=\((?P<mu_exp>-?\d+), (?P<sgd_exp>-?\d+)\) "
    r"muon_lr=(?P<muon_lr>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"tta_val_acc=(?P<tta_val_acc>\S+)"
)


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
    sgd_lr_mult: float
    name: str
    point: tuple[int, int]
    eval_logs: list[EvalLog] = field(default_factory=list)
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
    center: tuple[int, int]
    center_tta: float
    best: tuple[int, int]
    best_tta: float


@dataclass
class SearchComplete:
    batch_size: int
    best_point: tuple[int, int]
    muon_lr: float
    sgd_lr_mult: float
    tta_val_acc: float
    evaluated_points: int


def parse_exp_label(value):
    if value == "0":
        return 0
    if value.startswith("p"):
        return int(value[1:])
    if value.startswith("m"):
        return -int(value[1:])
    return int(value)


def point_from_name(name):
    match = NAME_POINT_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse exponent point from run name: {name}")
    return (
        parse_exp_label(match.group("mu_exp")),
        parse_exp_label(match.group("sgd_exp")),
    )


def parse_log(path):
    runs = []
    runs_by_name = {}
    search_steps = []
    completions = []
    cache_hits = []
    current = None

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
                    sgd_lr_mult=float(match.group("sgd_lr_mult")),
                    name=match.group("name"),
                    point=point_from_name(match.group("name")),
                )
                runs.append(current)
                runs_by_name[current.name] = current
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
                        center=(
                            int(match.group("center_mu")),
                            int(match.group("center_sgd")),
                        ),
                        center_tta=float(match.group("center_tta")),
                        best=(
                            int(match.group("best_mu")),
                            int(match.group("best_sgd")),
                        ),
                        best_tta=float(match.group("best_tta")),
                    )
                )
                continue

            match = SEARCH_COMPLETE_RE.match(line)
            if match:
                completions.append(
                    SearchComplete(
                        batch_size=int(match.group("batch_size")),
                        best_point=(
                            int(match.group("best_mu")),
                            int(match.group("best_sgd")),
                        ),
                        muon_lr=float(match.group("muon_lr")),
                        sgd_lr_mult=float(match.group("sgd_lr_mult")),
                        tta_val_acc=float(match.group("tta_val_acc")),
                        evaluated_points=int(match.group("evaluated_points")),
                    )
                )
                continue

            match = CACHE_HIT_RE.match(line)
            if match:
                cache_hits.append(
                    dict(
                        batch_size=int(match.group("batch_size")),
                        point=(
                            int(match.group("mu_exp")),
                            int(match.group("sgd_exp")),
                        ),
                        muon_lr=float(match.group("muon_lr")),
                        sgd_lr_mult=float(match.group("sgd_lr_mult")),
                        tta_val_acc=float(match.group("tta_val_acc")),
                    )
                )
                continue

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs, search_steps, completions, cache_hits


def batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def runs_for_batch(runs, batch_size):
    return sorted(
        [run for run in runs if run.batch_size == batch_size],
        key=lambda run: run.index,
    )


def metric_value(run, name):
    return getattr(run, name)


def finite(value):
    return value is not None and math.isfinite(value)


def point_label(point):
    return f"({point[0]}, {point[1]})"


def plot_search_surface(runs, search_steps, completions, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(7.2 * len(sizes), 6.2), squeeze=False
    )

    for ax, batch_size in zip(axes.ravel(), sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        xs = np.array([run.point[0] for run in batch_runs], dtype=float)
        ys = np.array([run.point[1] for run in batch_runs], dtype=float)
        vals = np.array([run.tta_val_acc for run in batch_runs], dtype=float)
        scatter = ax.scatter(
            xs,
            ys,
            c=vals,
            s=150,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        for run in batch_runs:
            ax.annotate(
                f"{run.index}\n{run.tta_val_acc:.4f}",
                run.point,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if run.tta_val_acc < np.nanmean(vals) else "black",
                zorder=4,
            )

        batch_steps = [step for step in search_steps if step.batch_size == batch_size]
        if batch_steps:
            path = [batch_steps[0].center] + [step.best for step in batch_steps]
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            ax.plot(path_x, path_y, color="tab:red", linewidth=2.0, zorder=2)
            for start, end in zip(path, path[1:]):
                if start == end:
                    continue
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0),
                    zorder=5,
                )

        completion = next(
            (row for row in completions if row.batch_size == batch_size), None
        )
        if completion is not None:
            ax.scatter(
                [completion.best_point[0]],
                [completion.best_point[1]],
                marker="*",
                s=420,
                color="gold",
                edgecolor="black",
                linewidth=1.0,
                zorder=6,
                label="selected best",
            )
            ax.legend(loc="best")

        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("Muon LR exponent, lr = 0.19 * 0.8^x")
        ax.set_ylabel("SGD LR multiplier exponent, mult = 1.0 * 0.8^y")
        ax.set_xticks(sorted(set(int(x) for x in xs)))
        ax.set_yticks(sorted(set(int(y) for y in ys)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="TTA val acc")

    fig.suptitle("Joint Line Search Samples and Path")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_heatmaps(runs, output_path, metric="tta_val_acc"):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(7.2 * len(sizes), 6.2), squeeze=False
    )

    for ax, batch_size in zip(axes.ravel(), sizes):
        batch_runs = runs_for_batch(runs, batch_size)
        x_values = sorted({run.point[0] for run in batch_runs})
        y_values = sorted({run.point[1] for run in batch_runs})
        x_index = {value: index for index, value in enumerate(x_values)}
        y_index = {value: index for index, value in enumerate(y_values)}
        grid = np.full((len(y_values), len(x_values)), np.nan)
        for run in batch_runs:
            grid[y_index[run.point[1]], x_index[run.point[0]]] = metric_value(
                run, metric
            )

        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        for run in batch_runs:
            value = metric_value(run, metric)
            ax.text(
                x_index[run.point[0]],
                y_index[run.point[1]],
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < np.nanmean(grid) else "black",
            )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("Muon exponent")
        ax.set_ylabel("SGD multiplier exponent")
        ax.set_xticks(range(len(x_values)), x_values)
        ax.set_yticks(range(len(y_values)), y_values)
        fig.colorbar(image, ax=ax, label=metric)

    fig.suptitle("Sampled TTA Validation Accuracy Surface")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_search_progress(search_steps, completions, output_path):
    sizes = sorted({step.batch_size for step in search_steps})
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(6.8 * len(sizes), 4.8), squeeze=False
    )

    for ax, batch_size in zip(axes.ravel(), sizes):
        steps = sorted(
            [step for step in search_steps if step.batch_size == batch_size],
            key=lambda step: step.move,
        )
        moves = [step.move for step in steps]
        center = [step.center_tta for step in steps]
        best = [step.best_tta for step in steps]
        ax.plot(moves, center, marker="o", label="center")
        ax.plot(moves, best, marker="s", label="best neighbor/center")
        for step in steps:
            ax.annotate(
                f"{point_label(step.center)} -> {point_label(step.best)}",
                (step.move, step.best_tta),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        completion = next(
            (row for row in completions if row.batch_size == batch_size), None
        )
        if completion is not None:
            ax.axhline(
                completion.tta_val_acc,
                color="tab:green",
                linestyle=":",
                label=f"selected {point_label(completion.best_point)}",
            )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("line-search move")
        ax.set_ylabel("TTA val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Line Search Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(runs, output_path):
    ordered = sorted(runs, key=lambda run: (run.batch_size, run.index))
    labels = [
        f"r{run.index}\nbs{run.batch_size}\n{point_label(run.point)}"
        for run in ordered
    ]
    x = np.arange(len(ordered))
    metrics = [
        ("TTA val acc", "tta_val_acc"),
        ("val acc", "val_acc"),
        ("train acc", "train_acc"),
        ("val loss", "val_loss"),
        ("train loss", "train_loss"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 13), sharex=True)
    for ax, (label, attr) in zip(axes, metrics):
        values = [
            metric_value(run, attr) if finite(metric_value(run, attr)) else np.nan
            for run in ordered
        ]
        colors = [
            "tab:blue" if run.batch_size == ordered[0].batch_size else "tab:orange"
            for run in ordered
        ]
        ax.bar(x, values, color=colors)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    fig.suptitle("Final Metrics by Evaluated Point")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_curves(runs, output_path):
    sizes = batch_sizes(runs)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(7.2 * len(sizes), 5.5), squeeze=False
    )

    for ax, batch_size in zip(axes.ravel(), sizes):
        for run in runs_for_batch(runs, batch_size):
            if not run.eval_logs:
                continue
            epochs = [row.epoch for row in run.eval_logs]
            accs = [row.val_acc for row in run.eval_logs]
            ax.plot(
                epochs,
                accs,
                linewidth=1.2,
                alpha=0.85,
                label=f"r{run.index} {point_label(run.point)}",
            )
        ax.set_title(f"batch_size={batch_size}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val acc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Validation Accuracy During Training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def write_summary(runs, search_steps, completions, cache_hits, output_path):
    ordered = sorted(runs, key=lambda run: (run.batch_size, run.index))
    lines = [
        "CIFAR Baseline Eval3 Exp1 Joint Line Search",
        "=" * 45,
        "",
        f"Unique evaluated runs: {len(runs)}",
        f"Cache hits: {len(cache_hits)}",
        "",
    ]

    for completion in sorted(completions, key=lambda row: row.batch_size):
        lines.extend(
            [
                f"batch_size={completion.batch_size} selected best",
                "-" * 35,
                f"point: {point_label(completion.best_point)}",
                f"muon_lr: {fmt(completion.muon_lr)}",
                f"sgd_lr_mult: {fmt(completion.sgd_lr_mult)}",
                f"TTA val acc: {fmt(completion.tta_val_acc)}",
                f"evaluated points: {completion.evaluated_points}",
                "",
            ]
        )

    headers = [
        "run",
        "batch",
        "point",
        "muon_lr",
        "sgd_mult",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_s",
    ]
    rows = [
        [
            run.index,
            run.batch_size,
            point_label(run.point),
            fmt(run.muon_lr),
            fmt(run.sgd_lr_mult),
            fmt(run.train_loss),
            fmt(run.val_loss),
            fmt(run.train_acc),
            fmt(run.val_acc),
            fmt(run.tta_val_acc),
            fmt(run.wall_time_seconds),
        ]
        for run in ordered
    ]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    lines.extend(["All Evaluated Points", "-" * 20])
    lines.append(
        "  ".join(str(value).ljust(width) for value, width in zip(headers, widths))
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(str(value).ljust(width) for value, width in zip(row, widths))
        )
    lines.append("")

    for batch_size in batch_sizes(runs):
        ranked = sorted(
            runs_for_batch(runs, batch_size),
            key=lambda run: run.tta_val_acc if run.tta_val_acc is not None else -1,
            reverse=True,
        )
        lines.extend([f"Ranking, batch_size={batch_size}", "-" * 24])
        for rank, run in enumerate(ranked, start=1):
            lines.append(
                f"{rank}. r{run.index} point={point_label(run.point)} "
                f"muon_lr={fmt(run.muon_lr)} sgd_lr_mult={fmt(run.sgd_lr_mult)} "
                f"tta_val_acc={fmt(run.tta_val_acc)}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    lines = [
        "run,batch_size,mu_exp,sgd_exp,muon_lr,sgd_lr_mult,train_loss,"
        "val_loss,train_acc,val_acc,tta_val_acc,wall_time_seconds,cuda_time_seconds"
    ]
    for run in sorted(runs, key=lambda row: row.index):
        lines.append(
            ",".join(
                str(value)
                for value in [
                    run.index,
                    run.batch_size,
                    run.point[0],
                    run.point[1],
                    fmt(run.muon_lr),
                    fmt(run.sgd_lr_mult),
                    fmt(run.train_loss),
                    fmt(run.val_loss),
                    fmt(run.train_acc),
                    fmt(run.val_acc),
                    fmt(run.tta_val_acc),
                    fmt(run.wall_time_seconds),
                    fmt(run.cuda_time_seconds),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval3 exp1 joint line-search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs, search_steps, completions, cache_hits = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_search_surface(
        runs, search_steps, completions, args.output_dir / "search_surface.png"
    )
    plot_metric_heatmaps(runs, args.output_dir / "tta_heatmap.png")
    plot_search_progress(
        search_steps, completions, args.output_dir / "search_progress.png"
    )
    plot_final_metrics(runs, args.output_dir / "final_metrics_by_run.png")
    plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    write_summary(
        runs,
        search_steps,
        completions,
        cache_hits,
        args.output_dir / "summary.txt",
    )
    write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} unique evaluated runs from {args.log}")
    print(f"Parsed {len(search_steps)} search steps and {len(cache_hits)} cache hits")
    for completion in sorted(completions, key=lambda row: row.batch_size):
        print(
            f"batch_size={completion.batch_size} best={point_label(completion.best_point)} "
            f"muon_lr={completion.muon_lr:.6g} sgd_lr_mult={completion.sgd_lr_mult:.6g} "
            f"tta={completion.tta_val_acc:.4f} evaluated={completion.evaluated_points}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

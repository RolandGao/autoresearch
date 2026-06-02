from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_exp3.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_exp3_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) batch_size=(?P<batch_size>\d+) "
    r"muon_lr=(?P<muon_lr>\S+) update=(?P<update>\S+)"
)
STEP_RE = re.compile(
    r"^step=(?P<step>\d+)/(?P<total_steps>\d+) epoch=(?P<epoch>\d+) "
    r"loss=(?P<loss>\S+) head_lr=(?P<head_lr>\S+) muon_lr=(?P<muon_lr>\S+)"
)
EVAL_RE = re.compile(
    r"^eval(?: run=(?P<run>\d+))? epoch=(?P<epoch>\d+) "
    r"val_acc=(?P<val_acc>\S+) time_seconds=(?P<time_seconds>\S+)"
)
FINAL_RE = re.compile(
    r"^eval epoch=final 25batch_train_loss=(?P<train25_loss>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)

UPDATE_LABELS = {
    "row_norm": "row norm",
    "zeropower_via_newtonschulz5": "Newton-Schulz",
}
UPDATE_COLORS = {
    "row_norm": "tab:blue",
    "zeropower_via_newtonschulz5": "tab:orange",
}


@dataclass
class Run:
    index: int
    batch_size: int
    muon_lr: float
    update: str
    steps: list[dict[str, float]] = field(default_factory=list)
    evals: list[dict[str, float]] = field(default_factory=list)
    train25_loss: float = math.nan
    val_acc: float = math.nan
    tta_val_acc: float = math.nan
    time_seconds: float = math.nan

    @property
    def label(self) -> str:
        return UPDATE_LABELS.get(self.update, self.update)


def parse_float(value: str) -> float:
    return float(value)


def parse_log(path: Path) -> list[Run]:
    runs: list[Run] = []
    current: Run | None = None

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if match := RUN_RE.match(line):
                current = Run(
                    index=int(match["run"]),
                    batch_size=int(match["batch_size"]),
                    muon_lr=parse_float(match["muon_lr"]),
                    update=match["update"],
                )
                runs.append(current)
                continue

            if current is None:
                continue

            if match := STEP_RE.match(line):
                current.steps.append(
                    {
                        "step": parse_float(match["step"]),
                        "total_steps": parse_float(match["total_steps"]),
                        "epoch": parse_float(match["epoch"]),
                        "loss": parse_float(match["loss"]),
                        "head_lr": parse_float(match["head_lr"]),
                        "muon_lr": parse_float(match["muon_lr"]),
                    }
                )
                continue

            if match := FINAL_RE.match(line):
                current.train25_loss = parse_float(match["train25_loss"])
                current.val_acc = parse_float(match["val_acc"])
                current.tta_val_acc = parse_float(match["tta_val_acc"])
                current.time_seconds = parse_float(match["time_seconds"])
                continue

            if match := EVAL_RE.match(line):
                current.evals.append(
                    {
                        "epoch": parse_float(match["epoch"]),
                        "val_acc": parse_float(match["val_acc"]),
                        "time_seconds": parse_float(match["time_seconds"]),
                    }
                )

    return runs


def sorted_runs(runs: list[Run]) -> list[Run]:
    return sorted(runs, key=lambda run: (run.batch_size, run.update))


def batch_sizes(runs: list[Run]) -> list[int]:
    return sorted({run.batch_size for run in runs})


def update_names(runs: list[Run]) -> list[str]:
    known = list(UPDATE_LABELS)
    present = {run.update for run in runs}
    ordered = [name for name in known if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def run_lookup(runs: list[Run]) -> dict[tuple[int, str], Run]:
    return {(run.batch_size, run.update): run for run in runs}


def plot_final_accuracy(runs: list[Run], output_path: Path) -> None:
    batches = batch_sizes(runs)
    updates = update_names(runs)
    lookup = run_lookup(runs)
    x = list(range(len(batches)))
    width = 0.8 / max(1, len(updates))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "val_acc", "Final validation accuracy"),
        (axes[1], "tta_val_acc", "Final TTA validation accuracy"),
    ]:
        for i, update in enumerate(updates):
            offset = (i - (len(updates) - 1) / 2) * width
            values = [
                getattr(lookup.get((batch, update)), metric, math.nan)
                for batch in batches
            ]
            bars = ax.bar(
                [pos + offset for pos in x],
                values,
                width=width,
                label=UPDATE_LABELS.get(update, update),
                color=UPDATE_COLORS.get(update),
                alpha=0.88,
            )
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
        ax.set_title(title)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.84, 0.945)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    axes[1].set_xticks(x, [str(batch) for batch in batches])
    axes[1].set_xlabel("batch size")
    fig.suptitle("cifar_baseline2_sgd_exp3")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_epoch_curves(runs: list[Run], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    markers = {125: "o", 500: "s", 2000: "^"}
    linestyles = {
        "row_norm": "--",
        "zeropower_via_newtonschulz5": "-",
    }

    for run in sorted_runs(runs):
        epochs = [row["epoch"] for row in run.evals]
        values = [row["val_acc"] for row in run.evals]
        ax.plot(
            epochs,
            values,
            marker=markers.get(run.batch_size, "o"),
            linestyle=linestyles.get(run.update, "-"),
            color=UPDATE_COLORS.get(run.update),
            linewidth=1.8,
            markersize=4.5,
            label=f"bs={run.batch_size} {run.label}",
        )

    ax.set_title("Validation accuracy by epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    queue = []
    for value in values:
        total += value
        queue.append(value)
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def plot_training_loss(runs: list[Run], output_path: Path) -> None:
    batches = batch_sizes(runs)
    fig, axes = plt.subplots(
        len(batches), 1, figsize=(11, 3.2 * len(batches)), sharex=True, constrained_layout=True
    )
    if len(batches) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        for run in [run for run in sorted_runs(runs) if run.batch_size == batch]:
            progress = [
                row["step"] / row["total_steps"]
                for row in run.steps
                if row["total_steps"] > 0
            ]
            losses = [row["loss"] for row in run.steps]
            smooth = rolling_mean(losses, max(1, len(losses) // 80))
            ax.plot(
                progress,
                smooth,
                label=run.label,
                color=UPDATE_COLORS.get(run.update),
                linewidth=1.5,
            )
        ax.set_title(f"Training loss, batch size {batch}")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("training progress")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(runs: list[Run], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "batch_size",
                "muon_lr",
                "update",
                "train25_loss",
                "val_acc",
                "tta_val_acc",
                "time_seconds",
            ]
        )
        for run in sorted_runs(runs):
            writer.writerow(
                [
                    run.index,
                    run.batch_size,
                    run.muon_lr,
                    run.update,
                    run.train25_loss,
                    run.val_acc,
                    run.tta_val_acc,
                    run.time_seconds,
                ]
            )


def write_summary(runs: list[Run], output_path: Path) -> None:
    best_tta = max(runs, key=lambda run: run.tta_val_acc)
    lines = [
        "run | batch_size | update | train25_loss | val_acc | tta_val_acc | seconds",
        "--- | ---: | --- | ---: | ---: | ---: | ---:",
    ]
    for run in sorted_runs(runs):
        lines.append(
            f"{run.index} | {run.batch_size} | {run.label} | "
            f"{run.train25_loss:.4f} | {run.val_acc:.4f} | "
            f"{run.tta_val_acc:.4f} | {run.time_seconds:.2f}"
        )
    lines.extend(
        [
            "",
            "Best TTA:",
            (
                f"run={best_tta.index} batch_size={best_tta.batch_size} "
                f"update={best_tta.label} tta_val_acc={best_tta.tta_val_acc:.4f}"
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def plot_all(runs: list[Run], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_final_accuracy(runs, output_dir / "final_accuracy.png")
    plot_epoch_curves(runs, output_dir / "epoch_val_accuracy.png")
    plot_training_loss(runs, output_dir / "training_loss.png")
    write_csv(runs, output_dir / "summary.csv")
    write_summary(runs, output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_exp3 row_norm vs Newton-Schulz runs."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = parse_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")

    plot_all(runs, args.output_dir)
    best = max(runs, key=lambda run: run.tta_val_acc)
    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best TTA: run={best.index} bs={best.batch_size} "
        f"update={best.label} tta={best.tta_val_acc:.4f}"
    )


if __name__ == "__main__":
    main()

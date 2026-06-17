#!/usr/bin/env python3
"""Plot CIFAR overfit LR/momentum-search logs."""

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


DEFAULT_LOG = Path(__file__).with_name("cifar_overfit_search_momentum.log")
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

RUN_RE = re.compile(
    rf"^cifar_baseline2_overfit_n_search "
    rf"run=(?P<run>\d+) "
    rf"batch_size=(?P<batch_size>\d+) "
    rf"N=(?P<N>\d+) "
    rf"M=(?P<M>\d+) "
    rf"final_k_granularity=(?P<final_k_granularity>{FLOAT_RE}) "
    rf"cooldown_final_k_granularity=(?P<cooldown_final_k_granularity>{FLOAT_RE}) "
    rf"muon_orthogonalize=(?P<muon_orthogonalize>\S+) "
    rf"initial_muon_lr=(?P<initial_muon_lr>{FLOAT_RE}) "
    rf"initial_muon_lr_k=(?P<initial_muon_lr_k>{FLOAT_RE}) "
    rf"initial_muon_momentum=(?P<initial_muon_momentum>{FLOAT_RE}) "
    rf"initial_muon_momentum_index=(?P<initial_muon_momentum_index>\d+)"
)
TRAIN_RE = re.compile(
    rf"^train_loss "
    rf"run=(?P<run>\d+) "
    rf"step=(?P<step>\d+)/(?P<total_steps>\d+) "
    rf"loss=(?P<loss>{FLOAT_RE}) "
    rf"head_lr=(?P<head_lr>{FLOAT_RE}) "
    rf"muon_lr=(?P<muon_lr>{FLOAT_RE}) "
    rf"muon_momentum=(?P<muon_momentum>{FLOAT_RE})"
)
INTERVAL_RE = re.compile(
    rf"^interval_muon_lr=(?P<muon_lr>{FLOAT_RE}) "
    rf"interval_muon_momentum=(?P<muon_momentum>{FLOAT_RE}) "
    rf"interval_loss=(?P<interval_loss>{FLOAT_RE})"
)
COOLDOWN_RE = re.compile(
    rf"^(?P<cooldown_muon_lr>{FLOAT_RE}) -> (?P<final_loss>{FLOAT_RE})$"
)
COOLDOWN_NONE_RE = re.compile(
    rf"^cooldown_muon_lr=none final_loss=(?P<final_loss>{FLOAT_RE})$"
)
BEST_RE = re.compile(
    rf"^best_interval_muon_lr=(?P<muon_lr>{FLOAT_RE}) "
    rf"best_interval_muon_momentum=(?P<muon_momentum>{FLOAT_RE}) "
    rf"best_cooldown_muon_lr=(?P<cooldown_muon_lr>none|{FLOAT_RE}) "
    rf"interval_loss=(?P<interval_loss>{FLOAT_RE}) "
    rf"final_loss=(?P<final_loss>{FLOAT_RE}) "
    rf"evaluated_interval_configs=(?P<evaluated_interval_configs>\d+) "
    rf"evaluated_configs=(?P<evaluated_configs>\d+)"
)
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z ]+):\s+(?P<value>\S+)")


@dataclass
class TrainPoint:
    step: int
    total_steps: int
    loss: float
    head_lr: float
    muon_lr: float
    muon_momentum: float


@dataclass
class CooldownEval:
    muon_lr: float
    final_loss: float


@dataclass
class IntervalEval:
    interval_index: int
    start_step: int
    muon_lr: float
    muon_momentum: float
    interval_loss: float
    cooldowns: list[CooldownEval] = field(default_factory=list)
    no_cooldown_final_loss: float | None = None

    @property
    def best_cooldown(self) -> CooldownEval | None:
        if not self.cooldowns:
            return None
        return min(self.cooldowns, key=lambda item: (item.final_loss, item.muon_lr))

    @property
    def final_loss(self) -> float:
        best = self.best_cooldown
        if best is not None:
            return best.final_loss
        if self.no_cooldown_final_loss is not None:
            return self.no_cooldown_final_loss
        return self.interval_loss


@dataclass
class IntervalChoice:
    interval_index: int
    start_step: int
    muon_lr: float
    muon_momentum: float
    cooldown_muon_lr: float | None
    interval_loss: float
    final_loss: float
    evaluated_interval_configs: int
    evaluated_configs: int


@dataclass
class Run:
    run: int
    batch_size: int
    N: int
    M: int
    final_k_granularity: float
    cooldown_final_k_granularity: float
    muon_orthogonalize: str
    initial_muon_lr: float
    initial_muon_lr_k: float
    initial_muon_momentum: float
    initial_muon_momentum_index: int
    train: list[TrainPoint] = field(default_factory=list)
    interval_evals: list[IntervalEval] = field(default_factory=list)
    interval_choices: list[IntervalChoice] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def final_loss(self) -> float | None:
        value = parse_optional_float(self.summary.get("Final train loss"))
        if value is not None:
            return value
        return self.train[-1].loss if self.train else None


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(path: Path) -> list[Run]:
    runs: list[Run] = []
    runs_by_id: dict[int, Run] = {}
    current_run: Run | None = None
    current_eval: IntervalEval | None = None
    current_interval_index = 0
    current_start_step = 0

    with path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current_run = Run(
                    run=int(match.group("run")),
                    batch_size=int(match.group("batch_size")),
                    N=int(match.group("N")),
                    M=int(match.group("M")),
                    final_k_granularity=float(match.group("final_k_granularity")),
                    cooldown_final_k_granularity=float(
                        match.group("cooldown_final_k_granularity")
                    ),
                    muon_orthogonalize=match.group("muon_orthogonalize"),
                    initial_muon_lr=float(match.group("initial_muon_lr")),
                    initial_muon_lr_k=float(match.group("initial_muon_lr_k")),
                    initial_muon_momentum=float(match.group("initial_muon_momentum")),
                    initial_muon_momentum_index=int(
                        match.group("initial_muon_momentum_index")
                    ),
                )
                runs.append(current_run)
                runs_by_id[current_run.run] = current_run
                current_eval = None
                current_interval_index = 0
                current_start_step = 0
                continue

            match = TRAIN_RE.match(line)
            if match:
                run = runs_by_id.get(int(match.group("run")), current_run)
                if run is None:
                    continue
                point = TrainPoint(
                    step=int(match.group("step")),
                    total_steps=int(match.group("total_steps")),
                    loss=float(match.group("loss")),
                    head_lr=float(match.group("head_lr")),
                    muon_lr=float(match.group("muon_lr")),
                    muon_momentum=float(match.group("muon_momentum")),
                )
                run.train.append(point)
                current_start_step = max(current_start_step, point.step)
                continue

            match = INTERVAL_RE.match(line)
            if match and current_run is not None:
                current_eval = IntervalEval(
                    interval_index=current_interval_index,
                    start_step=current_start_step,
                    muon_lr=float(match.group("muon_lr")),
                    muon_momentum=float(match.group("muon_momentum")),
                    interval_loss=float(match.group("interval_loss")),
                )
                current_run.interval_evals.append(current_eval)
                continue

            match = COOLDOWN_RE.match(line)
            if match and current_eval is not None:
                current_eval.cooldowns.append(
                    CooldownEval(
                        muon_lr=float(match.group("cooldown_muon_lr")),
                        final_loss=float(match.group("final_loss")),
                    )
                )
                continue

            match = COOLDOWN_NONE_RE.match(line)
            if match and current_eval is not None:
                current_eval.no_cooldown_final_loss = float(match.group("final_loss"))
                continue

            match = BEST_RE.match(line)
            if match and current_run is not None:
                current_run.interval_choices.append(
                    IntervalChoice(
                        interval_index=current_interval_index,
                        start_step=current_start_step,
                        muon_lr=float(match.group("muon_lr")),
                        muon_momentum=float(match.group("muon_momentum")),
                        cooldown_muon_lr=parse_optional_float(
                            match.group("cooldown_muon_lr")
                        ),
                        interval_loss=float(match.group("interval_loss")),
                        final_loss=float(match.group("final_loss")),
                        evaluated_interval_configs=int(
                            match.group("evaluated_interval_configs")
                        ),
                        evaluated_configs=int(match.group("evaluated_configs")),
                    )
                )
                current_interval_index += 1
                current_eval = None
                continue

            match = SUMMARY_RE.match(line)
            if match and current_run is not None:
                current_run.summary[match.group("key").strip()] = match.group("value")

    return runs


def write_csvs(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "train_losses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "step",
                "total_steps",
                "loss",
                "head_lr",
                "muon_lr",
                "muon_momentum",
            ],
        )
        writer.writeheader()
        for run in runs:
            for point in run.train:
                writer.writerow(
                    {
                        "run": run.run,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                        "muon_momentum": point.muon_momentum,
                    }
                )

    with (output_dir / "interval_choices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "interval_index",
                "start_step",
                "muon_lr",
                "muon_momentum",
                "cooldown_muon_lr",
                "interval_loss",
                "final_loss",
                "evaluated_interval_configs",
                "evaluated_configs",
            ],
        )
        writer.writeheader()
        for run in runs:
            for choice in run.interval_choices:
                writer.writerow({"run": run.run, **choice.__dict__})

    with (output_dir / "interval_evals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "interval_index",
                "start_step",
                "muon_lr",
                "muon_momentum",
                "interval_loss",
                "best_cooldown_muon_lr",
                "best_final_loss",
            ],
        )
        writer.writeheader()
        for run in runs:
            for item in run.interval_evals:
                best = item.best_cooldown
                writer.writerow(
                    {
                        "run": run.run,
                        "interval_index": item.interval_index,
                        "start_step": item.start_step,
                        "muon_lr": item.muon_lr,
                        "muon_momentum": item.muon_momentum,
                        "interval_loss": item.interval_loss,
                        "best_cooldown_muon_lr": best.muon_lr if best else None,
                        "best_final_loss": item.final_loss,
                    }
                )

    with (output_dir / "cooldown_evals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "interval_index",
                "start_step",
                "interval_muon_lr",
                "interval_muon_momentum",
                "cooldown_muon_lr",
                "final_loss",
            ],
        )
        writer.writeheader()
        for run in runs:
            for item in run.interval_evals:
                for cooldown in item.cooldowns:
                    writer.writerow(
                        {
                            "run": run.run,
                            "interval_index": item.interval_index,
                            "start_step": item.start_step,
                            "interval_muon_lr": item.muon_lr,
                            "interval_muon_momentum": item.muon_momentum,
                            "cooldown_muon_lr": cooldown.muon_lr,
                            "final_loss": cooldown.final_loss,
                        }
                    )


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> None:
    lines = [
        "CIFAR overfit LR/momentum search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"Run {run.run}",
                f"  train points: {len(run.train)}",
                f"  interval searches: {len(run.interval_choices)}",
                f"  interval configs evaluated: {len(run.interval_evals)}",
                f"  final loss: {format_number(run.final_loss)}",
                f"  final muon lr: {run.summary.get('Final Muon lr', 'NA')}",
                f"  final muon momentum: {run.summary.get('Final Muon momentum', 'NA')}",
                f"  final cooldown lr: {run.summary.get('Final cooldown lr', 'NA')}",
                "",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def format_number(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_train_loss(run: Run, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        [point.step for point in run.train],
        [point.loss for point in run.train],
        marker="o",
        markersize=3,
        linewidth=1.8,
        color="#1f77b4",
    )
    ax.set_title(f"Run {run.run}: train loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"run_{run.run}_train_loss.png", dpi=180)
    plt.close(fig)


def plot_choices(run: Run, output_dir: Path) -> None:
    choices = run.interval_choices
    fig, (ax_lr, ax_momentum, ax_loss) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True
    )
    steps = [choice.start_step for choice in choices]

    ax_lr.plot(
        steps,
        [choice.muon_lr for choice in choices],
        marker="o",
        label="interval lr",
        color="#1f77b4",
    )
    cooldown_steps = [choice.start_step for choice in choices if choice.cooldown_muon_lr]
    cooldown_lrs = [
        choice.cooldown_muon_lr for choice in choices if choice.cooldown_muon_lr
    ]
    if cooldown_steps:
        ax_lr.plot(
            cooldown_steps,
            cooldown_lrs,
            marker="s",
            label="cooldown lr",
            color="#ff7f0e",
        )
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Muon LR")
    ax_lr.legend()
    style_axes(ax_lr)

    ax_momentum.plot(
        steps,
        [choice.muon_momentum for choice in choices],
        marker="o",
        color="#2ca02c",
    )
    ax_momentum.set_ylim(-0.03, 0.93)
    ax_momentum.set_ylabel("Muon momentum")
    style_axes(ax_momentum)

    ax_loss.plot(
        steps,
        [choice.interval_loss for choice in choices],
        marker="o",
        label="interval loss",
        color="#9467bd",
    )
    ax_loss.plot(
        steps,
        [choice.final_loss for choice in choices],
        marker="s",
        label="after cooldown",
        color="#d62728",
    )
    ax_loss.set_xlabel("Interval start step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    style_axes(ax_loss)

    fig.suptitle(f"Run {run.run}: selected hyperparameters")
    fig.tight_layout()
    fig.savefig(output_dir / f"run_{run.run}_choices.png", dpi=180)
    plt.close(fig)


def plot_interval_landscapes(run: Run, output_dir: Path, max_panels: int) -> None:
    if not run.interval_evals:
        return

    interval_indexes = sorted({item.interval_index for item in run.interval_evals})
    interval_indexes = interval_indexes[:max_panels]
    cols = min(5, len(interval_indexes))
    rows = math.ceil(len(interval_indexes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows), squeeze=False)

    final_losses = [
        item.final_loss
        for item in run.interval_evals
        if item.interval_index in set(interval_indexes)
    ]
    vmin = min(final_losses)
    vmax = max(final_losses)

    for ax, interval_index in zip(axes.flat, interval_indexes):
        items = [
            item
            for item in run.interval_evals
            if item.interval_index == interval_index
        ]
        scatter = ax.scatter(
            [item.muon_lr for item in items],
            [item.muon_momentum for item in items],
            c=[item.final_loss for item in items],
            s=65,
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.3,
        )
        choice = next(
            item for item in run.interval_choices if item.interval_index == interval_index
        )
        ax.scatter(
            [choice.muon_lr],
            [choice.muon_momentum],
            s=120,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.6,
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 0.93)
        ax.set_title(f"interval {interval_index}, step {choice.start_step}")
        ax.set_xlabel("interval lr")
        ax.set_ylabel("momentum")
        style_axes(ax)

    for ax in axes.flat[len(interval_indexes) :]:
        ax.axis("off")

    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="best cooldown/final loss")
    fig.suptitle(f"Run {run.run}: interval search landscapes")
    fig.savefig(output_dir / f"run_{run.run}_interval_landscapes.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], output_dir: Path, max_panels: int) -> None:
    for run in runs:
        plot_train_loss(run, output_dir)
        plot_choices(run, output_dir)
        plot_interval_landscapes(run, output_dir, max_panels=max_panels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot CIFAR overfit LR/momentum-search logs."
    )
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG/CSV outputs. Defaults to <log stem>_plots.",
    )
    parser.add_argument(
        "--max-landscape-panels",
        type=int,
        default=25,
        help="Maximum interval landscape panels per run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log
    output_dir = args.output_dir or log_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(log_path)
    if not runs:
        raise SystemExit(f"No runs parsed from {log_path}")

    write_csvs(runs, output_dir)
    write_summary(runs, log_path, output_dir)
    plot_all(runs, output_dir, max_panels=args.max_landscape_panels)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

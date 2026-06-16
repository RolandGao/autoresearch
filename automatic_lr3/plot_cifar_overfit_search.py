#!/usr/bin/env python3
"""Plot and summarize CIFAR overfit LR-search logs."""

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


DEFAULT_LOG = Path(__file__).with_name("cifar_overfit_search_exp1.log")
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

RUN_RE = re.compile(
    rf"^cifar_baseline2_overfit_n_search "
    rf"run=(?P<run>\d+) "
    rf"batch_size=(?P<batch_size>\d+) "
    rf"N=(?P<N>\d+) "
    rf"M=(?P<M>\d+) "
    rf"final_k_granularity=(?P<final_k_granularity>{FLOAT_RE}) "
    rf"cooldown_final_k_granularity=(?P<cooldown_final_k_granularity>{FLOAT_RE}) "
    rf"initial_muon_lr=(?P<initial_muon_lr>{FLOAT_RE}) "
    rf"initial_muon_lr_k=(?P<initial_muon_lr_k>{FLOAT_RE})"
)
TRAIN_LOSS_RE = re.compile(
    rf"^train_loss "
    rf"run=(?P<run>\d+) "
    rf"step=(?P<step>\d+)/(?P<total_steps>\d+) "
    rf"loss=(?P<loss>{FLOAT_RE}) "
    rf"head_lr=(?P<head_lr>{FLOAT_RE}) "
    rf"muon_lr=(?P<muon_lr>{FLOAT_RE})"
)
SEARCH_RE = re.compile(
    rf"^muon_lr_search_complete "
    rf"search=(?P<search>\S+) "
    rf"(?P<fields>.*)$"
)
SEARCH_ID_RE = re.compile(
    rf"^run(?P<run>\d+)_.*_interval(?P<interval>\d+)_step(?P<step>\d+)"
    rf"(?:_cooldown_for_k(?P<cooldown_for_k>{FLOAT_RE}))?$"
)
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z ]+):\s+(?P<value>\S+)")


@dataclass
class TrainPoint:
    step: int
    total_steps: int
    loss: float
    head_lr: float
    muon_lr: float


@dataclass
class SearchChoice:
    search: str
    run: int
    interval: int | None
    step: int | None
    cooldown_for_k: float | None
    initial_muon_lr: float | None = None
    best_k: float | None = None
    best_muon_lr: float | None = None
    interval_steps: int | None = None
    cooldown_steps: int | None = None
    final_k_granularity: float | None = None
    cooldown_final_k_granularity: float | None = None
    interval_loss: float | None = None
    final_loss: float | None = None
    evaluated_lrs: int | None = None

    @property
    def is_cooldown(self) -> bool:
        return self.cooldown_for_k is not None


@dataclass
class Run:
    run: int
    batch_size: int
    N: int
    M: int
    final_k_granularity: float
    cooldown_final_k_granularity: float
    initial_muon_lr: float
    initial_muon_lr_k: float
    train: list[TrainPoint] = field(default_factory=list)
    searches: list[SearchChoice] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return (
            f"run={self.run} N={self.N} M={self.M} "
            f"G={format_number(self.final_k_granularity)} "
            f"CG={format_number(self.cooldown_final_k_granularity)}"
        )

    @property
    def granularity_label(self) -> str:
        return (
            f"G={format_number(self.final_k_granularity)}, "
            f"CG={format_number(self.cooldown_final_k_granularity)}"
        )

    @property
    def initial_loss(self) -> float | None:
        summary_loss = parse_optional_float(self.summary.get("Initial train loss"))
        if summary_loss is not None:
            return summary_loss
        return self.train[0].loss if self.train else None

    @property
    def final_loss(self) -> float | None:
        summary_loss = parse_optional_float(self.summary.get("Final train loss"))
        if summary_loss is not None:
            return summary_loss
        return self.train[-1].loss if self.train else None

    @property
    def final_muon_lr(self) -> float | None:
        summary_lr = parse_optional_float(self.summary.get("Final Muon lr"))
        if summary_lr is not None:
            return summary_lr
        return self.train[-1].muon_lr if self.train else None

    @property
    def final_muon_lr_k(self) -> float | None:
        return parse_optional_float(self.summary.get("Final Muon lr k"))

    @property
    def final_cooldown_lr(self) -> float | None:
        return parse_optional_float(self.summary.get("Final cooldown lr"))

    @property
    def final_cooldown_lr_k(self) -> float | None:
        return parse_optional_float(self.summary.get("Final cooldown lr k"))

    def loss_at_step(self, step: int) -> float | None:
        if step == 50:
            return self.final_loss
        for point in self.train:
            if point.step == step:
                return point.loss
        return None

    def first_step_below(self, threshold: float) -> TrainPoint | None:
        for point in sorted(self.train, key=lambda item: item.step):
            if point.loss < threshold:
                return point
        return None


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def parse_key_value_fields(text: str) -> dict[str, str]:
    fields = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_search(search: str, fields: dict[str, str]) -> SearchChoice:
    match = SEARCH_ID_RE.match(search)
    run = int(match.group("run")) if match else -1
    interval = int(match.group("interval")) if match else None
    step = int(match.group("step")) if match else None
    cooldown_for_k = (
        float(match.group("cooldown_for_k"))
        if match and match.group("cooldown_for_k") is not None
        else None
    )
    return SearchChoice(
        search=search,
        run=run,
        interval=interval,
        step=step,
        cooldown_for_k=cooldown_for_k,
        initial_muon_lr=parse_optional_float(fields.get("initial_muon_lr")),
        best_k=parse_optional_float(fields.get("best_k")),
        best_muon_lr=parse_optional_float(fields.get("best_muon_lr")),
        interval_steps=parse_optional_int(fields.get("interval_steps")),
        cooldown_steps=parse_optional_int(fields.get("cooldown_steps")),
        final_k_granularity=parse_optional_float(fields.get("final_k_granularity")),
        cooldown_final_k_granularity=parse_optional_float(
            fields.get("cooldown_final_k_granularity")
        ),
        interval_loss=parse_optional_float(fields.get("interval_loss")),
        final_loss=parse_optional_float(fields.get("final_loss")),
        evaluated_lrs=parse_optional_int(fields.get("evaluated_lrs")),
    )


def parse_log(path: Path) -> list[Run]:
    runs: list[Run] = []
    runs_by_id: dict[int, Run] = {}
    current: Run | None = None

    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue

            match = RUN_RE.match(line)
            if match:
                current = Run(
                    run=int(match.group("run")),
                    batch_size=int(match.group("batch_size")),
                    N=int(match.group("N")),
                    M=int(match.group("M")),
                    final_k_granularity=float(match.group("final_k_granularity")),
                    cooldown_final_k_granularity=float(
                        match.group("cooldown_final_k_granularity")
                    ),
                    initial_muon_lr=float(match.group("initial_muon_lr")),
                    initial_muon_lr_k=float(match.group("initial_muon_lr_k")),
                )
                runs.append(current)
                runs_by_id[current.run] = current
                continue

            match = TRAIN_LOSS_RE.match(line)
            if match:
                run_id = int(match.group("run"))
                run = runs_by_id.get(run_id, current)
                if run is None:
                    continue
                run.train.append(
                    TrainPoint(
                        step=int(match.group("step")),
                        total_steps=int(match.group("total_steps")),
                        loss=float(match.group("loss")),
                        head_lr=float(match.group("head_lr")),
                        muon_lr=float(match.group("muon_lr")),
                    )
                )
                continue

            match = SEARCH_RE.match(line)
            if match:
                fields = parse_key_value_fields(match.group("fields"))
                choice = parse_search(match.group("search"), fields)
                run = runs_by_id.get(choice.run, current)
                if run is not None:
                    run.searches.append(choice)
                continue

            match = SUMMARY_RE.match(line)
            if match and current is not None:
                current.summary[match.group("key").strip()] = match.group("value")

    return runs


def sorted_complete_runs(runs: list[Run]) -> list[Run]:
    return sorted(
        [run for run in runs if run.final_loss is not None],
        key=lambda run: (run.final_loss if run.final_loss is not None else math.inf, run.run),
    )


def write_csvs(runs: list[Run], output_dir: Path) -> None:
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "batch_size",
                "N",
                "M",
                "final_k_granularity",
                "cooldown_final_k_granularity",
                "initial_muon_lr",
                "initial_muon_lr_k",
                "initial_loss",
                "final_loss",
                "loss_delta",
                "final_muon_lr",
                "final_muon_lr_k",
                "final_cooldown_lr",
                "final_cooldown_lr_k",
                "train_points",
                "searches",
                "primary_searches",
                "cooldown_searches",
            ],
        )
        writer.writeheader()
        for run in runs:
            initial_loss = run.initial_loss
            final_loss = run.final_loss
            writer.writerow(
                {
                    "run": run.run,
                    "batch_size": run.batch_size,
                    "N": run.N,
                    "M": run.M,
                    "final_k_granularity": run.final_k_granularity,
                    "cooldown_final_k_granularity": run.cooldown_final_k_granularity,
                    "initial_muon_lr": run.initial_muon_lr,
                    "initial_muon_lr_k": run.initial_muon_lr_k,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "loss_delta": (
                        None
                        if initial_loss is None or final_loss is None
                        else final_loss - initial_loss
                    ),
                    "final_muon_lr": run.final_muon_lr,
                    "final_muon_lr_k": run.final_muon_lr_k,
                    "final_cooldown_lr": run.final_cooldown_lr,
                    "final_cooldown_lr_k": run.final_cooldown_lr_k,
                    "train_points": len(run.train),
                    "searches": len(run.searches),
                    "primary_searches": sum(not choice.is_cooldown for choice in run.searches),
                    "cooldown_searches": sum(choice.is_cooldown for choice in run.searches),
                }
            )

    with (output_dir / "train_losses.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "N",
                "M",
                "final_k_granularity",
                "cooldown_final_k_granularity",
                "step",
                "total_steps",
                "loss",
                "head_lr",
                "muon_lr",
            ],
        )
        writer.writeheader()
        for run in runs:
            for point in run.train:
                writer.writerow(
                    {
                        "run": run.run,
                        "N": run.N,
                        "M": run.M,
                        "final_k_granularity": run.final_k_granularity,
                        "cooldown_final_k_granularity": run.cooldown_final_k_granularity,
                        "step": point.step,
                        "total_steps": point.total_steps,
                        "loss": point.loss,
                        "head_lr": point.head_lr,
                        "muon_lr": point.muon_lr,
                    }
                )

    with (output_dir / "search_choices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "N",
                "M",
                "final_k_granularity",
                "cooldown_final_k_granularity",
                "search",
                "interval",
                "step",
                "is_cooldown",
                "cooldown_for_k",
                "initial_muon_lr",
                "best_k",
                "best_muon_lr",
                "interval_steps",
                "cooldown_steps",
                "interval_loss",
                "final_loss",
                "evaluated_lrs",
            ],
        )
        writer.writeheader()
        for run in runs:
            for choice in run.searches:
                writer.writerow(
                    {
                        "run": run.run,
                        "N": run.N,
                        "M": run.M,
                        "final_k_granularity": run.final_k_granularity,
                        "cooldown_final_k_granularity": run.cooldown_final_k_granularity,
                        "search": choice.search,
                        "interval": choice.interval,
                        "step": choice.step,
                        "is_cooldown": choice.is_cooldown,
                        "cooldown_for_k": choice.cooldown_for_k,
                        "initial_muon_lr": choice.initial_muon_lr,
                        "best_k": choice.best_k,
                        "best_muon_lr": choice.best_muon_lr,
                        "interval_steps": choice.interval_steps,
                        "cooldown_steps": choice.cooldown_steps,
                        "interval_loss": choice.interval_loss,
                        "final_loss": choice.final_loss,
                        "evaluated_lrs": choice.evaluated_lrs,
                    }
                )


def write_summary(runs: list[Run], log_path: Path, output_dir: Path) -> str:
    ranked = sorted_complete_runs(runs)
    best = ranked[0] if ranked else None
    lines: list[str] = [
        "CIFAR overfit LR search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Runs parsed: {len(runs)}",
        "",
    ]

    if best is not None:
        lines.extend(
            [
                "Best run",
                (
                    f"  {best.label}: final_loss={format_number(best.final_loss)} "
                    f"initial_loss={format_number(best.initial_loss)} "
                    f"final_muon_lr={format_number(best.final_muon_lr)} "
                    f"final_muon_lr_k={format_number(best.final_muon_lr_k)}"
                ),
                "",
            ]
        )

    if ranked:
        worst = ranked[-1]
        losses = [run.final_loss for run in ranked if run.final_loss is not None]
        lines.extend(
            [
                "Final-loss distribution",
                f"  min={format_number(min(losses))}",
                f"  median={format_number(median(losses))}",
                f"  max={format_number(max(losses))}",
                f"  spread={format_number(max(losses) - min(losses))}",
                (
                    f"  worst={worst.label}: "
                    f"final_loss={format_number(worst.final_loss)}"
                ),
                "",
            ]
        )

        add_loss_step_ranking(lines, runs, step=50, title="Ranking: loss after 50 steps")
        add_loss_step_ranking(lines, runs, step=10, title="Ranking: loss after 10 steps")
        add_loss_step_ranking(lines, runs, step=20, title="Ranking: loss after 20 steps")
        add_threshold_ranking(lines, runs, threshold=0.9)
        add_threshold_ranking(lines, runs, threshold=0.88)
        add_threshold_ranking(lines, runs, threshold=0.87)

    lines.extend(["Best run by N,M", "  N  M  run  G  CG  final_loss  final_muon_lr"])
    for (N, M), group in group_by_nm(ranked).items():
        run = group[0]
        lines.append(
            f"  {N:<1d}  {M:<1d}  {run.run:<3d}  "
            f"{format_number(run.final_k_granularity):<4s} "
            f"{format_number(run.cooldown_final_k_granularity):<4s} "
            f"{format_number(run.final_loss):<10s} "
            f"{format_number(run.final_muon_lr)}"
        )

    lines.extend(["", "Best run by granularity pair", "  G  CG  run  N  M  final_loss"])
    for granularity, group in group_by_granularity(ranked).items():
        run = group[0]
        lines.append(
            f"  {format_number(granularity[0]):<4s} "
            f"{format_number(granularity[1]):<4s} "
            f"{run.run:<3d} {run.N:<1d}  {run.M:<1d}  "
            f"{format_number(run.final_loss)}"
        )

    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    return text


def add_loss_step_ranking(lines: list[str], runs: list[Run], step: int, title: str) -> None:
    ranked = sorted(
        [run for run in runs if run.loss_at_step(step) is not None],
        key=lambda run: (run.loss_at_step(step), run.run),
    )
    lines.append(title)
    for rank, run in enumerate(ranked, start=1):
        loss = run.loss_at_step(step)
        lines.append(
            f"  {rank:2d}. {run.label}: "
            f"step={step} loss={format_number(loss)} "
            f"final_loss={format_number(run.final_loss)} "
            f"final_muon_lr={format_number(run.final_muon_lr)}"
        )
    missing = len(runs) - len(ranked)
    if missing:
        lines.append(f"  {missing} runs did not have step {step}.")
    lines.append("")


def add_threshold_ranking(lines: list[str], runs: list[Run], threshold: float) -> None:
    reached = [
        (run, point)
        for run in runs
        if (point := run.first_step_below(threshold)) is not None
    ]
    reached.sort(key=lambda item: (item[1].step, item[1].loss, item[0].run))

    lines.append(f"Ranking: first step below {threshold:g}")
    for rank, (run, point) in enumerate(reached, start=1):
        lines.append(
            f"  {rank:2d}. {run.label}: "
            f"first_step={point.step} loss={format_number(point.loss)} "
            f"final_loss={format_number(run.final_loss)}"
        )
    missing = len(runs) - len(reached)
    if missing:
        lines.append(f"  {missing} runs never went below {threshold:g}.")
    lines.append("")


def median(values: list[float]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def group_by_nm(runs: list[Run]) -> dict[tuple[int, int], list[Run]]:
    grouped: dict[tuple[int, int], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.N, run.M), []).append(run)
    return dict(sorted(grouped.items()))


def group_by_granularity(runs: list[Run]) -> dict[tuple[float, float], list[Run]]:
    grouped: dict[tuple[float, float], list[Run]] = {}
    for run in runs:
        grouped.setdefault(
            (run.final_k_granularity, run.cooldown_final_k_granularity), []
        ).append(run)
    return dict(sorted(grouped.items()))


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss_curves(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    top_runs = {run.run for run in ranked[:5]}

    fig, ax = plt.subplots(figsize=(10, 6))
    for run in runs:
        if not run.train:
            continue
        steps = [point.step for point in run.train]
        losses = [point.loss for point in run.train]
        is_top = run.run in top_runs
        ax.plot(
            steps,
            losses,
            linewidth=2.0 if is_top else 0.8,
            alpha=0.95 if is_top else 0.25,
            label=run.label if is_top else None,
        )

    ax.set_title("Train loss curves")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    style_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_final_loss_bars(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(ranked)))
    losses = [run.final_loss for run in ranked]
    colors = [f"C{run.M}" for run in ranked]
    ax.bar(x, losses, color=colors, alpha=0.8)
    ax.set_title("Final train loss by run, sorted best to worst")
    ax.set_xlabel("rank")
    ax.set_ylabel("final train loss")
    ax.set_xticks(x)
    ax.set_xticklabels([str(run.run) for run in ranked], rotation=90, fontsize=7)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_ranked.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    groups = group_by_granularity(ranked)
    Ns = sorted({run.N for run in ranked})
    Ms = sorted({run.M for run in ranked})

    if not groups or not Ns or not Ms:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    all_losses = [run.final_loss for run in ranked if run.final_loss is not None]
    vmin = min(all_losses)
    vmax = max(all_losses)

    for ax, (granularity, group) in zip(axes[0], groups.items()):
        by_nm = {(run.N, run.M): run for run in group}
        matrix = [
            [
                by_nm.get((N, M)).final_loss if by_nm.get((N, M)) else math.nan
                for M in Ms
            ]
            for N in Ns
        ]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"G={format_number(granularity[0])}, CG={format_number(granularity[1])}")
        ax.set_xlabel("M cooldown steps")
        ax.set_ylabel("N steps")
        ax.set_xticks(range(len(Ms)), [str(M) for M in Ms])
        ax.set_yticks(range(len(Ns)), [str(N) for N in Ns])
        for row_idx, N in enumerate(Ns):
            for col_idx, M in enumerate(Ms):
                run = by_nm.get((N, M))
                if run is None or run.final_loss is None:
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{run.final_loss:.6f}\n#{run.run}",
                    ha="center",
                    va="center",
                    color="white" if run.final_loss < median(all_losses) else "black",
                    fontsize=8,
                )

    fig.colorbar(image, ax=axes.ravel().tolist(), label="final train loss")
    fig.savefig(output_dir / "final_loss_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_loss_by_config(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    groups = group_by_granularity(ranked)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, (granularity, group) in zip(axes[0], groups.items()):
        for M in sorted({run.M for run in group}):
            subset = sorted([run for run in group if run.M == M], key=lambda run: run.N)
            ax.plot(
                [run.N for run in subset],
                [run.final_loss for run in subset],
                marker="o",
                linewidth=1.5,
                label=f"M={M}",
            )
        ax.set_title(f"G={format_number(granularity[0])}, CG={format_number(granularity[1])}")
        ax.set_xlabel("N steps")
        ax.set_ylabel("final train loss")
        style_axes(ax)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_by_config.png", dpi=180)
    plt.close(fig)


def plot_muon_lr_curves(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    top_runs = ranked[:8]

    fig, ax = plt.subplots(figsize=(10, 6))
    for run in top_runs:
        points = sorted(run.train, key=lambda point: point.step)
        ax.plot(
            [point.step for point in points],
            [point.muon_lr for point in points],
            marker=".",
            linewidth=1.2,
            label=run.label,
        )

    ax.set_title("Applied Muon LR for top runs")
    ax.set_xlabel("step")
    ax.set_ylabel("Muon LR")
    ax.set_yscale("log")
    style_axes(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "top_muon_lr_curves.png", dpi=180)
    plt.close(fig)


def plot_search_evaluation_counts(runs: list[Run], output_dir: Path) -> None:
    ranked = sorted_complete_runs(runs)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(ranked)))
    primary = [sum(not choice.is_cooldown for choice in run.searches) for run in ranked]
    cooldown = [sum(choice.is_cooldown for choice in run.searches) for run in ranked]

    ax.bar(x, primary, label="primary searches", color="C0", alpha=0.8)
    ax.bar(x, cooldown, bottom=primary, label="cooldown searches", color="C1", alpha=0.8)
    ax.set_title("LR search completions per run")
    ax.set_xlabel("run, sorted by final loss")
    ax.set_ylabel("search completions")
    ax.set_xticks(x)
    ax.set_xticklabels([str(run.run) for run in ranked], rotation=90, fontsize=7)
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "search_counts.png", dpi=180)
    plt.close(fig)


def plot_all(runs: list[Run], output_dir: Path) -> None:
    plot_loss_curves(runs, output_dir)
    plot_final_loss_bars(runs, output_dir)
    plot_heatmaps(runs, output_dir)
    plot_final_loss_by_config(runs, output_dir)
    plot_muon_lr_curves(runs, output_dir)
    plot_search_evaluation_counts(runs, output_dir)


def default_output_dir(log_path: Path) -> Path:
    return log_path.with_suffix("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a CIFAR overfit search log and write plots, CSV data, "
            "and a text summary."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help="Path to the log file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a directory with the log stem.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(log_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(log_path)
    if not runs:
        raise SystemExit(f"No runs parsed from {log_path}")

    write_csvs(runs, output_dir)
    plot_all(runs, output_dir)
    summary = write_summary(runs, log_path, output_dir)

    print(summary)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

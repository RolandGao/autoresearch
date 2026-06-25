#!/usr/bin/env python3
"""Plot the CIFAR simple scheduler search log."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "20260624_222539_669266" / "cifar_search_abs_diff.log"
DEFAULT_OUTPUT_DIR = (
    HERE / "20260624_222539_669266" / "cifar_search_abs_diff_plots"
)

OUTPUT_SUMMARY = "summary.txt"
OUTPUT_CURVES = "curves.png"

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 ]+):\s+(?P<value>.+)$")


@dataclass(frozen=True)
class HParams:
    muon_lr: float | None = None
    muon_momentum: float | None = None
    bias_lr: float | None = None
    head_lr: float | None = None

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "HParams":
        return cls(
            muon_lr=parse_optional_float(fields.get("muon_lr")),
            muon_momentum=parse_optional_float(fields.get("muon_momentum")),
            bias_lr=parse_optional_float(fields.get("bias_lr")),
            head_lr=parse_optional_float(fields.get("head_lr")),
        )


@dataclass
class SearchPathPoint:
    step: int
    hparams: HParams
    tta_val_acc: float


@dataclass
class SearchPath:
    interval: int
    group: int
    points: list[SearchPathPoint] = field(default_factory=list)

    @property
    def final_acc(self) -> float | None:
        return self.points[-1].tta_val_acc if self.points else None

    @property
    def final_hparams(self) -> HParams | None:
        return self.points[-1].hparams if self.points else None


@dataclass
class CooldownCandidate:
    interval: int
    eval_index: int
    hparams: HParams
    tta_val_acc: float | None
    blocked: bool


@dataclass
class MainEval:
    interval: int
    index: int
    hparams: HParams
    main_acc: float | None
    best_cooldown_acc: float | None
    blocked: bool
    candidates: list[CooldownCandidate] = field(default_factory=list)


@dataclass
class TrainInterval:
    run: int
    interval: int
    start_step: int
    completed_steps: int
    total_steps: int
    hparams: HParams
    muon_nesterov: str | None


@dataclass
class TrainPoint:
    run: int
    interval: int
    step: int
    loss: float


@dataclass
class Run:
    run: int
    header: dict[str, str] = field(default_factory=dict)
    summary: dict[str, str] = field(default_factory=dict)
    train_intervals: list[TrainInterval] = field(default_factory=list)
    train_loss: list[TrainPoint] = field(default_factory=list)
    main_evals: list[MainEval] = field(default_factory=list)
    cooldown_candidates: list[CooldownCandidate] = field(default_factory=list)
    search_paths: list[SearchPath] = field(default_factory=list)

    @property
    def final_tta_val_acc(self) -> float | None:
        return parse_optional_float(self.summary.get("TTA val acc"))

    @property
    def final_val_acc(self) -> float | None:
        return parse_optional_float(self.summary.get("Val acc"))

    @property
    def run_seconds(self) -> float | None:
        return parse_optional_float(self.summary.get("Run seconds"))

    @property
    def batch_size(self) -> int | None:
        return parse_optional_int(self.header.get("batch_size")) or parse_optional_int(
            self.summary.get("Batch size")
        )

    @property
    def train_steps(self) -> int | None:
        return parse_optional_int(self.summary.get("Train steps")) or parse_optional_int(
            self.header.get("total_steps")
        )


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().rstrip(",")
    if value.lower() in {"none", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(value: str | None) -> int | None:
    parsed = parse_optional_float(value)
    if parsed is None:
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def parse_kv_line(line: str) -> dict[str, str]:
    return {match["key"]: match["value"].rstrip(",") for match in KV_RE.finditer(line)}


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def style_axes(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def color_for_interval(interval: int) -> str:
    return f"C{interval % 10}"


def selected_path_by_interval(run: Run) -> dict[int, SearchPath]:
    selected: dict[int, SearchPath] = {}
    for path in run.search_paths:
        selected[path.interval] = path
    return selected


def assign_interval(
    interval: int,
    run: Run,
    pending_main_evals: list[MainEval],
    pending_candidates: list[CooldownCandidate],
    pending_paths: list[SearchPath],
) -> None:
    for main_eval in pending_main_evals:
        main_eval.interval = interval
        run.main_evals.append(main_eval)

    for candidate in pending_candidates:
        candidate.interval = interval
        run.cooldown_candidates.append(candidate)

    for path in pending_paths:
        path.interval = interval
        run.search_paths.append(path)


def parse_log(log_path: Path) -> Run:
    run = Run(run=0)
    pending_main_evals: list[MainEval] = []
    pending_candidates: list[CooldownCandidate] = []
    pending_paths: list[SearchPath] = []
    current_main_eval: MainEval | None = None
    current_path: SearchPath | None = None
    last_line_was_path = False

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                last_line_was_path = False
                continue

            if line.startswith("cifar_search_simple "):
                fields = parse_kv_line(line)
                run = Run(run=int(fields.get("run", "0")), header=fields)
                continue

            if line.startswith("main hparams: "):
                fields = parse_kv_line(line)
                current_main_eval = MainEval(
                    interval=-1,
                    index=len(run.main_evals) + len(pending_main_evals),
                    hparams=HParams.from_fields(fields),
                    main_acc=parse_optional_float(fields.get("main")),
                    best_cooldown_acc=parse_optional_float(fields.get("best_cooldown")),
                    blocked="-> blocked" in line,
                )
                pending_main_evals.append(current_main_eval)
                last_line_was_path = False
                continue

            if line.startswith("muon_lr=") and " -> " in line:
                fields = parse_kv_line(line)
                candidate = CooldownCandidate(
                    interval=-1,
                    eval_index=current_main_eval.index if current_main_eval else -1,
                    hparams=HParams.from_fields(fields),
                    tta_val_acc=parse_optional_float(fields.get("tta_val_acc")),
                    blocked="-> blocked" in line,
                )
                pending_candidates.append(candidate)
                if current_main_eval is not None:
                    current_main_eval.candidates.append(candidate)
                last_line_was_path = False
                continue

            if line.startswith("search_path "):
                fields = parse_kv_line(line)
                step = int(fields["step"])
                if step == 0 or not last_line_was_path or current_path is None:
                    current_path = SearchPath(
                        interval=-1,
                        group=len(run.search_paths) + len(pending_paths),
                    )
                    pending_paths.append(current_path)

                tta_val_acc = parse_optional_float(fields.get("tta_val_acc"))
                if tta_val_acc is not None:
                    current_path.points.append(
                        SearchPathPoint(
                            step=step,
                            hparams=HParams.from_fields(fields),
                            tta_val_acc=tta_val_acc,
                        )
                    )
                last_line_was_path = True
                continue

            if line.startswith("train_hparams "):
                fields = parse_kv_line(line)
                interval = int(fields["interval"])
                assign_interval(
                    interval,
                    run,
                    pending_main_evals,
                    pending_candidates,
                    pending_paths,
                )
                pending_main_evals = []
                pending_candidates = []
                pending_paths = []
                current_main_eval = None
                current_path = None
                last_line_was_path = False

                run.train_intervals.append(
                    TrainInterval(
                        run=int(fields.get("run", run.run)),
                        interval=interval,
                        start_step=int(fields["start_step"]),
                        completed_steps=int(fields["completed_steps"]),
                        total_steps=int(fields["total_steps"]),
                        hparams=HParams.from_fields(fields),
                        muon_nesterov=fields.get("muon_nesterov"),
                    )
                )
                continue

            if line.startswith("train_loss "):
                fields = parse_kv_line(line)
                run.train_loss.append(
                    TrainPoint(
                        run=int(fields.get("run", run.run)),
                        interval=int(fields["interval"]),
                        step=int(fields["step"]),
                        loss=float(fields["loss"]),
                    )
                )
                last_line_was_path = False
                continue

            summary_match = SUMMARY_RE.match(line)
            if summary_match:
                run.summary[summary_match["key"].strip()] = summary_match[
                    "value"
                ].strip()
                last_line_was_path = False
                continue

            last_line_was_path = False

    if pending_main_evals or pending_candidates or pending_paths:
        next_interval = (
            max((item.interval for item in run.train_intervals), default=-1) + 1
        )
        assign_interval(
            next_interval,
            run,
            pending_main_evals,
            pending_candidates,
            pending_paths,
        )

    return run


def interval_loss_ranges(run: Run) -> dict[int, tuple[float | None, float | None]]:
    ranges: dict[int, tuple[float | None, float | None]] = {}
    for interval in sorted({point.interval for point in run.train_loss}):
        points = [point for point in run.train_loss if point.interval == interval]
        if points:
            ranges[interval] = (points[0].loss, points[-1].loss)
    return ranges


def format_hparams(hparams: HParams, prefix: str = "") -> str:
    return (
        f"{prefix}muon_lr={format_number(hparams.muon_lr)} "
        f"{prefix}momentum={format_number(hparams.muon_momentum)} "
        f"{prefix}bias_lr={format_number(hparams.bias_lr)} "
        f"{prefix}head_lr={format_number(hparams.head_lr)}"
    )


def format_hparam_columns(hparams: HParams) -> str:
    return (
        f"muon_lr={format_number(hparams.muon_lr):<8} "
        f"momentum={format_number(hparams.muon_momentum):<8} "
        f"bias_lr={format_number(hparams.bias_lr):<8} "
        f"head_lr={format_number(hparams.head_lr):<8}"
    )


def format_candidate_row(
    phase: str,
    score: str,
    hparams: HParams,
) -> str:
    return (
        f"phase={phase:<8} score={score:<8} {format_hparam_columns(hparams)}"
    )


def best_cooldown_candidate(main_eval: MainEval) -> CooldownCandidate | None:
    return max(
        (item for item in main_eval.candidates if item.tta_val_acc is not None),
        key=lambda item: item.tta_val_acc or -math.inf,
        default=None,
    )


def hparams_key(hparams: HParams) -> tuple[float | None, float | None, float | None, float | None]:
    return (
        hparams.muon_lr,
        hparams.muon_momentum,
        hparams.bias_lr,
        hparams.head_lr,
    )


def selected_main_eval(run: Run, train_interval: TrainInterval) -> MainEval | None:
    selected_key = hparams_key(train_interval.hparams)
    for main_eval in run.main_evals:
        if (
            main_eval.interval == train_interval.interval
            and hparams_key(main_eval.hparams) == selected_key
        ):
            return main_eval
    return None


def write_summary(run: Run, log_path: Path, output_dir: Path) -> None:
    selected_paths = selected_path_by_interval(run)
    loss_ranges = interval_loss_ranges(run)
    best_main = max(
        (item for item in run.main_evals if item.main_acc is not None),
        key=lambda item: item.main_acc or -math.inf,
        default=None,
    )
    best_cooldown = max(
        (item for item in run.main_evals if item.best_cooldown_acc is not None),
        key=lambda item: item.best_cooldown_acc or -math.inf,
        default=None,
    )
    best_candidate = max(
        (item for item in run.cooldown_candidates if item.tta_val_acc is not None),
        key=lambda item: item.tta_val_acc or -math.inf,
        default=None,
    )

    lines = [
        "CIFAR search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Curves plot: {output_dir / OUTPUT_CURVES}",
        "",
        (
            f"Run {run.run}: batch_size={format_number(run.batch_size)} "
            f"train_epochs={run.header.get('train_epochs', 'NA')} "
            f"n_steps={run.header.get('n_steps', 'NA')} "
            f"m_steps={run.header.get('m_steps', 'NA')}"
        ),
        (
            f"Final val_acc={format_number(run.final_val_acc)} "
            f"tta_val_acc={format_number(run.final_tta_val_acc)} "
            f"run_seconds={format_number(run.run_seconds)}"
        ),
        (
            f"Parsed {len(run.main_evals)} main evaluations, "
            f"{sum(item.blocked for item in run.main_evals)} blocked main evaluations, "
            f"{len(run.cooldown_candidates)} cooldown candidates, "
            f"{sum(item.blocked for item in run.cooldown_candidates)} blocked candidates, "
            f"{len(run.train_loss)} train-loss points."
        ),
        "",
    ]

    if best_main is not None:
        lines.append(
            "Best main eval: "
            f"interval={best_main.interval} main={best_main.main_acc:.4f} "
            f"{format_hparams(best_main.hparams)}"
        )
    if best_cooldown is not None:
        lines.append(
            "Best eval cooldown: "
            f"interval={best_cooldown.interval} best_cooldown="
            f"{best_cooldown.best_cooldown_acc:.4f} "
            f"{format_hparams(best_cooldown.hparams, prefix='main_')}"
        )
    if best_candidate is not None:
        lines.append(
            "Best cooldown candidate: "
            f"interval={best_candidate.interval} tta_val_acc="
            f"{best_candidate.tta_val_acc:.4f} "
            f"{format_hparams(best_candidate.hparams)}"
        )
    lines.append("")

    lines.append("Selected training intervals")
    selected_main_lines = []
    selected_cooldown_lines = []
    for train_interval in sorted(run.train_intervals, key=lambda item: item.interval):
        path = selected_paths.get(train_interval.interval)
        start_loss, end_loss = loss_ranges.get(train_interval.interval, (None, None))
        main_eval = selected_main_eval(run, train_interval)
        cooldown = best_cooldown_candidate(main_eval) if main_eval is not None else None
        if cooldown is None:
            if path is not None and path.final_hparams is not None:
                cooldown_text = (
                    f"best_cooldown={format_number(path.final_acc)} "
                    f"{format_hparams(path.final_hparams, prefix='cooldown_')}"
                )
            else:
                cooldown_text = (
                    "best_cooldown=NA cooldown_muon_lr=NA cooldown_momentum=NA "
                    "cooldown_bias_lr=NA cooldown_head_lr=NA"
                )
        else:
            cooldown_text = (
                f"best_cooldown={format_number(cooldown.tta_val_acc)} "
                f"{format_hparams(cooldown.hparams, prefix='cooldown_')}"
            )
        selected_main_lines.append(
            f"interval={train_interval.interval} phase=main     start_step="
            f"{train_interval.start_step} steps={train_interval.completed_steps} "
            f"muon_lr={format_number(train_interval.hparams.muon_lr)} "
            f"momentum={format_number(train_interval.hparams.muon_momentum)} "
            f"bias_lr={format_number(train_interval.hparams.bias_lr)} "
            f"head_lr={format_number(train_interval.hparams.head_lr)} "
            f"path_final_tta={format_number(path.final_acc if path else None)} "
            f"loss={format_number(start_loss)}->{format_number(end_loss)}"
        )
        selected_cooldown_lines.append(
            f"interval={train_interval.interval} phase=cooldown {cooldown_text}"
        )
    lines.extend(selected_main_lines)
    lines.extend(selected_cooldown_lines)
    lines.append("")

    lines.append("Main interval candidates and best cooldown hparams")
    for interval in sorted({item.interval for item in run.main_evals}):
        lines.append(f"interval={interval}")
        interval_evals = [
            item for item in run.main_evals if item.interval == interval
        ]
        for main_eval in interval_evals:
            best = best_cooldown_candidate(main_eval)
            if best is None:
                cooldown_hparams = HParams()
                best_tta = format_number(main_eval.best_cooldown_acc)
            else:
                cooldown_hparams = best.hparams
                best_tta = format_number(best.tta_val_acc)
            main_status = "blocked" if main_eval.blocked else format_number(
                main_eval.main_acc
            )
            lines.append(
                format_candidate_row(
                    "main",
                    main_status,
                    main_eval.hparams,
                )
            )
            lines.append(
                format_candidate_row(
                    "cooldown",
                    best_tta,
                    cooldown_hparams,
                )
            )

    (output_dir / OUTPUT_SUMMARY).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_curves(run: Run, output_dir: Path) -> None:
    selected_paths = selected_path_by_interval(run)
    intervals = sorted({item.interval for item in run.train_intervals})
    train_intervals = sorted(run.train_intervals, key=lambda item: item.interval)
    end_step_by_interval = {
        item.interval: item.start_step + item.completed_steps for item in train_intervals
    }
    interval_steps = [
        end_step_by_interval.get(interval, interval) for interval in intervals
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.2))
    acc_ax, muon_lr_ax, bias_lr_ax, head_lr_ax, momentum_ax, empty_ax = axes.flat
    empty_ax.set_visible(False)

    best_main_by_interval = []
    best_cooldown_by_interval = []
    selected_by_interval = []
    for interval in intervals:
        main_values = [
            item.main_acc
            for item in run.main_evals
            if item.interval == interval and item.main_acc is not None
        ]
        cooldown_values = [
            item.best_cooldown_acc
            for item in run.main_evals
            if item.interval == interval and item.best_cooldown_acc is not None
        ]
        best_main_by_interval.append(max(main_values) if main_values else None)
        best_cooldown_by_interval.append(max(cooldown_values) if cooldown_values else None)
        path = selected_paths.get(interval)
        selected_by_interval.append(path.final_acc if path else None)

    def plot_optional_series(ax, label: str, values: list[float | None], marker: str):
        filtered = [
            (step, value)
            for step, value in zip(interval_steps, values)
            if value is not None
        ]
        if filtered:
            xs, ys = zip(*filtered)
            ax.plot(xs, ys, marker=marker, linewidth=1.8, label=label)

    plot_optional_series(acc_ax, "best main", best_main_by_interval, "o")
    plot_optional_series(acc_ax, "best cooldown", best_cooldown_by_interval, "s")
    plot_optional_series(acc_ax, "selected path", selected_by_interval, "^")
    final_tta = run.final_tta_val_acc
    if final_tta is not None:
        acc_ax.axhline(final_tta, color="#333333", linestyle="--", linewidth=1.1)
        acc_ax.text(
            interval_steps[-1] if interval_steps else 0,
            final_tta,
            f" final {final_tta:.4f}",
            va="bottom",
            fontsize=9,
        )
    acc_ax.set_title("Validation Accuracy by Step")
    acc_ax.set_xlabel("Training step")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend(fontsize=9)
    style_axes(acc_ax)

    lr_panels = [
        (muon_lr_ax, "Muon LR", "muon_lr"),
        (bias_lr_ax, "Bias LR", "bias_lr"),
        (head_lr_ax, "Head LR", "head_lr"),
    ]

    def plot_piecewise_hparam(ax, attr: str) -> None:
        plot_steps: list[int] = []
        plot_values: list[float] = []
        for train_interval in train_intervals:
            value = getattr(train_interval.hparams, attr)
            if value is None:
                continue
            plot_steps.append(train_interval.start_step)
            plot_values.append(value)
        if not plot_steps:
            return

        last_interval = max(
            train_intervals,
            key=lambda item: item.start_step + item.completed_steps,
        )
        last_step = last_interval.start_step + last_interval.completed_steps
        plot_steps.append(last_step)
        plot_values.append(plot_values[-1])
        ax.step(plot_steps, plot_values, where="post", linewidth=2.0)
        ax.set_xlim(plot_steps[0], last_step)

    for ax, title, attr in lr_panels:
        plot_piecewise_hparam(ax, attr)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Learning rate")
        style_axes(ax)

    plot_piecewise_hparam(momentum_ax, "muon_momentum")
    momentum_ax.set_ylim(-0.03, 1.03)
    momentum_ax.set_title("Selected Muon Momentum")
    momentum_ax.set_xlabel("Training step")
    momentum_ax.set_ylabel("Momentum")
    style_axes(momentum_ax)

    fig.suptitle("CIFAR search selected schedule")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / OUTPUT_CURVES, dpi=180)
    plt.close(fig)


def plot_search_paths(run: Run, output_dir: Path) -> None:
    selected_paths = selected_path_by_interval(run)
    intervals = sorted(selected_paths)
    if not intervals:
        return

    cols = min(3, len(intervals))
    rows = math.ceil(len(intervals) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.1 * cols, 3.8 * rows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, interval in zip(axes.flat, intervals):
        ax.set_visible(True)
        path = selected_paths[interval]
        xs = [point.step for point in path.points]
        ys = [point.tta_val_acc for point in path.points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color_for_interval(interval))
        if path.final_hparams is not None:
            final = path.final_hparams
            label = (
                f"final mu={format_number(final.muon_lr, 2)}\n"
                f"mom={format_number(final.muon_momentum, 2)} "
                f"b={format_number(final.bias_lr, 2)} "
                f"h={format_number(final.head_lr, 2)}"
            )
            ax.text(
                0.03,
                0.97,
                label,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
            )
        ax.set_title(f"Interval {interval} Selected Search Path")
        ax.set_xlabel("Search step")
        ax.set_ylabel("TTA val acc")
        style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_dir / OUTPUT_SEARCH_PATHS, dpi=180)
    plt.close(fig)


def scatter_hparam(ax, points, attr: str, ylabel: str, title: str, symlog=False) -> None:
    for interval in sorted({point.interval for point in points}):
        interval_points = [point for point in points if point.interval == interval]
        xs = [
            getattr(point.hparams, attr)
            for point in interval_points
            if getattr(point.hparams, attr) is not None
        ]
        ys = [
            getattr(point, ylabel)
            for point in interval_points
            if getattr(point.hparams, attr) is not None
        ]
        if xs and ys:
            ax.scatter(
                xs,
                ys,
                s=18,
                alpha=0.72,
                color=color_for_interval(interval),
                label=f"interval {interval}",
            )
    if symlog:
        ax.set_xscale("symlog", linthresh=1e-3)
    else:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(attr)
    ax.set_ylabel(ylabel)
    style_axes(ax)


def plot_main_search(run: Run, output_dir: Path) -> None:
    evals = [item for item in run.main_evals if item.main_acc is not None]
    if not evals:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    order_ax, muon_ax, bias_ax, head_ax = axes.flat

    order_ax.scatter(
        [item.index for item in evals],
        [item.main_acc for item in evals],
        s=18,
        alpha=0.72,
        label="main",
    )
    cooldown = [item for item in evals if item.best_cooldown_acc is not None]
    if cooldown:
        order_ax.scatter(
            [item.index for item in cooldown],
            [item.best_cooldown_acc for item in cooldown],
            s=18,
            alpha=0.72,
            label="best cooldown",
        )
    blocked_indexes = [item.index for item in run.main_evals if item.blocked]
    if blocked_indexes:
        ymin, ymax = order_ax.get_ylim()
        order_ax.scatter(
            blocked_indexes,
            [ymin + (ymax - ymin) * 0.02] * len(blocked_indexes),
            marker="x",
            s=20,
            color="#aa3333",
            label="blocked",
        )
    for interval in sorted({item.interval for item in run.main_evals}):
        interval_indexes = [
            item.index for item in run.main_evals if item.interval == interval
        ]
        if interval_indexes:
            order_ax.axvline(min(interval_indexes), color="#cccccc", linewidth=0.8)
    order_ax.set_title("Main Search Over Evaluation Order")
    order_ax.set_xlabel("Evaluation index")
    order_ax.set_ylabel("Accuracy")
    order_ax.legend(fontsize=9)
    style_axes(order_ax)

    scatter_hparam(muon_ax, evals, "muon_lr", "main_acc", "Main Acc vs Muon LR")
    scatter_hparam(bias_ax, evals, "bias_lr", "main_acc", "Main Acc vs Bias LR")
    scatter_hparam(
        head_ax, evals, "head_lr", "main_acc", "Main Acc vs Head LR", symlog=True
    )

    fig.tight_layout()
    fig.savefig(output_dir / OUTPUT_MAIN_SEARCH, dpi=180)
    plt.close(fig)


def plot_cooldown_candidates(run: Run, output_dir: Path) -> None:
    candidates = [
        item for item in run.cooldown_candidates if item.tta_val_acc is not None
    ]
    if not candidates:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    muon_ax, momentum_ax, bias_ax, head_ax = axes.flat

    for ax, attr, title, symlog in [
        (muon_ax, "muon_lr", "TTA Acc vs Cooldown Muon LR", False),
        (momentum_ax, "muon_momentum", "TTA Acc vs Cooldown Momentum", False),
        (bias_ax, "bias_lr", "TTA Acc vs Cooldown Bias LR", False),
        (head_ax, "head_lr", "TTA Acc vs Cooldown Head LR", True),
    ]:
        for interval in sorted({item.interval for item in candidates}):
            interval_points = [item for item in candidates if item.interval == interval]
            xs = [
                getattr(item.hparams, attr)
                for item in interval_points
                if getattr(item.hparams, attr) is not None
            ]
            ys = [
                item.tta_val_acc
                for item in interval_points
                if getattr(item.hparams, attr) is not None
            ]
            if xs and ys:
                ax.scatter(
                    xs,
                    ys,
                    s=18,
                    alpha=0.68,
                    color=color_for_interval(interval),
                    label=f"interval {interval}",
                )
        if attr == "muon_momentum":
            ax.set_xlim(-0.03, 1.03)
        elif symlog:
            ax.set_xscale("symlog", linthresh=1e-3)
        else:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(attr)
        ax.set_ylabel("tta_val_acc")
        style_axes(ax)

    handles, labels = muon_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / OUTPUT_COOLDOWN, dpi=180)
    plt.close(fig)


def plot_all(run: Run, log_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(run, log_path, output_dir)
    plot_curves(run, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot the CIFAR simple scheduler search log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file to plot. Defaults to {DEFAULT_LOG}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = parse_log(args.log)
    if not run.train_loss and not run.main_evals:
        raise SystemExit(f"No CIFAR search data parsed from {args.log}")

    plot_all(run, args.log, args.output_dir)

    print(f"Parsed run {run.run} from {args.log}")
    print(f"Wrote {args.output_dir / OUTPUT_SUMMARY}")
    print(f"Wrote {args.output_dir / OUTPUT_CURVES}")
    print(f"Final TTA val acc: {format_number(run.final_tta_val_acc)}")


if __name__ == "__main__":
    main()

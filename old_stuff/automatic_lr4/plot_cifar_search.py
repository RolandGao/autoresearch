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
LOG_FILE_PATH = HERE / "20260630_014033_070828" / "cifar_simplified_cooldown.log"

OUTPUT_SUMMARY = "summary.txt"
OUTPUT_CURVES = "curves.png"
OUTPUT_LANDSCAPES = "landscapes.png"
OUTPUT_LANDSCAPES_MORE = "landscapes_more.png"
MAX_COOLDOWN_LANDSCAPE_ROWS = 24

DEFAULT_SEARCH_HPARAMS = ["muon_lr", "muon_momentum", "bias_lr", "head_lr"]
DEFAULT_COOLDOWN_HPARAMS = ["muon_lr", "bias_lr", "head_lr"]
HPARAM_ATTRS = (
    "muon_lr",
    "muon_momentum",
    "bias_lr",
    "whiten_bias_lr",
    "bn_bias_lr",
    "head_lr",
)
HPARAM_DISPLAY_NAMES = {
    "muon_lr": "muon_lr",
    "muon_momentum": "momentum",
    "bias_lr": "bias_lr",
    "whiten_bias_lr": "whiten_bias_lr",
    "bn_bias_lr": "bn_bias_lr",
    "head_lr": "head_lr",
}
HPARAM_TITLES = {
    "muon_lr": "Muon LR",
    "muon_momentum": "Muon Momentum",
    "bias_lr": "Bias LR",
    "whiten_bias_lr": "Whiten Bias LR",
    "bn_bias_lr": "BN Bias LR",
    "head_lr": "Head LR",
}

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")
SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 ]+):\s+(?P<value>.+)$")


def default_output_dir(log_path: Path) -> Path:
    return log_path.parent / f"{log_path.stem}_plots"


def cooldown_landscape_output_path(output_dir: Path, page_index: int) -> Path:
    if page_index == 0:
        return output_dir / OUTPUT_LANDSCAPES_MORE
    output_name = Path(OUTPUT_LANDSCAPES_MORE)
    return output_dir / f"{output_name.stem}_{page_index + 1:03d}{output_name.suffix}"


@dataclass(frozen=True)
class HParams:
    muon_lr: float | None = None
    muon_momentum: float | None = None
    bias_lr: float | None = None
    whiten_bias_lr: float | None = None
    bn_bias_lr: float | None = None
    head_lr: float | None = None

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "HParams":
        return cls(
            muon_lr=parse_optional_float(fields.get("muon_lr")),
            muon_momentum=parse_optional_float(fields.get("muon_momentum")),
            bias_lr=parse_optional_float(fields.get("bias_lr")),
            whiten_bias_lr=parse_optional_float(fields.get("whiten_bias_lr")),
            bn_bias_lr=parse_optional_float(fields.get("bn_bias_lr")),
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
    if value.lower() in {"none", "nan", "inf", "+inf", "-inf"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def is_finite_number(value: float | int | None) -> bool:
    return value is not None and math.isfinite(value)


def finite_or_neg_inf(value: float | int | None) -> float:
    return float(value) if is_finite_number(value) else -math.inf


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
    if not is_finite_number(value):
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


def parse_hparam_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def search_hparam_names(run: Run) -> list[str]:
    return parse_hparam_names(run.header.get("search_hparams")) or list(
        DEFAULT_SEARCH_HPARAMS
    )


def cooldown_hparam_names(run: Run) -> list[str]:
    return parse_hparam_names(run.header.get("cooldown_search_hparams")) or list(
        DEFAULT_COOLDOWN_HPARAMS
    )


def hparam_value(hparams: HParams, attr: str) -> float | None:
    return getattr(hparams, attr, None)


def hparam_display_name(attr: str) -> str:
    return HPARAM_DISPLAY_NAMES.get(attr, attr)


def hparam_title(attr: str) -> str:
    return HPARAM_TITLES.get(attr, attr.replace("_", " ").title())


def present_hparam_attrs(hparams: HParams) -> list[str]:
    return [attr for attr in HPARAM_ATTRS if hparam_value(hparams, attr) is not None]


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


def format_hparams(
    hparams: HParams,
    prefix: str = "",
    attrs: list[str] | None = None,
) -> str:
    attrs = attrs or present_hparam_attrs(hparams) or list(DEFAULT_SEARCH_HPARAMS)
    return " ".join(
        f"{prefix}{hparam_display_name(attr)}={format_number(hparam_value(hparams, attr))}"
        for attr in attrs
    )


def format_hparam_columns(
    hparams: HParams,
    attrs: list[str] | None = None,
) -> str:
    attrs = attrs or present_hparam_attrs(hparams) or list(DEFAULT_SEARCH_HPARAMS)
    return " ".join(
        f"{hparam_display_name(attr)}={format_number(hparam_value(hparams, attr)):<8}"
        for attr in attrs
    )


def format_candidate_row(
    phase: str,
    score: str,
    hparams: HParams,
    attrs: list[str] | None = None,
) -> str:
    return (
        f"phase={phase:<8} score={score:<8} "
        f"{format_hparam_columns(hparams, attrs=attrs)}"
    )


def best_cooldown_candidate(main_eval: MainEval) -> CooldownCandidate | None:
    return max(
        (item for item in main_eval.candidates if is_finite_number(item.tta_val_acc)),
        key=lambda item: finite_or_neg_inf(item.tta_val_acc),
        default=None,
    )


def hparams_key(
    hparams: HParams,
    attrs: list[str] | tuple[str, ...] = HPARAM_ATTRS,
) -> tuple[float | None, ...]:
    return tuple(hparam_value(hparams, attr) for attr in attrs)


def same_hparam_value(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def hparams_match_except(candidate: HParams, center: HParams, varied_attr: str) -> bool:
    for attr in HPARAM_ATTRS:
        if attr == varied_attr:
            continue
        if not same_hparam_value(
            hparam_value(candidate, attr),
            hparam_value(center, attr),
        ):
            return False
    return True


def selected_main_eval(run: Run, train_interval: TrainInterval) -> MainEval | None:
    selected_attrs = search_hparam_names(run)
    selected_key = hparams_key(train_interval.hparams, selected_attrs)
    matches = [
        main_eval
        for main_eval in run.main_evals
        if (
            main_eval.interval == train_interval.interval
            and hparams_key(main_eval.hparams, selected_attrs) == selected_key
        )
    ]
    if not matches:
        return None

    return max(
        matches,
        key=lambda main_eval: (
            is_finite_number(main_eval.best_cooldown_acc),
            finite_or_neg_inf(main_eval.best_cooldown_acc),
            finite_or_neg_inf(main_eval.main_acc),
            len(main_eval.candidates),
            main_eval.index,
        ),
    )


def write_summary(run: Run, log_path: Path, output_dir: Path) -> None:
    search_names = search_hparam_names(run)
    selected_paths = selected_path_by_interval(run)
    best_main = max(
        (item for item in run.main_evals if is_finite_number(item.main_acc)),
        key=lambda item: finite_or_neg_inf(item.main_acc),
        default=None,
    )
    best_cooldown = max(
        (item for item in run.main_evals if is_finite_number(item.best_cooldown_acc)),
        key=lambda item: finite_or_neg_inf(item.best_cooldown_acc),
        default=None,
    )
    best_candidate = max(
        (item for item in run.cooldown_candidates if is_finite_number(item.tta_val_acc)),
        key=lambda item: finite_or_neg_inf(item.tta_val_acc),
        default=None,
    )

    lines = [
        "CIFAR search summary",
        f"Input log: {log_path}",
        f"Output directory: {output_dir}",
        f"Curves plot: {output_dir / OUTPUT_CURVES}",
        f"Landscape plot: {output_dir / OUTPUT_LANDSCAPES}",
        f"More landscape plot: {output_dir / OUTPUT_LANDSCAPES_MORE}",
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
            f"{format_hparams(best_main.hparams, attrs=search_names)}"
        )
    if best_cooldown is not None:
        lines.append(
            "Best eval cooldown: "
            f"interval={best_cooldown.interval} best_cooldown="
            f"{best_cooldown.best_cooldown_acc:.4f} "
            f"{format_hparams(best_cooldown.hparams, prefix='main_', attrs=search_names)}"
        )
    if best_candidate is not None:
        lines.append(
            "Best cooldown candidate: "
            f"interval={best_candidate.interval} tta_val_acc="
            f"{best_candidate.tta_val_acc:.4f} "
            f"{format_hparams(best_candidate.hparams, attrs=search_names)}"
        )
    lines.append("")

    lines.append("main:")
    selected_main_lines = []
    selected_cooldown_lines = []
    for train_interval in sorted(run.train_intervals, key=lambda item: item.interval):
        path = selected_paths.get(train_interval.interval)
        main_eval = selected_main_eval(run, train_interval)
        cooldown = best_cooldown_candidate(main_eval) if main_eval is not None else None
        if cooldown is None:
            if path is not None and path.final_hparams is not None:
                cooldown_hparams = path.final_hparams
            else:
                cooldown_hparams = HParams()
        else:
            cooldown_hparams = cooldown.hparams
        selected_main_lines.append(
            f"interval={train_interval.interval} "
            f"{format_hparams(train_interval.hparams, attrs=search_names)} "
            f"path_final_tta={format_number(path.final_acc if path else None)}"
        )
        selected_cooldown_lines.append(
            f"interval={train_interval.interval} "
            f"{format_hparams(cooldown_hparams, attrs=search_names)}"
        )
    lines.extend(selected_main_lines)
    lines.append("")
    lines.append("cooldown:")
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
                    attrs=search_names,
                )
            )
            lines.append(
                format_candidate_row(
                    "cooldown",
                    best_tta,
                    cooldown_hparams,
                    attrs=search_names,
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

    curve_hparams = [
        attr
        for attr in search_hparam_names(run)
        if any(
            is_finite_number(hparam_value(item.hparams, attr))
            for item in train_intervals
        )
    ]
    panel_count = 1 + len(curve_hparams)
    col_count = min(3, max(1, panel_count))
    row_count = math.ceil(panel_count / col_count)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(5.7 * col_count, 3.55 * row_count),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    acc_ax = flat_axes[0]
    for ax in flat_axes[panel_count:]:
        ax.set_visible(False)

    best_main_by_interval = []
    best_cooldown_by_interval = []
    selected_by_interval = []
    for interval in intervals:
        main_values = [
            item.main_acc
            for item in run.main_evals
            if item.interval == interval and is_finite_number(item.main_acc)
        ]
        cooldown_values = [
            item.best_cooldown_acc
            for item in run.main_evals
            if item.interval == interval and is_finite_number(item.best_cooldown_acc)
        ]
        best_main_by_interval.append(max(main_values) if main_values else None)
        best_cooldown_by_interval.append(max(cooldown_values) if cooldown_values else None)
        path = selected_paths.get(interval)
        selected_by_interval.append(path.final_acc if path else None)

    def plot_optional_series(ax, label: str, values: list[float | None], marker: str):
        filtered = [
            (step, value)
            for step, value in zip(interval_steps, values)
            if is_finite_number(value)
        ]
        if filtered:
            xs, ys = zip(*filtered)
            ax.plot(xs, ys, marker=marker, linewidth=1.8, label=label)

    plot_optional_series(acc_ax, "best main", best_main_by_interval, "o")
    plot_optional_series(acc_ax, "best cooldown", best_cooldown_by_interval, "s")
    plot_optional_series(acc_ax, "selected path", selected_by_interval, "^")
    final_tta = run.final_tta_val_acc
    if is_finite_number(final_tta):
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

    def plot_piecewise_hparam(ax, attr: str) -> None:
        plot_steps: list[int] = []
        plot_values: list[float] = []
        for train_interval in train_intervals:
            value = hparam_value(train_interval.hparams, attr)
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

    for ax, attr in zip(flat_axes[1:], curve_hparams):
        plot_piecewise_hparam(ax, attr)
        ax.set_title(f"Selected {hparam_title(attr)}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Momentum" if attr == "muon_momentum" else "Learning rate")
        if attr == "muon_momentum":
            ax.set_ylim(-0.03, 1.03)
        style_axes(ax)

    fig.suptitle("CIFAR search selected schedule")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / OUTPUT_CURVES, dpi=180)
    plt.close(fig)


def hparam_axis_scale(attr: str) -> tuple[str, dict[str, float]]:
    if attr == "head_lr":
        return "symlog", {"linthresh": 1e-3}
    if attr.endswith("_lr"):
        return "log", {}
    return "linear", {}


def unique_accuracy_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    best_by_x: dict[float, float] = {}
    for x_value, score in points:
        if not is_finite_number(x_value) or not is_finite_number(score):
            continue
        best_by_x[x_value] = max(score, best_by_x.get(x_value, -math.inf))
    return sorted(best_by_x.items())


def candidate_accuracy_landscape(
    candidates,
    center: HParams,
    varied_attr: str,
    score_attr: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for candidate in candidates:
        x_value = hparam_value(candidate.hparams, varied_attr)
        score = getattr(candidate, score_attr)
        if (
            x_value is None
            or score is None
            or not is_finite_number(x_value)
            or not is_finite_number(score)
            or not hparams_match_except(candidate.hparams, center, varied_attr)
        ):
            continue
        points.append((x_value, score))
    return unique_accuracy_points(points)


def landscape_ylabel(score_attr: str) -> str:
    if score_attr == "best_cooldown_acc":
        return "best cooldown tta_val_acc"
    return score_attr


def main_landscape_score_attr(run: Run, interval: int) -> str:
    if any(
        item.interval == interval and is_finite_number(item.best_cooldown_acc)
        for item in run.main_evals
    ):
        return "best_cooldown_acc"
    return "main_acc"


def best_main_interval_score(run: Run, interval: int, score_attr: str) -> float | None:
    scores = [
        getattr(item, score_attr)
        for item in run.main_evals
        if item.interval == interval and is_finite_number(getattr(item, score_attr))
    ]
    return max(scores) if scores else None


def plot_baseline(ax, value: float | None, label: str) -> None:
    if not is_finite_number(value):
        return
    ax.axhline(value, color="#555555", linestyle=":", linewidth=1.0)
    ax.text(
        0.98,
        0.95,
        f"{label}={format_number(value, 4)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.85,
        },
    )


def plot_accuracy_landscapes(run: Run, output_dir: Path) -> None:
    train_intervals = sorted(run.train_intervals, key=lambda item: item.interval)
    if not train_intervals:
        return

    search_names = search_hparam_names(run)
    cooldown_names = cooldown_hparam_names(run) or search_names
    columns = search_names
    row_count = len(train_intervals) * 2
    col_count = len(columns)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(4.2 * col_count, 2.75 * row_count),
        squeeze=False,
    )
    row_scores: dict[int, list[float]] = {row: [] for row in range(row_count)}

    for row_index, train_interval in enumerate(train_intervals):
        main_eval = selected_main_eval(run, train_interval)
        main_score_attr = main_landscape_score_attr(run, train_interval.interval)
        main_candidates = [
            item
            for item in run.main_evals
            if (
                item.interval == train_interval.interval
                and is_finite_number(getattr(item, main_score_attr))
            )
        ]
        cooldown = best_cooldown_candidate(main_eval) if main_eval is not None else None
        cooldown_candidates = (
            [
                item
                for item in main_eval.candidates
                if is_finite_number(item.tta_val_acc)
            ]
            if main_eval is not None
            else []
        )
        row_specs = [
            (
                row_index * 2,
                "main",
                train_interval.hparams,
                main_candidates,
                set(search_names),
                main_score_attr,
            ),
            (
                row_index * 2 + 1,
                "cooldown",
                cooldown.hparams if cooldown is not None else HParams(),
                cooldown_candidates,
                set(cooldown_names),
                "tta_val_acc",
            ),
        ]

        for row, phase, center, candidates, active_names, score_attr in row_specs:
            for col, attr in enumerate(columns):
                ax = axes[row][col]
                if attr not in active_names:
                    ax.set_visible(False)
                    continue

                points = candidate_accuracy_landscape(candidates, center, attr, score_attr)
                center_value = hparam_value(center, attr)
                color = color_for_interval(train_interval.interval)
                if points:
                    xs, ys = zip(*points)
                    row_scores[row].extend(ys)
                    ax.plot(xs, ys, marker="o", linewidth=1.6, color=color)
                    if center_value is not None:
                        ax.axvline(
                            center_value,
                            color="#333333",
                            linestyle="--",
                            linewidth=1.0,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "no 1D slice",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#777777",
                    )

                scale, scale_kwargs = hparam_axis_scale(attr)
                ax.set_xscale(scale, **scale_kwargs)
                ax.set_title(f"Interval {train_interval.interval} {phase}: {attr}")
                ax.set_xlabel(attr)
                ax.set_ylabel(landscape_ylabel(score_attr))
                if attr == "muon_momentum":
                    ax.set_xlim(-0.03, 1.03)
                style_axes(ax)

                if center_value is not None:
                    ax.text(
                        0.03,
                        0.95,
                        f"chosen={format_number(center_value, 3)}",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "#dddddd",
                            "alpha": 0.85,
                        },
                    )

    for row, scores in row_scores.items():
        if not scores:
            continue
        ymin = min(scores)
        ymax = max(scores)
        padding = (ymax - ymin) * 0.08 if ymax > ymin else 0.001
        for ax in axes[row]:
            if ax.get_visible():
                ax.set_ylim(ymin - padding, ymax + padding)

    fig.suptitle(
        "1D accuracy landscapes around selected main and cooldown hparams",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_dir / OUTPUT_LANDSCAPES, dpi=180)
    plt.close(fig)


def plot_all_cooldown_landscapes(run: Run, output_dir: Path) -> None:
    cooldown_names = cooldown_hparam_names(run)
    main_evals = [
        item
        for item in sorted(
            run.main_evals,
            key=lambda main_eval: (main_eval.interval, main_eval.index),
        )
        if item.candidates
    ]
    if not main_evals:
        return

    col_count = len(cooldown_names)
    for page_index, page_start in enumerate(
        range(0, len(main_evals), MAX_COOLDOWN_LANDSCAPE_ROWS)
    ):
        page_evals = main_evals[
            page_start : page_start + MAX_COOLDOWN_LANDSCAPE_ROWS
        ]
        row_count = len(page_evals)
        fig, axes = plt.subplots(
            row_count,
            col_count,
            figsize=(4.4 * col_count, 1.75 * row_count),
            squeeze=False,
        )
        row_scores: dict[int, list[float]] = {
            row: [] for row in range(row_count)
        }

        for row, main_eval in enumerate(page_evals):
            best = best_cooldown_candidate(main_eval)
            center = best.hparams if best is not None else HParams()
            baseline = best_main_interval_score(
                run, main_eval.interval, "best_cooldown_acc"
            )
            candidates = [
                item
                for item in main_eval.candidates
                if is_finite_number(item.tta_val_acc)
            ]
            color = color_for_interval(main_eval.interval)
            if baseline is not None:
                row_scores[row].append(baseline)

            for col, attr in enumerate(cooldown_names):
                ax = axes[row][col]
                points = candidate_accuracy_landscape(
                    candidates,
                    center,
                    attr,
                    "tta_val_acc",
                )
                center_value = hparam_value(center, attr)
                if points:
                    xs, ys = zip(*points)
                    row_scores[row].extend(ys)
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        markersize=3.5,
                        linewidth=1.2,
                        color=color,
                    )
                    if center_value is not None:
                        ax.axvline(
                            center_value,
                            color="#333333",
                            linestyle="--",
                            linewidth=0.9,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "no 1D slice",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#777777",
                    )
                plot_baseline(ax, baseline, "interval_best")

                scale, scale_kwargs = hparam_axis_scale(attr)
                ax.set_xscale(scale, **scale_kwargs)
                if col == 0:
                    ax.set_ylabel("tta_val_acc")
                else:
                    ax.set_ylabel("")
                ax.set_xlabel(attr)
                ax.set_title(
                    f"Interval {main_eval.interval} eval {main_eval.index}: {attr}",
                    fontsize=9,
                )
                style_axes(ax)
                if center_value is not None:
                    ax.text(
                        0.03,
                        0.95,
                        f"best={format_number(center_value, 3)}",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=7,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "#dddddd",
                            "alpha": 0.85,
                        },
                    )

        for row, scores in row_scores.items():
            if not scores:
                continue
            ymin = min(scores)
            ymax = max(scores)
            padding = (ymax - ymin) * 0.08 if ymax > ymin else 0.001
            for ax in axes[row]:
                if ax.get_visible():
                    ax.set_ylim(ymin - padding, ymax + padding)

        page_end = page_start + len(page_evals)
        fig.suptitle(
            (
                "Cooldown accuracy landscapes for every evaluated main hparam tuple "
                f"({page_start + 1}-{page_end} of {len(main_evals)})"
            ),
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.992))
        fig.savefig(cooldown_landscape_output_path(output_dir, page_index), dpi=140)
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
            label = "\n".join(
                format_hparams(final, attrs=search_hparam_names(run)).split()
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
        pairs = [
            (hparam_value(point.hparams, attr), getattr(point, ylabel))
            for point in interval_points
            if is_finite_number(hparam_value(point.hparams, attr))
            and is_finite_number(getattr(point, ylabel))
        ]
        if pairs:
            xs, ys = zip(*pairs)
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
    evals = [item for item in run.main_evals if is_finite_number(item.main_acc)]
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
    cooldown = [item for item in evals if is_finite_number(item.best_cooldown_acc)]
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
        item for item in run.cooldown_candidates if is_finite_number(item.tta_val_acc)
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
            pairs = [
                (hparam_value(item.hparams, attr), item.tta_val_acc)
                for item in interval_points
                if is_finite_number(hparam_value(item.hparams, attr))
                and is_finite_number(item.tta_val_acc)
            ]
            if pairs:
                xs, ys = zip(*pairs)
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
    plot_accuracy_landscapes(run, output_dir)
    plot_all_cooldown_landscapes(run, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot the CIFAR simple scheduler search log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=LOG_FILE_PATH,
        help=f"Log file to plot. Defaults to {LOG_FILE_PATH}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a sibling directory named <log-stem>_plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log
    output_dir = args.output_dir or default_output_dir(log_path)

    run = parse_log(log_path)
    if not run.train_loss and not run.main_evals:
        raise SystemExit(f"No CIFAR search data parsed from {log_path}")

    plot_all(run, log_path, output_dir)

    print(f"Parsed run {run.run} from {log_path}")
    print(f"Wrote {output_dir / OUTPUT_SUMMARY}")
    print(f"Wrote {output_dir / OUTPUT_CURVES}")
    print(f"Wrote {output_dir / OUTPUT_LANDSCAPES}")
    print(f"Wrote {output_dir / OUTPUT_LANDSCAPES_MORE}")
    print(f"Final TTA val acc: {format_number(run.final_tta_val_acc)}")


if __name__ == "__main__":
    main()

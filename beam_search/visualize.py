#!/usr/bin/env python3
"""Visualize beam_search3.log.

The log contains several beam-search runs with different fixed-prefix lengths.
This script parses the progress lines, selected survivors, final metrics, and
the summary table, then writes:

  - beam_search3_overview.png
  - beam_search3_details.pdf
  - beam_search3_lr_models.png
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

RUN_START_RE = re.compile(
    rf"^Beam search k=(?P<k>\d+), fixed prefix=(?P<prefix>\d+) steps"
)
INITIAL_LR_RE = re.compile(rf"^Fixed prefix LR_MULT:\s+(?P<lr>{FLOAT_RE})")
SEGMENT_STEPS_RE = re.compile(r"^Segment steps:\s+(?P<steps>\d+)")
ROUND_RE = re.compile(
    rf"^Beam round (?P<round>\d+): steps (?P<start>\d+)-(?P<end>\d+)"
    rf"\s+\|\s+current LR_MULT=(?P<lr>{FLOAT_RE})"
)
CANDIDATE_RE = re.compile(
    rf"^\s+candidate (?P<idx>\d+)/(?P<total>\d+)"
    rf"\s+\|\s+parent_step=(?P<parent>\d+)"
    rf"\s+\|\s+LR_MULT=(?P<lr>{FLOAT_RE})"
)
STEP_RE = re.compile(
    rf"step\s+(?P<step>\d+)(?:/\d+)?\s+\|\s+LR_MULT\s+(?P<lr>{FLOAT_RE})"
    rf"\s+\|\s+loss\s+(?P<loss>{FLOAT_RE})"
    rf"\s+\|\s+avg3\s+(?P<avg3>{FLOAT_RE})"
)
AVG_LOSS_RE = re.compile(
    rf"^\s+avg_loss=(?P<loss>{FLOAT_RE})\s+\|\s+(?P<status>kept|discarded)"
)
SURVIVOR_RE = re.compile(
    rf"^\s+#(?P<rank>\d+): avg_loss=(?P<loss>{FLOAT_RE}),"
    rf"\s+LR_MULT=(?P<lr>{FLOAT_RE}),"
    r"\s+path=round_(?P<round>\d+)_cand_(?P<candidate>\d+)_step_(?P<step>\d+)_"
)
BEST_LOSS_RE = re.compile(
    rf"^Best prefix=(?P<prefix>\d+) k=(?P<k>\d+) avg_loss:\s+(?P<loss>{FLOAT_RE})"
)
BEST_SCHEDULE_RE = re.compile(
    rf"^Best prefix=(?P<prefix>\d+) LR_MULT schedule:"
    rf"\s+(?P<initial>{FLOAT_RE}) for (?P<prefix_steps>\d+) steps,"
    r"\s+then \[(?P<schedule>.*)\]"
)
METRIC_RE = re.compile(r"^\s*(?P<key>[a-zA-Z_]+):\s+(?P<value>.+?)\s*$")
SUMMARY_RE = re.compile(
    rf"^prefix=(?P<prefix>\d+)\s+\|\s+k=(?P<k>\d+)"
    rf"\s+\|\s+steps=(?P<steps>\d+)"
    rf"\s+\|\s+val_bpb=(?P<val_bpb>{FLOAT_RE})"
    rf"\s+\|\s+best_avg_loss=(?P<best_avg_loss>{FLOAT_RE})"
    rf"\s+\|\s+final_lr_mult=(?P<final_lr_mult>{FLOAT_RE})"
)

METRIC_KEYS = {
    "k",
    "prefix_steps",
    "val_bpb",
    "best_avg_loss",
    "final_lr_mult",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
    "checkpoint_dir",
}


@dataclass(frozen=True)
class StepPoint:
    step: int
    lr: float
    loss: float
    avg3: float


@dataclass
class Candidate:
    round_idx: int
    candidate_idx: int
    total_candidates: int
    parent_step: int
    lr: float
    step_start: int
    step_end: int
    current_lr: float
    steps: list[StepPoint] = field(default_factory=list)
    avg_loss: float | None = None
    status: str | None = None

    @property
    def final_step(self) -> int:
        return self.steps[-1].step if self.steps else self.step_end


@dataclass(frozen=True)
class Survivor:
    rank: int
    round_idx: int
    candidate_idx: int
    step: int
    lr: float
    avg_loss: float


@dataclass
class Run:
    prefix: int
    k: int
    initial_lr: float = 1.5
    segment_steps: int = 10
    fixed_steps: list[StepPoint] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    survivors: list[Survivor] = field(default_factory=list)
    best_avg_loss: float | None = None
    best_schedule: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int | None:
        value = self.summary.get("steps", self.metrics.get("num_steps"))
        return int(value) if value is not None else None

    @property
    def val_bpb(self) -> float | None:
        value = self.summary.get("val_bpb", self.metrics.get("val_bpb"))
        return float(value) if value is not None else None


@dataclass(frozen=True)
class CommonLrModelFit:
    name: str
    formula: str
    avg_log_mse: float
    avg_log_rmse: float
    weighted_r2_log: float
    per_run_rmse: dict[int, float]
    grid_steps: np.ndarray
    grid_lrs: np.ndarray


def parse_number(value: str) -> int | float | str:
    value = value.strip()
    try:
        parsed = float(value)
    except ValueError:
        return value

    if parsed.is_integer() and re.fullmatch(r"[+-]?\d+", value):
        return int(parsed)
    return parsed


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(match.group(0)) for match in re.finditer(FLOAT_RE, value)]


def parse_log(path: Path) -> list[Run]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    runs: list[Run] = []
    current_run: Run | None = None
    current_round: tuple[int, int, int, float] | None = None
    current_candidate: Candidate | None = None

    for line in lines:
        if match := RUN_START_RE.search(line):
            if current_run is not None:
                runs.append(current_run)
            current_run = Run(
                prefix=int(match.group("prefix")),
                k=int(match.group("k")),
            )
            current_round = None
            current_candidate = None
            continue

        if current_run is None:
            continue

        if match := INITIAL_LR_RE.search(line):
            current_run.initial_lr = float(match.group("lr"))
            continue

        if match := SEGMENT_STEPS_RE.search(line):
            current_run.segment_steps = int(match.group("steps"))
            continue

        if match := ROUND_RE.search(line):
            current_round = (
                int(match.group("round")),
                int(match.group("start")),
                int(match.group("end")),
                float(match.group("lr")),
            )
            current_candidate = None
            continue

        if match := CANDIDATE_RE.search(line):
            if current_round is None:
                raise ValueError(f"Candidate found before a beam round: {line}")
            round_idx, step_start, step_end, current_lr = current_round
            current_candidate = Candidate(
                round_idx=round_idx,
                candidate_idx=int(match.group("idx")),
                total_candidates=int(match.group("total")),
                parent_step=int(match.group("parent")),
                lr=float(match.group("lr")),
                step_start=step_start,
                step_end=step_end,
                current_lr=current_lr,
            )
            current_run.candidates.append(current_candidate)
            continue

        step_matches = list(STEP_RE.finditer(line))
        if step_matches:
            points = [
                StepPoint(
                    step=int(match.group("step")),
                    lr=float(match.group("lr")),
                    loss=float(match.group("loss")),
                    avg3=float(match.group("avg3")),
                )
                for match in step_matches
            ]
            if current_candidate is not None:
                current_candidate.steps.extend(points)
            else:
                current_run.fixed_steps.extend(points)
            continue

        if match := AVG_LOSS_RE.search(line):
            if current_candidate is None:
                raise ValueError(f"Candidate result found without candidate: {line}")
            current_candidate.avg_loss = float(match.group("loss"))
            current_candidate.status = match.group("status")
            current_candidate = None
            continue

        if match := SURVIVOR_RE.search(line):
            current_run.survivors.append(
                Survivor(
                    rank=int(match.group("rank")),
                    round_idx=int(match.group("round")),
                    candidate_idx=int(match.group("candidate")),
                    step=int(match.group("step")),
                    lr=float(match.group("lr")),
                    avg_loss=float(match.group("loss")),
                )
            )
            continue

        if match := BEST_LOSS_RE.search(line):
            current_run.best_avg_loss = float(match.group("loss"))
            continue

        if match := BEST_SCHEDULE_RE.search(line):
            current_run.initial_lr = float(match.group("initial"))
            current_run.best_schedule = parse_float_list(match.group("schedule"))
            continue

        if match := METRIC_RE.search(line):
            key = match.group("key")
            if key in METRIC_KEYS:
                current_run.metrics[key] = parse_number(match.group("value"))

    if current_run is not None:
        runs.append(current_run)

    by_prefix = {run.prefix: run for run in runs}
    for line in lines:
        match = SUMMARY_RE.search(line)
        if not match:
            continue
        prefix = int(match.group("prefix"))
        run = by_prefix.get(prefix)
        if run is None:
            continue
        run.summary.update(
            {
                "k": int(match.group("k")),
                "steps": int(match.group("steps")),
                "val_bpb": float(match.group("val_bpb")),
                "best_avg_loss": float(match.group("best_avg_loss")),
                "final_lr_mult": float(match.group("final_lr_mult")),
            }
        )

    return sorted(runs, key=lambda item: item.prefix)


def selected_survivors(run: Run) -> list[Survivor]:
    return sorted(
        (survivor for survivor in run.survivors if survivor.rank == 1),
        key=lambda survivor: survivor.round_idx,
    )


def selected_candidates(run: Run) -> list[Candidate]:
    by_key = {
        (candidate.round_idx, candidate.candidate_idx): candidate
        for candidate in run.candidates
    }
    chosen: list[Candidate] = []
    for survivor in selected_survivors(run):
        candidate = by_key.get((survivor.round_idx, survivor.candidate_idx))
        if candidate is not None:
            chosen.append(candidate)
    return chosen


def selected_points(run: Run) -> list[StepPoint]:
    points = sorted(run.fixed_steps, key=lambda item: item.step)
    for candidate in selected_candidates(run):
        points.extend(candidate.steps)
    return points


def rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    averaged: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        count = min(index + 1, window)
        averaged.append(total / count)
    return averaged


def decay_counts(run: Run) -> np.ndarray:
    y = np.asarray(run.best_schedule, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    counts = np.log(y / run.initial_lr) / math.log(0.8)
    return np.rint(counts).astype(float)


def post_prefix_rounds(run: Run) -> np.ndarray:
    return np.arange(1, len(run.best_schedule) + 1, dtype=float)


def post_prefix_steps(run: Run) -> np.ndarray:
    return post_prefix_rounds(run) * run.segment_steps


def common_step_series(
    runs: list[Run],
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    series: list[tuple[int, np.ndarray, np.ndarray]] = []
    steps: list[np.ndarray] = []
    log_lrs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for run in runs:
        lr_values = np.asarray(run.best_schedule, dtype=float)
        if lr_values.size == 0:
            continue
        step_values = post_prefix_steps(run)
        series.append((run.prefix, step_values, np.log(lr_values)))
        steps.append(step_values)
        log_lrs.append(np.log(lr_values))
        weights.append(np.full(lr_values.size, 1.0 / (len(runs) * lr_values.size)))

    if not series:
        empty = np.array([], dtype=float)
        return series, empty, empty, empty

    return series, np.concatenate(steps), np.concatenate(log_lrs), np.concatenate(weights)


def weighted_least_squares(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_weights = np.sqrt(weights)
    return np.linalg.lstsq(x * sqrt_weights[:, None], y * sqrt_weights, rcond=None)[0]


def average_run_log_mse(
    series: list[tuple[int, np.ndarray, np.ndarray]], predicted_log_lrs: np.ndarray
) -> float:
    offset = 0
    losses: list[float] = []
    for _, step_values, log_lrs in series:
        size = step_values.size
        predicted = predicted_log_lrs[offset : offset + size]
        losses.append(float(np.mean((log_lrs - predicted) ** 2)))
        offset += size
    return float(np.mean(losses))


def per_run_log_rmse(
    series: list[tuple[int, np.ndarray, np.ndarray]], predicted_log_lrs: np.ndarray
) -> dict[int, float]:
    offset = 0
    result: dict[int, float] = {}
    for prefix, step_values, log_lrs in series:
        size = step_values.size
        predicted = predicted_log_lrs[offset : offset + size]
        result[prefix] = float(np.sqrt(np.mean((log_lrs - predicted) ** 2)))
        offset += size
    return result


def weighted_log_r2(log_lrs: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * log_lrs))
    total = float(np.sum(weights * (log_lrs - mean) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum(weights * (log_lrs - predicted) ** 2))
    return 1.0 - residual / total


def make_common_step_fit(
    name: str,
    formula: str,
    series: list[tuple[int, np.ndarray, np.ndarray]],
    all_steps: np.ndarray,
    log_lrs: np.ndarray,
    weights: np.ndarray,
    predicted_log_lrs: np.ndarray,
    grid_steps: np.ndarray,
    grid_log_lrs: np.ndarray,
) -> CommonLrModelFit:
    avg_mse = average_run_log_mse(series, predicted_log_lrs)
    return CommonLrModelFit(
        name=name,
        formula=formula,
        avg_log_mse=avg_mse,
        avg_log_rmse=math.sqrt(avg_mse),
        weighted_r2_log=weighted_log_r2(log_lrs, predicted_log_lrs, weights),
        per_run_rmse=per_run_log_rmse(series, predicted_log_lrs),
        grid_steps=grid_steps,
        grid_lrs=np.exp(grid_log_lrs),
    )


def fit_common_step_lr_models(runs: list[Run]) -> list[CommonLrModelFit]:
    series, all_steps, log_lrs, weights = common_step_series(runs)
    if all_steps.size == 0:
        return []

    grid_steps = np.linspace(float(np.min(all_steps)), float(np.max(all_steps)), 600)
    fits: list[CommonLrModelFit] = []

    design = np.column_stack([np.ones_like(all_steps), all_steps])
    coef = weighted_least_squares(design, log_lrs, weights)
    predicted = design @ coef
    grid_design = np.column_stack([np.ones_like(grid_steps), grid_steps])
    fits.append(
        make_common_step_fit(
            "exponential",
            f"{math.exp(float(coef[0])):.6g}*exp({float(coef[1]):.6g}*s)",
            series,
            all_steps,
            log_lrs,
            weights,
            predicted,
            grid_steps,
            grid_design @ coef,
        )
    )

    slope = float(np.sum(weights * all_steps * (log_lrs - math.log(1.5))))
    slope /= float(np.sum(weights * all_steps * all_steps))
    predicted = math.log(1.5) + slope * all_steps
    fits.append(
        make_common_step_fit(
            "anchored exponential",
            f"1.5*exp({slope:.6g}*s)",
            series,
            all_steps,
            log_lrs,
            weights,
            predicted,
            grid_steps,
            math.log(1.5) + slope * grid_steps,
        )
    )

    log_steps = np.log(all_steps)
    grid_log_steps = np.log(grid_steps)
    design = np.column_stack([np.ones_like(all_steps), log_steps])
    coef = weighted_least_squares(design, log_lrs, weights)
    predicted = design @ coef
    grid_design = np.column_stack([np.ones_like(grid_steps), grid_log_steps])
    fits.append(
        make_common_step_fit(
            "power",
            f"{math.exp(float(coef[0])):.6g}*s^{float(coef[1]):.6g}",
            series,
            all_steps,
            log_lrs,
            weights,
            predicted,
            grid_steps,
            grid_design @ coef,
        )
    )

    for degree in (2, 3):
        design = np.column_stack([log_steps**power for power in range(degree + 1)])
        coef = weighted_least_squares(design, log_lrs, weights)
        predicted = design @ coef
        grid_design = np.column_stack(
            [grid_log_steps**power for power in range(degree + 1)]
        )
        formula_terms = [f"{float(coef[0]):.5g}"]
        for power in range(1, degree + 1):
            formula_terms.append(f"{float(coef[power]):+.5g}*ln(s)^{power}")
        fits.append(
            make_common_step_fit(
                f"log polynomial {degree}",
                "exp(" + " ".join(formula_terms) + ")",
                series,
                all_steps,
                log_lrs,
                weights,
                predicted,
                grid_steps,
                grid_design @ coef,
            )
        )

    best_shift: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for shift in np.linspace(0.0, 1000.0, 2001):
        shifted_log_steps = np.log(all_steps + shift)
        design = np.column_stack([np.ones_like(all_steps), shifted_log_steps])
        coef = weighted_least_squares(design, log_lrs, weights)
        predicted = design @ coef
        score = average_run_log_mse(series, predicted)
        if best_shift is None or score < best_shift[0]:
            grid_design = np.column_stack(
                [np.ones_like(grid_steps), np.log(grid_steps + shift)]
            )
            best_shift = (score, shift, coef, predicted, grid_design @ coef)
    assert best_shift is not None
    _, shift, coef, predicted, grid_predicted = best_shift
    fits.append(
        make_common_step_fit(
            "shifted inverse power",
            f"{math.exp(float(coef[0])):.6g}/(s+{shift:.1f})^{-float(coef[1]):.6g}",
            series,
            all_steps,
            log_lrs,
            weights,
            predicted,
            grid_steps,
            grid_predicted,
        )
    )

    for degree in (2, 3):
        best_log1p: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        for scale in np.linspace(1.0, 1000.0, 1000):
            basis = np.log1p(all_steps / scale)
            design = np.column_stack([basis**power for power in range(degree + 1)])
            coef = weighted_least_squares(design, log_lrs, weights)
            predicted = design @ coef
            score = average_run_log_mse(series, predicted)
            if best_log1p is None or score < best_log1p[0]:
                grid_basis = np.log1p(grid_steps / scale)
                grid_design = np.column_stack(
                    [grid_basis**power for power in range(degree + 1)]
                )
                best_log1p = (score, scale, coef, predicted, grid_design @ coef)
        assert best_log1p is not None
        _, scale, coef, predicted, grid_predicted = best_log1p
        terms = [f"{float(coef[0]):.5g}"]
        for power in range(1, degree + 1):
            terms.append(f"{float(coef[power]):+.5g}*z^{power}")
        fits.append(
            make_common_step_fit(
                f"log1p polynomial {degree}",
                f"exp({' '.join(terms)}), z=ln(1+s/{scale:.0f})",
                series,
                all_steps,
                log_lrs,
                weights,
                predicted,
                grid_steps,
                grid_predicted,
            )
        )

    actual_decay = (log_lrs - math.log(1.5)) / math.log(0.8)
    best_decay: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for exponent in np.linspace(0.1, 2.5, 2401):
        basis = all_steps**exponent
        coefficient = float(np.sum(weights * basis * actual_decay))
        coefficient /= float(np.sum(weights * basis * basis))
        predicted_decay = coefficient * basis
        predicted = math.log(1.5) + math.log(0.8) * predicted_decay
        score = average_run_log_mse(series, predicted)
        if best_decay is None or score < best_decay[0]:
            grid_decay = coefficient * (grid_steps**exponent)
            grid_predicted = math.log(1.5) + math.log(0.8) * grid_decay
            best_decay = (score, exponent, coefficient, predicted, grid_predicted)
    assert best_decay is not None
    _, exponent, coefficient, predicted, grid_predicted = best_decay
    fits.append(
        make_common_step_fit(
            "decay power",
            f"1.5*0.8^({coefficient:.6g}*s^{exponent:.6g})",
            series,
            all_steps,
            log_lrs,
            weights,
            predicted,
            grid_steps,
            grid_predicted,
        )
    )

    return sorted(fits, key=lambda fit: fit.avg_log_mse)


def schedule_from_run(run: Run) -> tuple[list[int], list[float]]:
    if run.best_schedule:
        xs = [0, run.prefix]
        ys = [run.initial_lr, run.initial_lr]
        cursor = run.prefix
        for lr in run.best_schedule:
            xs.extend([cursor, cursor + run.segment_steps])
            ys.extend([lr, lr])
            cursor += run.segment_steps
        return xs, ys

    chosen = selected_candidates(run)
    if not chosen:
        return [0, run.prefix], [run.initial_lr, run.initial_lr]

    xs = [0, run.prefix]
    ys = [run.initial_lr, run.initial_lr]
    for candidate in chosen:
        xs.extend([candidate.step_start - 1, candidate.step_end])
        ys.extend([candidate.lr, candidate.lr])
    return xs, ys


def metric(run: Run, key: str) -> float | int | None:
    value = run.summary.get(key, run.metrics.get(key))
    if isinstance(value, (int, float)):
        return value
    return None


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def annotate_points(ax: plt.Axes, xs: list[int], ys: list[float], fmt: str) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def plot_overview(runs: list[Run], output: Path) -> None:
    prefixes = [run.prefix for run in runs]
    val_bpbs = [metric(run, "val_bpb") for run in runs]
    avg_losses = [
        metric(run, "best_avg_loss")
        if metric(run, "best_avg_loss") is not None
        else run.best_avg_loss
        for run in runs
    ]
    num_steps = [metric(run, "steps") or metric(run, "num_steps") for run in runs]
    training_seconds = [metric(run, "training_seconds") for run in runs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle("beam_search3.log overview", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(prefixes, val_bpbs, marker="o", linewidth=2.0, color="tab:blue")
    annotate_points(ax, prefixes, val_bpbs, "{:.3f}")
    ax.set_title("Validation BPB by fixed prefix")
    ax.set_xlabel("Fixed prefix steps")
    ax.set_ylabel("Validation BPB")

    ax = axes[0, 1]
    ax.plot(prefixes, avg_losses, marker="o", linewidth=2.0, color="tab:green")
    annotate_points(ax, prefixes, avg_losses, "{:.3f}")
    ax.set_title("Best beam average loss")
    ax.set_xlabel("Fixed prefix steps")
    ax.set_ylabel("Average loss")

    ax = axes[1, 0]
    ax.plot(prefixes, num_steps, marker="o", linewidth=2.0, color="tab:purple", label="steps")
    ax.set_title("Run size")
    ax.set_xlabel("Fixed prefix steps")
    ax.set_ylabel("Total steps")
    ax2 = ax.twinx()
    ax2.plot(
        prefixes,
        training_seconds,
        marker="s",
        linewidth=1.8,
        color="tab:orange",
        label="training seconds",
    )
    ax2.set_ylabel("Training seconds")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="best")

    ax = axes[1, 1]
    for run in runs:
        xs, ys = schedule_from_run(run)
        ax.step(xs, ys, where="post", linewidth=1.8, label=f"prefix={run.prefix}")
    ax.set_yscale("log")
    ax.set_title("Selected LR_MULT schedules")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR_MULT")
    ax.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_candidate_scatter(ax: plt.Axes, run: Run) -> None:
    selected_keys = {
        (survivor.round_idx, survivor.candidate_idx)
        for survivor in selected_survivors(run)
    }

    for candidate in run.candidates:
        if candidate.avg_loss is None:
            continue
        is_selected = (candidate.round_idx, candidate.candidate_idx) in selected_keys
        if is_selected:
            color = "black"
            marker = "o"
            size = 18
            alpha = 0.95
            zorder = 3
        elif candidate.status == "kept":
            color = "tab:orange"
            marker = "."
            size = 16
            alpha = 0.5
            zorder = 2
        else:
            color = "0.65"
            marker = "x"
            size = 14
            alpha = 0.45
            zorder = 1
        ax.scatter(
            candidate.round_idx,
            candidate.avg_loss,
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0.8,
            zorder=zorder,
        )

    survivors = selected_survivors(run)
    ax.plot(
        [survivor.round_idx for survivor in survivors],
        [survivor.avg_loss for survivor in survivors],
        color="black",
        linewidth=1.2,
        label="selected survivor",
    )
    ax.set_title("Beam-round candidate losses")
    ax.set_xlabel("Beam round")
    ax.set_ylabel("Candidate avg loss")
    ax.legend(loc="best", fontsize=8)


def plot_selected_trace(ax: plt.Axes, run: Run) -> None:
    points = selected_points(run)
    if not points:
        ax.text(0.5, 0.5, "No selected points parsed", ha="center", va="center")
        return

    xs = [point.step for point in points]
    losses = [point.loss for point in points]
    smooth = rolling_mean(losses, window=25)

    ax.plot(xs, losses, color="tab:blue", alpha=0.18, linewidth=0.8, label="loss")
    ax.plot(xs, smooth, color="tab:blue", linewidth=1.8, label="rolling mean, 25")
    survivors = selected_survivors(run)
    if survivors:
        ax.scatter(
            [survivor.step for survivor in survivors],
            [survivor.avg_loss for survivor in survivors],
            color="black",
            s=13,
            alpha=0.85,
            label="selected segment avg",
            zorder=4,
        )
    ax.axvline(run.prefix, color="0.25", linestyle="--", linewidth=1.0, label="fixed prefix")
    ax.set_title("Selected training path")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=8)


def plot_lr_schedule(ax: plt.Axes, run: Run) -> None:
    xs, ys = schedule_from_run(run)
    ax.step(xs, ys, where="post", color="tab:green", linewidth=1.8)
    ax.axvline(run.prefix, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Selected LR_MULT")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR_MULT")


def plot_run_metrics(ax: plt.Axes, run: Run) -> None:
    ax.axis("off")
    rows = [
        ("prefix", run.prefix),
        ("k", run.k),
        ("val_bpb", run.val_bpb),
        ("best_avg_loss", metric(run, "best_avg_loss") or run.best_avg_loss),
        ("final_lr_mult", metric(run, "final_lr_mult")),
        ("num_steps", run.num_steps),
        ("training_seconds", metric(run, "training_seconds")),
        ("mfu_percent", metric(run, "mfu_percent")),
        ("total_tokens_M", metric(run, "total_tokens_M")),
        ("selected rounds", len(selected_survivors(run))),
        ("fixed points parsed", len(run.fixed_steps)),
        ("candidate trials parsed", len(run.candidates)),
    ]
    text_lines = []
    for key, value in rows:
        if isinstance(value, float):
            rendered = f"{value:.6g}"
        else:
            rendered = str(value)
        text_lines.append(f"{key:18s} {rendered}")
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    ax.set_title("Parsed metrics", loc="left")


def plot_lr_model_overview(runs: list[Run], output: Path) -> None:
    common_models = fit_common_step_lr_models(runs)
    best_common = common_models[0] if common_models else None
    structural = next(
        (model for model in common_models if model.name == "decay power"),
        best_common,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle("Post-prefix LR_MULT models", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for run in runs:
        step_values = post_prefix_steps(run)
        lrs = np.asarray(run.best_schedule, dtype=float)
        ax.step(step_values, lrs, where="post", linewidth=1.4, label=f"prefix={run.prefix}")
    if best_common is not None:
        ax.plot(
            best_common.grid_steps,
            best_common.grid_lrs,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=f"best common: {best_common.name}",
        )
    ax.set_yscale("log")
    ax.set_title("Schedules modeled by post-prefix steps")
    ax.set_xlabel("s = steps after fixed-prefix phase")
    ax.set_ylabel("LR_MULT")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for run in runs:
        step_values = post_prefix_steps(run)
        counts = decay_counts(run)
        ax.step(step_values, counts, where="post", linewidth=1.4, label=f"prefix={run.prefix}")
    if structural is not None:
        structural_counts = np.log(structural.grid_lrs / runs[0].initial_lr) / math.log(0.8)
        ax.plot(
            structural.grid_steps,
            structural_counts,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="structural decay-power fit",
        )
    ax.set_title("Discrete decay count by post-prefix steps")
    ax.set_xlabel("s = steps after fixed-prefix phase")
    ax.set_ylabel("d in LR_MULT = 1.5 * 0.8^d")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    model_names = [model.name for model in common_models]
    errors = [model.avg_log_rmse for model in common_models]
    x = np.arange(len(model_names))
    ax.bar(x, errors, color="tab:blue", alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_title("Balanced average fit error across four runs")
    ax.set_xlabel("Common step-based model")
    ax.set_ylabel("Average RMSE in log(LR_MULT)")

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Objective: minimize mean per-run MSE in log(LR_MULT)", ""]
    if best_common is not None:
        lines.append(f"Best common model: {best_common.name}")
        lines.append(f"avg log RMSE: {best_common.avg_log_rmse:.4f}")
        lines.append(f"weighted R2:   {best_common.weighted_r2_log:.4f}")
        lines.append(best_common.formula)
        lines.append("")
        lines.append("Per-run log RMSE:")
        for prefix, value in best_common.per_run_rmse.items():
            lines.append(f"  prefix={prefix:<4} {value:.4f}")
    if structural is not None and structural is not best_common:
        lines.append("")
        lines.append("Best structural grid model:")
        lines.append(structural.formula)
        lines.append(f"avg log RMSE: {structural.avg_log_rmse:.4f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.7,
    )
    ax.set_title("Model summaries", loc="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_details(runs: list[Run], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for run in runs:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
            fig.suptitle(
                f"beam_search3.log detail: fixed prefix {run.prefix}",
                fontsize=15,
                fontweight="bold",
            )
            plot_selected_trace(axes[0, 0], run)
            plot_candidate_scatter(axes[0, 1], run)
            plot_lr_schedule(axes[1, 0], run)
            plot_run_metrics(axes[1, 1], run)
            pdf.savefig(fig)
            plt.close(fig)


def print_lr_model_summary(runs: list[Run]) -> None:
    common_models = fit_common_step_lr_models(runs)
    best_common = common_models[0] if common_models else None
    structural = next(
        (model for model in common_models if model.name == "decay power"),
        None,
    )
    print()
    print("Post-prefix LR_MULT models in steps")
    print("  s = training steps after the fixed-prefix phase")
    print("  objective = mean per-run MSE in log(LR_MULT)")
    if best_common is not None:
        print(
            f"  best common model: {best_common.name} | "
            f"avg_log_rmse={best_common.avg_log_rmse:.4f} | "
            f"weighted_R2={best_common.weighted_r2_log:.4f}"
        )
        print(f"    {best_common.formula}")
        print("    per-run log RMSE: " + ", ".join(
            f"prefix={prefix}:{value:.3f}"
            for prefix, value in best_common.per_run_rmse.items()
        ))
    if structural is not None:
        print(
            f"  best structural grid model: avg_log_rmse={structural.avg_log_rmse:.4f}"
        )
        print(f"    {structural.formula}")
    print("  candidates ranked by average log RMSE:")
    for model in common_models:
        print(
            f"    {model.name:<24} {model.avg_log_rmse:.4f}  "
            f"{model.formula}"
        )


def print_summary(runs: list[Run], outputs: list[Path]) -> None:
    print(f"Parsed {len(runs)} beam-search runs")
    for run in runs:
        print(
            "  "
            f"prefix={run.prefix:<4} "
            f"steps={run.num_steps:<5} "
            f"val_bpb={run.val_bpb:.6f} "
            f"best_avg_loss={(metric(run, 'best_avg_loss') or run.best_avg_loss):.6f} "
            f"rounds={len(selected_survivors(run))}"
        )
    print_lr_model_summary(runs)
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        default="beam_search3.log",
        type=Path,
        help="Path to the beam-search log. Defaults to beam_search3.log.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("visualizations"),
        type=Path,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    log_path = args.log_path
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    configure_matplotlib()
    runs = parse_log(log_path)
    if not runs:
        raise SystemExit(f"No beam-search runs found in {log_path}")

    stem = log_path.stem
    overview_path = args.out_dir / f"{stem}_overview.png"
    details_path = args.out_dir / f"{stem}_details.pdf"
    lr_models_path = args.out_dir / f"{stem}_lr_models.png"

    plot_overview(runs, overview_path)
    plot_details(runs, details_path)
    plot_lr_model_overview(runs, lr_models_path)
    print_summary(runs, [overview_path, details_path, lr_models_path])

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval_n_search.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval_n_search_plots")

RUN_RE = re.compile(
    r"^cifar_baseline2 run=(?P<run>\d+) train_steps=(?P<train_steps>\d+) "
    r"batch_size=(?P<batch_size>\d+) N=(?P<N>\d+) "
    r"initial_lr=(?P<initial_lr>\S+) initial_lr_k=(?P<initial_lr_k>-?\d+) "
    r"muon_momentum=(?P<muon_momentum>\S+) sgd_lr_mult=(?P<sgd_lr_mult>\S+) "
    r"name=(?P<name>\S+) search=(?P<search>\S+) "
    r"muon_lr_schedule=(?P<muon_lr_schedule>\S+) "
    r"sgd_lr_schedule=(?P<sgd_lr_schedule>\S+)"
    r"(?: ema=(?P<ema>\S+) applied_lr_source=(?P<applied_lr_source>\S+))?"
)
N_SEARCH_TRAIN_LOSS_RE = re.compile(
    r"^n_search_train_loss run=(?P<run>\d+) interval=(?P<interval>\d+) "
    r"train_step=(?P<train_step>\d+) "
    r"candidate_step=(?P<candidate_step>\d+)/(?P<candidate_steps>\d+) "
    r"k=(?P<k>-?\d+) lr=(?P<lr>\S+) train_loss=(?P<train_loss>\S+)"
)
N_SEARCH_CANDIDATE_RE = re.compile(
    r"^n_search candidate run=(?P<run>\d+) interval=(?P<interval>\d+) "
    r"start_step=(?P<start_step>\d+) steps=(?P<steps>\d+) "
    r"k=(?P<k>-?\d+) lr=(?P<lr>\S+) train_loss=(?P<train_loss>\S+)"
)
APPLIED_LR_RE = re.compile(
    r"^applied_lr step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) muon_lr=(?P<muon_lr>\S+)"
)
STEP_TRAIN_LOSS_RE = re.compile(
    r"^step_train_loss step=(?P<step>\d+)/(?P<total_steps>\d+) "
    r"name=(?P<name>\S+) train_loss=(?P<train_loss>\S+)"
)
INTERVAL_SELECTED_RE = re.compile(
    r"^n_search interval_selected run=(?P<run>\d+) interval=(?P<interval>\d+) "
    r"steps=(?P<start_step>\d+)-(?P<end_step>\d+) N=(?P<N>\d+) "
    r"selected_k=(?P<selected_k>-?\d+) selected_lr=(?P<selected_lr>\S+) "
    r"train_loss=(?P<train_loss>\S+) "
    r"evaluated_candidates=(?P<evaluated_candidates>\d+)"
)
INTERVAL_SELECTED_WITH_EMA_RE = re.compile(
    r"^n_search interval_selected run=(?P<run>\d+) interval=(?P<interval>\d+) "
    r"steps=(?P<start_step>\d+)-(?P<end_step>\d+) N=(?P<N>\d+) "
    r"initial_lr=(?P<initial_lr>\S+) "
    r"selected_k=(?P<selected_k>-?\d+) selected_lr=(?P<selected_lr>\S+) "
    r"next_initial_lr=(?P<next_initial_lr>\S+) ema=(?P<ema>\S+) "
    r"applied_lr_source=(?P<applied_lr_source>\S+) "
    r"applied_k=(?P<applied_k>-?\d+) applied_lr=(?P<applied_lr>\S+) "
    r"train_loss=(?P<train_loss>\S+) "
    r"evaluated_candidates=(?P<evaluated_candidates>\d+)"
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


@dataclass
class SearchTrainLoss:
    interval: int
    train_step: int
    candidate_step: int
    candidate_steps: int
    k: int
    lr: float
    train_loss: float


@dataclass
class CandidateResult:
    interval: int
    start_step: int
    steps: int
    k: int
    lr: float
    train_loss: float


@dataclass
class IntervalSelection:
    interval: int
    start_step: int
    end_step: int
    selected_k: int
    selected_lr: float
    train_loss: float
    evaluated_candidates: int
    initial_lr: float | None = None
    next_initial_lr: float | None = None
    ema: float | None = None
    applied_lr_source: str | None = None
    applied_k: int | None = None
    applied_lr: float | None = None


@dataclass
class Run:
    index: int
    train_steps: int
    batch_size: int
    interval_steps: int
    initial_lr: float
    initial_lr_k: int
    muon_momentum: float
    sgd_lr_mult: float
    name: str
    muon_lr_schedule: str
    sgd_lr_schedule: str
    ema: float | None = None
    applied_lr_source: str | None = None
    applied_lr: list[tuple[int, float]] = field(default_factory=list)
    step_train_losses: list[tuple[int, float]] = field(default_factory=list)
    searched_train_losses: list[SearchTrainLoss] = field(default_factory=list)
    candidate_results: list[CandidateResult] = field(default_factory=list)
    interval_selections: list[IntervalSelection] = field(default_factory=list)
    eval_train_loss: float | None = None
    wall_time_seconds: float | None = None
    cuda_time_seconds: float | None = None

    @property
    def final_step_train_loss(self):
        if not self.step_train_losses:
            return None
        return self.step_train_losses[-1][1]


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def fmt_sigfig(value, sigfigs=3):
    if value is None:
        return "NA"
    return f"{value:.{sigfigs}g}"


def fmt_sequence(values, max_items=14):
    values = [fmt(value) for value in values]
    if len(values) <= max_items:
        return " ".join(values)
    head_count = max_items // 2
    tail_count = max_items - head_count
    return " ".join(values[:head_count] + ["..."] + values[-tail_count:])


def parse_float(value):
    return float(value)


def parse_log(path):
    runs = []
    runs_by_name = {}
    runs_by_index = {}
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
                    interval_steps=int(match.group("N")),
                    initial_lr=parse_float(match.group("initial_lr")),
                    initial_lr_k=int(match.group("initial_lr_k")),
                    muon_momentum=parse_float(match.group("muon_momentum")),
                    sgd_lr_mult=parse_float(match.group("sgd_lr_mult")),
                    name=match.group("name"),
                    muon_lr_schedule=match.group("muon_lr_schedule"),
                    sgd_lr_schedule=match.group("sgd_lr_schedule"),
                    ema=(
                        parse_float(match.group("ema"))
                        if match.group("ema") is not None
                        else None
                    ),
                    applied_lr_source=match.group("applied_lr_source"),
                )
                runs.append(current)
                runs_by_name[current.name] = current
                runs_by_index[current.index] = current
                continue

            match = N_SEARCH_TRAIN_LOSS_RE.match(line)
            if match:
                run = runs_by_index.get(int(match.group("run")))
                if run is not None:
                    run.searched_train_losses.append(
                        SearchTrainLoss(
                            interval=int(match.group("interval")),
                            train_step=int(match.group("train_step")),
                            candidate_step=int(match.group("candidate_step")),
                            candidate_steps=int(match.group("candidate_steps")),
                            k=int(match.group("k")),
                            lr=parse_float(match.group("lr")),
                            train_loss=parse_float(match.group("train_loss")),
                        )
                    )
                continue

            match = N_SEARCH_CANDIDATE_RE.match(line)
            if match:
                run = runs_by_index.get(int(match.group("run")))
                if run is not None:
                    run.candidate_results.append(
                        CandidateResult(
                            interval=int(match.group("interval")),
                            start_step=int(match.group("start_step")),
                            steps=int(match.group("steps")),
                            k=int(match.group("k")),
                            lr=parse_float(match.group("lr")),
                            train_loss=parse_float(match.group("train_loss")),
                        )
                    )
                continue

            match = APPLIED_LR_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.applied_lr.append(
                        (int(match.group("step")), parse_float(match.group("muon_lr")))
                    )
                continue

            match = STEP_TRAIN_LOSS_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"), current)
                if run is not None:
                    run.step_train_losses.append(
                        (int(match.group("step")), parse_float(match.group("train_loss")))
                    )
                continue

            match = INTERVAL_SELECTED_WITH_EMA_RE.match(line)
            if match:
                run = runs_by_index.get(int(match.group("run")))
                if run is not None:
                    run.interval_selections.append(
                        IntervalSelection(
                            interval=int(match.group("interval")),
                            start_step=int(match.group("start_step")),
                            end_step=int(match.group("end_step")),
                            selected_k=int(match.group("selected_k")),
                            selected_lr=parse_float(match.group("selected_lr")),
                            train_loss=parse_float(match.group("train_loss")),
                            evaluated_candidates=int(match.group("evaluated_candidates")),
                            initial_lr=parse_float(match.group("initial_lr")),
                            next_initial_lr=parse_float(match.group("next_initial_lr")),
                            ema=parse_float(match.group("ema")),
                            applied_lr_source=match.group("applied_lr_source"),
                            applied_k=int(match.group("applied_k")),
                            applied_lr=parse_float(match.group("applied_lr")),
                        )
                    )
                continue

            match = INTERVAL_SELECTED_RE.match(line)
            if match:
                run = runs_by_index.get(int(match.group("run")))
                if run is not None:
                    run.interval_selections.append(
                        IntervalSelection(
                            interval=int(match.group("interval")),
                            start_step=int(match.group("start_step")),
                            end_step=int(match.group("end_step")),
                            selected_k=int(match.group("selected_k")),
                            selected_lr=parse_float(match.group("selected_lr")),
                            train_loss=parse_float(match.group("train_loss")),
                            evaluated_candidates=int(match.group("evaluated_candidates")),
                        )
                    )
                continue

            match = FINAL_RE.match(line)
            if match and current is not None:
                current.eval_train_loss = parse_float(match.group("train_loss"))
                continue

            match = RUN_TIME_RE.match(line)
            if match:
                run = runs_by_name.get(match.group("name"))
                if run is not None:
                    run.wall_time_seconds = parse_float(match.group("wall_time_seconds"))
                    run.cuda_time_seconds = parse_float(match.group("cuda_time_seconds"))

    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def sorted_batch_sizes(runs):
    return sorted({run.batch_size for run in runs})


def sorted_interval_steps(runs):
    return sorted({run.interval_steps for run in runs})


def run_variant_key(run):
    return (run.ema, run.applied_lr_source)


def run_variant_label(run):
    if run.ema is None and run.applied_lr_source is None:
        return "default"
    ema = fmt(run.ema) if run.ema is not None else "NA"
    source = run.applied_lr_source or "NA"
    return f"ema={ema}, {source}"


def run_variant_slug(run):
    if run.ema is None and run.applied_lr_source is None:
        return "default"
    ema = fmt(run.ema).replace("-", "m").replace(".", "p")
    source = (run.applied_lr_source or "NA").replace("_", "")
    return f"ema{ema}_{source}"


def sorted_variants(runs):
    variants = {}
    for run in runs:
        variants[run_variant_key(run)] = run_variant_label(run)
    return sorted(variants.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))


def has_variants(runs):
    return len({run_variant_key(run) for run in runs}) > 1


def final_loss(run):
    if run.final_step_train_loss is not None:
        return run.final_step_train_loss
    return run.eval_train_loss


def run_for(runs, batch_size, interval_steps):
    for run in runs:
        if run.batch_size == batch_size and run.interval_steps == interval_steps:
            return run
    return None


def best_run_for(runs, batch_size, interval_steps):
    matches = [
        run
        for run in runs
        if run.batch_size == batch_size and run.interval_steps == interval_steps
    ]
    if not matches:
        return None
    return min(matches, key=final_loss)


def selected_steps_and_losses(run):
    pairs = sorted(run.step_train_losses)
    return [step for step, _ in pairs], [loss for _, loss in pairs]


def selected_loss_at_step(run, step):
    losses_by_step = dict(run.step_train_losses)
    return losses_by_step.get(step)


def selected_steps_and_lrs(run):
    pairs = sorted(run.applied_lr)
    return [step for step, _ in pairs], [lr for _, lr in pairs]


def candidates_by_interval(run):
    grouped = {}
    for candidate in run.candidate_results:
        grouped.setdefault(candidate.interval, []).append(candidate)
    return grouped


def selections_by_interval(run):
    return {selection.interval: selection for selection in run.interval_selections}


def plot_final_loss_by_interval_steps(runs, output_path):
    interval_steps_list = sorted_interval_steps(runs)
    batch_sizes = sorted_batch_sizes(runs)
    if has_variants(runs):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(13.0, 8.6),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        variants = sorted_variants(runs)
        for ax, batch_size in zip(axes.ravel(), batch_sizes):
            for variant_key, variant_label in variants:
                rows = [
                    run
                    for run in runs
                    if run.batch_size == batch_size and run_variant_key(run) == variant_key
                ]
                by_n = {run.interval_steps: run for run in rows}
                ys = [
                    final_loss(by_n[interval_steps])
                    if interval_steps in by_n
                    else np.nan
                    for interval_steps in interval_steps_list
                ]
                ax.plot(
                    interval_steps_list,
                    ys,
                    marker="o",
                    linewidth=1.6,
                    label=variant_label,
                )
            ax.set_title(f"bs={batch_size}")
            ax.set_xticks(interval_steps_list)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("N")
            ax.set_ylabel("final train loss")
            ax.legend(fontsize=7)
        fig.suptitle("N-Search Final Train Loss by Variant", y=0.995)
    else:
        fig, ax = plt.subplots(figsize=(9.0, 5.6))
        for batch_size in batch_sizes:
            ys = []
            for interval_steps in interval_steps_list:
                run = run_for(runs, batch_size, interval_steps)
                ys.append(final_loss(run) if run is not None else np.nan)
            ax.plot(
                interval_steps_list,
                ys,
                marker="o",
                linewidth=2.0,
                label=f"bs={batch_size}",
            )
        ax.set_xlabel("N, constant-LR interval length")
        ax.set_ylabel("final selected-path train loss")
        ax.set_title("N-Search Final Train Loss")
        ax.set_xticks(interval_steps_list)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_grid(runs, output_path, draw_fn, title, ylabel):
    batch_sizes = sorted_batch_sizes(runs)
    interval_steps_list = sorted_interval_steps(runs)
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(interval_steps_list),
        figsize=(4.2 * len(interval_steps_list), 3.1 * len(batch_sizes)),
        squeeze=False,
        sharex=True,
    )
    for row, batch_size in enumerate(batch_sizes):
        for col, interval_steps in enumerate(interval_steps_list):
            ax = axes[row][col]
            run = run_for(runs, batch_size, interval_steps)
            if run is not None:
                draw_fn(ax, run)
            ax.set_title(f"bs={batch_size}, N={interval_steps}", fontsize=10)
            ax.grid(True, alpha=0.2)
            if row == len(batch_sizes) - 1:
                ax.set_xlabel("train step")
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_step_train_losses(runs, output_path):
    if has_variants(runs):
        plot_variant_batch_curves(
            runs,
            output_path,
            selected_steps_and_losses,
            "Selected-Path Train Losses by Variant",
            "train loss",
            use_step=False,
        )
        return

    def draw(ax, run):
        steps, losses = selected_steps_and_losses(run)
        if steps:
            ax.plot(steps, losses, color="black", linewidth=1.7)
            ax.scatter(steps[-1], losses[-1], color="black", s=14, zorder=3)

    plot_grid(
        runs,
        output_path,
        draw,
        "Selected-Path Train Losses",
        "train loss",
    )


def plot_selected_lr_schedules(runs, output_path):
    if has_variants(runs):
        plot_variant_batch_curves(
            runs,
            output_path,
            selected_steps_and_lrs,
            "Selected Muon LR Schedules by Variant",
            "Muon LR",
            use_step=True,
            log_y=True,
        )
        return

    def draw(ax, run):
        steps, lrs = selected_steps_and_lrs(run)
        if steps:
            ax.step(steps, lrs, where="post", color="#1f77b4", linewidth=1.7)
            ax.scatter(steps, lrs, color="#1f77b4", s=8, alpha=0.65)
            ymin = min(lr for lr in lrs if lr > 0)
            ymax = max(lrs)
            if ymin > 0 and ymax > 0:
                ax.set_yscale("log")
                ax.set_ylim(ymin / 1.4, ymax * 1.4)

    plot_grid(
        runs,
        output_path,
        draw,
        "Selected Muon LR Schedules",
        "Muon LR",
    )


def plot_variant_batch_curves(
    runs,
    output_path,
    series_fn,
    title,
    ylabel,
    use_step=False,
    log_y=False,
):
    variants = sorted_variants(runs)
    batch_sizes = sorted_batch_sizes(runs)
    interval_steps_list = sorted_interval_steps(runs)
    fig, axes = plt.subplots(
        len(variants),
        len(batch_sizes),
        figsize=(4.2 * len(batch_sizes), 2.8 * len(variants)),
        squeeze=False,
        sharex=True,
    )
    cmap = plt.get_cmap("viridis")
    colors = {
        interval_steps: cmap(index / max(1, len(interval_steps_list) - 1))
        for index, interval_steps in enumerate(interval_steps_list)
    }

    for row, (variant_key, variant_label) in enumerate(variants):
        for col, batch_size in enumerate(batch_sizes):
            ax = axes[row][col]
            y_values = []
            for interval_steps in interval_steps_list:
                matches = [
                    run
                    for run in runs
                    if run.batch_size == batch_size
                    and run.interval_steps == interval_steps
                    and run_variant_key(run) == variant_key
                ]
                if not matches:
                    continue
                run = min(matches, key=final_loss)
                steps, values = series_fn(run)
                if not steps:
                    continue
                y_values.extend(value for value in values if value > 0)
                plot = ax.step if use_step else ax.plot
                kwargs = dict(
                    color=colors[interval_steps],
                    linewidth=1.5,
                    label=f"N={interval_steps}",
                )
                if use_step:
                    plot(steps, values, where="post", **kwargs)
                else:
                    plot(steps, values, **kwargs)
            if log_y and y_values:
                ax.set_yscale("log")
                ax.set_ylim(min(y_values) / 1.5, max(y_values) * 1.5)
            if row == 0:
                ax.set_title(f"bs={batch_size}")
            if col == 0:
                ax.set_ylabel(f"{variant_label}\n{ylabel}")
            if row == len(variants) - 1:
                ax.set_xlabel("train step")
            ax.grid(True, alpha=0.25)
            if row == 0 and col == len(batch_sizes) - 1:
                ax.legend(fontsize=7)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_lr_schedulers_by_batch_size(runs, output_path):
    batch_sizes = sorted_batch_sizes(runs)
    interval_steps_list = sorted_interval_steps(runs)
    if has_variants(runs):
        variants = sorted_variants(runs)
        fig, axes = plt.subplots(
            len(variants) * 2,
            len(batch_sizes),
            figsize=(4.2 * len(batch_sizes), 4.6 * len(variants)),
            squeeze=False,
            sharex=True,
        )
        cmap = plt.get_cmap("viridis")
        colors = {
            interval_steps: cmap(index / max(1, len(interval_steps_list) - 1))
            for index, interval_steps in enumerate(interval_steps_list)
        }
        for variant_row, (variant_key, variant_label) in enumerate(variants):
            lr_row = 2 * variant_row
            loss_row = lr_row + 1
            for col, batch_size in enumerate(batch_sizes):
                lr_ax = axes[lr_row][col]
                loss_ax = axes[loss_row][col]
                all_lrs = []
                for interval_steps in interval_steps_list:
                    matches = [
                        run
                        for run in runs
                        if run.batch_size == batch_size
                        and run.interval_steps == interval_steps
                        and run_variant_key(run) == variant_key
                    ]
                    if not matches:
                        continue
                    run = min(matches, key=final_loss)
                    steps, lrs = selected_steps_and_lrs(run)
                    if steps:
                        all_lrs.extend(lrs)
                        lr_ax.step(
                            steps,
                            lrs,
                            where="post",
                            linewidth=1.5,
                            color=colors[interval_steps],
                            label=f"N={interval_steps}",
                        )
                    loss_steps, losses = selected_steps_and_losses(run)
                    if loss_steps:
                        loss_ax.plot(
                            loss_steps,
                            losses,
                            linewidth=1.4,
                            color=colors[interval_steps],
                            label=f"N={interval_steps}",
                        )
                if all_lrs:
                    lr_ax.set_yscale("log")
                    lr_ax.set_ylim(min(all_lrs) / 1.5, max(all_lrs) * 1.5)
                if variant_row == 0:
                    lr_ax.set_title(f"bs={batch_size}")
                if col == 0:
                    lr_ax.set_ylabel(f"{variant_label}\nMuon LR")
                    loss_ax.set_ylabel("train loss")
                if loss_row == len(variants) * 2 - 1:
                    loss_ax.set_xlabel("train step")
                lr_ax.grid(True, alpha=0.25)
                loss_ax.grid(True, alpha=0.25)
                if variant_row == 0 and col == len(batch_sizes) - 1:
                    lr_ax.legend(fontsize=7)

        fig.suptitle(
            "LR Schedulers and Loss Curves by EMA and Applied LR Source",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        2,
        len(batch_sizes),
        figsize=(4.2 * len(batch_sizes), 7.6),
        squeeze=False,
        sharex=True,
    )
    cmap = plt.get_cmap("viridis")
    colors = {
        interval_steps: cmap(index / max(1, len(interval_steps_list) - 1))
        for index, interval_steps in enumerate(interval_steps_list)
    }

    for col, batch_size in enumerate(batch_sizes):
        lr_ax = axes[0][col]
        loss_ax = axes[1][col]
        all_lrs = []
        for interval_steps in interval_steps_list:
            run = best_run_for(runs, batch_size, interval_steps)
            if run is None:
                continue
            steps, lrs = selected_steps_and_lrs(run)
            if steps:
                all_lrs.extend(lrs)
                lr_ax.step(
                    steps,
                    lrs,
                    where="post",
                    linewidth=1.8,
                    color=colors[interval_steps],
                    label=f"N={interval_steps}",
                )
            loss_steps, losses = selected_steps_and_losses(run)
            if loss_steps:
                loss_ax.plot(
                    loss_steps,
                    losses,
                    linewidth=1.6,
                    color=colors[interval_steps],
                    label=f"N={interval_steps}",
                )
        if all_lrs:
            lr_ax.set_yscale("log")
            lr_ax.set_ylim(min(all_lrs) / 1.5, max(all_lrs) * 1.5)
        lr_ax.set_title(f"bs={batch_size}")
        lr_ax.set_ylabel("selected Muon LR")
        lr_ax.grid(True, alpha=0.25)
        lr_ax.legend(fontsize=8)

        loss_ax.set_xlabel("train step")
        loss_ax.set_ylabel("train loss")
        loss_ax.grid(True, alpha=0.25)

    fig.suptitle("Best LR Schedulers and Loss Curves by Batch Size", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_searched_train_losses(runs, output_path):
    all_ks = [event.k for run in runs for event in run.searched_train_losses]
    if all_ks:
        norm = Normalize(min(all_ks), max(all_ks))
    else:
        norm = Normalize(0, 1)
    cmap = plt.get_cmap("turbo")

    batch_sizes = sorted_batch_sizes(runs)
    interval_steps_list = sorted_interval_steps(runs)
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(interval_steps_list),
        figsize=(4.2 * len(interval_steps_list), 3.1 * len(batch_sizes)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for row, batch_size in enumerate(batch_sizes):
        for col, interval_steps in enumerate(interval_steps_list):
            ax = axes[row][col]
            run = run_for(runs, batch_size, interval_steps)
            if run is not None and run.searched_train_losses:
                events = run.searched_train_losses
                ax.scatter(
                    [event.train_step for event in events],
                    [event.train_loss for event in events],
                    c=[event.k for event in events],
                    cmap=cmap,
                    norm=norm,
                    s=8,
                    alpha=0.28,
                    linewidths=0,
                )
                steps, losses = selected_steps_and_losses(run)
                if steps:
                    ax.plot(steps, losses, color="black", linewidth=1.5, label="selected")
            ax.set_title(f"bs={batch_size}, N={interval_steps}", fontsize=10)
            ax.grid(True, alpha=0.2)
            if row == len(batch_sizes) - 1:
                ax.set_xlabel("train step")
            if col == 0:
                ax.set_ylabel("train loss")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="LR exponent k", shrink=0.88)
    fig.suptitle("All Searched Train Losses", y=0.995)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_interval_lr_losses_for_run(run, output_path):
    by_interval = candidates_by_interval(run)
    by_selection = selections_by_interval(run)
    interval_numbers = sorted(by_interval)
    if not interval_numbers:
        return

    nplots = len(interval_numbers)
    ncols = min(10, max(1, math.ceil(math.sqrt(nplots))))
    nrows = math.ceil(nplots / ncols)
    fig_width = max(8.0, ncols * 2.35)
    fig_height = max(4.8, nrows * 2.1)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for ax in axes.ravel()[nplots:]:
        ax.axis("off")

    for ax, interval in zip(axes.ravel(), interval_numbers):
        candidates = sorted(by_interval[interval], key=lambda candidate: candidate.lr)
        lrs = [candidate.lr for candidate in candidates]
        losses = [candidate.train_loss for candidate in candidates]
        ax.plot(lrs, losses, marker="o", linewidth=1.1, markersize=3.0)
        selection = by_selection.get(interval)
        if selection is not None:
            ax.scatter(
                [selection.selected_lr],
                [selection.train_loss],
                marker="*",
                s=52,
                color="crimson",
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.22)
        if selection is None:
            ax.set_title(f"interval {interval}", fontsize=8)
        else:
            ax.set_title(
                f"interval {interval}: steps {selection.start_step}-{selection.end_step}",
                fontsize=8,
            )
        ax.tick_params(labelsize=7)

    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("LR", fontsize=8)
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("loss", fontsize=8)

    fig.suptitle(
        f"LR Search Losses: bs={run.batch_size}, N={run.interval_steps}, "
        f"{run_variant_label(run)}",
        y=0.998,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_interval_lr_losses(runs, output_dir):
    include_variant = has_variants(runs)
    for run in sorted(runs, key=lambda run: (run.batch_size, run.interval_steps)):
        variant_suffix = f"_{run_variant_slug(run)}" if include_variant else ""
        plot_interval_lr_losses_for_run(
            run,
            output_dir
            / (
                f"interval_lr_losses_bs{run.batch_size}_N{run.interval_steps}"
                f"{variant_suffix}.png"
            ),
        )


def write_summary(runs, output_path):
    include_variant = has_variants(runs)
    header = ["bs", "N"]
    if include_variant:
        header.extend(["ema", "applied_lr_source"])
    header.extend(["step_100", "step_10", "step_20"])
    rows = [tuple(header)]
    for run in sorted(runs, key=lambda run: (run.batch_size, run.interval_steps)):
        row = [str(run.batch_size), str(run.interval_steps)]
        if include_variant:
            row.extend([fmt(run.ema), run.applied_lr_source or "NA"])
        row.extend(
            [
                fmt_sigfig(selected_loss_at_step(run, 100)),
                fmt_sigfig(selected_loss_at_step(run, 10)),
                fmt_sigfig(selected_loss_at_step(run, 20)),
            ]
        )
        rows.append(tuple(row))

    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = [
        "  ".join(value.rjust(widths[col]) for col, value in enumerate(row))
        for row in rows
    ]
    Path(output_path).write_text("\n".join(lines) + "\n")


def write_csv(runs, output_path):
    fieldnames = [
        "run",
        "name",
        "batch_size",
        "N",
        "ema",
        "applied_lr_source",
        "final_train_loss",
        "final_eval_train_loss",
        "selected_lr_count",
        "searched_train_loss_count",
        "candidate_count",
        "total_evaluated_candidates",
        "selected_lr_sequence",
        "selected_k_sequence",
        "wall_time_seconds",
        "cuda_time_seconds",
    ]
    with Path(output_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted(runs, key=lambda run: (run.batch_size, run.interval_steps)):
            writer.writerow(
                {
                    "run": run.index,
                    "name": run.name,
                    "batch_size": run.batch_size,
                    "N": run.interval_steps,
                    "ema": run.ema,
                    "applied_lr_source": run.applied_lr_source,
                    "final_train_loss": final_loss(run),
                    "final_eval_train_loss": run.eval_train_loss,
                    "selected_lr_count": len(run.applied_lr),
                    "searched_train_loss_count": len(run.searched_train_losses),
                    "candidate_count": len(run.candidate_results),
                    "total_evaluated_candidates": sum(
                        selection.evaluated_candidates
                        for selection in run.interval_selections
                    ),
                    "selected_lr_sequence": " ".join(
                        fmt(selection.selected_lr)
                        for selection in run.interval_selections
                    ),
                    "selected_k_sequence": " ".join(
                        str(selection.selected_k)
                        for selection in run.interval_selections
                    ),
                    "wall_time_seconds": run.wall_time_seconds,
                    "cuda_time_seconds": run.cuda_time_seconds,
                }
            )


def main(default_log=DEFAULT_LOG, default_output_dir=DEFAULT_OUTPUT_DIR):
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval_n_search.py log output."
    )
    parser.add_argument("--log", type=Path, default=default_log)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    args = parser.parse_args()

    runs = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_loss_by_interval_steps(
        runs, args.output_dir / "final_train_loss_by_N.png"
    )
    plot_selected_lr_schedules(runs, args.output_dir / "selected_lr_schedules.png")
    plot_best_lr_schedulers_by_batch_size(
        runs, args.output_dir / "best_lr_schedulers_by_bs.png"
    )
    plot_selected_step_train_losses(
        runs, args.output_dir / "selected_step_train_losses.png"
    )
    if not has_variants(runs):
        plot_searched_train_losses(runs, args.output_dir / "searched_train_losses.png")
    write_summary(runs, args.output_dir / "summary.txt")
    write_csv(runs, args.output_dir / "summary.csv")

    best_overall = min(runs, key=lambda run: final_loss(run))
    print(f"Parsed {len(runs)} runs from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        "Best final train loss: "
        f"bs={best_overall.batch_size}, N={best_overall.interval_steps}, "
        f"loss={fmt(final_loss(best_overall))}"
    )


if __name__ == "__main__":
    main()

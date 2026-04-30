#!/usr/bin/env python3
"""Visualize optimizer hyperparameter search logs."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np


DEFAULT_LOG_NAME = "scalar_optimizers_logging5.log"
DEFAULT_OUT_DIR = Path(__file__).with_name("scalar_optimizers_logging5_dir")
SSE_KEY = "clean_train_sse"
SUMMARY_FILENAME = "optimizer_hparam_summary.txt"
TOP_K = 3
BETA1_SPLIT_HPARAM = "beta1"
PREFERRED_OPTIMIZER_ORDER = (
    "AdamW",
    "Adam2",
    "AdamH",
    "Muon",
    "MuonH",
    "SGD",
    "SGD2",
)
H_NORM_ORDER = ("matrix", "row")
ADAM2_OPTION_KEYS = ("nesterov", "disable_bias1", "adaptive_norm")
PREFERRED_HPARAM_ORDER = (
    "lr",
    "lr_decay",
    "lr_power",
    "beta1",
    "beta2",
    "eps",
    "momentum",
    "wd",
)
LOG_SCALE_HPARAMS = {"lr", "eps", "wd"}
NON_SEARCH_KEYS = {
    "actual_samples",
    "batch_size",
    "candidate_idx",
    "clean_sse",
    "clean_train_sse",
    "compiled",
    "duration_sec",
    "elapsed_sec",
    "final_train_loss",
    "grid_idx",
    "h_norm",
    "lr_schedule",
    "log_scalar_max",
    "log_scalar_mean",
    "log_scalar_min",
    "log_scalar_std",
    "num_candidates",
    "num_samples",
    "optimizer",
    "peak_allocated_bytes",
    "peak_allocated_mib",
    "peak_reserved_bytes",
    "peak_reserved_mib",
    "predicted_lr",
    "sample_idx",
    "sample_mode",
    "scalar_count",
    "scalar_max",
    "scalar_mean",
    "scalar_min",
    "scalar_std",
    "setting_key",
    "step_size",
    "steps",
    "target_met",
    "training_elapsed_sec",
    "variant",
    "effective_weight_matrix_norm",
    "effective_weight_row_norm_mean",
    "effective_weight_row_norm_std",
    "weight_matrix_norm",
    "weight_row_norm_mean",
    "weight_row_norm_std",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Create one figure per optimizer variant and beta1 from {DEFAULT_LOG_NAME}. "
            "Each hparam row plots the three best observed hparam values across "
            "batch sizes and sample counts."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"Input log file. Defaults to the single {DEFAULT_LOG_NAME} under this repo.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where optimizer PNG figures are written.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser.parse_args()


def default_log_path() -> Path:
    candidates = sorted(
        Path(__file__).resolve().parents[1].glob(f"**/{DEFAULT_LOG_NAME}")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {DEFAULT_LOG_NAME}, found {len(candidates)}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def is_plot_value(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str)) and value is not None


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
            return str(int(round(value)))
        return f"{value:.10g}"
    return str(value)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def sample_budget(row: dict[str, Any]) -> int | None:
    value = row.get("num_samples", row.get("actual_samples"))
    if value is None:
        return None
    return int(value)


def category_sort_key(value: Any) -> tuple[str, float | str]:
    if isinstance(value, bool):
        return ("bool", float(value))
    if isinstance(value, (int, float)):
        return ("number", float(value))
    return ("string", str(value))


def grouped_rows(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def read_run_summaries(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_rows = 0
    skipped_nonfinite = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or " " not in line:
                continue
            event, payload = line.split(" ", 1)
            if event != "RUN_SUMMARY":
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {log_path}:{line_num}") from exc
            if SSE_KEY in row and "optimizer" in row and "batch_size" in row:
                candidate_rows += 1
                try:
                    sse = float(row[SSE_KEY])
                except (TypeError, ValueError):
                    skipped_nonfinite += 1
                    continue
                if not math.isfinite(sse):
                    skipped_nonfinite += 1
                    continue
                rows.append(row)
    if not rows:
        if candidate_rows:
            raise ValueError(
                f"no finite RUN_SUMMARY rows with {SSE_KEY!r} found in {log_path}; "
                f"skipped {skipped_nonfinite} invalid/non-finite rows"
            )
        raise ValueError(f"no RUN_SUMMARY rows with {SSE_KEY!r} found in {log_path}")
    return rows


def read_search_plan(log_path: Path) -> dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or " " not in line:
                continue
            event, payload = line.split(" ", 1)
            if event != "SEARCH_PLAN":
                continue
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {log_path}:{line_num}") from exc
    return {}


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row[SSE_KEY]))


def optimizer_sort_key(name: str) -> tuple[int, str]:
    if name in PREFERRED_OPTIMIZER_ORDER:
        return (PREFERRED_OPTIMIZER_ORDER.index(name), name)
    return (len(PREFERRED_OPTIMIZER_ORDER), name)


def plot_group_name(row: dict[str, Any]) -> str:
    variant = row.get("variant")
    if isinstance(variant, str) and variant:
        base_name = variant
    else:
        optimizer = str(row["optimizer"])
        h_norm = row.get("h_norm")
        if h_norm is None:
            base_name = optimizer
        else:
            base_name = f"{optimizer}_{h_norm}"

    if row.get("optimizer") == "Adam2":
        option_suffix = "__".join(
            f"{key}={int(bool(row[key]))}" for key in ADAM2_OPTION_KEYS if key in row
        )
        if option_suffix:
            return f"{base_name}__{option_suffix}"
    return base_name


def beta1_split_value(row: dict[str, Any]) -> Any | None:
    value = row.get(BETA1_SPLIT_HPARAM)
    if not is_plot_value(value):
        return None
    return normalize_value(value)


def add_beta1_split_to_plot_group(plot_group: str, value: Any | None) -> str:
    if value is None:
        return plot_group
    return f"{plot_group}__{BETA1_SPLIT_HPARAM}={format_value(value)}"


def plot_group_base_name(name: str) -> str:
    return name.split("__", 1)[0]


def plot_group_option_sort_key(name: str) -> tuple[int, ...]:
    option_values: dict[str, int] = {}
    for part in name.split("__")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key in ADAM2_OPTION_KEYS:
            option_values[key] = int(value)
    return tuple(option_values.get(key, -1) for key in ADAM2_OPTION_KEYS)


def plot_group_sort_key(name: str) -> tuple[int, int, tuple[int, ...], str]:
    base_name = plot_group_base_name(name)
    base_optimizer = base_name
    h_norm_index = -1
    for idx, h_norm in enumerate(H_NORM_ORDER):
        suffix = f"_{h_norm}"
        if base_name.endswith(suffix):
            base_optimizer = base_name[: -len(suffix)]
            h_norm_index = idx
            break
    optimizer_index = optimizer_sort_key(base_optimizer)[0]
    return (optimizer_index, h_norm_index, plot_group_option_sort_key(name), name)


def build_plot_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return grouped_rows(
        rows,
        lambda row: add_beta1_split_to_plot_group(
            plot_group_name(row), beta1_split_value(row)
        ),
    )


def hparams_for_optimizer(rows: list[dict[str, Any]]) -> list[str]:
    names = {
        key
        for row in rows
        for key, value in row.items()
        if key not in NON_SEARCH_KEYS and is_plot_value(value)
    }
    varying = [
        name
        for name in names
        if len(
            {
                normalize_value(row.get(name))
                for row in rows
                if row.get(name) is not None
            }
        )
        > 1
    ]
    preferred = [name for name in PREFERRED_HPARAM_ORDER if name in varying]
    rest = sorted(name for name in varying if name not in preferred)
    return preferred + rest


def tuple_key_map(value: Any) -> dict[tuple[Any, ...], tuple[float, float]]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[tuple[Any, ...], tuple[float, float]] = {}
    for raw_key, raw_bounds in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_bounds, list):
            continue
        if len(raw_bounds) != 2:
            continue
        try:
            key = ast.literal_eval(raw_key)
            lo = float(raw_bounds[0])
            hi = float(raw_bounds[1])
        except (SyntaxError, ValueError, TypeError):
            continue
        if not isinstance(key, tuple):
            continue
        if math.isfinite(lo) and math.isfinite(hi):
            parsed[key] = (lo, hi)
    return parsed


def numeric_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if math.isfinite(lo) and math.isfinite(hi):
        return lo, hi
    return None


def optimizer_lr_ranges(search_plan: dict[str, Any]) -> dict[str, tuple[float, float]]:
    optimizers = [
        str(optimizer)
        for optimizer in search_plan.get("optimizers", [])
        if isinstance(optimizer, str)
    ]
    ranges: dict[str, tuple[float, float]] = {}

    for optimizer, bounds in tuple_key_map(
        search_plan.get("optimizer_lr_ranges")
    ).items():
        if len(optimizer) == 1 and isinstance(optimizer[0], str):
            ranges[optimizer[0]] = bounds

    adam_bounds = numeric_pair(search_plan.get("adam_lr_range"))
    if adam_bounds is not None:
        for optimizer in optimizers or ["AdamW", "Adam", "AdamH"]:
            if optimizer.lower().startswith("adam"):
                ranges[optimizer] = adam_bounds

    sgd_bounds = numeric_pair(search_plan.get("sgd_lr_range"))
    if sgd_bounds is not None:
        for optimizer in optimizers or ["SGD", "SGD2"]:
            if optimizer.lower().startswith("sgd"):
                ranges[optimizer] = sgd_bounds

    return ranges


def build_search_space(search_plan: dict[str, Any]) -> dict[str, Any]:
    global_bounds: dict[str, tuple[float, float]] = {}
    beta1_momentum_values = search_plan.get("beta1_momentum_values")
    if isinstance(beta1_momentum_values, list) and beta1_momentum_values:
        values = [float(value) for value in beta1_momentum_values]
        global_bounds["beta1"] = (min(values), max(values))
        global_bounds["momentum"] = (min(values), max(values))

    for plan_key, hparam in (("adam_beta2s", "beta2"), ("adam_eps", "eps")):
        values = search_plan.get(plan_key)
        if isinstance(values, list) and values:
            numeric_values = [float(value) for value in values]
            global_bounds[hparam] = (min(numeric_values), max(numeric_values))

    for plan_key, hparam in (("adam_beta2", "beta2"), ("adam_eps", "eps")):
        value = search_plan.get(plan_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            global_bounds[hparam] = (float(value), float(value))

    lr_grid = search_plan.get("adam_lr_grid")
    if isinstance(lr_grid, list) and lr_grid:
        numeric_values = [float(value) for value in lr_grid]
        if all(math.isfinite(value) for value in numeric_values):
            global_bounds["lr"] = (min(numeric_values), max(numeric_values))

    lr_range = numeric_pair(search_plan.get("lr_range"))
    if lr_range is not None:
        global_bounds["lr"] = lr_range

    for hparam in ("lr_decay", "lr_power", "wd"):
        value = search_plan.get(hparam)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            global_bounds[hparam] = (float(value), float(value))
    return {
        "lr_ranges": tuple_key_map(search_plan.get("lr_ranges")),
        "lr_decay_ranges": tuple_key_map(search_plan.get("lr_decay_ranges")),
        "optimizer_lr_ranges": optimizer_lr_ranges(search_plan),
        "global_bounds": global_bounds,
    }


def plot_group_components(plot_group: str) -> tuple[str, str | None]:
    plot_group = plot_group_base_name(plot_group)
    for h_norm in H_NORM_ORDER:
        suffix = f"_{h_norm}"
        if plot_group.endswith(suffix):
            return plot_group[: -len(suffix)], h_norm
    return plot_group, None


def observed_hparam_bounds(
    rows: list[dict[str, Any]], hparam: str
) -> tuple[float, float] | None:
    values = [
        float(value)
        for row in rows
        if (value := row.get(hparam)) is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not values:
        return None
    return min(values), max(values)


def search_space_bounds(
    rows: list[dict[str, Any]],
    hparam: str,
    plot_group: str,
    batch_size: int,
    num_samples: int,
    search_space: dict[str, Any],
) -> tuple[float, float] | None:
    if rows:
        optimizer = str(rows[0]["optimizer"])
        h_norm = str(rows[0]["h_norm"]) if rows[0].get("h_norm") is not None else None
    else:
        optimizer, h_norm = plot_group_components(plot_group)
    if hparam == "lr":
        bounds = search_space["lr_ranges"].get(
            (optimizer, h_norm, num_samples, batch_size)
        )
        if bounds is not None:
            return bounds
        bounds = search_space["optimizer_lr_ranges"].get(optimizer)
        if bounds is not None:
            return bounds
        bounds = search_space["global_bounds"].get(hparam)
        if bounds is not None:
            return bounds
    if hparam == "lr_decay":
        bounds = search_space["lr_decay_ranges"].get(
            (optimizer, num_samples, batch_size)
        )
        if bounds is not None:
            return bounds
    bounds = search_space["global_bounds"].get(hparam)
    if bounds is not None:
        return bounds
    return observed_hparam_bounds(rows, hparam)


def finite_positive(values: list[float] | np.ndarray) -> bool:
    arr = np.array(values, dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def top_rows(rows: list[dict[str, Any]], count: int = TOP_K) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[SSE_KEY]))[:count]


def summarize_hparam(rows: list[dict[str, Any]], hparam: str) -> dict[str, Any]:
    values = [normalize_value(row[hparam]) for row in rows]
    best_rows = top_rows(rows)
    top_values = [normalize_value(row[hparam]) for row in best_rows]
    top_sses = [float(row[SSE_KEY]) for row in best_rows]

    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return {
            "kind": "numeric",
            "top_values": [float(value) for value in top_values],
            "top_sses": top_sses,
            "observed_best_value": float(top_values[0]),
            "observed_best_sse": top_sses[0],
        }

    categories = sorted(set(values), key=category_sort_key)
    return {
        "kind": "categorical",
        "categories": categories,
        "top_values": top_values,
        "top_sses": top_sses,
        "observed_best_value": top_values[0],
        "observed_best_sse": top_sses[0],
    }


def format_top_values(summary: dict[str, Any]) -> str:
    parts = []
    for idx, (value, sse) in enumerate(
        zip(summary["top_values"], summary["top_sses"], strict=False),
        start=1,
    ):
        parts.append(f"top{idx}={format_value(value)} (sse={format_value(float(sse))})")
    return "; ".join(parts)


def draw_bound_lines(
    ax: plt.Axes,
    bounds_by_x: dict[int, tuple[float, float] | None],
) -> None:
    items = sorted(
        (x, bounds) for x, bounds in bounds_by_x.items() if bounds is not None
    )
    if not items:
        return
    xs = np.array([x for x, _ in items], dtype=float)
    lows = np.array([bounds[0] for _, bounds in items], dtype=float)
    highs = np.array([bounds[1] for _, bounds in items], dtype=float)
    line_kwargs = {
        "color": "black",
        "linewidth": 0.9,
        "alpha": 0.85,
        "marker": "_",
        "markersize": 7,
        "zorder": 1,
    }
    ax.plot(xs, lows, **line_kwargs)
    if not np.allclose(lows, highs, rtol=0.0, atol=1e-12):
        ax.plot(xs, highs, **line_kwargs)


def draw_hparam_trend_subplot(
    ax: plt.Axes,
    summaries_by_x: dict[int, dict[str, Any]],
    bounds_by_x: dict[int, tuple[float, float] | None],
    hparam: str,
    x_label: str,
    title: str,
) -> None:
    xs = sorted(summaries_by_x)
    first = summaries_by_x[xs[0]]
    rank_styles = (
        ("#b83232", "o", "rank 1"),
        ("#2f6f9f", "s", "rank 2"),
        ("#6f8f2f", "^", "rank 3"),
    )

    if first["kind"] == "numeric":
        draw_bound_lines(ax, bounds_by_x)
        finite_values: list[float] = []
        for rank, (color, marker, label) in enumerate(rank_styles):
            rank_xs = [x for x in xs if rank < len(summaries_by_x[x]["top_values"])]
            rank_values = [
                float(summaries_by_x[x]["top_values"][rank]) for x in rank_xs
            ]
            finite_values.extend(rank_values)
            if rank_xs:
                ax.plot(
                    rank_xs,
                    rank_values,
                    color=color,
                    marker=marker,
                    markersize=3.8,
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                    zorder=3,
                )
        for bounds in bounds_by_x.values():
            if bounds is not None:
                finite_values.extend(bounds)
        if hparam in LOG_SCALE_HPARAMS and finite_positive(np.array(finite_values)):
            ax.set_yscale("log")
        ax.set_ylabel(hparam)
    else:
        category_union = sorted(
            {
                category
                for summary in summaries_by_x.values()
                for category in summary["categories"]
            },
            key=category_sort_key,
        )
        positions = {value: idx for idx, value in enumerate(category_union)}
        for rank, (color, marker, label) in enumerate(rank_styles):
            rank_xs = [x for x in xs if rank < len(summaries_by_x[x]["top_values"])]
            rank_values = [
                positions[summaries_by_x[x]["top_values"][rank]] for x in rank_xs
            ]
            if rank_xs:
                ax.plot(
                    rank_xs,
                    rank_values,
                    color=color,
                    marker=marker,
                    markersize=3.8,
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                    zorder=3,
                )
        ax.set_yticks(range(len(category_union)))
        ax.set_yticklabels([format_value(value) for value in category_union])
        ax.set_ylabel(hparam)

    if finite_positive(np.array(xs, dtype=float)):
        ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xs]))
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=8)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="best", fontsize=6, frameon=False)


def plot_optimizer_figure(
    plot_group: str,
    plot_rows: list[dict[str, Any]],
    out_dir: Path,
    dpi: int,
    search_space: dict[str, Any],
) -> tuple[Path, list[str], dict[tuple[int, int, str], dict[str, Any]]]:
    hparams = hparams_for_optimizer(plot_rows)
    if not hparams:
        raise ValueError(f"no varying searched hparams found for {plot_group}")

    cell_groups = grouped_rows(
        plot_rows,
        lambda row: (int(row["batch_size"]), int(sample_budget(row) or 0)),
    )
    batch_sizes = sorted({batch_size for batch_size, _ in cell_groups})
    sample_budgets = sorted({num_samples for _, num_samples in cell_groups})
    summaries: dict[tuple[int, int, str], dict[str, Any]] = {}
    bounds: dict[tuple[int, int, str], tuple[float, float] | None] = {}
    for batch_size in batch_sizes:
        for num_samples in sample_budgets:
            cell_rows = cell_groups[(batch_size, num_samples)]
            for hparam in hparams:
                hparam_rows = [row for row in cell_rows if hparam in row]
                summaries[(batch_size, num_samples, hparam)] = summarize_hparam(
                    hparam_rows, hparam
                )
                bounds[(batch_size, num_samples, hparam)] = search_space_bounds(
                    hparam_rows,
                    hparam,
                    plot_group,
                    batch_size,
                    num_samples,
                    search_space,
                )

    ncols = len(batch_sizes) + len(sample_budgets)
    nrows = len(hparams)
    fig_width = 3.45 * ncols
    fig_height = max(2.65 * nrows, 4.6)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.45, wspace=0.28)

    for row_idx, hparam in enumerate(hparams):
        for col_idx, batch_size in enumerate(batch_sizes):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            draw_hparam_trend_subplot(
                ax,
                {
                    num_samples: summaries[(batch_size, num_samples, hparam)]
                    for num_samples in sample_budgets
                },
                {
                    num_samples: bounds[(batch_size, num_samples, hparam)]
                    for num_samples in sample_budgets
                },
                hparam,
                "num samples",
                f"batch={batch_size}",
            )
        offset = len(batch_sizes)
        for sample_idx, num_samples in enumerate(sample_budgets):
            ax = fig.add_subplot(grid[row_idx, offset + sample_idx])
            draw_hparam_trend_subplot(
                ax,
                {
                    batch_size: summaries[(batch_size, num_samples, hparam)]
                    for batch_size in batch_sizes
                },
                {
                    batch_size: bounds[(batch_size, num_samples, hparam)]
                    for batch_size in batch_sizes
                },
                hparam,
                "batch size",
                f"samples={num_samples}",
            )

    fig.suptitle(f"{plot_group} hparam search", fontsize=15, y=0.998)
    out_path = out_dir / f"optimizer_{safe_filename(plot_group)}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, hparams, summaries


def clean_output_dir(out_dir: Path) -> None:
    for path in out_dir.glob("optimizer_*.png"):
        path.unlink()
    summary_path = out_dir / SUMMARY_FILENAME
    if summary_path.exists():
        summary_path.unlink()


def write_summary(
    out_path: Path,
    log_path: Path,
    rows: list[dict[str, Any]],
    plot_groups: dict[str, list[dict[str, Any]]],
    plot_hparams: dict[str, list[str]],
    plot_summaries: dict[str, dict[tuple[int, int, str], dict[str, Any]]],
) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Optimizer hparam plot summary\n")
        handle.write("=============================\n\n")
        handle.write(f"Input log: {log_path}\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Split hparam: {BETA1_SPLIT_HPARAM}\n")
        handle.write(
            "Plot groups: "
            + ", ".join(sorted(plot_groups, key=plot_group_sort_key))
            + "\n\n"
        )

        for plot_group in sorted(plot_groups, key=plot_group_sort_key):
            batch_sizes = sorted(
                {int(row["batch_size"]) for row in plot_groups[plot_group]}
            )
            hparams = plot_hparams[plot_group]
            summaries = plot_summaries[plot_group]
            handle.write(f"{plot_group}\n")
            handle.write("-" * len(plot_group) + "\n")
            handle.write(f"Figure: optimizer_{safe_filename(plot_group)}.png\n")
            handle.write(f"Rows: {len(plot_groups[plot_group])}\n")
            sample_budgets = sorted(
                {
                    budget
                    for row in plot_groups[plot_group]
                    if (budget := sample_budget(row)) is not None
                }
            )
            if sample_budgets:
                handle.write(
                    "Num samples: "
                    + ", ".join(str(budget) for budget in sample_budgets)
                    + "\n"
                )
            handle.write(
                f"Batch sizes: {', '.join(str(batch) for batch in batch_sizes)}\n"
            )
            handle.write(f"Hparams: {', '.join(hparams)}\n")
            for hparam in hparams:
                handle.write(f"\n{hparam}\n")
                for num_samples in sample_budgets:
                    handle.write(f"  num_samples={num_samples}\n")
                    for batch_size in batch_sizes:
                        summary = summaries[(batch_size, num_samples, hparam)]
                        handle.write(
                            f"    batch_size={batch_size}: "
                            f"{format_top_values(summary)}\n"
                        )
            handle.write("\n")

        all_batch_sizes = sorted({int(row["batch_size"]) for row in rows})
        all_sample_budgets = sorted(
            {budget for row in rows if (budget := sample_budget(row)) is not None}
        )
        handle.write("SSE comparison by batch size x num_samples\n")
        handle.write("==========================================\n")
        for batch_size in all_batch_sizes:
            for num_samples in all_sample_budgets:
                comparisons: list[tuple[float, str]] = []
                for plot_group, group_rows in plot_groups.items():
                    cell_rows = [
                        row
                        for row in group_rows
                        if int(row["batch_size"]) == batch_size
                        and sample_budget(row) == num_samples
                    ]
                    if not cell_rows:
                        continue
                    comparisons.append(
                        (float(best_row(cell_rows)[SSE_KEY]), plot_group)
                    )
                comparisons.sort()
                handle.write(f"\nbatch_size={batch_size}, num_samples={num_samples}\n")
                for sse, plot_group in comparisons:
                    handle.write(f"  {plot_group}: best_sse={format_value(sse)}\n")


def main() -> None:
    args = parse_args()
    log_path = args.log if args.log is not None else default_log_path()
    rows = read_run_summaries(log_path)
    search_space = build_search_space(read_search_plan(log_path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(args.out_dir)

    plot_groups = build_plot_groups(rows)
    outputs: list[Path] = []
    plot_hparams: dict[str, list[str]] = {}
    plot_summaries: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = {}

    for plot_group in sorted(plot_groups, key=plot_group_sort_key):
        output_path, hparams, summaries = plot_optimizer_figure(
            plot_group,
            plot_groups[plot_group],
            args.out_dir,
            args.dpi,
            search_space,
        )
        outputs.append(output_path)
        plot_hparams[plot_group] = hparams
        plot_summaries[plot_group] = summaries

    summary_path = args.out_dir / SUMMARY_FILENAME
    write_summary(
        summary_path,
        log_path,
        rows,
        plot_groups,
        plot_hparams,
        plot_summaries,
    )

    print(f"read {len(rows)} rows from {log_path}")
    print(f"wrote {len(outputs)} optimizer figures to {args.out_dir}")
    print(summary_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

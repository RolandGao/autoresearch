#!/usr/bin/env python3
"""Visualize optimizer hyperparameter search logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_NAME = "optimizers_logging5.log"
DEFAULT_OUT_DIR = Path(__file__).with_name("optimizer_hparam_plots5")
SSE_KEY = "clean_train_sse"
SUMMARY_FILENAME = "optimizer_hparam_summary.txt"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_QUANTILES = (2.5, 50.0, 97.5)
BOOTSTRAP_SAMPLE_FRACTION = 0.25
PREFERRED_OPTIMIZER_ORDER = ("AdamW", "AdamH", "Muon", "MuonH", "SGD", "SGD2")
H_NORM_ORDER = ("matrix", "row")
PREFERRED_HPARAM_ORDER = (
    "lr",
    "lr_decay",
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
    "h_norm",
    "lr_power",
    "lr_schedule",
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
            f"Create one figure per optimizer variant from {DEFAULT_LOG_NAME}. "
            "Each hparam row contains SSE-vs-hparam subplots for every batch "
            "size and one bootstrap hparam-CI-vs-batch-size subplot."
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
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples used for empirical 95%% hparam intervals.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row[SSE_KEY]))


def optimizer_sort_key(name: str) -> tuple[int, str]:
    if name in PREFERRED_OPTIMIZER_ORDER:
        return (PREFERRED_OPTIMIZER_ORDER.index(name), name)
    return (len(PREFERRED_OPTIMIZER_ORDER), name)


def plot_group_name(row: dict[str, Any]) -> str:
    variant = row.get("variant")
    if isinstance(variant, str) and variant:
        return variant
    optimizer = str(row["optimizer"])
    h_norm = row.get("h_norm")
    if h_norm is None:
        return optimizer
    return f"{optimizer}_{h_norm}"


def plot_group_sort_key(name: str) -> tuple[int, int, str]:
    base_optimizer = name
    h_norm_index = -1
    for idx, h_norm in enumerate(H_NORM_ORDER):
        suffix = f"_{h_norm}"
        if name.endswith(suffix):
            base_optimizer = name[: -len(suffix)]
            h_norm_index = idx
            break
    optimizer_index = optimizer_sort_key(base_optimizer)[0]
    return (optimizer_index, h_norm_index, name)


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


def finite_positive(values: list[float] | np.ndarray) -> bool:
    arr = np.array(values, dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def bootstrap_best_hparam_values(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
    if num_samples <= 0:
        return np.array([], dtype=object)
    sses = np.array([float(row[SSE_KEY]) for row in rows], dtype=float)
    values = np.array([normalize_value(row[hparam]) for row in rows], dtype=object)
    sample_size = max(1, round(len(rows) * BOOTSTRAP_SAMPLE_FRACTION))
    sampled_idx = rng.integers(0, len(rows), size=(num_samples, sample_size))
    sampled_sses = sses[sampled_idx]
    best_positions = np.argmin(sampled_sses, axis=1)
    best_idx = sampled_idx[np.arange(num_samples), best_positions]
    return values[best_idx]


def summarize_hparam(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    values = [normalize_value(row[hparam]) for row in rows]
    sses = np.array([float(row[SSE_KEY]) for row in rows], dtype=float)
    observed_best = best_row(rows)
    boot_values = bootstrap_best_hparam_values(rows, hparam, rng, bootstrap_samples)

    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        observed_value = float(observed_best[hparam])
        if len(boot_values):
            lo, median, hi = np.percentile(
                np.array(boot_values, dtype=float), BOOTSTRAP_QUANTILES
            )
        else:
            lo = median = hi = observed_value
        return {
            "kind": "numeric",
            "values": np.array(values, dtype=float),
            "sses": sses,
            "observed_best_value": observed_value,
            "observed_best_sse": float(observed_best[SSE_KEY]),
            "boot_lo": float(lo),
            "boot_median": float(median),
            "boot_hi": float(hi),
        }

    categories = sorted(set(values), key=category_sort_key)
    positions = {value: idx for idx, value in enumerate(categories)}
    encoded_values = np.array([positions[value] for value in values], dtype=float)
    observed_value = normalize_value(observed_best[hparam])
    if len(boot_values):
        boot_encoded = np.array(
            [positions[value] for value in boot_values], dtype=float
        )
        lo, median, hi = np.percentile(boot_encoded, BOOTSTRAP_QUANTILES)
    else:
        lo = median = hi = float(positions[observed_value])
    return {
        "kind": "categorical",
        "values": encoded_values,
        "sses": sses,
        "categories": categories,
        "positions": positions,
        "observed_best_value": observed_value,
        "observed_best_sse": float(observed_best[SSE_KEY]),
        "boot_lo": float(lo),
        "boot_median": float(median),
        "boot_hi": float(hi),
    }


def draw_sse_hparam_subplot(
    ax: plt.Axes,
    summary: dict[str, Any],
    hparam: str,
    batch_size: int,
    rng: np.random.Generator,
) -> None:
    if summary["kind"] == "numeric":
        x = summary["values"]
        ax.scatter(x, summary["sses"], s=18, alpha=0.72, linewidths=0, color="#256d85")
        ax.axvspan(
            summary["boot_lo"],
            summary["boot_hi"],
            color="#b83232",
            alpha=0.14,
            linewidth=0,
        )
        ax.axvline(summary["boot_median"], color="#b83232", linewidth=1.4)
        ax.scatter(
            [summary["observed_best_value"]],
            [summary["observed_best_sse"]],
            s=52,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        if hparam in LOG_SCALE_HPARAMS and finite_positive(x):
            ax.set_xscale("log")
    else:
        x = summary["values"]
        jitter = rng.uniform(-0.08, 0.08, size=len(x))
        ax.scatter(
            x + jitter,
            summary["sses"],
            s=18,
            alpha=0.72,
            linewidths=0,
            color="#256d85",
        )
        ax.axvspan(
            summary["boot_lo"] - 0.18,
            summary["boot_hi"] + 0.18,
            color="#b83232",
            alpha=0.14,
            linewidth=0,
        )
        ax.axvline(summary["boot_median"], color="#b83232", linewidth=1.4)
        ax.scatter(
            [summary["positions"][summary["observed_best_value"]]],
            [summary["observed_best_sse"]],
            s=52,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        ax.set_xticks(range(len(summary["categories"])))
        ax.set_xticklabels([format_value(value) for value in summary["categories"]])
        ax.set_xlim(-0.5, len(summary["categories"]) - 0.5)

    ax.set_yscale("log")
    ax.set_title(
        f"batch={batch_size}\n"
        f"median={format_value(summary['boot_median'])}, "
        f"best={format_value(summary['observed_best_value'])}",
        fontsize=8,
    )
    ax.set_xlabel(hparam)
    ax.grid(True, which="both", alpha=0.22)


def draw_ci_vs_batch_subplot(
    ax: plt.Axes,
    summaries_by_batch: dict[int, dict[str, Any]],
    hparam: str,
) -> None:
    batch_sizes = sorted(summaries_by_batch)
    first = summaries_by_batch[batch_sizes[0]]

    if first["kind"] == "numeric":
        observed = np.array(
            [
                float(summaries_by_batch[batch]["observed_best_value"])
                for batch in batch_sizes
            ],
            dtype=float,
        )
        medians = np.array(
            [float(summaries_by_batch[batch]["boot_median"]) for batch in batch_sizes],
            dtype=float,
        )
        lo = np.array(
            [float(summaries_by_batch[batch]["boot_lo"]) for batch in batch_sizes],
            dtype=float,
        )
        hi = np.array(
            [float(summaries_by_batch[batch]["boot_hi"]) for batch in batch_sizes],
            dtype=float,
        )
        ax.fill_between(batch_sizes, lo, hi, color="#b83232", alpha=0.14)
        ax.plot(batch_sizes, medians, color="#b83232", marker="o", linewidth=1.5)
        ax.scatter(batch_sizes, observed, color="#256d85", s=34, zorder=4)
        if hparam in LOG_SCALE_HPARAMS and finite_positive(
            np.concatenate([observed, medians, lo, hi])
        ):
            ax.set_yscale("log")
        ax.set_ylabel(hparam)
    else:
        category_union = sorted(
            {
                category
                for summary in summaries_by_batch.values()
                for category in summary["categories"]
            },
            key=category_sort_key,
        )
        positions = {category: idx for idx, category in enumerate(category_union)}
        observed = np.array(
            [
                positions[summaries_by_batch[batch]["observed_best_value"]]
                for batch in batch_sizes
            ],
            dtype=float,
        )
        medians = np.array(
            [float(summaries_by_batch[batch]["boot_median"]) for batch in batch_sizes],
            dtype=float,
        )
        lo = np.array(
            [float(summaries_by_batch[batch]["boot_lo"]) for batch in batch_sizes],
            dtype=float,
        )
        hi = np.array(
            [float(summaries_by_batch[batch]["boot_hi"]) for batch in batch_sizes],
            dtype=float,
        )
        ax.fill_between(batch_sizes, lo, hi, color="#b83232", alpha=0.14)
        ax.plot(batch_sizes, medians, color="#b83232", marker="o", linewidth=1.5)
        ax.scatter(batch_sizes, observed, color="#256d85", s=34, zorder=4)
        ax.set_yticks(range(len(category_union)))
        ax.set_yticklabels([format_value(value) for value in category_union])
        ax.set_ylabel(hparam)

    ax.set_xticks(batch_sizes)
    ax.set_xlabel("batch size")
    ax.set_title("hparam CI vs batch size", fontsize=8)
    ax.grid(True, which="both", alpha=0.22)


def format_ci(summary: dict[str, Any]) -> str:
    if summary["kind"] == "numeric":
        return (
            f"median={format_value(summary['boot_median'])}, "
            f"95% CI=[{format_value(summary['boot_lo'])}, "
            f"{format_value(summary['boot_hi'])}]"
        )
    categories = summary["categories"]
    lo = categories[int(np.clip(round(summary["boot_lo"]), 0, len(categories) - 1))]
    median = categories[
        int(np.clip(round(summary["boot_median"]), 0, len(categories) - 1))
    ]
    hi = categories[int(np.clip(round(summary["boot_hi"]), 0, len(categories) - 1))]
    return (
        f"median={format_value(median)}, "
        f"95% CI=[{format_value(lo)}, {format_value(hi)}]"
    )


def plot_optimizer_figure(
    plot_group: str,
    plot_rows: list[dict[str, Any]],
    out_dir: Path,
    rng: np.random.Generator,
    bootstrap_samples: int,
    dpi: int,
) -> tuple[Path, list[str], dict[tuple[int, str], dict[str, Any]]]:
    hparams = hparams_for_optimizer(plot_rows)
    if not hparams:
        raise ValueError(f"no varying searched hparams found for {plot_group}")

    batch_groups = grouped_rows(plot_rows, lambda row: int(row["batch_size"]))
    batch_sizes = sorted(batch_groups)
    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for batch_size in batch_sizes:
        for hparam in hparams:
            summaries[(batch_size, hparam)] = summarize_hparam(
                [row for row in batch_groups[batch_size] if hparam in row],
                hparam,
                rng,
                bootstrap_samples,
            )

    ncols = len(batch_sizes) + 1
    nrows = len(hparams)
    fig_width = 4.4 * ncols
    fig_height = max(3.0 * nrows, 5.0)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.45, wspace=0.28)

    for row_idx, hparam in enumerate(hparams):
        for col_idx, batch_size in enumerate(batch_sizes):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            draw_sse_hparam_subplot(
                ax,
                summaries[(batch_size, hparam)],
                hparam,
                batch_size,
                rng,
            )
            ax.set_ylabel("clean train SSE" if col_idx == 0 else "")
        ax = fig.add_subplot(grid[row_idx, len(batch_sizes)])
        draw_ci_vs_batch_subplot(
            ax,
            {batch_size: summaries[(batch_size, hparam)] for batch_size in batch_sizes},
            hparam,
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
    plot_summaries: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Optimizer hparam plot summary\n")
        handle.write("=============================\n\n")
        handle.write(f"Input log: {log_path}\n")
        handle.write(f"Rows: {len(rows)}\n")
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
            handle.write(
                f"Batch sizes: {', '.join(str(batch) for batch in batch_sizes)}\n"
            )
            handle.write(f"Hparams: {', '.join(hparams)}\n")
            for hparam in hparams:
                handle.write(f"\n{hparam}\n")
                for batch_size in batch_sizes:
                    summary = summaries[(batch_size, hparam)]
                    handle.write(
                        f"  batch_size={batch_size}: {format_ci(summary)}; "
                        f"observed_best={format_value(summary['observed_best_value'])}; "
                        f"best_sse={format_value(summary['observed_best_sse'])}\n"
                    )
            handle.write("\n")


def main() -> None:
    args = parse_args()
    log_path = args.log if args.log is not None else default_log_path()
    rows = read_run_summaries(log_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    plot_groups = grouped_rows(rows, plot_group_name)
    outputs: list[Path] = []
    plot_hparams: dict[str, list[str]] = {}
    plot_summaries: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}

    for plot_group in sorted(plot_groups, key=plot_group_sort_key):
        output_path, hparams, summaries = plot_optimizer_figure(
            plot_group,
            plot_groups[plot_group],
            args.out_dir,
            rng,
            args.bootstrap_samples,
            args.dpi,
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

#!/usr/bin/env python3
"""Visualize SGDH hparam search results from train_learnable_softmax2.py logs."""

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


DEFAULT_LOG = Path(__file__).with_name("sgdh_ablation2.log")
DEFAULT_OUT_DIR = Path(__file__).with_name("sgdh_ablation2_variant_plots")
RMSE_KEY = "clean_train_rmse"
BOOTSTRAP_FRACTION = 0.25
OPTIMIZER_FIELDS = ("g_projection", "g_norm", "nesterov", "m_projection")
SEARCH_HPARAM_KEYS = ("lr", "lr_decay", "momentum")
LOG_SCALE_HPARAMS = {"lr"}
SUMMARY_FILENAME = "sgdh_ablation2_variant_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one figure per SGDH boolean variant with batch/hparam subplots "
            "and best-hparam confidence intervals versus batch size."
        )
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Input log file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples used for best-hparam intervals.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser.parse_args()


def read_run_summaries(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            if RMSE_KEY in row and all(field in row for field in OPTIMIZER_FIELDS):
                rows.append(row)
    if not rows:
        raise ValueError(f"no RUN_SUMMARY rows with {RMSE_KEY!r} found in {log_path}")
    return rows


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
            return str(int(round(value)))
        return f"{value:.10g}"
    return str(value)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def bool_token(value: Any) -> str:
    return "T" if bool(value) else "F"


def variant_label(row: dict[str, Any]) -> str:
    return ",".join(f"{field}={bool_token(row[field])}" for field in OPTIMIZER_FIELDS)


def short_variant_label(row: dict[str, Any]) -> str:
    names = {
        "g_projection": "gp",
        "g_norm": "gn",
        "nesterov": "nes",
        "m_projection": "mp",
    }
    return "_".join(f"{names[field]}{bool_token(row[field])}" for field in OPTIMIZER_FIELDS)


def variant_sort_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(bool(row[field])) for field in OPTIMIZER_FIELDS)


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


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row[RMSE_KEY]))


def searched_hparams(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for hparam in SEARCH_HPARAM_KEYS:
        values = {
            normalize_value(row.get(hparam))
            for row in rows
            if row.get(hparam) is not None
        }
        if len(values) > 1:
            names.append(hparam)
    return names


def finite_positive(values: list[float] | np.ndarray) -> bool:
    arr = np.array(values, dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def discrete_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("cannot compute quantile of an empty array")
    sorted_values = np.sort(np.array(values, dtype=float))
    idx = int(math.ceil(quantile * len(sorted_values))) - 1
    idx = int(np.clip(idx, 0, len(sorted_values) - 1))
    return float(sorted_values[idx])


def bootstrap_best_hparam_values(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
    if num_samples <= 0:
        return np.array([], dtype=object)
    rmses = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    values = np.array([normalize_value(row[hparam]) for row in rows], dtype=object)
    sample_size = max(1, int(round(BOOTSTRAP_FRACTION * len(rows))))
    sampled_idx = rng.integers(0, len(rows), size=(num_samples, sample_size))
    sampled_rmses = rmses[sampled_idx]
    best_positions = np.argmin(sampled_rmses, axis=1)
    best_idx = sampled_idx[np.arange(num_samples), best_positions]
    return values[best_idx]


def clean_output_dir(out_dir: Path) -> None:
    for path in out_dir.glob("*.png"):
        path.unlink()
    summary_path = out_dir / SUMMARY_FILENAME
    if summary_path.exists():
        summary_path.unlink()


def summarize_hparam(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    values = [normalize_value(row[hparam]) for row in rows]
    rmses = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    observed_best = best_row(rows)
    boot_values = bootstrap_best_hparam_values(rows, hparam, rng, bootstrap_samples)

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        observed_value = float(observed_best[hparam])
        if len(boot_values):
            boot_numeric = np.array(boot_values, dtype=float)
            lo, median, hi = np.percentile(boot_numeric, [2.5, 50.0, 97.5])
        else:
            lo = median = hi = observed_value
        return {
            "kind": "numeric",
            "values": np.array(values, dtype=float),
            "rmses": rmses,
            "observed_best_value": observed_value,
            "observed_best_rmse": float(observed_best[RMSE_KEY]),
            "boot_lo": float(lo),
            "boot_median": float(median),
            "boot_hi": float(hi),
        }

    categories = sorted(set(values), key=category_sort_key)
    positions = {value: idx for idx, value in enumerate(categories)}
    encoded_values = np.array([positions[value] for value in values], dtype=float)
    if len(boot_values):
        boot_encoded = np.array([positions[value] for value in boot_values], dtype=float)
        lo = discrete_quantile(boot_encoded, 0.025)
        median = discrete_quantile(boot_encoded, 0.5)
        hi = discrete_quantile(boot_encoded, 0.975)
    else:
        observed_pos = float(positions[normalize_value(observed_best[hparam])])
        lo = median = hi = observed_pos
    return {
        "kind": "categorical",
        "values": encoded_values,
        "rmses": rmses,
        "categories": categories,
        "positions": positions,
        "observed_best_value": normalize_value(observed_best[hparam]),
        "observed_best_rmse": float(observed_best[RMSE_KEY]),
        "boot_lo": float(lo),
        "boot_median": float(median),
        "boot_hi": float(hi),
    }


def draw_rmse_hparam_subplot(
    ax: plt.Axes,
    summary: dict[str, Any],
    hparam: str,
    batch_size: int,
    rng: np.random.Generator,
    title_prefix: str = "",
) -> None:
    if summary["kind"] == "numeric":
        x = summary["values"]
        ax.scatter(x, summary["rmses"], s=18, alpha=0.7, linewidths=0, color="#286983")
        ax.axvspan(summary["boot_lo"], summary["boot_hi"], color="#b83232", alpha=0.14, linewidth=0)
        ax.axvline(summary["boot_median"], color="#b83232", linewidth=1.3)
        ax.scatter(
            [summary["observed_best_value"]],
            [summary["observed_best_rmse"]],
            s=54,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        if hparam in LOG_SCALE_HPARAMS and finite_positive(x):
            ax.set_xscale("log")
        unique_values = sorted(set(float(value) for value in x))
        if len(unique_values) <= 8:
            ax.set_xticks(unique_values)
            ax.set_xticklabels([format_value(value) for value in unique_values])
    else:
        x = summary["values"]
        jitter = rng.uniform(-0.08, 0.08, size=len(x))
        ax.scatter(x + jitter, summary["rmses"], s=18, alpha=0.7, linewidths=0, color="#286983")
        ax.axvspan(summary["boot_lo"] - 0.18, summary["boot_hi"] + 0.18, color="#b83232", alpha=0.14, linewidth=0)
        ax.axvline(summary["boot_median"], color="#b83232", linewidth=1.3)
        ax.scatter(
            [summary["positions"][summary["observed_best_value"]]],
            [summary["observed_best_rmse"]],
            s=54,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        ax.set_xticks(range(len(summary["categories"])))
        ax.set_xticklabels([format_value(value) for value in summary["categories"]])
        ax.set_xlim(-0.5, len(summary["categories"]) - 0.5)

    ax.set_yscale("log")
    title = (
        f"{title_prefix}batch={batch_size}\n"
        f"best={format_value(summary['observed_best_value'])}, "
        f"RMSE={format_value(summary['observed_best_rmse'])}"
    )
    ax.set_title(title.strip(), fontsize=8)
    ax.grid(True, which="both", alpha=0.22)


def draw_ci_vs_batch_subplot(
    ax: plt.Axes,
    summaries_by_batch: dict[int, dict[str, Any]],
    hparam: str,
) -> None:
    batch_sizes = sorted(summaries_by_batch)
    first_summary = summaries_by_batch[batch_sizes[0]]

    if first_summary["kind"] == "numeric":
        observed = np.array(
            [float(summaries_by_batch[batch]["observed_best_value"]) for batch in batch_sizes],
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
        ax.plot(batch_sizes, medians, color="#b83232", marker="o", linewidth=1.6, label="bootstrap median")
        ax.scatter(batch_sizes, observed, color="#286983", s=34, zorder=4, label="observed best")
        if hparam in LOG_SCALE_HPARAMS and finite_positive(np.concatenate([observed, medians, lo, hi])):
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
            [positions[summaries_by_batch[batch]["observed_best_value"]] for batch in batch_sizes],
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
        ax.plot(batch_sizes, medians, color="#b83232", marker="o", linewidth=1.6, label="bootstrap median")
        ax.scatter(batch_sizes, observed, color="#286983", s=34, zorder=4, label="observed best")
        ax.set_yticks(range(len(category_union)))
        ax.set_yticklabels([format_value(value) for value in category_union])
        ax.set_ylabel(hparam)

    ax.set_xticks(batch_sizes)
    ax.set_xlabel("batch size")
    ax.set_title(f"{hparam} CI vs batch size", fontsize=9)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=7, loc="best")


def format_ci(summary: dict[str, Any]) -> str:
    if summary["kind"] == "numeric":
        return (
            f"median={format_value(summary['boot_median'])}, "
            f"95% CI=[{format_value(summary['boot_lo'])}, {format_value(summary['boot_hi'])}]"
        )
    categories = summary["categories"]
    lo_idx = int(round(summary["boot_lo"]))
    median_idx = int(round(summary["boot_median"]))
    hi_idx = int(round(summary["boot_hi"]))
    lo_value = categories[lo_idx]
    median_value = categories[median_idx]
    hi_value = categories[hi_idx]
    return (
        f"median={format_value(median_value)}, "
        f"95% CI=[{format_value(lo_value)}, {format_value(hi_value)}]"
    )


def ci_bounds_text(summary: dict[str, Any]) -> str:
    if summary["kind"] == "numeric":
        return f"[{format_value(summary['boot_lo'])}, {format_value(summary['boot_hi'])}]"
    categories = summary["categories"]
    lo_value = categories[int(round(summary["boot_lo"]))]
    hi_value = categories[int(round(summary["boot_hi"]))]
    return f"[{format_value(lo_value)}, {format_value(hi_value)}]"


def build_optimizer_comparison(
    batch_sizes: list[int],
    ordered_variants: list[str],
    variant_groups: dict[str, list[dict[str, Any]]],
) -> tuple[dict[int, list[tuple[str, float]]], dict[str, dict[str, Any]]]:
    best_rmse_by_variant: dict[str, dict[int, float]] = {}
    for variant in ordered_variants:
        batch_groups = grouped_rows(variant_groups[variant], lambda row: int(row["batch_size"]))
        best_rmse_by_variant[variant] = {
            batch_size: float(best_row(batch_groups[batch_size])[RMSE_KEY])
            for batch_size in batch_sizes
        }

    ranked_by_batch: dict[int, list[tuple[str, float]]] = {}
    for batch_size in batch_sizes:
        ranked_by_batch[batch_size] = sorted(
            ((variant, best_rmse_by_variant[variant][batch_size]) for variant in ordered_variants),
            key=lambda item: (item[1], item[0]),
        )

    rank_by_batch: dict[int, dict[str, int]] = {}
    for batch_size, ranked in ranked_by_batch.items():
        rank_by_batch[batch_size] = {
            variant: rank for rank, (variant, _) in enumerate(ranked, start=1)
        }

    overall: dict[str, dict[str, Any]] = {}
    for variant in ordered_variants:
        ranks = [rank_by_batch[batch_size][variant] for batch_size in batch_sizes]
        rmses = [best_rmse_by_variant[variant][batch_size] for batch_size in batch_sizes]
        overall[variant] = {
            "mean_rank": float(np.mean(ranks)),
            "worst_rank": int(max(ranks)),
            "wins": int(sum(rank == 1 for rank in ranks)),
            "mean_best_rmse": float(np.mean(rmses)),
            "best_rmse_by_batch": best_rmse_by_variant[variant],
        }

    return ranked_by_batch, overall


def median_value_text(summary: dict[str, Any]) -> str:
    if summary["kind"] == "numeric":
        return format_value(summary["boot_median"])
    categories = summary["categories"]
    median_value = categories[int(round(summary["boot_median"]))]
    return format_value(median_value)


def write_summary(
    out_path: Path,
    log_path: Path,
    rows: list[dict[str, Any]],
    hparams: list[str],
    ordered_variants: list[str],
    variant_groups: dict[str, list[dict[str, Any]]],
    variant_summaries: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> None:
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("SGDH ablation2 plot summary\n")
        handle.write("==========================\n\n")
        handle.write(f"Input log: {log_path}\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Batch sizes: {', '.join(str(batch) for batch in batch_sizes)}\n")
        handle.write(f"Hparams: {', '.join(hparams)}\n\n")

        for variant in ordered_variants:
            example_row = variant_groups[variant][0]
            handle.write(f"Variant: {variant}\n")
            handle.write(f"Figure: variant_{safe_filename(short_variant_label(example_row))}.png\n")
            handle.write("-" * 72 + "\n")
            summaries = variant_summaries[variant]

            handle.write("RMSE vs hparam subplots\n")
            for hparam in hparams:
                handle.write(f"{hparam}\n")
                for batch_size in batch_sizes:
                    summary = summaries[(batch_size, hparam)]
                    handle.write(
                        f"  batch_size={batch_size}: {format_ci(summary)}; "
                        f"observed_best={format_value(summary['observed_best_value'])}; "
                        f"best_rmse={format_value(summary['observed_best_rmse'])}\n"
                    )

            handle.write("CI vs batch size subplots\n")
            for hparam in hparams:
                median_path = " | ".join(
                    f"{batch_size}:{format_value(summaries[(batch_size, hparam)]['boot_median'])}"
                    for batch_size in batch_sizes
                )
                ci_path = " | ".join(
                    f"{batch_size}:{ci_bounds_text(summaries[(batch_size, hparam)])}"
                    for batch_size in batch_sizes
                )
                observed_path = " | ".join(
                    f"{batch_size}:{format_value(summaries[(batch_size, hparam)]['observed_best_value'])}"
                    for batch_size in batch_sizes
                )
                handle.write(f"{hparam}\n")
                handle.write(f"  median path: {median_path}\n")
                handle.write(f"  95% CI path: {ci_path}\n")
                handle.write(f"  observed best path: {observed_path}\n")
            handle.write("\n")

        ranked_by_batch, overall = build_optimizer_comparison(
            batch_sizes,
            ordered_variants,
            variant_groups,
        )
        overall_ranked = sorted(
            ordered_variants,
            key=lambda variant: (
                overall[variant]["mean_rank"],
                overall[variant]["worst_rank"],
                overall[variant]["mean_best_rmse"],
                variant,
            ),
        )

        handle.write("Optimizer comparison across batch sizes\n")
        handle.write("======================================\n\n")
        handle.write("Per-batch rankings (top 5)\n")
        for batch_size in batch_sizes:
            handle.write(f"batch_size={batch_size}\n")
            for rank, (variant, rmse) in enumerate(ranked_by_batch[batch_size][:5], start=1):
                handle.write(
                    f"  {rank}. {variant}  best_rmse={format_value(rmse)}\n"
                )
            handle.write("\n")

        handle.write("Overall ranking across batch sizes\n")
        handle.write(
            "rank | mean_rank | worst_rank | wins | mean_best_rmse | "
            + " | ".join(f"batch_{batch}" for batch in batch_sizes)
            + " | variant\n"
        )
        for rank, variant in enumerate(overall_ranked, start=1):
            stats = overall[variant]
            batch_rmse_text = " | ".join(
                f"{format_value(stats['best_rmse_by_batch'][batch_size]):>11}"
                for batch_size in batch_sizes
            )
            handle.write(
                f"{rank:>4} | "
                f"{stats['mean_rank']:>9.2f} | "
                f"{stats['worst_rank']:>10} | "
                f"{stats['wins']:>4} | "
                f"{format_value(stats['mean_best_rmse']):>14} | "
                f"{batch_rmse_text} | {variant}\n"
            )

        handle.write("\nRecommended bootstrap medians by optimizer\n")
        handle.write("=========================================\n\n")
        handle.write(
            "These are the bootstrap median hparams from the plotted summaries. "
            "The overall recommendation picks the batch size where that optimizer "
            "achieved its lowest observed best RMSE.\n\n"
        )
        for variant in ordered_variants:
            summaries = variant_summaries[variant]
            stats = overall[variant]
            best_batch_size = min(
                batch_sizes,
                key=lambda batch_size: (
                    stats["best_rmse_by_batch"][batch_size],
                    batch_size,
                ),
            )
            overall_triplet = ", ".join(
                f"{hparam}={median_value_text(summaries[(best_batch_size, hparam)])}"
                for hparam in hparams
            )
            handle.write(f"{variant}\n")
            handle.write(
                f"  overall recommended batch_size={best_batch_size}: "
                f"{overall_triplet}; "
                f"best_rmse={format_value(stats['best_rmse_by_batch'][best_batch_size])}\n"
            )
            handle.write("  per-batch medians\n")
            for batch_size in batch_sizes:
                triplet = ", ".join(
                    f"{hparam}={median_value_text(summaries[(batch_size, hparam)])}"
                    for hparam in hparams
                )
                handle.write(
                    f"    batch_size={batch_size}: {triplet}; "
                    f"best_rmse={format_value(stats['best_rmse_by_batch'][batch_size])}\n"
                )
            handle.write("\n")


def plot_variant_figure(
    variant_rows: list[dict[str, Any]],
    hparams: list[str],
    out_dir: Path,
    rng: np.random.Generator,
    bootstrap_samples: int,
    dpi: int,
) -> tuple[Path, dict[tuple[int, str], dict[str, Any]]]:
    batch_groups = grouped_rows(variant_rows, lambda row: int(row["batch_size"]))
    batch_sizes = sorted(batch_groups)
    example_row = variant_rows[0]

    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for batch_size in batch_sizes:
        for hparam in hparams:
            summaries[(batch_size, hparam)] = summarize_hparam(
                batch_groups[batch_size],
                hparam,
                rng,
                bootstrap_samples,
            )

    ncols = len(batch_sizes) + 1
    nrows = len(hparams)
    fig = plt.figure(figsize=(4.6 * ncols, 3.2 * nrows), constrained_layout=True)
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.45, wspace=0.28)

    for row_idx, hparam in enumerate(hparams):
        for col_idx, batch_size in enumerate(batch_sizes):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            title_prefix = f"{hparam}\n" if col_idx == 0 else ""
            draw_rmse_hparam_subplot(
                ax,
                summaries[(batch_size, hparam)],
                hparam,
                batch_size,
                rng,
                title_prefix=title_prefix,
            )
            ax.set_ylabel(f"{hparam}\nclean train RMSE" if col_idx == 0 else "clean train RMSE")
            ax.set_xlabel(hparam)
        ax = fig.add_subplot(grid[row_idx, len(batch_sizes)])
        draw_ci_vs_batch_subplot(
            ax,
            {batch_size: summaries[(batch_size, hparam)] for batch_size in batch_sizes},
            hparam,
        )

    fig.suptitle(
        f"SGDH variant: {variant_label(example_row)}",
        fontsize=14,
        y=0.995,
    )
    out_path = out_dir / f"variant_{safe_filename(short_variant_label(example_row))}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, summaries


def main() -> None:
    args = parse_args()
    rows = read_run_summaries(args.log)
    hparams = searched_hparams(rows)
    if not hparams:
        raise ValueError("no varying searched hyperparameters found in the log")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    variant_groups = grouped_rows(rows, variant_label)
    ordered_variants = sorted(
        variant_groups,
        key=lambda label: variant_sort_key(variant_groups[label][0]),
    )

    outputs: list[Path] = []
    variant_summaries: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for variant in ordered_variants:
        output_path, summaries = plot_variant_figure(
            variant_groups[variant],
            hparams,
            args.out_dir,
            rng,
            args.bootstrap_samples,
            args.dpi,
        )
        outputs.append(output_path)
        variant_summaries[variant] = summaries

    summary_path = args.out_dir / SUMMARY_FILENAME
    write_summary(
        summary_path,
        args.log,
        rows,
        hparams,
        ordered_variants,
        variant_groups,
        variant_summaries,
    )

    print(f"read {len(rows)} rows from {args.log}")
    print(f"searched hparams: {', '.join(hparams)}")
    print(f"wrote {len(outputs)} figures to {args.out_dir}")
    print(summary_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

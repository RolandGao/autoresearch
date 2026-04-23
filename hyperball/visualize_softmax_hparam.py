#!/usr/bin/env python3
"""Plot softmax hparam search results from train_learnable_softmax.py logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("softmax2.log")
DEFAULT_OUT_DIR = Path(__file__).with_name("softmax_hparam_plots")
RMSE_KEY = "clean_train_rmse"
BOOTSTRAP_FRACTION = 0.25
GROUP_KEYS = ("variant", "batch_size")
NON_HPARAM_KEYS = {
    "actual_samples",
    "batch_size",
    "candidate_idx",
    "clean_train_rmse",
    "compiled",
    "elapsed_sec",
    "final_train_loss",
    "num_candidates",
    "peak_allocated_bytes",
    "peak_allocated_mib",
    "peak_reserved_bytes",
    "peak_reserved_mib",
    "round",
    "sample_idx",
    "continuous_sample_idx",
    "steps",
    "num_samples",
    "target_met",
    "training_elapsed_sec",
    "variant",
    "weight_row_norm_mean",
    "weight_row_norm_std",
}
PREFERRED_HPARAM_ORDER = (
    "lr",
    "lr_decay",
    "lr_power",
    "sample_mode",
    "beta1",
    "beta2",
    "eps",
    "momentum",
    "nesterov",
    "weight_decay",
    "lr_schedule",
)
LOG_X_HPARAMS = {"lr", "eps", "weight_decay"}
SUMMARY_FILENAME = "bootstrap_hparam_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one RMSE-vs-hparam scatter figure per optimizer and hparam."
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
        help="Bootstrap resamples used for the empirical 95%% best-hparam interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )
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
            if RMSE_KEY in row and all(key in row for key in GROUP_KEYS):
                rows.append(row)
    if not rows:
        raise ValueError(f"no RUN_SUMMARY rows with {RMSE_KEY!r} found in {log_path}")
    return rows


def hparam_names(rows: list[dict[str, Any]]) -> list[str]:
    names = {
        key
        for row in rows
        for key, value in row.items()
        if key not in NON_HPARAM_KEYS and is_plot_value(value)
    }
    varying = [name for name in names if len({normalize_value(row.get(name)) for row in rows}) > 1]
    preferred = [name for name in PREFERRED_HPARAM_ORDER if name in varying]
    rest = sorted(name for name in varying if name not in preferred)
    return preferred + rest


def is_plot_value(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) and value is not None


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def grouped_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), int(row["batch_size"]))].append(row)
    return groups


def variant_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["variant"])].append(row)
    return groups


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row[RMSE_KEY]))


def bootstrap_best_values(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
    if num_samples <= 0:
        return np.array([], dtype=object)
    rmses = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    values = np.array([normalize_value(row.get(hparam)) for row in rows], dtype=object)
    sample_size = max(1, int(round(BOOTSTRAP_FRACTION * len(rows))))
    sampled_idx = rng.integers(0, len(rows), size=(num_samples, sample_size))
    sampled_rmses = rmses[sampled_idx]
    best_positions = np.argmin(sampled_rmses, axis=1)
    best_idx = sampled_idx[np.arange(num_samples), best_positions]
    return values[best_idx]


def is_numeric_values(values: list[Any]) -> bool:
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


def encode_categorical(values: list[Any]) -> tuple[np.ndarray, list[Any]]:
    categories = sorted({value for value in values}, key=category_sort_key)
    positions = {value: idx for idx, value in enumerate(categories)}
    encoded = np.array([positions[value] for value in values], dtype=float)
    return encoded, categories


def category_sort_key(value: Any) -> tuple[str, float | str]:
    if isinstance(value, bool):
        return ("bool", float(value))
    if isinstance(value, (int, float)):
        return ("number", float(value))
    return ("string", str(value))


def numeric_bootstrap_summary(values: np.ndarray) -> tuple[float, float, float] | None:
    if len(values) == 0:
        return None
    numeric = np.array(values, dtype=float)
    lo, median, hi = np.percentile(numeric, [2.5, 50.0, 97.5])
    return float(lo), float(median), float(hi)


def discrete_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("cannot compute a discrete quantile of an empty array")
    sorted_values = np.sort(np.array(values, dtype=float))
    idx = int(math.ceil(quantile * len(sorted_values))) - 1
    idx = int(np.clip(idx, 0, len(sorted_values) - 1))
    return float(sorted_values[idx])


def finite_positive(values: list[float]) -> bool:
    return all(math.isfinite(value) and value > 0 for value in values)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
            return str(int(round(value)))
        return f"{value:.10g}"
    return str(value)


def bootstrap_numeric_summary(
    rows: list[dict[str, Any]],
    hparam: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float, float, float]:
    boot_values = bootstrap_best_values(rows, hparam, rng, bootstrap_samples)
    summary = numeric_bootstrap_summary(boot_values)
    if summary is not None:
        return summary
    fallback = float(best_row(rows)[hparam])
    return fallback, fallback, fallback


def bootstrap_categorical_summary(
    rows: list[dict[str, Any]],
    hparam: str,
    categories: list[Any],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float, float, float, Any]:
    positions = {value: idx for idx, value in enumerate(categories)}
    boot_values = bootstrap_best_values(rows, hparam, rng, bootstrap_samples)
    boot_y = np.array([positions[value] for value in boot_values if value in positions], dtype=float)
    if len(boot_y):
        lo = discrete_quantile(boot_y, 0.025)
        median = discrete_quantile(boot_y, 0.5)
        hi = discrete_quantile(boot_y, 0.975)
    else:
        median_value = normalize_value(best_row(rows)[hparam])
        median = float(positions[median_value])
        lo = hi = median
    median_idx = int(np.clip(round(float(median)), 0, len(categories) - 1))
    return float(lo), float(median), float(hi), categories[median_idx]


def draw_numeric_scatter(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    hparam: str,
    summary: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    x = np.array([float(row[hparam]) for row in rows], dtype=float)
    y = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    lo, best_x, hi = summary

    ax.scatter(x, y, s=18, alpha=0.62, linewidths=0, color="#31688e")
    ax.axvline(best_x, color="#b83232", linewidth=1.8, label="bootstrap median best")
    if lo <= hi:
        ax.axvspan(lo, hi, color="#b83232", alpha=0.14, label="95% bootstrap CI")
    if hparam in LOG_X_HPARAMS and finite_positive(list(x)):
        ax.set_xscale("log")
    ax.set_title(f"{hparam}: median best={best_x:.4g}", fontsize=10)
    ax.set_xlabel(hparam)
    ax.set_ylabel("clean train RMSE")
    ax.grid(True, which="both", alpha=0.22)


def draw_categorical_scatter(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    hparam: str,
    categories: list[Any],
    summary: tuple[float, float, float, Any],
    rng: np.random.Generator,
) -> None:
    raw_values = [normalize_value(row[hparam]) for row in rows]
    positions = {value: idx for idx, value in enumerate(categories)}
    x = np.array([positions[value] for value in raw_values], dtype=float)
    y = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    lo, best_x, hi, best_value = summary

    jitter = rng.uniform(-0.08, 0.08, size=len(x))
    ax.scatter(x + jitter, y, s=18, alpha=0.62, linewidths=0, color="#31688e")
    ax.axvline(best_x, color="#b83232", linewidth=1.8, label="bootstrap median best")
    ax.axvspan(lo - 0.18, hi + 0.18, color="#b83232", alpha=0.14, label="95% bootstrap CI")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(value) for value in categories], rotation=30, ha="right")
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_title(f"{hparam}: median best={best_value}", fontsize=10)
    ax.set_xlabel(hparam)
    ax.set_ylabel("clean train RMSE")
    ax.grid(True, axis="y", alpha=0.22)


def draw_numeric_batch_summary(
    ax: plt.Axes,
    summaries_by_batch: dict[int, tuple[float, float, float]],
    hparam: str,
) -> None:
    batch_sizes = np.array(sorted(summaries_by_batch), dtype=float)
    summaries = [summaries_by_batch[int(batch_size)] for batch_size in batch_sizes]
    los = np.array([item[0] for item in summaries], dtype=float)
    medians = np.array([item[1] for item in summaries], dtype=float)
    his = np.array([item[2] for item in summaries], dtype=float)
    yerr = np.vstack([medians - los, his - medians])

    ax.errorbar(
        batch_sizes,
        medians,
        yerr=yerr,
        fmt="o-",
        color="#b83232",
        ecolor="#b83232",
        elinewidth=1.5,
        capsize=4,
    )
    ax.set_xscale("log", base=2)
    if hparam in LOG_X_HPARAMS and finite_positive(list(medians) + list(los) + list(his)):
        ax.set_yscale("log")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(int(batch_size)) for batch_size in batch_sizes])
    ax.set_xlabel("batch size")
    ax.set_ylabel(hparam)
    ax.set_title("median best hparam by batch size", fontsize=10)
    ax.grid(True, which="both", alpha=0.22)


def draw_categorical_batch_summary(
    ax: plt.Axes,
    summaries_by_batch: dict[int, tuple[float, float, float, Any]],
    hparam: str,
    categories: list[Any],
) -> None:
    batch_sizes = np.array(sorted(summaries_by_batch), dtype=float)
    summaries = [summaries_by_batch[int(batch_size)] for batch_size in batch_sizes]
    los = np.array([item[0] for item in summaries], dtype=float)
    medians = np.array([item[1] for item in summaries], dtype=float)
    his = np.array([item[2] for item in summaries], dtype=float)
    yerr = np.vstack([medians - los, his - medians])

    ax.errorbar(
        batch_sizes,
        medians,
        yerr=yerr,
        fmt="o-",
        color="#b83232",
        ecolor="#b83232",
        elinewidth=1.5,
        capsize=4,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(int(batch_size)) for batch_size in batch_sizes])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([str(value) for value in categories])
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.set_xlabel("batch size")
    ax.set_ylabel(hparam)
    ax.set_title("median best hparam by batch size", fontsize=10)
    ax.grid(True, axis="both", alpha=0.22)


def plot_variant_hparam(
    rows: list[dict[str, Any]],
    variant: str,
    hparam: str,
    out_dir: Path,
    rng: np.random.Generator,
    bootstrap_samples: int,
    dpi: int,
) -> tuple[Path, list[dict[str, Any]]] | None:
    rows_by_batch = {
        batch_size: group
        for (_, batch_size), group in grouped_rows(rows).items()
        if all(hparam in row for row in group)
    }
    if not rows_by_batch:
        return None
    batch_sizes = sorted(rows_by_batch)
    all_values = [normalize_value(row[hparam]) for group in rows_by_batch.values() for row in group]
    is_numeric = is_numeric_values(all_values)
    categories = None if is_numeric else encode_categorical(all_values)[1]
    numeric_summaries: dict[int, tuple[float, float, float]] = {}
    categorical_summaries: dict[int, tuple[float, float, float, Any]] = {}
    if is_numeric:
        numeric_summaries = {
            batch_size: bootstrap_numeric_summary(
                batch_rows, hparam, rng, bootstrap_samples
            )
            for batch_size, batch_rows in rows_by_batch.items()
        }
    else:
        assert categories is not None
        categorical_summaries = {
            batch_size: bootstrap_categorical_summary(
                batch_rows, hparam, categories, rng, bootstrap_samples
            )
            for batch_size, batch_rows in rows_by_batch.items()
        }

    summary_rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch_rows = rows_by_batch[batch_size]
        observed_best = best_row(batch_rows)
        if is_numeric:
            lo, median, hi = numeric_summaries[batch_size]
            median_value = median
            ci_low_value = lo
            ci_high_value = hi
            categories_text = ""
        else:
            assert categories is not None
            lo, median, hi, median_value = categorical_summaries[batch_size]
            ci_low_value = categories[int(lo)]
            ci_high_value = categories[int(hi)]
            categories_text = ", ".join(str(value) for value in categories)
        summary_rows.append(
            {
                "variant": variant,
                "hparam": hparam,
                "batch_size": batch_size,
                "n_pairs": len(batch_rows),
                "bootstrap_fraction": BOOTSTRAP_FRACTION,
                "bootstrap_samples": bootstrap_samples,
                "hparam_type": "numeric" if is_numeric else "categorical",
                "bootstrap_ci_low": ci_low_value,
                "bootstrap_median_best": median_value,
                "bootstrap_ci_high": ci_high_value,
                "bootstrap_ci_low_code": lo,
                "bootstrap_median_best_code": median,
                "bootstrap_ci_high_code": hi,
                "observed_best_hparam": normalize_value(observed_best[hparam]),
                "observed_best_rmse": float(observed_best[RMSE_KEY]),
                "categories": categories_text,
            }
        )

    panel_count = len(batch_sizes) + 1
    ncols = min(3, panel_count)
    nrows = math.ceil(panel_count / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.1 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    axes_flat = list(axes.ravel())

    for ax, batch_size in zip(axes_flat, batch_sizes):
        batch_rows = rows_by_batch[batch_size]
        if is_numeric:
            draw_numeric_scatter(
                ax, batch_rows, hparam, numeric_summaries[batch_size], rng
            )
        else:
            assert categories is not None
            draw_categorical_scatter(
                ax,
                batch_rows,
                hparam,
                categories,
                categorical_summaries[batch_size],
                rng,
            )
        ax.set_yscale("log")
        ax.set_title(f"batch size {batch_size}", fontsize=10)

    summary_ax = axes_flat[len(batch_sizes)]
    if is_numeric:
        draw_numeric_batch_summary(summary_ax, numeric_summaries, hparam)
    else:
        assert categories is not None
        draw_categorical_batch_summary(
            summary_ax, categorical_summaries, hparam, categories
        )

    for ax in axes_flat[panel_count:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(
        f"{variant}: {hparam} search across batch sizes (n={len(rows)})",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"{safe_filename(variant)}_{safe_filename(hparam)}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, summary_rows


def write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Bootstrap hyperparameter summary\n")
        handle.write("===============================\n\n")
        handle.write(
            "Each batch-size entry uses the same bootstrap result shown in the "
            "matching plot panel.\n"
        )
        handle.write(
            f"Bootstrap algorithm: sample {BOOTSTRAP_FRACTION:.0%} of pairs with "
            "replacement, pick the sampled pair with minimum RMSE, repeat N times.\n\n"
        )

        variants = sorted({str(row["variant"]) for row in rows})
        for variant in variants:
            handle.write(f"Optimizer: {variant}\n")
            handle.write("-" * (len("Optimizer: ") + len(variant)) + "\n\n")
            variant_rows_ = [row for row in rows if row["variant"] == variant]
            hparams = sorted(
                {str(row["hparam"]) for row in variant_rows_},
                key=lambda name: (
                    PREFERRED_HPARAM_ORDER.index(name)
                    if name in PREFERRED_HPARAM_ORDER
                    else len(PREFERRED_HPARAM_ORDER)
                ),
            )
            for hparam in hparams:
                hparam_rows = [
                    row for row in variant_rows_ if row["hparam"] == hparam
                ]
                hparam_rows.sort(key=lambda row: int(row["batch_size"]))
                first = hparam_rows[0]
                handle.write(f"Hparam: {hparam}\n")
                handle.write(f"Type: {first['hparam_type']}\n")
                handle.write(
                    f"Bootstrap repeats: {format_value(first['bootstrap_samples'])}\n"
                )
                if first["categories"]:
                    handle.write(f"Categories: {first['categories']}\n")
                    handle.write("Category codes follow the category order above.\n")
                handle.write("\n")
                for row in hparam_rows:
                    handle.write(f"  Batch size {row['batch_size']} (n={row['n_pairs']})\n")
                    handle.write(
                        "    Bootstrap median best: "
                        f"{format_value(row['bootstrap_median_best'])}\n"
                    )
                    handle.write(
                        "    95% bootstrap CI: "
                        f"[{format_value(row['bootstrap_ci_low'])}, "
                        f"{format_value(row['bootstrap_ci_high'])}]\n"
                    )
                    if row["hparam_type"] == "categorical":
                        handle.write(
                            "    Encoded median/CI: "
                            f"{format_value(row['bootstrap_median_best_code'])} "
                            f"[{format_value(row['bootstrap_ci_low_code'])}, "
                            f"{format_value(row['bootstrap_ci_high_code'])}]\n"
                        )
                    handle.write(
                        "    Observed best run: "
                        f"{format_value(row['observed_best_hparam'])} "
                        f"(RMSE={format_value(row['observed_best_rmse'])})\n"
                    )
                handle.write("\n")


def main() -> None:
    args = parse_args()
    rows = read_run_summaries(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    outputs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for variant, group in sorted(variant_rows(rows).items()):
        for hparam in hparam_names(group):
            result = plot_variant_hparam(
                group,
                variant,
                hparam,
                args.out_dir,
                rng,
                args.bootstrap_samples,
                args.dpi,
            )
            if result is not None:
                output, rows_for_plot = result
                outputs.append(output)
                summary_rows.extend(rows_for_plot)

    summary_path = args.out_dir / SUMMARY_FILENAME
    write_summary(summary_path, summary_rows)

    print(f"wrote {len(outputs)} figures to {args.out_dir}")
    print(summary_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

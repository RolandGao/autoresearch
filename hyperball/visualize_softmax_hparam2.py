#!/usr/bin/env python3
"""Visualize SGDH ablation results from train_learnable_softmax2.py logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("sgdh_ablation.log")
DEFAULT_OUT_DIR = Path(__file__).with_name("sgd_ablation")
RMSE_KEY = "clean_train_rmse"
BOOTSTRAP_FRACTION = 0.25
OPTIMIZER_FIELDS = ("g_projection", "g_norm", "nesterov", "m_projection")
SEARCH_HPARAM_KEYS = ("lr", "lr_decay", "momentum")
LOG_X_HPARAMS = {"lr"}
SUMMARY_FILENAME = "categorical_bootstrap_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RMSE plots and categorical bootstrap summaries."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Input log file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where plots and text summaries are written.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples used for categorical best-category summaries.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser.parse_args()


def read_run_summaries(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.startswith("RUN_SUMMARY "):
                continue
            try:
                row = json.loads(line.split(" ", 1)[1])
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


def finite_positive(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values > 0.0))


def plot_rmse_vs_hparam(
    rows: list[dict[str, Any]],
    batch_size: int,
    variant: str,
    hparam: str,
    out_dir: Path,
    rng: np.random.Generator,
    dpi: int,
) -> Path:
    values = [normalize_value(row[hparam]) for row in rows]
    rmses = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
    observed_best = best_row(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        x = np.array(values, dtype=float)
        ax.scatter(x, rmses, s=28, alpha=0.68, linewidths=0, color="#286983")
        ax.scatter(
            [float(observed_best[hparam])],
            [float(observed_best[RMSE_KEY])],
            s=88,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            label="best run",
            zorder=4,
        )
        if hparam in LOG_X_HPARAMS and finite_positive(x):
            ax.set_xscale("log")
        ax.set_xlabel(hparam)
    else:
        categories = sorted(set(values), key=category_sort_key)
        positions = {value: idx for idx, value in enumerate(categories)}
        x = np.array([positions[value] for value in values], dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(x))
        ax.scatter(x + jitter, rmses, s=28, alpha=0.68, linewidths=0, color="#286983")
        ax.scatter(
            [positions[normalize_value(observed_best[hparam])]],
            [float(observed_best[RMSE_KEY])],
            s=88,
            color="#b83232",
            edgecolors="white",
            linewidths=0.8,
            label="best run",
            zorder=4,
        )
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([format_value(value) for value in categories])
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.set_xlabel(hparam)

    ax.set_yscale("log")
    ax.set_ylabel("clean train RMSE")
    ax.set_title(
        f"batch_size={batch_size}, {variant}\n"
        f"best {hparam}={format_value(normalize_value(observed_best[hparam]))}, "
        f"RMSE={format_value(float(observed_best[RMSE_KEY]))}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = out_dir / (
        f"batch_{batch_size}_{safe_filename(variant)}_{safe_filename(hparam)}.png"
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def categorical_bootstrap(
    rows: list[dict[str, Any]],
    category_fn: Callable[[dict[str, Any]], Any],
    rng: np.random.Generator,
    num_samples: int,
) -> dict[str, Any]:
    categories = sorted({category_fn(row) for row in rows}, key=category_sort_key)
    if num_samples <= 0:
        counts = Counter({category: 0 for category in categories})
    else:
        rmses = np.array([float(row[RMSE_KEY]) for row in rows], dtype=float)
        row_categories = np.array([category_fn(row) for row in rows], dtype=object)
        sample_size = max(1, int(round(BOOTSTRAP_FRACTION * len(rows))))
        sampled_idx = rng.integers(0, len(rows), size=(num_samples, sample_size))
        sampled_rmses = rmses[sampled_idx]
        best_positions = np.argmin(sampled_rmses, axis=1)
        best_idx = sampled_idx[np.arange(num_samples), best_positions]
        counts = Counter(row_categories[best_idx])
        for category in categories:
            counts.setdefault(category, 0)

    ranked = sorted(
        categories,
        key=lambda category: (-counts[category], category_sort_key(category)),
    )
    cumulative = 0
    ci_set: list[Any] = []
    threshold = 0.95 * max(1, num_samples)
    for category in ranked:
        if counts[category] <= 0 and ci_set:
            break
        ci_set.append(category)
        cumulative += counts[category]
        if cumulative >= threshold:
            break

    observed_groups = grouped_rows(rows, category_fn)
    observed = {
        category: {
            "n": len(group),
            "best_rmse": float(best_row(group)[RMSE_KEY]),
            "median_rmse": float(np.median([float(row[RMSE_KEY]) for row in group])),
            "target_rate": float(np.mean([bool(row.get("target_met", False)) for row in group])),
        }
        for category, group in observed_groups.items()
    }
    return {
        "categories": categories,
        "counts": counts,
        "ranked": ranked,
        "ci_set": ci_set,
        "observed": observed,
        "num_samples": num_samples,
    }


def bootstrap_probability(result: dict[str, Any], category: Any) -> float:
    num_samples = int(result["num_samples"])
    if num_samples <= 0:
        return 0.0
    return float(result["counts"][category]) / float(num_samples)


def plot_categorical_probabilities(
    result: dict[str, Any],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
    horizontal: bool = True,
) -> Path:
    ranked = result["ranked"]
    labels = [format_value(category) for category in ranked]
    probs = np.array([bootstrap_probability(result, category) for category in ranked])
    ci_set = set(result["ci_set"])
    colors = ["#b83232" if category in ci_set else "#286983" for category in ranked]

    if horizontal:
        fig_height = max(4.8, 0.42 * len(labels) + 1.6)
        fig, ax = plt.subplots(figsize=(10.5, fig_height))
        y = np.arange(len(labels))
        ax.barh(y, probs, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("bootstrap probability of being best")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="x", alpha=0.22)
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(labels))
        ax.bar(x, probs, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("bootstrap probability of being best")
        ax.grid(True, axis="y", alpha=0.22)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_field_probabilities(
    field_results: dict[str, dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), squeeze=False)
    for ax, field in zip(axes.ravel(), OPTIMIZER_FIELDS):
        result = field_results[field]
        categories = [False, True]
        probs = [bootstrap_probability(result, category) for category in categories]
        colors = [
            "#b83232" if category in set(result["ci_set"]) else "#286983"
            for category in categories
        ]
        ax.bar([0, 1], probs, color=colors, alpha=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["False", "True"])
        ax.set_ylim(0.0, max(1.0, max(probs) * 1.15))
        ax.set_title(field)
        ax.set_ylabel("bootstrap probability")
        ax.grid(True, axis="y", alpha=0.22)
    fig.suptitle("Optimizer field bootstrap probability of being best", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_summary(
    summary_path: Path,
    rows: list[dict[str, Any]],
    variant_result: dict[str, Any],
    field_results: dict[str, dict[str, Any]],
    batch_variant_results: dict[int, dict[str, Any]],
) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("SGDH categorical bootstrap summary\n")
        handle.write("==================================\n\n")
        handle.write(f"Input rows: {len(rows)}\n")
        handle.write(f"RMSE key: {RMSE_KEY}\n")
        handle.write(
            f"Bootstrap algorithm: sample {BOOTSTRAP_FRACTION:.0%} of rows with "
            "replacement, pick the sampled row with minimum RMSE, record its category.\n"
        )
        handle.write(
            "Categorical 95% CI: smallest probability-ranked category set whose "
            "cumulative bootstrap mass reaches 95%.\n\n"
        )

        handle.write("16 optimizer variants ranked independently\n")
        handle.write("------------------------------------------\n")
        handle.write(
            "rank | bootstrap_prob | observed_best_rmse | median_rmse | "
            "target_rate | n | variant\n"
        )
        for rank, category in enumerate(variant_result["ranked"], start=1):
            obs = variant_result["observed"][category]
            handle.write(
                f"{rank:>4} | "
                f"{bootstrap_probability(variant_result, category):>14.4f} | "
                f"{format_value(obs['best_rmse']):>18} | "
                f"{format_value(obs['median_rmse']):>11} | "
                f"{obs['target_rate']:>11.3f} | "
                f"{obs['n']:>3} | {category}\n"
            )
        handle.write(
            "\n95% categorical CI set for best variant:\n  "
            + "\n  ".join(format_value(category) for category in variant_result["ci_set"])
            + "\n\n"
        )

        handle.write("16 optimizer variants ranked by batch size\n")
        handle.write("------------------------------------------\n")
        for batch_size in sorted(batch_variant_results):
            result = batch_variant_results[batch_size]
            handle.write(f"\nBatch size {batch_size}\n")
            handle.write(
                "rank | bootstrap_prob | observed_best_rmse | target_rate | n | variant\n"
            )
            for rank, category in enumerate(result["ranked"], start=1):
                obs = result["observed"][category]
                handle.write(
                    f"{rank:>4} | "
                    f"{bootstrap_probability(result, category):>14.4f} | "
                    f"{format_value(obs['best_rmse']):>18} | "
                    f"{obs['target_rate']:>11.3f} | "
                    f"{obs['n']:>3} | {category}\n"
                )
            handle.write(
                "95% categorical CI set:\n  "
                + "\n  ".join(format_value(category) for category in result["ci_set"])
                + "\n"
            )

        handle.write("\nOptimizer fields treated independently\n")
        handle.write("--------------------------------------\n")
        for field in OPTIMIZER_FIELDS:
            result = field_results[field]
            handle.write(f"\nField: {field}\n")
            handle.write("rank | value | bootstrap_prob | observed_best_rmse | target_rate | n\n")
            for rank, category in enumerate(result["ranked"], start=1):
                obs = result["observed"][category]
                handle.write(
                    f"{rank:>4} | {format_value(category):>5} | "
                    f"{bootstrap_probability(result, category):>14.4f} | "
                    f"{format_value(obs['best_rmse']):>18} | "
                    f"{obs['target_rate']:>11.3f} | "
                    f"{obs['n']:>4}\n"
                )
            handle.write(
                "95% categorical CI set: "
                + ", ".join(format_value(category) for category in result["ci_set"])
                + "\n"
            )


def main() -> None:
    args = parse_args()
    rows = read_run_summaries(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    outputs: list[Path] = []
    hparams = searched_hparams(rows)
    group_key = lambda row: (int(row["batch_size"]), variant_label(row))
    groups = grouped_rows(rows, group_key)
    variant_name_by_group = {
        key: short_variant_label(group[0]) for key, group in groups.items()
    }

    for (batch_size, variant), group in sorted(groups.items()):
        for hparam in hparams:
            outputs.append(
                plot_rmse_vs_hparam(
                    group,
                    batch_size,
                    variant_name_by_group[(batch_size, variant)],
                    hparam,
                    args.out_dir,
                    rng,
                    args.dpi,
                )
            )

    variant_result = categorical_bootstrap(
        rows,
        lambda row: variant_label(row),
        rng,
        args.bootstrap_samples,
    )
    outputs.append(
        plot_categorical_probabilities(
            variant_result,
            "16 optimizer variants: bootstrap probability of best RMSE",
            "optimizer variant",
            args.out_dir / "variant_bootstrap_probabilities.png",
            args.dpi,
            horizontal=True,
        )
    )

    field_results = {
        field: categorical_bootstrap(
            rows,
            lambda row, field=field: bool(row[field]),
            rng,
            args.bootstrap_samples,
        )
        for field in OPTIMIZER_FIELDS
    }
    outputs.append(
        plot_field_probabilities(
            field_results,
            args.out_dir / "optimizer_field_bootstrap_probabilities.png",
            args.dpi,
        )
    )

    batch_variant_results: dict[int, dict[str, Any]] = {}
    for batch_size, batch_rows in sorted(
        grouped_rows(rows, lambda row: int(row["batch_size"])).items()
    ):
        result = categorical_bootstrap(
            batch_rows,
            lambda row: variant_label(row),
            rng,
            args.bootstrap_samples,
        )
        batch_variant_results[int(batch_size)] = result
        outputs.append(
            plot_categorical_probabilities(
                result,
                f"batch_size={batch_size}: variant bootstrap probability of best RMSE",
                "optimizer variant",
                args.out_dir / f"batch_{batch_size}_variant_bootstrap_probabilities.png",
                args.dpi,
                horizontal=True,
            )
        )

    summary_path = args.out_dir / SUMMARY_FILENAME
    write_summary(
        summary_path,
        rows,
        variant_result,
        field_results,
        batch_variant_results,
    )

    print(f"read {len(rows)} rows from {args.log}")
    print(f"searched hparams: {', '.join(hparams)}")
    print(f"wrote {len(outputs)} figures to {args.out_dir}")
    print(summary_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

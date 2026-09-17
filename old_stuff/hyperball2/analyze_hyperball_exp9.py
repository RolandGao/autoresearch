#!/usr/bin/env python3
"""Analyze hyperball_exp9 shared-LR sweep results.

Default usage:
    python hyperball2/analyze_hyperball_exp9.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


DEFAULT_LOG_PATH = Path(__file__).with_name("hyperball_exp9.log")
METRIC = "step_29_smoothed_train_loss"


def parse_json_payload(line):
    line = line.strip()
    if not line:
        return None
    if line.startswith("STEP29_RESULT "):
        line = line[len("STEP29_RESULT ") :]
    elif not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("type") != "sweep_result":
        return None
    return payload


def load_results(path):
    prefixed = {}
    fallback = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = parse_json_payload(line)
            if payload is None:
                continue
            run_idx = payload.get("run_idx")
            if run_idx is None:
                continue
            if line.lstrip().startswith("STEP29_RESULT "):
                prefixed[run_idx] = payload
            else:
                fallback.setdefault(run_idx, payload)

    results_by_run = dict(fallback)
    results_by_run.update(prefixed)
    return [results_by_run[key] for key in sorted(results_by_run)]


def valid_results(results):
    valid = []
    failed = []
    for result in results:
        metric = result.get(METRIC)
        if result.get("failed") or metric is None or not math.isfinite(float(metric)):
            failed.append(result)
        else:
            valid.append(result)
    return valid, failed


def fmt(value, digits=5):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def print_rows(rows, headers):
    widths = [
        max(len(str(header)), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    ]
    print("  ".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(width) for header, width in zip(headers, widths)))


def group_stats(results, key):
    groups = defaultdict(list)
    for result in results:
        groups[result[key]].append(float(result[METRIC]))
    rows = []
    for group_value in sorted(groups):
        values = groups[group_value]
        rows.append(
            {
                key: fmt(group_value),
                "count": len(values),
                "mean": fmt(mean(values)),
                "median": fmt(median(values)),
                "best": fmt(min(values)),
                "worst": fmt(max(values)),
                "std": fmt(pstdev(values) if len(values) > 1 else 0.0),
            }
        )
    return rows


def build_grid(results):
    directions = sorted({result["direction_start_lr"] for result in results})
    scales = sorted({result["scale_start_lr"] for result in results})
    grid = {
        (result["direction_start_lr"], result["scale_start_lr"]): float(result[METRIC])
        for result in results
    }
    return directions, scales, grid


def print_loss_grid(results):
    directions, scales, grid = build_grid(results)
    best_value = min(grid.values())
    headers = ["dir\\scale", *(fmt(scale) for scale in scales)]
    rows = []
    for direction in directions:
        row = {"dir\\scale": fmt(direction)}
        for scale in scales:
            value = grid.get((direction, scale))
            cell = fmt(value, digits=4)
            if value == best_value:
                cell = f"{cell}*"
            row[fmt(scale)] = cell
        rows.append(row)
    print_rows(rows, headers)


def local_neighborhood(results, best, max_rows):
    best_direction = best["direction_start_lr"]
    best_scale = best["scale_start_lr"]
    ranked = sorted(
        results,
        key=lambda result: (
            abs(math.log(result["direction_start_lr"] / best_direction))
            + abs(math.log(result["scale_start_lr"] / best_scale)),
            float(result[METRIC]),
        ),
    )
    return ranked[:max_rows]


def write_csv(results, path):
    fieldnames = [
        "run_idx",
        "direction_start_lr",
        "scale_start_lr",
        METRIC,
        "train_loss",
        "mfu_percent",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "failed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.get("run_idx", -1)):
            writer.writerow({field: result.get(field) for field in fieldnames})


def analyze(path, top_k, csv_path):
    results = load_results(path)
    valid, failed = valid_results(results)
    if not results:
        raise SystemExit(f"No sweep results found in {path}")
    if not valid:
        raise SystemExit(f"No valid {METRIC} values found in {path}")

    metric_values = [float(result[METRIC]) for result in valid]
    ranked = sorted(valid, key=lambda result: float(result[METRIC]))
    best = ranked[0]
    worst = ranked[-1]

    print(f"Log: {path}")
    print(f"Parsed runs: {len(results)} ({len(valid)} valid, {len(failed)} failed)")
    print(f"Metric: {METRIC}")
    print(
        "Best: "
        f"run {best['run_idx']} | "
        f"direction_start_lr={fmt(best['direction_start_lr'])} | "
        f"scale_start_lr={fmt(best['scale_start_lr'])} | "
        f"{METRIC}={fmt(best[METRIC])}"
    )
    print(
        "Worst: "
        f"run {worst['run_idx']} | "
        f"direction_start_lr={fmt(worst['direction_start_lr'])} | "
        f"scale_start_lr={fmt(worst['scale_start_lr'])} | "
        f"{METRIC}={fmt(worst[METRIC])}"
    )
    print(
        "Spread: "
        f"mean={fmt(mean(metric_values))}, "
        f"median={fmt(median(metric_values))}, "
        f"std={fmt(pstdev(metric_values))}, "
        f"best-to-worst={fmt(float(worst[METRIC]) - float(best[METRIC]))}"
    )

    print(f"\nTop {top_k}")
    top_rows = [
        {
            "rank": i,
            "run": result["run_idx"],
            "direction_lr": fmt(result["direction_start_lr"]),
            "scale_lr": fmt(result["scale_start_lr"]),
            METRIC: fmt(result[METRIC]),
            "train_loss": fmt(result.get("train_loss")),
        }
        for i, result in enumerate(ranked[:top_k], start=1)
    ]
    print_rows(top_rows, ["rank", "run", "direction_lr", "scale_lr", METRIC, "train_loss"])

    print("\nMean By Direction LR")
    print_rows(
        group_stats(valid, "direction_start_lr"),
        ["direction_start_lr", "count", "mean", "median", "best", "worst", "std"],
    )

    print("\nMean By Scale LR")
    print_rows(
        group_stats(valid, "scale_start_lr"),
        ["scale_start_lr", "count", "mean", "median", "best", "worst", "std"],
    )

    print("\nLoss Grid")
    print_loss_grid(valid)
    print("* marks the best run.")

    print("\nBest Neighborhood")
    neighbor_rows = [
        {
            "run": result["run_idx"],
            "direction_lr": fmt(result["direction_start_lr"]),
            "scale_lr": fmt(result["scale_start_lr"]),
            METRIC: fmt(result[METRIC]),
            "delta": fmt(float(result[METRIC]) - float(best[METRIC])),
        }
        for result in local_neighborhood(valid, best, top_k)
    ]
    print_rows(neighbor_rows, ["run", "direction_lr", "scale_lr", METRIC, "delta"])

    if failed:
        print("\nFailed Runs")
        failed_rows = [
            {
                "run": result.get("run_idx"),
                "direction_lr": fmt(result.get("direction_start_lr")),
                "scale_lr": fmt(result.get("scale_start_lr")),
                "error": result.get("error", "failed"),
            }
            for result in failed
        ]
        print_rows(failed_rows, ["run", "direction_lr", "scale_lr", "error"])

    if csv_path is not None:
        write_csv(valid, csv_path)
        print(f"\nWrote CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to sweep log. Defaults to {DEFAULT_LOG_PATH}",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write deduplicated valid results as CSV.",
    )
    args = parser.parse_args()
    analyze(args.log_path, args.top_k, args.csv)


if __name__ == "__main__":
    main()

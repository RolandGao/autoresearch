#!/usr/bin/env python3
"""Analyze hyperball_exp10 shared-profile sweep results.

Default usage:
    python hyperball2/analyze_hyperball_exp10.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


DEFAULT_LOG_PATH = Path(__file__).with_name("hyperball_exp10.log")
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
    results = [results_by_run[key] for key in sorted(results_by_run)]
    for result in results:
        add_profile_columns(result)
    return results


def profile_to_dict(profile):
    return {int(step): float(multiplier) for step, multiplier in profile}


def add_profile_columns(result):
    direction_profile = profile_to_dict(result.get("direction_profile", []))
    scale_profile = profile_to_dict(result.get("scale_profile", []))
    result["d3"] = direction_profile.get(3)
    result["d10"] = direction_profile.get(10)
    result["d30"] = direction_profile.get(30)
    result["s3"] = scale_profile.get(3)
    result["s10"] = scale_profile.get(10)
    result["s30"] = scale_profile.get(30)


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
    if not rows:
        print("(none)")
        return
    widths = [
        max(len(str(header)), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    ]
    print("  ".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                str(row.get(header, "")).ljust(width)
                for header, width in zip(headers, widths)
            )
        )


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


def compact_profile(result, prefix):
    if prefix == "d":
        return f"d3={fmt(result['d3'])}, d10={fmt(result['d10'])}, d30={fmt(result['d30'])}"
    return f"s3={fmt(result['s3'])}, s10={fmt(result['s10'])}, s30={fmt(result['s30'])}"


def ranked_rows(results, top_k):
    ranked = sorted(results, key=lambda result: float(result[METRIC]))
    rows = []
    for rank, result in enumerate(ranked[:top_k], start=1):
        family = result.get("variant_family", "-")
        profile = (
            compact_profile(result, "d")
            if family == "direction_profile"
            else compact_profile(result, "s")
        )
        rows.append(
            {
                "rank": rank,
                "run": result["run_idx"],
                "family": family,
                "variant": result.get("variant_idx"),
                METRIC: fmt(result[METRIC]),
                "train_loss": fmt(result.get("train_loss")),
                "profile": profile,
            }
        )
    return rows


def print_family_summary(valid):
    rows = []
    for family in sorted({result.get("variant_family") for result in valid}):
        family_results = [result for result in valid if result.get("variant_family") == family]
        ranked = sorted(family_results, key=lambda result: float(result[METRIC]))
        values = [float(result[METRIC]) for result in family_results]
        best = ranked[0]
        prefix = "d" if family == "direction_profile" else "s"
        rows.append(
            {
                "family": family,
                "count": len(family_results),
                "best_run": best["run_idx"],
                "best": fmt(best[METRIC]),
                "mean": fmt(mean(values)),
                "median": fmt(median(values)),
                "std": fmt(pstdev(values) if len(values) > 1 else 0.0),
                "best_profile": compact_profile(best, prefix),
            }
        )
    print_rows(
        rows,
        ["family", "count", "best_run", "best", "mean", "median", "std", "best_profile"],
    )


def print_multiplier_stats(results, keys):
    for key in keys:
        print(f"\nBy {key}")
        print_rows(group_stats(results, key), [key, "count", "mean", "median", "best", "worst", "std"])


def print_grid(results, row_key, col_key, title):
    row_values = sorted({result[row_key] for result in results})
    col_values = sorted({result[col_key] for result in results})
    values = {
        (result[row_key], result[col_key]): float(result[METRIC])
        for result in results
    }
    best = min(values.values())
    rows = []
    headers = [f"{row_key}\\{col_key}", *(fmt(value) for value in col_values)]
    for row_value in row_values:
        row = {f"{row_key}\\{col_key}": fmt(row_value)}
        for col_value in col_values:
            value = values.get((row_value, col_value))
            cell = fmt(value, digits=4)
            if value == best:
                cell = f"{cell}*"
            row[fmt(col_value)] = cell
        rows.append(row)
    print(f"\n{title}")
    print_rows(rows, headers)


def print_profile_grids(results, family):
    family_results = [
        result for result in results if result.get("variant_family") == family
    ]
    if family == "direction_profile":
        for d30 in sorted({result["d30"] for result in family_results}):
            subset = [result for result in family_results if result["d30"] == d30]
            print_grid(subset, "d3", "d10", f"Direction Profile Grid at d30={fmt(d30)}")
    else:
        for s30 in sorted({result["s30"] for result in family_results}):
            subset = [result for result in family_results if result["s30"] == s30]
            print_grid(subset, "s3", "s10", f"Scale Profile Grid at s30={fmt(s30)}")
    print("* marks the best run within that grid.")


def write_csv(results, path):
    fieldnames = [
        "run_idx",
        "variant_family",
        "variant_idx",
        "direction_start_lr",
        "scale_start_lr",
        "d3",
        "d10",
        "d30",
        "s3",
        "s10",
        "s30",
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

    ranked = sorted(valid, key=lambda result: float(result[METRIC]))
    best = ranked[0]
    worst = ranked[-1]
    values = [float(result[METRIC]) for result in valid]

    print(f"Log: {path}")
    print(f"Parsed runs: {len(results)} ({len(valid)} valid, {len(failed)} failed)")
    print(f"Metric: {METRIC}")
    print(
        "Best: "
        f"run {best['run_idx']} | family={best.get('variant_family')} | "
        f"variant={best.get('variant_idx')} | {METRIC}={fmt(best[METRIC])}"
    )
    print(
        "Best profile: "
        f"{compact_profile(best, 'd')} | {compact_profile(best, 's')}"
    )
    print(
        "Worst: "
        f"run {worst['run_idx']} | family={worst.get('variant_family')} | "
        f"variant={worst.get('variant_idx')} | {METRIC}={fmt(worst[METRIC])}"
    )
    print(
        "Spread: "
        f"mean={fmt(mean(values))}, "
        f"median={fmt(median(values))}, "
        f"std={fmt(pstdev(values))}, "
        f"best-to-worst={fmt(float(worst[METRIC]) - float(best[METRIC]))}"
    )

    print(f"\nTop {top_k}")
    print_rows(
        ranked_rows(valid, top_k),
        ["rank", "run", "family", "variant", METRIC, "train_loss", "profile"],
    )

    print("\nFamily Summary")
    print_family_summary(valid)

    direction_results = [
        result for result in valid if result.get("variant_family") == "direction_profile"
    ]
    scale_results = [
        result for result in valid if result.get("variant_family") == "scale_profile"
    ]

    print("\nDirection Profile Multipliers")
    print_multiplier_stats(direction_results, ["d3", "d10", "d30"])

    print("\nScale Profile Multipliers")
    print_multiplier_stats(scale_results, ["s3", "s10", "s30"])

    print_profile_grids(valid, "direction_profile")
    print_profile_grids(valid, "scale_profile")

    if failed:
        print("\nFailed Runs")
        failed_rows = [
            {
                "run": result.get("run_idx"),
                "family": result.get("variant_family"),
                "variant": result.get("variant_idx"),
                "error": result.get("error", "failed"),
            }
            for result in failed
        ]
        print_rows(failed_rows, ["run", "family", "variant", "error"])

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
    parser.add_argument("--top-k", type=int, default=12)
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

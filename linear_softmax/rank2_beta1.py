#!/usr/bin/env python3
"""Rank beta1 choices from scalar_optimizers_logging7.log.

The log is newline-delimited text where RUN_SUMMARY records contain JSON.  This
script streams those summaries, groups them by (batch_size, num_samples), and
compares beta1 values by the best clean_sse each beta1 achieved in that group.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LOG = Path(__file__).with_name("scalar_optimizers_logging7.log")
SUMMARY_PREFIX = "RUN_SUMMARY "
METRIC = "clean_sse"

DISPLAY_HPARAMS = (
    "lr",
    "disable_bias1",
    "adaptive_norm",
    "lr_in_momentum",
    "flush_last",
    "beta2",
    "eps",
    "lr_decay",
    "lr_power",
    "optimizer",
    "variant",
    "h_norm",
    "wd",
)
FLAG_FIELDS = ("disable_bias1", "adaptive_norm", "lr_in_momentum", "flush_last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract RUN_SUMMARY records and rank beta1 values for each "
            "(batch_size, num_samples) pair."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"path to log file (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--top-beta1",
        type=int,
        default=7,
        help="number of beta1 rows to show per group (default: all 7 values)",
    )
    parser.add_argument(
        "--final-top",
        type=int,
        default=2,
        help="number of rows to show per group in the final compact table (default: 2)",
    )
    parser.add_argument(
        "--metric",
        default=METRIC,
        help=f"metric to minimize from RUN_SUMMARY records (default: {METRIC})",
    )
    parser.add_argument(
        "--flag-patterns",
        default=None,
        help=(
            "comma-separated flag patterns to keep, using T/F in the order "
            "disable_bias1,adaptive_norm,lr_in_momentum,flush_last; "
            "for example: TFFT,FFFT"
        ),
    )
    return parser.parse_args()


def iter_run_summaries(log_path: Path) -> Any:
    with log_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.startswith(SUMMARY_PREFIX):
                continue
            payload = line[len(SUMMARY_PREFIX) :]
            try:
                yield line_no, json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON at {log_path}:{line_no}: {exc}") from exc


def sort_key(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def parse_flag_patterns(patterns: str | None) -> set[tuple[bool, ...]] | None:
    if not patterns:
        return None

    parsed = set()
    for raw_pattern in patterns.split(","):
        pattern = raw_pattern.strip().upper()
        if len(pattern) != len(FLAG_FIELDS) or any(char not in "TF" for char in pattern):
            fields = ",".join(FLAG_FIELDS)
            raise SystemExit(
                f"Invalid flag pattern {raw_pattern!r}; expected {len(FLAG_FIELDS)} "
                f"T/F chars in order {fields}"
            )
        parsed.add(tuple(char == "T" for char in pattern))
    return parsed


def row_pattern(row: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(bool(row[field]) for field in FLAG_FIELDS)


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None, metric: str) -> bool:
    if incumbent is None:
        return True

    candidate_metric = candidate.get(metric)
    incumbent_metric = incumbent.get(metric)
    if not isinstance(candidate_metric, (int, float)) or not math.isfinite(candidate_metric):
        return False
    if not isinstance(incumbent_metric, (int, float)) or not math.isfinite(incumbent_metric):
        return True

    if candidate_metric != incumbent_metric:
        return candidate_metric < incumbent_metric

    candidate_target = bool(candidate.get("target_met"))
    incumbent_target = bool(incumbent.get("target_met"))
    if candidate_target != incumbent_target:
        return candidate_target

    return int(candidate.get("candidate_idx", 10**12)) < int(
        incumbent.get("candidate_idx", 10**12)
    )


def summarize(
    log_path: Path, metric: str, flag_patterns: set[tuple[bool, ...]] | None = None
) -> dict[tuple[int, int], dict[str, Any]]:
    groups: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "best": None, "best_by_beta1": {}}
    )

    for _, row in iter_run_summaries(log_path):
        if metric not in row:
            continue
        if flag_patterns is not None and row_pattern(row) not in flag_patterns:
            continue

        batch_size = int(row["batch_size"])
        num_samples = int(row["num_samples"])
        beta1 = float(row["beta1"])
        group = groups[(batch_size, num_samples)]
        group["count"] += 1

        if is_better(row, group["best"], metric):
            group["best"] = row

        by_beta1 = group["best_by_beta1"]
        if is_better(row, by_beta1.get(beta1), metric):
            by_beta1[beta1] = row

    return groups


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def hparams(row: dict[str, Any]) -> str:
    return ", ".join(f"{name}={format_value(row[name])}" for name in DISPLAY_HPARAMS if name in row)


def compact_winner(row: dict[str, Any], metric: str) -> str:
    return (
        f"beta1={format_value(row['beta1'])}, {metric}={row[metric]:.6g}, "
        f"target_met={format_value(row.get('target_met', False))}, {hparams(row)}"
    )


def pattern_name(pattern: tuple[bool, ...]) -> str:
    return "".join("T" if value else "F" for value in pattern)


def print_results(
    groups: dict[tuple[int, int], dict[str, Any]],
    metric: str,
    top_beta1: int,
    final_top: int,
    flag_patterns: set[tuple[bool, ...]] | None = None,
) -> None:
    total_runs = sum(group["count"] for group in groups.values())
    print(f"Parsed {total_runs} RUN_SUMMARY rows across {len(groups)} groups.")
    print(f"Ranking beta1 by the lowest {metric} achieved in each group.\n")
    if flag_patterns is not None:
        patterns = ", ".join(pattern_name(pattern) for pattern in sorted(flag_patterns))
        fields = ",".join(FLAG_FIELDS)
        print(f"Constrained to flag patterns ({fields}): {patterns}\n")

    beta1_wins: dict[float, int] = defaultdict(int)
    final_rows: list[tuple[int, int, int, dict[str, Any]]] = []

    for (batch_size, num_samples) in sorted(groups):
        group = groups[(batch_size, num_samples)]
        best = group["best"]
        if best is None:
            continue
        beta1_wins[float(best["beta1"])] += 1

        print(f"batch_size={batch_size}, num_samples={num_samples}, runs={group['count']}")
        print(f"  best: {compact_winner(best, metric)}")

        ranked = sorted(
            group["best_by_beta1"].items(),
            key=lambda item: (
                item[1][metric],
                not bool(item[1].get("target_met")),
                sort_key(item[0]),
            ),
        )
        for rank, (_, row) in enumerate(ranked[:final_top], start=1):
            final_rows.append((batch_size, num_samples, rank, row))

        print("  beta1 ranking:")
        for rank, (beta1, row) in enumerate(ranked[:top_beta1], start=1):
            print(
                f"    {rank}. beta1={format_value(beta1):>4} "
                f"{metric}={row[metric]:.6g} "
                f"target_met={format_value(row.get('target_met', False))} "
                f"{hparams(row)}"
            )
        print()

    print("Overall beta1 win counts by group:")
    for beta1, count in sorted(beta1_wins.items(), key=lambda item: (-item[1], item[0])):
        print(f"  beta1={format_value(beta1):>4}: {count}")

    print(f"\nTop {final_top} group rows:")
    header = (
        "batch_size  num_samples  rank  beta1        clean_sse  target  lr       "
        "disable_bias1  adaptive_norm  lr_in_momentum  flush_last"
    )
    print(header)
    print("-" * len(header))
    for batch_size, num_samples, rank, row in sorted(final_rows):
        print(
            f"{batch_size:>10}  {num_samples:>11}  {rank:>4}  "
            f"{format_value(row['beta1']):>5}  {row[metric]:>15.6g}  "
            f"{format_value(row.get('target_met', False)):>6}  "
            f"{format_value(row['lr']):>7}  "
            f"{format_value(row['disable_bias1']):>13}  "
            f"{format_value(row['adaptive_norm']):>13}  "
            f"{format_value(row['lr_in_momentum']):>14}  "
            f"{format_value(row['flush_last']):>10}"
        )


def main() -> None:
    args = parse_args()
    if not args.log_path.exists():
        raise SystemExit(f"Log file not found: {args.log_path}")

    flag_patterns = parse_flag_patterns(args.flag_patterns)
    groups = summarize(args.log_path, args.metric, flag_patterns)
    if not groups:
        raise SystemExit(f"No RUN_SUMMARY rows with metric {args.metric!r} found in {args.log_path}")
    print_results(groups, args.metric, args.top_beta1, args.final_top, flag_patterns)


if __name__ == "__main__":
    main()

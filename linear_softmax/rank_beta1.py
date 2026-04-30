#!/usr/bin/env python3
"""Rank beta1 values across batch-size / num-sample settings.

For each (batch_size, num_samples) situation, each beta1 is scored by the
best clean_sse achieved by that beta1 across all logged learning rates. The
ranking is computed separately for each optimizer variant. Beta1 values are
then dense-ranked by score; exact ties receive the same rank.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        default=Path(__file__).with_name("scalar_optimizers_logging6.log"),
        type=Path,
        help="Path to scalar optimizer log file.",
    )
    return parser.parse_args()


def load_run_summaries(log_path: Path) -> list[dict]:
    rows = []
    skipped_nonfinite = 0
    with log_path.open() as f:
        for line in f:
            if line.startswith("RUN_SUMMARY "):
                row = json.loads(line.split(" ", 1)[1])
                score = float(row["clean_sse"])
                if not math.isfinite(score):
                    skipped_nonfinite += 1
                    continue
                rows.append(row)
    if not rows:
        raise SystemExit(f"No RUN_SUMMARY rows found in {log_path}")
    if skipped_nonfinite:
        print(f"Skipped {skipped_nonfinite} RUN_SUMMARY rows with non-finite clean_sse")
    return rows


def dense_ranks_by_score(scores: dict[float, float]) -> dict[float, int]:
    """Return dense ranks for beta1 -> score, lower score is better."""
    ranks = {}
    previous_score = None
    rank = 0
    for beta1, score in sorted(scores.items(), key=lambda item: (item[1], item[0])):
        if previous_score is None or score != previous_score:
            rank += 1
            previous_score = score
        ranks[beta1] = rank
    return ranks


def optimizer_variant(row: dict) -> str:
    variant = row.get("variant")
    if isinstance(variant, str) and variant:
        parts = [variant]
    elif row["optimizer"] == "AdamW":
        parts = ["AdamW_row"]
    elif row["optimizer"] == "Adam2":
        parts = ["Adam2_row"]
    else:
        h_norm = row.get("h_norm")
        if h_norm is None:
            parts = [str(row["optimizer"])]
        else:
            parts = [f"{row['optimizer']}_{h_norm}"]

    for key in ("nesterov", "disable_bias1", "adaptive_norm", "flush_last"):
        if key in row:
            parts.append(f"{key}={bool(row[key])}")
    return "__".join(parts)


def rank_rows_by_average(rows: list[list[str]], avg_rank_index: int) -> list[list[str]]:
    ranked_rows = []
    previous_avg_rank = None
    display_rank = 0
    for row in rows:
        avg_rank = float(row[avg_rank_index])
        if previous_avg_rank is None or avg_rank != previous_avg_rank:
            display_rank += 1
            previous_avg_rank = avg_rank
        ranked_rows.append([str(display_rank), *row])
    return ranked_rows


def format_beta1(beta1: float) -> str:
    return f"{beta1:g}"


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    widths = [
        max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))
    ]
    print("  ".join(str(value).rjust(widths[i]) for i, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).rjust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    rows = load_run_summaries(args.log_path)

    for row in rows:
        row["optimizer_variant"] = optimizer_variant(row)

    best_sse = {}
    best_lr = {}
    for row in rows:
        key = (
            row["optimizer_variant"],
            row["batch_size"],
            row["num_samples"],
            float(row["beta1"]),
        )
        score = float(row["clean_sse"])
        if key not in best_sse or score < best_sse[key]:
            best_sse[key] = score
            best_lr[key] = float(row["lr"])

    optimizer_variants = sorted({key[0] for key in best_sse})
    batch_sizes = sorted({key[1] for key in best_sse})
    num_samples_values = sorted({key[2] for key in best_sse})
    beta1_values = sorted({key[3] for key in best_sse})

    for variant in optimizer_variants:
        print(f"\n{'=' * 100}\n{variant}\n{'=' * 100}")

        situation_ranks = {}
        for batch_size in batch_sizes:
            for num_samples in num_samples_values:
                scores = {
                    beta1: best_sse[(variant, batch_size, num_samples, beta1)]
                    for beta1 in beta1_values
                }
                situation_ranks[(batch_size, num_samples)] = dense_ranks_by_score(
                    scores
                )

        ranks_by_beta1 = defaultdict(list)
        ranks_by_batch_beta1 = defaultdict(list)
        for (batch_size, num_samples), ranks in situation_ranks.items():
            del num_samples
            for beta1, rank in ranks.items():
                ranks_by_beta1[beta1].append(rank)
                ranks_by_batch_beta1[(batch_size, beta1)].append(rank)

        global_rows = []
        for beta1, ranks in ranks_by_beta1.items():
            avg_rank = sum(ranks) / len(ranks)
            global_rows.append(
                [format_beta1(beta1), f"{avg_rank:.4f}", ",".join(map(str, ranks))]
            )
        global_rows.sort(key=lambda row: (float(row[1]), float(row[0])))

        print_table(
            "Global Beta1 Ranking By Average Rank",
            ["global_rank", "beta1", "avg_rank", "ranks"],
            rank_rows_by_average(global_rows, avg_rank_index=1),
        )

        per_batch_rows = []
        for batch_size in batch_sizes:
            batch_rows = []
            for beta1 in beta1_values:
                ranks = ranks_by_batch_beta1[(batch_size, beta1)]
                avg_rank = sum(ranks) / len(ranks)
                batch_rows.append(
                    [format_beta1(beta1), f"{avg_rank:.4f}", ",".join(map(str, ranks))]
                )
            batch_rows.sort(key=lambda row: (float(row[1]), float(row[0])))
            for row in rank_rows_by_average(batch_rows, avg_rank_index=1):
                per_batch_rows.append([str(batch_size), *row])

        print_table(
            "Per-Batch Beta1 Ranking By Average Rank",
            ["batch", "batch_rank", "beta1", "avg_rank", "ranks"],
            per_batch_rows,
        )


if __name__ == "__main__":
    main()

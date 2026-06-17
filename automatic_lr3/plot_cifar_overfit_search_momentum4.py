#!/usr/bin/env python3
"""Plot the fourth CIFAR overfit LR/momentum-search log."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_cifar_overfit_search_momentum import (
    parse_log,
    plot_all,
    write_csvs,
    write_summary,
)


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "cifar_overfit_search_momentum4.log"
DEFAULT_OUTPUT_DIR = HERE / "cifar_overfit_search_momentum4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and plot cifar_overfit_search_momentum4.log."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file to plot. Defaults to {DEFAULT_LOG.name}.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG/CSV outputs. Defaults to {DEFAULT_OUTPUT_DIR.name}.",
    )
    parser.add_argument(
        "--max-landscape-panels",
        type=int,
        default=25,
        help="Maximum interval landscape panels per run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(log_path)
    if not runs:
        raise SystemExit(f"No runs parsed from {log_path}")

    write_csvs(runs, output_dir)
    write_summary(runs, log_path, output_dir)
    plot_all(runs, output_dir, max_panels=args.max_landscape_panels)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

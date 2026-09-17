#!/usr/bin/env python3
"""Summarize per-step train losses from an automatic_lr3 log file."""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path


RUN_RE = re.compile(
    r"^cifar_baseline2_overfit_n_search "
    r"run=(?P<run>\d+) "
    r"batch_size=(?P<batch_size>\d+) "
    r"N=(?P<N>\d+) "
    r"initial_muon_lr=(?P<initial_muon_lr>\S+) "
    r"initial_muon_lr_k=(?P<initial_muon_lr_k>\S+)"
)
TRAIN_LOSS_RE = re.compile(
    r"^train_loss "
    r"run=(?P<run>\d+) "
    r"step=(?P<step>\d+)/(?:\d+) "
    r"loss=(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_log(log_path: Path) -> OrderedDict[int, dict[str, object]]:
    runs: OrderedDict[int, dict[str, object]] = OrderedDict()

    with log_path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.strip()

            run_match = RUN_RE.match(line)
            if run_match:
                run = int(run_match["run"])
                runs[run] = {
                    "line": line_number,
                    "batch_size": int(run_match["batch_size"]),
                    "N": int(run_match["N"]),
                    "initial_muon_lr": run_match["initial_muon_lr"],
                    "initial_muon_lr_k": run_match["initial_muon_lr_k"],
                    "losses": [],
                }
                continue

            loss_match = TRAIN_LOSS_RE.match(line)
            if loss_match:
                run = int(loss_match["run"])
                if run not in runs:
                    runs[run] = {"line": None, "losses": []}
                runs[run]["losses"].append(
                    (int(loss_match["step"]), float(loss_match["loss"]))
                )

    return runs


def write_summary(log_path: Path, output_path: Path) -> None:
    runs = parse_log(log_path)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"Log: {log_path.name}\n")
        output_file.write(f"Configurations: {len(runs)}\n\n")

        for run, info in runs.items():
            losses = info["losses"]
            output_file.write(
                "run={run} batch_size={batch_size} N={N} "
                "initial_muon_lr={initial_muon_lr} "
                "initial_muon_lr_k={initial_muon_lr_k}\n".format(
                    run=run,
                    batch_size=info.get("batch_size", "unknown"),
                    N=info.get("N", "unknown"),
                    initial_muon_lr=info.get("initial_muon_lr", "unknown"),
                    initial_muon_lr_k=info.get("initial_muon_lr_k", "unknown"),
                )
            )
            output_file.write(f"loss_per_step = {losses}\n\n")


def default_output_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_loss_summary.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a txt summary of loss per step for each run configuration."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=Path("cifar_baseline2_overfit_n_search_exp1.log"),
        help="Path to the log file to summarize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output txt path. Defaults to <log stem>_loss_summary.txt next to the log.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output else default_output_path(log_path)
    )
    write_summary(log_path, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

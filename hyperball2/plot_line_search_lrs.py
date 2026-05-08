#!/usr/bin/env python3
"""Plot LR choices from line-search training logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("line_search_exp7.log")
DEFAULT_OUTPUT = Path(__file__).with_name("line_search_exp7_lrs.png")
LR_MARKER = "LR_SEARCH "


def parse_lr_records(log_path: Path) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            idx = line.find(LR_MARKER)
            if idx < 0:
                continue
            payload = line[idx + len(LR_MARKER) :].strip()
            if not payload.startswith("{"):
                skipped += 1
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if record.get("type") == "lr_search" and "step" in record:
                records.append(record)

    records.sort(
        key=lambda record: (
            str(record.get("experiment", "")),
            str(record.get("val_set", "")),
            int(record["step"]),
        )
    )
    return records, skipped


def experiment_key(record: dict) -> tuple[str, str]:
    return (
        str(record.get("experiment", "unknown_experiment")),
        str(record.get("val_set", "unknown_val_set")),
    )


def collect_lr_series(
    records: list[dict], lr_field: str
) -> dict[str, tuple[list[int], list[float]]]:
    series = defaultdict(lambda: ([], []))
    for record in records:
        lrs = record.get(lr_field) or {}
        for name, value in lrs.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                continue
            steps, values = series[str(name)]
            steps.append(int(record["step"]))
            values.append(value)
    return dict(series)


def color_for_names(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}


def plot_experiment(
    ax,
    records: list[dict],
    title: str,
    show_depth2: bool,
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    best_series = collect_lr_series(records, "best_lrs")
    depth2_series = collect_lr_series(records, "best_depth2_lrs")
    names = sorted(set(best_series) | (set(depth2_series) if show_depth2 else set()))

    for name in names:
        color = colors[name]
        if name in best_series:
            steps, values = best_series[name]
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=color,
                label=name,
            )
        if show_depth2 and name in depth2_series:
            steps, values = depth2_series[name]
            ax.plot(
                steps,
                values,
                marker="x",
                markersize=3,
                linewidth=1.4,
                linestyle="--",
                color=color,
                label=f"{name} depth2",
            )

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize="small", ncols=2)


def plot_lr_curves(
    records: list[dict],
    output_path: Path,
    show_depth2: bool,
    dpi: int,
) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[experiment_key(record)].append(record)

    if not grouped:
        raise ValueError("No LR_SEARCH records found.")

    experiments = sorted(grouped)
    cols = min(2, len(experiments))
    rows = math.ceil(len(experiments) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7.5 * cols, 4.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, key in zip(axes.flat, experiments):
        experiment, val_set = key
        records_for_experiment = sorted(grouped[key], key=lambda record: record["step"])
        names = sorted(
            set(collect_lr_series(records_for_experiment, "best_lrs"))
            | set(collect_lr_series(records_for_experiment, "best_depth2_lrs"))
        )
        colors = color_for_names(names)
        plot_experiment(
            ax,
            records_for_experiment,
            f"{experiment} ({val_set})",
            show_depth2,
            colors,
        )

    for ax in axes.flat[len(experiments) :]:
        ax.axis("off")

    fig.suptitle("Line-search LR curves")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_summary(records: list[dict], skipped: int, output_path: Path) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[experiment_key(record)].append(record)

    print(f"parsed {len(records)} LR_SEARCH records; skipped {skipped}")
    for key in sorted(grouped):
        experiment, val_set = key
        run_records = grouped[key]
        first_step = min(record["step"] for record in run_records)
        last_step = max(record["step"] for record in run_records)
        names = sorted(
            {
                name
                for record in run_records
                for name in (record.get("best_lrs") or {})
            }
        )
        print(
            f"{experiment} ({val_set}): {len(run_records)} records, "
            f"steps {first_step}-{last_step}, groups {', '.join(names)}"
        )
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best_lrs curves from LR_SEARCH records."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file to parse (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--hide-depth2",
        action="store_true",
        help="Do not plot best_depth2_lrs dashed curves.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, skipped = parse_lr_records(args.log)
    plot_lr_curves(
        records,
        args.output,
        show_depth2=not args.hide_depth2,
        dpi=args.dpi,
    )
    print_summary(records, skipped, args.output)


if __name__ == "__main__":
    main()

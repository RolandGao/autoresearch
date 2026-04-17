#!/usr/bin/env python3
"""Plot norm traces from hyperball training logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_LOG = Path(__file__).with_name("more_logging.log")


def parse_norm_records(log_path: Path) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    marker = "NORM_LOG "
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            idx = line.find(marker)
            if idx < 0:
                continue
            payload = line[idx + len(marker):].strip()
            if not payload.startswith("{"):
                skipped += 1
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if record.get("type") == "norms" and "step" in record:
                records.append(record)
    records.sort(key=lambda record: record["step"])
    return records, skipped


def collect_series(records: list[dict], section: str) -> dict[str, tuple[list[int], list[float]]]:
    series = defaultdict(lambda: ([], []))
    for record in records:
        step = record["step"]
        for name, value in record.get(section, {}).items():
            if value is None:
                continue
            steps, values = series[name]
            steps.append(step)
            values.append(float(value))
    return dict(series)


def collect_ratio_series(
    records: list[dict],
    numerator_section: str,
    denominator_section: str,
) -> dict[str, tuple[list[int], list[float]]]:
    series = defaultdict(lambda: ([], []))
    for record in records:
        step = record["step"]
        numerators = record.get(numerator_section, {})
        denominators = record.get(denominator_section, {})
        for name, numerator in numerators.items():
            denominator = denominators.get(name)
            if numerator is None or denominator is None:
                continue
            numerator = float(numerator)
            denominator = float(denominator)
            if denominator == 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
                continue
            steps, values = series[name]
            steps.append(step)
            values.append(numerator / denominator)
    return dict(series)


def filter_series(
    series: dict[str, tuple[list[int], list[float]]],
    min_step: int | None = None,
    positive_only: bool = False,
) -> dict[str, tuple[list[int], list[float]]]:
    filtered = {}
    for name, (steps, values) in series.items():
        kept_steps = []
        kept_values = []
        for step, value in zip(steps, values):
            if min_step is not None and step < min_step:
                continue
            if positive_only and value <= 0:
                continue
            kept_steps.append(step)
            kept_values.append(value)
        if kept_steps:
            filtered[name] = (kept_steps, kept_values)
    return filtered


def layer_index(name: str) -> int:
    match = re.match(r"h\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def strip_layer(name: str) -> str:
    return re.sub(r"^h\.\d+\.", "", name)


def natural_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def group_series(series: dict[str, tuple[list[int], list[float]]]) -> list[tuple[str, list[str]]]:
    groups = [
        ("Embeddings And Head", []),
        ("Attention QKV", []),
        ("Attention And MLP Projections", []),
        ("Lambda Scalars", []),
        ("Value Embeddings And Gates", []),
        ("Other", []),
    ]
    by_title = {title: names for title, names in groups}

    for name in series:
        suffix = strip_layer(name)
        if not name.startswith("h."):
            by_title["Embeddings And Head"].append(name)
        elif suffix in {"q", "k", "v"}:
            by_title["Attention QKV"].append(name)
        elif suffix in {"attn.c_proj", "mlp.c_fc", "mlp.c_proj"}:
            by_title["Attention And MLP Projections"].append(name)
        elif suffix in {"resid_lambdas", "x0_lambdas"}:
            by_title["Lambda Scalars"].append(name)
        elif suffix in {"ve", "attn.ve_gate"}:
            by_title["Value Embeddings And Gates"].append(name)
        else:
            by_title["Other"].append(name)

    result = []
    for title, names in groups:
        if names:
            result.append((title, sorted(names, key=series_key)))
    return result


def series_key(name: str) -> tuple:
    return (strip_layer(name), layer_index(name), natural_key(name))


def split_long_group(names: list[str], max_lines: int) -> list[list[str]]:
    return [names[i:i + max_lines] for i in range(0, len(names), max_lines)]


def positive_range(values: list[float]) -> tuple[float, float] | None:
    positives = [value for value in values if value > 0 and math.isfinite(value)]
    if not positives:
        return None
    return min(positives), max(positives)


def set_norm_scale(ax, names: list[str], series: dict[str, tuple[list[int], list[float]]]) -> None:
    values = []
    for name in names:
        values.extend(series[name][1])
    value_range = positive_range(values)
    if value_range is None:
        return
    smallest, largest = value_range
    if largest / smallest > 50:
        ax.set_yscale("symlog", linthresh=smallest / 10)


def plot_series_page(
    pdf: PdfPages,
    title: str,
    names: list[str],
    series: dict[str, tuple[list[int], list[float]]],
    ylabel: str,
    force_linear_y: bool = False,
    y_scale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    for name in names:
        steps, values = series[name]
        ax.plot(steps, values, linewidth=1.25, label=name)
    if y_scale is not None:
        ax.set_yscale(y_scale)
    elif not force_linear_y:
        set_norm_scale(ax, names, series)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ncol = 1 if len(names) <= 12 else 2 if len(names) <= 28 else 3
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=ncol)
    pdf.savefig(fig)
    plt.close(fig)


def plot_section(
    pdf: PdfPages,
    section_title: str,
    series: dict[str, tuple[list[int], list[float]]],
    max_lines_per_page: int,
    force_linear_y: bool = False,
    y_scale: str | None = None,
) -> int:
    pages = 0
    for group_title, names in group_series(series):
        chunks = split_long_group(names, max_lines_per_page)
        for idx, chunk in enumerate(chunks, start=1):
            suffix = f" ({idx}/{len(chunks)})" if len(chunks) > 1 else ""
            plot_series_page(
                pdf,
                f"{section_title}: {group_title}{suffix}",
                chunk,
                series,
                section_title.lower(),
                force_linear_y=force_linear_y,
                y_scale=y_scale,
            )
            pages += 1
    return pages


def plot_activations(pdf: PdfPages, series: dict[str, tuple[list[int], list[float]]]) -> int:
    if not series:
        return 0
    names = sorted(series, key=series_key)
    plot_series_page(pdf, "Activation L2 Norms After Each Block", names, series, "activation L2 norm")
    return 1


def plot_all(records: list[dict], output_path: Path, max_lines_per_page: int) -> int:
    weight_series = collect_series(records, "weight_norms")
    grad_series = collect_series(records, "grad_norms")
    update_series = collect_series(records, "update_norms")
    effective_lr_series = collect_ratio_series(records, "update_norms", "weight_norms")
    effective_lr_log_series = filter_series(effective_lr_series, positive_only=True)
    effective_lr_linear_series = filter_series(effective_lr_series, min_step=20)
    activation_series = collect_series(records, "activation_l2_norms")

    pages = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pages += plot_activations(pdf, activation_series)
        pages += plot_section(pdf, "Weight Norms", weight_series, max_lines_per_page)
        pages += plot_section(pdf, "Gradient Norms", grad_series, max_lines_per_page)
        pages += plot_section(pdf, "Update Norms", update_series, max_lines_per_page)
        pages += plot_section(
            pdf,
            "Effective LR (Log Scale)",
            effective_lr_log_series,
            max_lines_per_page,
            y_scale="log",
        )
        pages += plot_section(
            pdf,
            "Effective LR (Linear, Step >= 20)",
            effective_lr_linear_series,
            max_lines_per_page,
            force_linear_y=True,
        )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NORM_LOG traces from a hyperball train log.")
    parser.add_argument("log", nargs="?", default=DEFAULT_LOG, type=Path, help="Training log to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <log_stem>_norms.pdf next to the log.",
    )
    parser.add_argument(
        "--max-lines-per-page",
        type=int,
        default=32,
        help="Maximum number of plotted traces on each PDF page.",
    )
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    if args.output is None:
        output_path = log_path.with_name(f"{log_path.stem}_norms.pdf")
    else:
        output_path = args.output.expanduser().resolve()

    records, skipped = parse_norm_records(log_path)
    if not records:
        raise SystemExit(f"No complete NORM_LOG records found in {log_path}")

    pages = plot_all(records, output_path, args.max_lines_per_page)
    first_step = records[0]["step"]
    last_step = records[-1]["step"]
    print(f"parsed_records: {len(records)}")
    print(f"step_range:     {first_step}..{last_step}")
    print(f"skipped_lines:   {skipped}")
    print(f"pages:           {pages}")
    print(f"output:          {output_path}")


if __name__ == "__main__":
    main()

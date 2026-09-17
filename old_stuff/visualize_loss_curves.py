#!/usr/bin/env python3
"""Visualize training loss curves from autoresearch logs."""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LOGS = ("bs_e17.log", "run.log", "run2.log")
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+"
    r"\((?P<percent>\d+(?:\.\d+)?)%\)\s+\|\s+"
    r"loss:\s+(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
VAL_BPB_RE = re.compile(
    r"val_bpb:\s+(?P<val_bpb>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True)
class LossCurve:
    label: str
    path: Path
    steps: list[int]
    percents: list[float]
    losses: list[float]
    val_bpb: float | None

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def min_loss(self) -> float:
        return min(self.losses)


def parse_loss_curve(path: Path) -> LossCurve:
    text = path.read_text(encoding="utf-8", errors="replace")
    steps: list[int] = []
    percents: list[float] = []
    losses: list[float] = []

    for match in STEP_RE.finditer(text):
        steps.append(int(match.group("step")))
        percents.append(float(match.group("percent")))
        losses.append(float(match.group("loss")))

    if not losses:
        raise ValueError(f"No step/loss entries found in {path}")

    val_match = VAL_BPB_RE.search(text)
    val_bpb = float(val_match.group("val_bpb")) if val_match else None

    return LossCurve(
        label=path.name,
        path=path,
        steps=steps,
        percents=percents,
        losses=losses,
        val_bpb=val_bpb,
    )


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]

    smoothed: list[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
            divisor = window
        else:
            divisor = index + 1
        smoothed.append(running_total / divisor)
    return smoothed


def format_label(curve: LossCurve) -> str:
    val_text = f", val_bpb {curve.val_bpb:.4f}" if curve.val_bpb is not None else ""
    return f"{curve.label} (final {curve.final_loss:.3f}, min {curve.min_loss:.3f}{val_text})"


def plot_with_matplotlib(
    curves: list[LossCurve],
    output: Path,
    x_axis: str,
    smooth_window: int,
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        xs = curve.percents if x_axis == "percent" else curve.steps
        smooth = moving_average(curve.losses, smooth_window)
        ax.plot(xs, curve.losses, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(xs, smooth, color=color, linewidth=2.2, label=format_label(curve))

    xlabel = "Training progress (%)" if x_axis == "percent" else "Training step"
    ax.set_title("Training Loss Curves")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if math.isclose(in_min, in_max):
        return (out_min + out_max) / 2.0
    return out_min + ((value - in_min) / (in_max - in_min)) * (out_max - out_min)


def nice_ticks(low: float, high: float, count: int) -> list[float]:
    if count <= 1 or math.isclose(low, high):
        return [low]

    raw_step = (high - low) / (count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice_step = magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(low / nice_step) * nice_step
    ticks: list[float] = []
    value = start
    while value <= high + nice_step * 0.5:
        if low <= value <= high:
            ticks.append(value)
        value += nice_step
    return ticks


def polyline_points(
    xs: list[float],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> str:
    points: list[str] = []
    for x, y in zip(xs, ys):
        px = scale(x, x_min, x_max, left, right)
        py = scale(y, y_min, y_max, bottom, top)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def plot_with_svg(
    curves: list[LossCurve],
    output: Path,
    x_axis: str,
    smooth_window: int,
) -> None:
    width = 1280
    height = 760
    left = 88
    right = width - 42
    top = 64
    bottom = height - 118
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    all_xs = [x for curve in curves for x in (curve.percents if x_axis == "percent" else curve.steps)]
    all_losses = [loss for curve in curves for loss in curve.losses]
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_losses), max(all_losses)
    y_padding = max((y_max - y_min) * 0.06, 0.05)
    y_min = max(0.0, y_min - y_padding)
    y_max += y_padding

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}",
        ".title{font-size:26px;font-weight:700}",
        ".axis{stroke:#222;stroke-width:1.3}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".tick{font-size:13px;fill:#333}",
        ".label{font-size:16px;font-weight:600}",
        ".legend{font-size:14px}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text class="title" x="88" y="38">Training Loss Curves</text>',
    ]

    for tick in nice_ticks(y_min, y_max, 7):
        y = scale(tick, y_min, y_max, bottom, top)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        svg.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>')

    x_tick_count = 6
    for index in range(x_tick_count):
        tick = x_min + (x_max - x_min) * index / (x_tick_count - 1)
        x = scale(tick, x_min, x_max, left, right)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>')
        label = f"{tick:.0f}" if x_axis == "step" else f"{tick:.0f}%"
        svg.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 24}" text-anchor="middle">{label}</text>')

    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<text class="label" x="{(left + right) / 2:.2f}" y="{height - 34}" text-anchor="middle">'
            + ("Training progress (%)" if x_axis == "percent" else "Training step")
            + "</text>",
            f'<text class="label" x="24" y="{(top + bottom) / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {(top + bottom) / 2:.2f})">Loss</text>',
        ]
    )

    legend_y = 66
    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        xs = curve.percents if x_axis == "percent" else curve.steps
        smooth = moving_average(curve.losses, smooth_window)
        raw_points = polyline_points(xs, curve.losses, x_min, x_max, y_min, y_max, left, right, top, bottom)
        smooth_points = polyline_points(xs, smooth, x_min, x_max, y_min, y_max, left, right, top, bottom)
        svg.append(f'<polyline points="{raw_points}" fill="none" stroke="{color}" stroke-width="1" opacity="0.18"/>')
        svg.append(f'<polyline points="{smooth_points}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        svg.append(f'<line x1="{right - 426}" y1="{legend_y}" x2="{right - 388}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text class="legend" x="{right - 378}" y="{legend_y + 5}">'
            f"{html.escape(format_label(curve))}</text>"
        )
        legend_y += 24

    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def resolve_logs(logs: list[str]) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    paths = []
    for log in logs:
        path = Path(log)
        if not path.is_absolute():
            path = base_dir / path
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        default=list(DEFAULT_LOGS),
        help="Log files to parse. Defaults to bs_e17.log, run.log, and run2.log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="loss_curves.svg",
        help="Output image path. Defaults to loss_curves.svg beside this script.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("percent", "step"),
        default="percent",
        help="Use comparable training percent or raw step as the x-axis.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Moving average window for the bold curves. Use 1 to disable.",
    )
    parser.add_argument(
        "--no-matplotlib",
        action="store_true",
        help="Force the built-in SVG renderer even if matplotlib is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_paths = resolve_logs(args.logs)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output

    curves = [parse_loss_curve(path) for path in log_paths]

    if output.suffix.lower() != ".svg" and args.no_matplotlib:
        raise ValueError("The built-in renderer only supports .svg output.")

    used_matplotlib = False
    if not args.no_matplotlib:
        try:
            plot_with_matplotlib(curves, output, args.x_axis, args.smooth)
            used_matplotlib = True
        except ModuleNotFoundError:
            if output.suffix.lower() != ".svg":
                output = output.with_suffix(".svg")

    if not used_matplotlib:
        plot_with_svg(curves, output, args.x_axis, args.smooth)

    print(f"Wrote {output}")
    for curve in curves:
        val_text = f", val_bpb={curve.val_bpb:.6f}" if curve.val_bpb is not None else ""
        print(
            f"{curve.label}: points={len(curve.losses)}, "
            f"final_loss={curve.final_loss:.6f}, min_loss={curve.min_loss:.6f}{val_text}"
        )


if __name__ == "__main__":
    main()

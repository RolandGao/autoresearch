#!/usr/bin/env python3
"""Fit non-lambda effective LR traces from a NORM_LOG file.

The effective LR used by visualize.py is update_norm / weight_norm.  This
script ignores the lambda scalar traces and fits one shared time-shape with a
per-parameter amplitude.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).with_name("more_logging2.log")
DEFAULT_MIN_STEP = 20
DEFAULT_KNOTS = (20, 200, 400, 700, 1000, 1200, 1349)


def parse_records(log_path: Path) -> list[dict]:
    records = []
    marker = "NORM_LOG "
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            idx = line.find(marker)
            if idx < 0:
                continue
            try:
                record = json.loads(line[idx + len(marker) :])
            except json.JSONDecodeError:
                continue
            if record.get("type") == "norms" and "step" in record:
                records.append(record)
    records.sort(key=lambda record: record["step"])
    return records


def is_lambda_scalar(name: str) -> bool:
    return name.endswith(".resid_lambdas") or name.endswith(".x0_lambdas")


def collect_effective_lr(
    records: list[dict], min_step: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    series: dict[str, dict[int, float]] = {}
    for record in records:
        step = int(record["step"])
        if step < min_step:
            continue
        update_norms = record.get("update_norms", {})
        weight_norms = record.get("weight_norms", {})
        for name, update_norm in update_norms.items():
            if is_lambda_scalar(name):
                continue
            weight_norm = weight_norms.get(name)
            if update_norm is None or weight_norm is None or float(weight_norm) == 0:
                continue
            value = float(update_norm) / float(weight_norm)
            if value > 0 and math.isfinite(value):
                series.setdefault(name, {})[step] = value

    if not series:
        raise SystemExit("No positive non-lambda effective LR values found")

    common_steps = sorted(set.intersection(*(set(values) for values in series.values())))
    names = sorted(series)
    values = np.array([[series[name][step] for name in names] for step in common_steps])
    return np.array(common_steps, dtype=float), names, values


def fit_with_log_shape(log_values: np.ndarray, log_shape: np.ndarray) -> tuple[float, float, float]:
    amplitudes = (log_values - log_shape[:, None]).mean(axis=0)
    residual = log_values - (log_shape[:, None] + amplitudes[None, :])
    rmse = float(np.sqrt(np.mean(residual**2)))
    median_abs = float(np.median(np.abs(residual)))
    p95_abs = float(np.percentile(np.abs(residual), 95))
    return rmse, median_abs, p95_abs


def fit_stretched_exp(
    steps: np.ndarray,
    values: np.ndarray,
    min_p: float = 0.05,
    max_p: float = 3.5,
    grid: int = 700,
) -> tuple[float, float, float, float, float]:
    log_values = np.log(values)
    t = (steps - steps[0]) / (steps[-1] - steps[0])
    centered_values = log_values - log_values.mean(axis=0, keepdims=True)
    best = None
    for p in np.linspace(min_p, max_p, grid):
        z = -(t**p)
        centered_z = z - z.mean()
        c = float(
            (centered_z[:, None] * centered_values).sum()
            / ((centered_z**2).sum() * values.shape[1])
        )
        if c < 0:
            continue
        log_shape = c * z
        rmse, median_abs, p95_abs = fit_with_log_shape(log_values, log_shape)
        if best is None or rmse < best[0]:
            best = (rmse, median_abs, p95_abs, c, float(p))
    assert best is not None
    return best


def fit_piecewise_log_linear(
    steps: np.ndarray, values: np.ndarray, knots: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    log_values = np.log(values)
    centered = log_values - log_values.mean(axis=0, keepdims=True)
    mean_shape = centered.mean(axis=1)
    t = (steps - steps[0]) / (steps[-1] - steps[0])
    knot_t = (np.array(knots, dtype=float) - steps[0]) / (steps[-1] - steps[0])
    knot_log_values = np.interp(knot_t, t, mean_shape)
    log_shape = np.interp(t, knot_t, knot_log_values)
    rmse, median_abs, p95_abs = fit_with_log_shape(log_values, log_shape)
    relative_log_values = knot_log_values - knot_log_values[0]
    return np.array(knots), relative_log_values, rmse, median_abs, p95_abs


def summarize_amplitudes(
    steps: np.ndarray,
    names: list[str],
    values: np.ndarray,
    knot_steps: np.ndarray,
    knot_log_ratios: np.ndarray,
) -> list[tuple[str, float]]:
    log_values = np.log(values)
    log_shape = np.interp(steps, knot_steps, knot_log_ratios)
    amplitudes = np.exp((log_values - log_shape[:, None]).mean(axis=0))
    return list(zip(names, amplitudes))


def suffix_name(name: str) -> str:
    if not name.startswith("h."):
        return name
    return ".".join(name.split(".")[2:])


def write_plot(
    output_path: Path,
    steps: np.ndarray,
    names: list[str],
    values: np.ndarray,
    knot_steps: np.ndarray,
    knot_log_ratios: np.ndarray,
    amplitudes: list[tuple[str, float]],
) -> None:
    amp_by_name = dict(amplitudes)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for idx, name in enumerate(names):
        ax.plot(steps, values[:, idx], color="0.75", alpha=0.35, linewidth=0.7)
    median_amp = float(np.median([amp for _, amp in amplitudes]))
    dense_steps = np.linspace(steps[0], steps[-1], 1000)
    fit = median_amp * np.exp(np.interp(dense_steps, knot_steps, knot_log_ratios))
    ax.plot(dense_steps, fit, color="tab:red", linewidth=2.2, label="median fit")
    ax.scatter(knot_steps, median_amp * np.exp(knot_log_ratios), color="tab:red", s=22)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("effective LR = update_norm / weight_norm")
    ax.set_title("Non-lambda effective LR fit")
    ax.grid(True, alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--min-step", type=int, default=DEFAULT_MIN_STEP)
    parser.add_argument("--plot", type=Path, default=Path("effective_lr_fit.png"))
    args = parser.parse_args()

    records = parse_records(args.log)
    steps, names, values = collect_effective_lr(records, args.min_step)
    exp_rmse, exp_median, exp_p95, c, p = fit_stretched_exp(steps, values)
    knot_steps, knot_log_ratios, pw_rmse, pw_median, pw_p95 = fit_piecewise_log_linear(
        steps, values, DEFAULT_KNOTS
    )
    amplitudes = summarize_amplitudes(steps, names, values, knot_steps, knot_log_ratios)

    print(f"records: {len(records)}")
    print(f"series:  {len(names)} non-lambda weights")
    print(f"fit_step_range: {int(steps[0])}..{int(steps[-1])}")
    print()
    print("stretched_exponential:")
    print(f"  t = (step - {int(steps[0])}) / {int(steps[-1] - steps[0])}")
    print(f"  effective_lr_i(step) ~= A_i * exp(-{c:.6g} * t^{p:.6g})")
    print(f"  log_rmse={exp_rmse:.4f}  typical_factor={math.exp(exp_rmse):.3f}")
    print(f"  log_p95={exp_p95:.4f}   p95_factor={math.exp(exp_p95):.3f}")
    print()
    print("piecewise_log_linear:")
    print("  effective_lr_i(step) ~= A_i * exp(linear_interp(step, knots))")
    print("  knot relative multipliers:")
    for step, log_ratio in zip(knot_steps, knot_log_ratios):
        print(f"    {int(step):4d}: {math.exp(float(log_ratio)):.6g}")
    print(f"  log_rmse={pw_rmse:.4f}  typical_factor={math.exp(pw_rmse):.3f}")
    print(f"  log_p95={pw_p95:.4f}   p95_factor={math.exp(pw_p95):.3f}")
    print()
    print("median A_i by weight kind, for the piecewise fit:")
    grouped: dict[str, list[float]] = {}
    for name, amplitude in amplitudes:
        grouped.setdefault(suffix_name(name), []).append(amplitude)
    for suffix in sorted(grouped):
        vals = grouped[suffix]
        print(f"  {suffix:14s} {np.median(vals):.6g}  n={len(vals)}")

    write_plot(args.plot, steps, names, values, knot_steps, knot_log_ratios, amplitudes)
    print()
    print(f"plot: {args.plot.resolve()}")


if __name__ == "__main__":
    main()

"""Plot and compare head optimizer grids from train_cifar.py's text logs.

Usage:
    python plot_cifar_grid.py [logs/run1.log logs/run2.log ...] [--metric val_acc]
    python plot_cifar_grid.py logs/run.log --run exp6_head_sgd_grid

Defaults to the original input-conditioned grid, its larger-LR extension, and
the SGD/Lion comparison log. Compatible LR extensions are joined into one grid.
Writes comparison curves, per-grid heatmaps and LR curves (PNG/PDF), and a CSV under
untracked_logs/<last log filename>/plots_combined/ (plots/ for one log).
Comparison curves select the best momentum at each LR, separately for each
optimizer, schedule, momentum version, and Nesterov setting. Does not import training code.
"""

import argparse
import copy
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np


DEFAULT_LOG = Path("logs/cifar_baseline_20260928_223330_094572Z_125999.log")
DEFAULT_LOGS = [
    DEFAULT_LOG,
    Path("logs/cifar_baseline_20260929_001548_198729Z_135547.log"),
    Path("logs/cifar_baseline_20260929_020116_300964Z_141192.log"),
]
GRID_FIELDS = (
    "head.initial_lr", "head.momentum", "head.decay", "head.momentum_version", "head.nesterov",
)
ALGORITHM_LABELS = {"input_conditioned": "Input-conditioned", "sgd": "SGD", "lion": "Lion"}


def run_names(path):
    _, marker, output = path.read_text(encoding="utf-8").partition("\n# Run output\n")
    if not marker:
        raise ValueError(f"{path}: log has no '# Run output' marker")
    return re.findall(r"^base_config run=(\S+)\s*$", output, re.MULTILINE)


def parse_log(path, run):
    """Reconstruct each trial from the base config, never the preceding trial.

    The source snapshot and the final debug replay are excluded. Incomplete
    trials have no score and are left out, allowing plots of a running search.
    """
    text = path.read_text(encoding="utf-8")
    _, marker, output = text.partition("\n# Run output\n")
    if not marker:
        raise ValueError("Log has no '# Run output' marker")
    match = re.search(rf"^base_config run={re.escape(run)}\s*$", output, re.MULTILINE)
    if match is None:
        raise ValueError(f"No base config for {run!r}")
    body = output[match.end():].lstrip()
    config, end = json.JSONDecoder().raw_decode(body)
    if config["hparam_tuning"]["algorithm"] != "grid":
        raise ValueError(f"{run!r} is not a grid search")
    specs = config["hparam_tuning"]["params"]
    if not set(GRID_FIELDS[:3]) <= set(specs) <= set(GRID_FIELDS):
        raise ValueError("Expected a head LR/momentum/decay grid with optional momentum version and Nesterov")

    rows, trial, head = [], None, None
    seen = set()
    for line in body[end:].splitlines():
        if line.startswith(("search_best ", "debug_final_run ", "base_config ")):
            break
        match = re.fullmatch(r"config_diff run=(\S+) base_run=(\S+)", line)
        if match:
            trial = match[1] if match[2] == run else None
            head = copy.deepcopy(config["param_groups"]["head"]) if trial else None
            continue
        if trial is None:
            continue
        match = re.fullmatch(r"\s+head\.([^:]+): .* -> (.*)", line)
        if match:
            head[match[1]] = json.loads(match[2])
        elif line.startswith("val_acc="):
            metrics = dict(field.split("=", 1) for field in line.split())
            lines = head["lr_scheduler"]
            if len(lines) != 1 or (len(lines[0]) == 3 and lines[0][2] != 0):
                raise ValueError(f"{trial}: expected a constant or linear-to-zero schedule")
            decay = "constant" if len(lines[0]) == 2 else "linear_decay"
            point = (lines[0][1], head["momentum"], decay, head["momentum_version"], head["nesterov"])
            if point in seen:
                raise ValueError(f"Duplicate grid point: {point}")
            for field, value in zip(GRID_FIELDS, point):
                if field in specs and value not in specs[field]["choices"]:
                    raise ValueError(f"{trial}: unexpected {field}={value}")
            seen.add(point)
            rows.append(dict(
                source_log=str(path),
                run=run,
                algorithm=head["algorithm"],
                trial=trial,
                initial_lr=point[0],
                momentum=point[1],
                decay=point[2],
                momentum_version=point[3],
                nesterov=point[4],
                val_acc=float(metrics["val_acc"]),
                tta_val_acc=float(metrics["tta_val_acc"]),
                seconds=float(metrics["seconds"]),
            ))
            trial = None
    if not rows:
        raise ValueError(f"No completed trials for {run!r}")
    return config, rows


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)


def merge_lr_extensions(datasets):
    """Join disjoint LR grids only when every other config setting matches.

    Keep original run/trial/log identifiers in each row for the CSV. Refuse
    overlapping LR choices rather than silently replacing repeated measurements.
    """
    merged = {}
    for name, config, rows in datasets:
        signature = copy.deepcopy(config)
        del signature["hparam_tuning"]["params"]["head.initial_lr"]
        key = json.dumps(signature, sort_keys=True)
        if key not in merged:
            merged[key] = (name, copy.deepcopy(config), list(rows))
            continue
        _, combined_config, combined_rows = merged[key]
        lr_spec = combined_config["hparam_tuning"]["params"]["head.initial_lr"]
        new_lrs = config["hparam_tuning"]["params"]["head.initial_lr"]["choices"]
        overlap = set(lr_spec["choices"]) & set(new_lrs)
        if overlap:
            raise ValueError(f"Overlapping LR grids for {name}: {sorted(overlap)}")
        lr_spec["choices"] = sorted(set(lr_spec["choices"]) | set(new_lrs))
        combined_rows.extend(rows)
    return list(merged.values())


def plot_results(config, rows, directory, metric, heatmap_min):
    specs = config["hparam_tuning"]["params"]
    lrs = sorted(specs["head.initial_lr"]["choices"])
    momenta = sorted(specs["head.momentum"]["choices"])
    decays = specs["head.decay"]["choices"]
    versions = specs.get("head.momentum_version", {"choices": [
        config["param_groups"]["head"]["momentum_version"]
    ]})["choices"]
    nesterov_choices = specs.get("head.nesterov", {"choices": [
        config["param_groups"]["head"]["nesterov"]
    ]})["choices"]
    columns = [(decay, nesterov) for decay in decays for nesterov in nesterov_choices]
    expected = math.prod(len(spec["choices"]) for spec in specs.values())
    label = (
        "TTA validation accuracy (%)"
        if metric == "tta_val_acc"
        else "Validation accuracy (%)"
    )
    finite = [row for row in rows if math.isfinite(row[metric])]
    if not finite:
        raise ValueError(f"No finite {metric} scores")
    best = max(finite, key=lambda row: row[metric])
    maximum = max(heatmap_min + 0.1, math.ceil(best[metric] * 1000) / 10)
    head = config["param_groups"]["head"]
    algorithm = ALGORITHM_LABELS.get(head["algorithm"], head["algorithm"])
    conditioning = ""
    if head["algorithm"] == "input_conditioned":
        order = "before" if head["gradient_momentum_before_conditioning"] else "after"
        conditioning = f" | momentum {order} conditioning"
    title = (
        f"{algorithm} head grid | {len(rows)}/{expected} trials\n"
        f"Conv: {config['param_groups']['conv']['algorithm']} | "
        f"batch size {config['batch_size']} | {config['num_epochs']} epochs"
        f"{conditioning}"
    )
    shape = (len(versions), len(columns))
    fig, axes = plt.subplots(
        *shape, figsize=(max(16, len(lrs) * 1.2, len(columns) * 7), 4.5 * len(versions)),
        squeeze=False, layout="constrained",
    )
    curves, curve_axes = plt.subplots(
        *shape, figsize=(7 * len(columns), 4.5 * len(versions)),
        squeeze=False, sharex=True, sharey=True, layout="constrained"
    )
    fig.suptitle(title, fontsize=15)
    curves.suptitle(title, fontsize=15)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#f8d6d2")
    cmap.set_bad("#dddddd")
    norm = Normalize(vmin=heatmap_min, vmax=maximum)
    colors = plt.get_cmap("tab10")
    for i, version in enumerate(versions):
        for j, (decay, nesterov) in enumerate(columns):
            ax, curve_ax = axes[i, j], curve_axes[i, j]
            subset = [
                row for row in rows
                if row["momentum_version"] == version and row["decay"] == decay
                and row["nesterov"] == nesterov
            ]
            lookup = {
                (row["momentum"], row["initial_lr"]): row[metric] * 100
                for row in subset
            }
            values = np.array([
                [lookup.get((m, lr), np.nan) for lr in lrs] for m in momenta
            ])
            values[~np.isfinite(values)] = np.nan
            plotted = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
            panel = (
                f"{decay.replace('_', ' ').capitalize()} | momentum version {version}"
                f" | Nesterov {nesterov}"
            )
            if np.isfinite(values).any():
                winner = np.unravel_index(np.nanargmax(values), values.shape)
                panel += (
                    f"\nBest {values[winner]:.2f}% | LR {lrs[winner[1]]:g}, "
                    f"momentum {momenta[winner[0]]:g}"
                )
                ax.add_patch(Rectangle(
                    (winner[1] - 0.48, winner[0] - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="#e35b00", linewidth=2.5,
                ))
            ax.set_title(panel, fontsize=11)
            curve_ax.set_title(panel, fontsize=11)
            ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs], rotation=45)
            ax.set_yticks(range(len(momenta)), [f"{m:g}" for m in momenta])
            ax.set_xlabel("Head initial learning rate")
            ax.set_ylabel("Head momentum")
            for row_index, m in enumerate(momenta):
                for column, value in enumerate(values[row_index]):
                    text = "—" if not np.isfinite(value) else f"{value:.2f}"
                    dark = (
                        np.isfinite(value)
                        and heatmap_min <= value < heatmap_min + 0.55 * (maximum - heatmap_min)
                    )
                    ax.text(
                        column, row_index, text, ha="center", va="center",
                        fontsize=8, color="white" if dark else "black",
                    )
                curve_ax.plot(
                    lrs, values[row_index], marker="o", markersize=4,
                    color=colors(row_index), label=f"momentum {m:g}",
                )
            curve_ax.set_xscale("log")
            curve_ax.set_xlabel("Head initial learning rate (log scale)")
            curve_ax.set_ylabel(label)
            curve_ax.grid(alpha=0.25)
            curve_ax.legend(fontsize=8, loc="lower left")
    below = any(row[metric] * 100 < heatmap_min for row in finite)
    fig.colorbar(
        plotted, ax=axes, label=label,
        extend="min" if below else "neither", shrink=0.8,
    )
    fig.supxlabel(
        f"Color scale starts at {heatmap_min:g}%; pink cells are below it. "
        "Labels show exact scores; gray cells are missing/nonfinite. Orange outline: panel best.",
        fontsize=10,
    )
    save_figure(fig, directory, f"{metric}_heatmaps")
    plt.close(fig)
    save_figure(curves, directory, f"{metric}_lr_curves")
    # A second view makes small differences visible without hiding failed runs
    # from the full-range figure or the annotated heatmaps.
    curve_axes[0, 0].set_ylim(heatmap_min, maximum + 0.1)
    curves.supxlabel(
        f"Zoomed to {heatmap_min:g}% and above; see the full-range curves for lower scores.",
        fontsize=10,
    )
    save_figure(curves, directory, f"{metric}_lr_curves_zoom")
    plt.close(curves)
    print(f"Completed {len(rows)}/{expected} trials")
    print(
        f"Best {metric}={best[metric]:.4f}: LR={best['initial_lr']:g}, "
        f"momentum={best['momentum']:g}, decay={best['decay']}, "
        f"momentum_version={best['momentum_version']}, nesterov={best['nesterov']} ({best['trial']})"
    )


def plot_comparison(datasets, directory, metric, floor):
    """Compare observed scores, maximizing over momentum at each learning rate."""
    rows = [row for _, _, records in datasets for row in records]
    versions = sorted({row["momentum_version"] for row in rows})
    decays = sorted({row["decay"] for row in rows})
    nesterov_choices = sorted({row["nesterov"] for row in rows}, reverse=True)
    columns = [(decay, nesterov) for decay in decays for nesterov in nesterov_choices]
    maximum = math.ceil(max(row[metric] for row in rows if math.isfinite(row[metric])) * 1000) / 10
    fig, axes = plt.subplots(
        len(versions), len(columns), figsize=(7 * len(columns), 4 * len(versions)),
        squeeze=False, sharex=True, sharey=True, layout="constrained",
    )
    fig.suptitle(f"Head optimizer comparison | {len(rows)} trials\nBest momentum at each LR", fontsize=15)
    label = "TTA validation accuracy (%)" if metric == "tta_val_acc" else "Validation accuracy (%)"
    for i, version in enumerate(versions):
        for j, (decay, nesterov) in enumerate(columns):
            ax = axes[i, j]
            for index, (name, _, records) in enumerate(datasets):
                subset = [
                    r for r in records if r["momentum_version"] == version
                    and r["decay"] == decay and r["nesterov"] == nesterov
                ]
                lrs = sorted({r["initial_lr"] for r in subset})
                scores = []
                for lr in lrs:
                    values = [r[metric] * 100 for r in subset if r["initial_lr"] == lr and math.isfinite(r[metric])]
                    scores.append(max(values) if values else np.nan)
                if lrs:
                    ax.plot(lrs, scores, marker="o", markersize=4, color=f"C{index % 10}", label=name)
            ax.set_title(
                f"{decay.replace('_', ' ').capitalize()} | momentum version {version}"
                f" | Nesterov {nesterov}"
            )
            ax.set_xscale("log")
            ax.set_xlabel("Head initial learning rate (log scale)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
    fig.supxlabel("Each point selects the best tested momentum. Missing optimizer/version combinations are omitted.", fontsize=10)
    save_figure(fig, directory, f"{metric}_comparison")
    axes[0, 0].set_ylim(floor, maximum + 0.1)
    fig.supxlabel(f"Zoomed to {floor:g}% and above. Each point selects the best tested momentum; lower scores appear in the full-range plot.", fontsize=10)
    save_figure(fig, directory, f"{metric}_comparison_zoom")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--run", action="append", help="Select a run name (repeatable); default: all runs")
    parser.add_argument("--metric", choices=("tta_val_acc", "val_acc"), default="tta_val_acc")
    parser.add_argument(
        "--heatmap-min", type=float, default=90.0,
        help="Color scale and curve zoom floor in percent; annotations remain exact",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = list(dict.fromkeys(path.resolve() for path in args.logs))
    datasets = []
    found = set()
    for path in paths:
        for run in run_names(path):
            if args.run and run not in args.run:
                continue
            config, rows = parse_log(path, run)
            found.add(run)
            algorithm = rows[0]["algorithm"]
            datasets.append((ALGORITHM_LABELS.get(algorithm, algorithm), config, rows))
    if args.run and set(args.run) - found:
        parser.error(f"Runs not found: {sorted(set(args.run) - found)}")
    if not datasets:
        parser.error("No completed grids found")
    datasets = merge_lr_extensions(datasets)
    # Make separate runs of the same optimizer distinguishable in the legend.
    names = [name for name, _, _ in datasets]
    datasets = [
        (f"{name} ({Path(rows[0]['source_log']).stem}, {rows[0]['run']})" if names.count(name) > 1 else name, config, rows)
        for name, config, rows in datasets
    ]
    rows = [row for _, _, records in datasets for row in records]
    directory = args.output_dir or Path("untracked_logs") / paths[-1].name / (
        "plots_combined" if len(paths) > 1 else "plots"
    )
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)
    for index, (_, config, records) in enumerate(datasets):
        run_directory = directory if len(datasets) == 1 else directory / f"{index + 1}_{records[0]['run']}"
        run_directory.mkdir(parents=True, exist_ok=True)
        plot_results(config, records, run_directory, args.metric, args.heatmap_min)
    if len(datasets) > 1:
        plot_comparison(datasets, directory, args.metric, args.heatmap_min)


if __name__ == "__main__":
    main()

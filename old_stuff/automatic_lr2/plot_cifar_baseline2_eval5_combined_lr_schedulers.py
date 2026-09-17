import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_cifar_baseline2_eval5_exp3 import (
    group_key as schedule_group_key,
    parse_log as parse_schedule_log,
    run_for_completion,
    runs_for_group as schedule_runs_for_group,
)
from plot_cifar_baseline2_eval5_no_multiplier import parse_log as parse_min_loss_log


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "cifar_baseline2_eval5_combined_lr_schedulers_plots"


@dataclass
class Curve:
    label: str
    source: str
    batch_size: int
    train_steps: int
    lrs: list[float]
    train_loss: float | None
    val_loss: float | None
    val_acc: float | None
    tta_val_acc: float | None
    run_index: int | None
    detail: str
    step_train_losses: list[float]


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def curve_key(curve):
    return curve.batch_size, curve.train_steps, curve.label


def run_lrs(run):
    return list(run.schedule or run.applied_lr or [])


def best_schedule_curve(label, source, path):
    runs, _search_steps, completions, _cache_hits = parse_schedule_log(path)
    curves = {}
    for completion in completions:
        run = run_for_completion(runs, completion)
        detail = f"selected_point={completion.best_point}"
        if run is None:
            group_runs = [
                row
                for row in schedule_runs_for_group(runs, schedule_group_key(completion))
                if row.train_loss is not None
            ]
            run = min(group_runs, key=lambda row: row.train_loss) if group_runs else None
            detail = f"selected_point={completion.best_point}; used lowest logged run"
        lrs = run_lrs(run) if run is not None else list(completion.lrs)
        train_steps, batch_size = schedule_group_key(completion)
        curves[(batch_size, train_steps)] = Curve(
            label=label,
            source=source,
            batch_size=batch_size,
            train_steps=train_steps,
            lrs=lrs,
            train_loss=run.train_loss if run is not None else completion.train_loss,
            val_loss=run.val_loss if run is not None else None,
            val_acc=run.val_acc if run is not None else None,
            tta_val_acc=run.tta_val_acc if run is not None else None,
            run_index=run.index if run is not None else None,
            detail=detail,
            step_train_losses=[],
        )
    return curves


def min_loss_curves(path):
    runs = parse_min_loss_log(path)
    curves = {}
    for run in runs:
        label = "min_loss_m%s" % fmt(run.muon_momentum)
        curves[(run.batch_size, run.train_steps, label)] = Curve(
            label=label,
            source=path.name,
            batch_size=run.batch_size,
            train_steps=run.train_steps,
            lrs=list(run.applied_lr),
            train_loss=run.train_loss,
            val_loss=run.val_loss,
            val_acc=run.val_acc,
            tta_val_acc=run.tta_val_acc,
            run_index=run.index,
            detail=f"momentum={fmt(run.muon_momentum)}, multiplier=1",
            step_train_losses=[entry.best_loss for entry in run.best_lr_logs],
        )
    return curves


def load_curves(paths):
    exp2_curves = best_schedule_curve(
        "schedule_search_m0.6",
        paths["exp2"].name,
        paths["exp2"],
    )
    exp3_curves = best_schedule_curve(
        "schedule_search_m0.6",
        paths["exp3"].name,
        paths["exp3"],
    )
    schedule_momentum_curves = {**exp2_curves, **exp3_curves}
    no_momentum_curves = best_schedule_curve(
        "schedule_search_m0",
        paths["no_momentum"].name,
        paths["no_momentum"],
    )
    min_loss_by_key = min_loss_curves(paths["no_multiplier"])

    curves = []
    curves.extend(schedule_momentum_curves.values())
    curves.extend(no_momentum_curves.values())
    curves.extend(min_loss_by_key.values())
    return sorted(curves, key=curve_key)


def plot_lr_schedulers(curves, output_path):
    batch_sizes = sorted({curve.batch_size for curve in curves})
    train_steps_list = sorted({curve.train_steps for curve in curves})
    by_group = {}
    for curve in curves:
        by_group.setdefault((curve.batch_size, curve.train_steps), []).append(curve)

    styles = {
        "schedule_search_m0.6": dict(color="tab:blue", marker="o", linewidth=2.1),
        "schedule_search_m0": dict(color="tab:orange", marker="s", linewidth=2.1),
        "min_loss_m0": dict(color="tab:green", marker="^", linewidth=1.8),
        "min_loss_m0.6": dict(color="tab:red", marker="D", linewidth=1.8),
    }

    fig, axes = plt.subplots(
        len(batch_sizes),
        len(train_steps_list),
        figsize=(4.3 * len(train_steps_list), 3.25 * len(batch_sizes)),
        squeeze=False,
        sharey=True,
    )

    for row, batch_size in enumerate(batch_sizes):
        for col, train_steps in enumerate(train_steps_list):
            ax = axes[row, col]
            group_curves = sorted(
                by_group.get((batch_size, train_steps), []),
                key=lambda curve: list(styles).index(curve.label),
            )
            if not group_curves:
                ax.set_visible(False)
                continue
            for curve in group_curves:
                steps = np.arange(1, len(curve.lrs) + 1)
                style = styles[curve.label]
                ax.plot(
                    steps,
                    curve.lrs,
                    label=f"{curve.label} loss={fmt(curve.train_loss)}",
                    **style,
                )
            if row == 0:
                ax.set_title(f"{train_steps} train steps")
            if col == 0:
                ax.set_ylabel(f"bs={batch_size}\nMuon LR")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=5.4)

    fig.suptitle("Selected LR Schedulers Across Eval5 Runs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_train_losses(curves, output_path):
    batch_sizes = sorted({curve.batch_size for curve in curves})
    train_steps_list = sorted({curve.train_steps for curve in curves})
    by_group = {}
    for curve in curves:
        by_group.setdefault((curve.batch_size, curve.train_steps), []).append(curve)

    styles = {
        "min_loss_m0": dict(color="tab:green", marker="^", linewidth=2.0),
        "min_loss_m0.6": dict(color="tab:red", marker="D", linewidth=2.0),
    }

    fig, axes = plt.subplots(
        len(batch_sizes),
        len(train_steps_list),
        figsize=(4.3 * len(train_steps_list), 3.25 * len(batch_sizes)),
        squeeze=False,
        sharey=False,
    )

    for row, batch_size in enumerate(batch_sizes):
        for col, train_steps in enumerate(train_steps_list):
            ax = axes[row, col]
            group_curves = [
                curve
                for curve in by_group.get((batch_size, train_steps), [])
                if curve.step_train_losses
            ]
            if not group_curves:
                ax.set_visible(False)
                continue
            for curve in sorted(group_curves, key=lambda row: row.label):
                steps = np.arange(1, len(curve.step_train_losses) + 1)
                ax.plot(
                    steps,
                    curve.step_train_losses,
                    label=f"{curve.label} final={fmt(curve.train_loss)}",
                    **styles[curve.label],
                )
            if row == 0:
                ax.set_title(f"{train_steps} train steps")
            if col == 0:
                ax.set_ylabel(f"bs={batch_size}\ntrain loss")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6.1)

    fig.suptitle("Per-Step min_loss Train Losses")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(curves, output_path):
    lines = [
        "CIFAR Baseline Eval5 Combined LR Schedulers",
        "=" * 47,
        "",
        "Curves per subplot:",
        "schedule_search_m0.6: selected schedule search result from exp2 for bs=500, exp3 otherwise",
        "schedule_search_m0: selected schedule search result from no_momentum",
        "min_loss_m0: no_multiplier direct min_loss run with Muon momentum 0",
        "min_loss_m0.6: no_multiplier direct min_loss run with Muon momentum 0.6",
        "",
        "Per-step train-loss traces are available for min_loss curves only.",
        "Schedule-search logs record selected final train loss, not per-step train loss.",
        "",
        f"Total selected curves: {len(curves)}",
        "",
    ]

    groups = sorted({(curve.batch_size, curve.train_steps) for curve in curves})
    for batch_size, train_steps in groups:
        group_curves = [curve for curve in curves if (curve.batch_size, curve.train_steps) == (batch_size, train_steps)]
        lines.extend(
            [
                f"batch_size={batch_size}, train_steps={train_steps}",
                "-" * 56,
            ]
        )
        for curve in sorted(group_curves, key=lambda row: row.label):
            lines.append(
                f"{curve.label}: source={curve.source} run={fmt(curve.run_index)} "
                f"train_loss={fmt(curve.train_loss)} "
                f"lrs={','.join(fmt(value) for value in curve.lrs)} "
                f"step_train_losses={','.join(fmt(value) for value in curve.step_train_losses) or 'NA'} "
                f"{curve.detail}"
            )
        if len(group_curves) != 4:
            lines.append(f"warning: expected 4 curves, found {len(group_curves)}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine Eval5 logs and plot selected LR schedulers."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exp2-log",
        type=Path,
        default=BASE_DIR / "cifar_baseline2_eval5_exp2.log",
    )
    parser.add_argument(
        "--exp3-log",
        type=Path,
        default=BASE_DIR / "cifar_baseline2_eval5_exp3.log",
    )
    parser.add_argument(
        "--no-momentum-log",
        type=Path,
        default=BASE_DIR / "cifar_baseline2_eval5_no_momentum.log",
    )
    parser.add_argument(
        "--no-multiplier-log",
        type=Path,
        default=BASE_DIR / "cifar_baseline2_eval5_no_multiplier.log",
    )
    args = parser.parse_args()

    paths = {
        "exp2": args.exp2_log,
        "exp3": args.exp3_log,
        "no_momentum": args.no_momentum_log,
        "no_multiplier": args.no_multiplier_log,
    }
    curves = load_curves(paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "lr_schedulers.png"
    loss_plot_path = args.output_dir / "per_step_train_losses.png"
    summary_path = args.output_dir / "summary.txt"
    plot_lr_schedulers(curves, plot_path)
    plot_step_train_losses(curves, loss_plot_path)
    write_summary(curves, summary_path)

    print(f"Selected {len(curves)} curves")
    print(plot_path)
    print(loss_plot_path)
    print(summary_path)


if __name__ == "__main__":
    main()

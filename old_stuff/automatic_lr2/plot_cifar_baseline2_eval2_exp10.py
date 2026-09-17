import argparse
from pathlib import Path

import plot_cifar_baseline2_eval2_exp8 as base


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp10.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp10_plots")

SCHEDULER_ORDER = {
    "fixed": 0,
    "constant": 1,
    "linear_2_to_0.1": 2,
    "linear": 3,
    "last2_linear": 4,
}


def run_kind(run):
    if run.best_lr_strategy is None:
        return "fixed"
    return getattr(run, "best_lr_scheduler", "constant")


def run_label(run):
    labels = {
        "fixed": "fixed",
        "constant": "best_lr const",
        "linear_2_to_0.1": "best_lr 2->0.1",
        "linear_2_to_0.01": "best_lr 2->0.1",
        "linear": "best_lr 1->0.1",
        "last2_linear": "best_lr last2",
    }
    return labels.get(run_kind(run), run_kind(run))


def sorted_runs(runs):
    return sorted(
        runs,
        key=lambda run: (
            run.batch_size,
            SCHEDULER_ORDER.get(run_kind(run), 99),
            run.index,
        ),
    )


def patch_labels():
    base.run_kind = run_kind
    base.run_label = run_label
    base.run_sorted = sorted_runs


def write_summary(runs, output_path):
    patch_labels()
    base.write_summary(runs, output_path)
    text = output_path.read_text()
    text = text.replace("CIFAR Baseline Eval2 Exp8 Results", "CIFAR Baseline Eval2 Exp10 Results")
    output_path.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 exp10 results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    patch_labels()
    runs = base.parse_exp8_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base.plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    base.plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    base.plot_lr_traces(runs, args.output_dir / "applied_lr_traces.png")
    base.plot_best_lr_traces(runs, args.output_dir / "best_lr_traces.png")
    base.plot_final_post_loss_distributions(
        runs, args.output_dir / "final_post_loss_distributions.png"
    )
    write_summary(runs, args.output_dir / "summary.txt")
    base.write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in sorted_runs(runs):
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        print(
            f"run={run.index} name={run.name} kind={run_label(run)} "
            f"bs={run.batch_size} steps={len(run.applied_lrs)}/{total_steps} "
            f"tta={run.tta_val_acc} wall={getattr(run, 'wall_time_seconds', None)}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

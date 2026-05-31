from pathlib import Path
import argparse

import plot_cifar_baseline2_eval2_n_search as plotter


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_n_search4.log")
DEFAULT_BASELINE_LOG = Path(__file__).with_name("cifar_baseline2_eval2_n_search2.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_n_search4_plots")


def baseline_runs_for(primary_runs, baseline_log):
    batch_sizes = {run.batch_size for run in primary_runs}
    return [
        run
        for run in plotter.parse_log(baseline_log)
        if run.strategy is None and run.batch_size in batch_sizes
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 n_search4 results with baseline."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--baseline-log", type=Path, default=DEFAULT_BASELINE_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = plotter.parse_log(args.log)
    baseline_runs = baseline_runs_for(runs, args.baseline_log)
    runs = baseline_runs + runs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plotter.plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    plotter.plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    plotter.plot_lr_schedules(runs, args.output_dir / "applied_lr_schedules.png")
    plotter.plot_n_search_step_train_losses(
        runs, args.output_dir / "n_search_step_train_losses.png"
    )
    plotter.plot_n_search_selected_lrs(
        runs, args.output_dir / "n_search_selected_lrs.png"
    )
    plotter.write_summary(runs, args.output_dir / "summary.txt")
    plotter.write_csv(runs, args.output_dir / "summary.csv")

    print(
        f"Parsed {len(runs) - len(baseline_runs)} runs from {args.log} "
        f"and {len(baseline_runs)} baseline run from {args.baseline_log}"
    )
    print(f"Wrote plots and summaries to {args.output_dir}")
    for run in plotter.sorted_runs(runs):
        print(
            f"run={run.index} bs={run.batch_size} kind={run.label} "
            f"train_loss={plotter.fmt(run.train_loss)} "
            f"val_acc={plotter.fmt(run.val_acc)} tta={plotter.fmt(run.tta_val_acc)}"
        )


if __name__ == "__main__":
    main()

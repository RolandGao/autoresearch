import argparse
from pathlib import Path

import plot_cifar_baseline2_eval2_exp3 as base


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_exp4.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp4_plots")


def write_progress_summary(runs, output_path):
    lines = [
        "run,name,batch_size,steps,total_steps,progress,best_lr_logs,"
        "eval_points,is_complete,final_val_acc,final_tta_val_acc"
    ]
    for run in runs:
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        steps = len(run.applied_lrs)
        progress = ""
        if total_steps:
            progress = "%.8g" % (steps / total_steps)
        is_complete = bool(run.tta_val_acc is not None)
        fields = [
            run.index,
            run.name,
            run.batch_size,
            steps,
            total_steps,
            progress,
            len(run.best_lr_logs),
            len(run.eval_logs),
            is_complete,
            "" if run.val_acc is None else "%.8g" % run.val_acc,
            "" if run.tta_val_acc is None else "%.8g" % run.tta_val_acc,
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval2 exp4 results, including partial logs."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = base.parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base.plot_final_metrics(runs, args.output_dir / "final_metrics.png")
    base.plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    base.plot_applied_lr_traces(runs, args.output_dir / "applied_lr_traces.png")
    base.plot_best_lr_traces(runs, args.output_dir / "best_lr_traces.png")
    base.plot_lr_landscape_snapshots(
        runs, args.output_dir / "best_lr_landscapes.png"
    )
    base.write_summary(runs, args.output_dir / "summary.csv")
    write_progress_summary(runs, args.output_dir / "progress_summary.csv")

    print(f"Parsed {len(runs)} runs from {args.log}")
    for run in runs:
        total_steps = max((row.total_steps for row in run.applied_lrs), default="")
        progress = ""
        if total_steps:
            progress = f"{len(run.applied_lrs) / total_steps:.3f}"
        print(
            f"run={run.index} name={run.name} steps={len(run.applied_lrs)}/"
            f"{total_steps} progress={progress} best_lr_logs={len(run.best_lr_logs)} "
            f"complete={run.tta_val_acc is not None}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import plot_cifar_baseline2_eval2_exp3 as base


DEFAULT_LOGS = [
    ("exp3", Path(__file__).with_name("cifar_baseline2_eval2_exp3.log")),
    ("exp4", Path(__file__).with_name("cifar_baseline2_eval2_exp4.log")),
]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval2_exp3_exp4_plots")


def parse_log_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("logs must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("log label cannot be empty")
    return label, Path(path)


def parse_combined_logs(logs):
    combined = []
    for source, path in logs:
        runs = base.parse_log(path)
        for run in runs:
            run.index = len(combined)
            run.name = f"{run.name}_{source}"
            combined.append(run)
    return combined


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
        fields = [
            run.index,
            run.name,
            run.batch_size,
            steps,
            total_steps,
            progress,
            len(run.best_lr_logs),
            len(run.eval_logs),
            run.tta_val_acc is not None,
            "" if run.val_acc is None else "%.8g" % run.val_acc,
            "" if run.tta_val_acc is None else "%.8g" % run.tta_val_acc,
        ]
        lines.append(",".join(str(field) for field in fields))
    output_path.write_text("\n".join(lines) + "\n")


def fmt(value):
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def write_important_info(runs, output_path):
    complete_runs = [run for run in runs if run.tta_val_acc is not None]
    ranked = sorted(
        complete_runs,
        key=lambda run: (
            -run.tta_val_acc,
            -(run.val_acc if run.val_acc is not None else float("-inf")),
        ),
    )
    lines = [
        "Combined exp3 + exp4 CIFAR Baseline Results",
        "=" * 46,
        "",
        f"Runs: {len(runs)}",
        f"Complete runs: {len(complete_runs)}",
        "",
    ]

    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "Best overall by TTA val acc",
                "-" * 28,
                f"Name: {best.name}",
                f"Batch size: {best.batch_size}",
                f"TTA val acc: {fmt(best.tta_val_acc)}",
                f"Val acc: {fmt(best.val_acc)}",
                f"25-batch train loss: {fmt(best.train25_loss)}",
                "",
                "Best by batch size",
                "-" * 18,
            ]
        )
        for batch_size in sorted({run.batch_size for run in complete_runs}):
            batch_runs = [run for run in complete_runs if run.batch_size == batch_size]
            batch_best = max(batch_runs, key=lambda run: run.tta_val_acc)
            lines.extend(
                [
                    f"batch_size={batch_size}",
                    f"  name: {batch_best.name}",
                    f"  TTA val acc: {fmt(batch_best.tta_val_acc)}",
                    f"  val acc: {fmt(batch_best.val_acc)}",
                ]
            )
        lines.append("")

    lines.extend(["Rankings by batch size", "-" * 22])
    for batch_size in sorted({run.batch_size for run in complete_runs}):
        batch_runs = [
            run for run in ranked if run.batch_size == batch_size
        ]
        lines.extend(["", f"batch_size={batch_size}"])
        for rank, run in enumerate(batch_runs, start=1):
            final_applied_lr = run.applied_lrs[-1].lr if run.applied_lrs else ""
            final_searched_lr = (
                run.best_lr_logs[-1].searched_lr if run.best_lr_logs else ""
            )
            final_best_lr_ema = (
                run.best_lr_logs[-1].best_lr_ema if run.best_lr_logs else ""
            )
            lines.extend(
                [
                    f"{rank}. {run.name}",
                    f"   TTA val acc: {fmt(run.tta_val_acc)}",
                    f"   Val acc: {fmt(run.val_acc)}",
                    f"   25-batch train loss: {fmt(run.train25_loss)}",
                    f"   Strategy: {run.best_lr_strategy or 'fixed'}",
                    f"   Linear decay: {run.best_lr_linear_decay}",
                    f"   Base Muon LR: {fmt(run.muon_lr)}",
                    f"   Steps: {len(run.applied_lrs)}",
                    f"   Final applied LR: {fmt(final_applied_lr)}",
                    f"   Final searched LR: {fmt(final_searched_lr)}",
                    f"   Final best-LR EMA: {fmt(final_best_lr_ema)}",
                ]
            )

    incomplete_runs = [run for run in runs if run.tta_val_acc is None]
    if incomplete_runs:
        lines.extend(["", "Incomplete runs", "-" * 15])
        for run in incomplete_runs:
            total_steps = max((row.total_steps for row in run.applied_lrs), default="")
            lines.append(f"{run.name}: steps={len(run.applied_lrs)}/{total_steps}")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined cifar_baseline2_eval2 exp3 and exp4 results."
    )
    parser.add_argument(
        "--log",
        action="append",
        type=parse_log_arg,
        dest="logs",
        help="Input log in LABEL=PATH form. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logs = args.logs if args.logs else DEFAULT_LOGS
    runs = parse_combined_logs(logs)
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
    write_important_info(runs, args.output_dir / "important_info.txt")

    print(f"Parsed {len(runs)} runs from {len(logs)} logs")
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

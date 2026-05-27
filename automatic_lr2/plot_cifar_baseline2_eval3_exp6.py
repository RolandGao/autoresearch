import argparse
from pathlib import Path

import plot_cifar_baseline2_eval3_exp1 as base


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval3_exp6.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval3_exp6_plots")


def write_summary(runs, search_steps, completions, cache_hits, output_path):
    ordered = sorted(runs, key=lambda run: (run.batch_size, run.index))
    lines = [
        "CIFAR Baseline Eval3 Exp6 Joint Line Search",
        "=" * 45,
        "",
        "Muon LR and SGD LR multiplier are rounded to 2 significant figures before running.",
        "",
        f"Unique evaluated runs: {len(runs)}",
        f"Search steps: {len(search_steps)}",
        f"Cache hits: {len(cache_hits)}",
        "",
    ]

    for completion in sorted(completions, key=lambda row: row.batch_size):
        lines.extend(
            [
                f"batch_size={completion.batch_size} selected best",
                "-" * 35,
                f"point: {base.point_label(completion.best_point)}",
                f"muon_lr: {base.fmt(completion.muon_lr)}",
                f"sgd_lr_mult: {base.fmt(completion.sgd_lr_mult)}",
                f"TTA val acc: {base.fmt(completion.tta_val_acc)}",
                f"evaluated points: {completion.evaluated_points}",
                "",
            ]
        )

    headers = [
        "run",
        "batch",
        "point",
        "muon_lr",
        "sgd_mult",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "tta_val_acc",
        "wall_s",
    ]
    rows = [
        [
            run.index,
            run.batch_size,
            base.point_label(run.point),
            base.fmt(run.muon_lr),
            base.fmt(run.sgd_lr_mult),
            base.fmt(run.train_loss),
            base.fmt(run.val_loss),
            base.fmt(run.train_acc),
            base.fmt(run.val_acc),
            base.fmt(run.tta_val_acc),
            base.fmt(run.wall_time_seconds),
        ]
        for run in ordered
    ]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]

    lines.extend(["All Evaluated Points", "-" * 20])
    lines.append(
        "  ".join(str(value).ljust(width) for value, width in zip(headers, widths))
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(str(value).ljust(width) for value, width in zip(row, widths))
        )
    lines.append("")

    for batch_size in base.batch_sizes(runs):
        ranked = sorted(
            base.runs_for_batch(runs, batch_size),
            key=lambda run: run.tta_val_acc if run.tta_val_acc is not None else -1,
            reverse=True,
        )
        lines.extend([f"Ranking, batch_size={batch_size}", "-" * 24])
        for rank, run in enumerate(ranked, start=1):
            lines.append(
                f"{rank}. r{run.index} point={base.point_label(run.point)} "
                f"muon_lr={base.fmt(run.muon_lr)} "
                f"sgd_lr_mult={base.fmt(run.sgd_lr_mult)} "
                f"tta_val_acc={base.fmt(run.tta_val_acc)} "
                f"val_acc={base.fmt(run.val_acc)} "
                f"train_loss={base.fmt(run.train_loss)}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_eval3 exp6 joint line-search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs, search_steps, completions, cache_hits = base.parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base.plot_search_surface(
        runs, search_steps, completions, args.output_dir / "search_surface.png"
    )
    base.plot_metric_heatmaps(runs, args.output_dir / "tta_heatmap.png")
    base.plot_search_progress(
        search_steps, completions, args.output_dir / "search_progress.png"
    )
    base.plot_final_metrics(runs, args.output_dir / "final_metrics_by_run.png")
    base.plot_eval_curves(runs, args.output_dir / "eval_curves.png")
    write_summary(
        runs,
        search_steps,
        completions,
        cache_hits,
        args.output_dir / "summary.txt",
    )
    base.write_csv(runs, args.output_dir / "summary.csv")

    print(f"Parsed {len(runs)} unique evaluated runs from {args.log}")
    print(f"Parsed {len(search_steps)} search steps and {len(cache_hits)} cache hits")
    for completion in sorted(completions, key=lambda row: row.batch_size):
        print(
            f"batch_size={completion.batch_size} "
            f"best={base.point_label(completion.best_point)} "
            f"muon_lr={completion.muon_lr:.6g} "
            f"sgd_lr_mult={completion.sgd_lr_mult:.6g} "
            f"tta={completion.tta_val_acc:.4f} "
            f"evaluated={completion.evaluated_points}"
        )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()

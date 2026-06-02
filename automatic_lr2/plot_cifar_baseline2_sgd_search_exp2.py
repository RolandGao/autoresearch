from pathlib import Path
import argparse

import plot_cifar_baseline2_sgd_search_exp1 as plotter


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_search_exp2.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_search_exp2_plots")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_search_exp2 LR search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    searches = plotter.parse_log(args.log)
    if not searches:
        raise SystemExit(f"No LR searches parsed from {args.log}")

    plotter.plot_all(searches, args.output_dir)
    best = max(
        (search.best_eval for search in searches), key=lambda row: row.tta_val_acc
    )
    print(f"Parsed {len(searches)} LR searches from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best overall: search={best.search} bs={best.batch_size} "
        f"update={best.label} lr={best.lr:g} tta={best.tta_val_acc:.4f}"
    )


if __name__ == "__main__":
    main()

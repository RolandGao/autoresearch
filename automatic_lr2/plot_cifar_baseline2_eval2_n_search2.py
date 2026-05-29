from pathlib import Path

import plot_cifar_baseline2_eval2_n_search as plotter


plotter.DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval2_n_search2.log")
plotter.DEFAULT_OUTPUT_DIR = Path(__file__).with_name(
    "cifar_baseline2_eval2_n_search2_plots"
)


if __name__ == "__main__":
    plotter.main()

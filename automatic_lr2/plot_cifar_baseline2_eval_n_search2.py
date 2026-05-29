from pathlib import Path

from plot_cifar_baseline2_eval_n_search import main


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_eval_n_search2.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_eval_n_search2_plots")


if __name__ == "__main__":
    main(default_log=DEFAULT_LOG, default_output_dir=DEFAULT_OUTPUT_DIR)

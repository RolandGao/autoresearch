from pathlib import Path

from plot_cifar_baseline2_eval_constant_lr import main


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    main(
        default_log=script_dir / "cifar_baseline2_eval_constant_lr_exp2.log",
        default_output_dir=script_dir / "cifar_baseline2_eval_constant_lr_exp2_plots",
    )

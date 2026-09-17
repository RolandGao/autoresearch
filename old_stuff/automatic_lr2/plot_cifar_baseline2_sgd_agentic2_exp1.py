from __future__ import annotations

import argparse
from pathlib import Path

import plot_cifar_baseline2_sgd_agentic_exp1 as agentic_plotter


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_agentic2_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_agentic2_exp1_plots")

UPDATE_LABELS = {
    "track3_muon": "Track3 Muon",
    "track3_soft_muon_p01": "Soft Muon p=0.1",
    "track3_contra_muon": "Contra Muon",
    "track3_contra_to_soft_muon": "Contra to Soft Muon",
    "normon_muon": "NorMuon",
    "normon_soft_muon": "NorMuon Soft",
    "normon_contra_to_soft_muon": "NorMuon Contra to Soft",
    "aurora_row_balanced_muon": "Aurora row balanced",
    "aurora_half_balanced_muon": "Aurora half balanced",
    "aurora_normon_muon": "Aurora NorMuon",
    "muown_row_control_muon": "Muown row control",
    "muown_normon_muon": "Muown NorMuon",
    "soap_muon": "SOAP Muon",
    "soap_normon_muon": "SOAP NorMuon",
    "soap_contra_soft_normon": "SOAP Contra Soft NorMuon",
    "sinksoap_normon_muon": "SinkSOAP NorMuon",
    "kl_soap_muon": "KL-SOAP Muon",
    "shampoo_muon": "Shampoo Muon",
    "radial_brake_soft_normon": "Radial brake Soft NorMuon",
    "soda_contra_soft_normon": "SODA Contra Soft NorMuon",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_agentic2_exp1 LR search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    agentic_plotter.UPDATE_LABELS.clear()
    agentic_plotter.UPDATE_LABELS.update(UPDATE_LABELS)

    searches = agentic_plotter.search_plotter.parse_log(args.log)
    if not searches:
        raise SystemExit(f"No LR searches parsed from {args.log}")

    agentic_plotter.plot_all(searches, args.output_dir, top_n=args.top_n)
    best = agentic_plotter.best_rows(searches)[0]
    print(f"Parsed {len(searches)} LR searches from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best overall: search={best['search']} update={best['label']} "
        f"lr={best['best_lr']:g} tta={best['tta_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()

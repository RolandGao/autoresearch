from __future__ import annotations

import argparse
from pathlib import Path

import plot_cifar_baseline2_sgd_agentic_exp1 as agentic_plotter


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_agentic3_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_agentic3_exp1_plots")

UPDATE_LABELS = {
    "ns4_track3_blend25": "NS4 + Track3 25%",
    "ns4_track3_blend50": "NS4 + Track3 50%",
    "ns4_track3_blend75": "NS4 + Track3 75%",
    "ns4_aurora_half_blend50": "NS4 Aurora half + Track3",
    "ns4_aurora_quarter": "NS4 Aurora quarter",
    "ns4_aurora_threequarter": "NS4 Aurora three-quarter",
    "ns4_centered_aurora_half": "NS4 centered Aurora half",
    "ns4_normon_beta90": "NS4 NorMuon beta=0.90",
    "ns4_aurora_half_normon": "NS4 Aurora half NorMuon",
    "ns5_muon": "NS5 Muon",
    "ns5_aurora_half": "NS5 Aurora half",
    "ns6_muon": "NS6 Muon",
    "ns3_aurora_half": "NS3 Aurora half",
    "soft_aurora_quarter": "Soft Aurora quarter",
    "soft_aurora_half": "Soft Aurora half",
    "soft_aurora_threequarter": "Soft Aurora three-quarter",
    "soft_aurora_half_normon": "Soft Aurora half NorMuon",
    "contra_soft_aurora_half": "Contra Soft Aurora half",
    "contra_fast_soft_aurora_half": "Fast Contra Soft Aurora half",
    "ns4_qr_aurora_half_blend": "NS4 QR Aurora half blend",
    "adamw_precond_norm": "AdamW preconditioned",
    "adamh_radial_precond": "AdamH radial preconditioned",
    "adafactor_norm": "Adafactor",
    "adafactor_clipped_norm": "Clipped Adafactor",
    "signum_norm": "Signum",
    "lion_sign_norm": "Lion sign",
    "signum_rms_hybrid": "Signum RMS hybrid",
    "raw_shampoo_norm": "Raw Shampoo",
    "spectral_descent_raw": "Spectral Descent",
    "raw_soap_norm": "Raw SOAP",
    "raw_kl_soap_norm": "Raw KL-SOAP",
    "sinksoap_raw_norm": "Raw SinkSOAP",
    "psgd_kron_whiten_norm": "PSGD Kronecker whitening",
    "pmuon_bilateral_power_raw": "PMuon bilateral power",
    "right_newton_cov_raw": "Right Newton covariance",
    "left_newton_cov_raw": "Left Newton covariance",
    "adam_soda_anchor": "Adam SODA anchor",
    "adam_radial_brake": "Adam radial brake",
    "muloco_update_extrap": "MuLoCo extrapolation",
    "late_rre_update_extrap": "Late RRE extrapolation",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_agentic3_exp1 LR search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    agentic_plotter.UPDATE_LABELS.clear()
    agentic_plotter.UPDATE_LABELS.update(UPDATE_LABELS)

    searches = agentic_plotter.search_plotter.parse_log(args.log)
    if not searches:
        raise SystemExit(f"No LR searches parsed from {args.log}")

    plot_all(searches, args.output_dir, top_n=args.top_n)
    best = agentic_plotter.best_rows(searches)[0]
    print(f"Parsed {len(searches)} LR searches from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best overall: search={best['search']} update={best['label']} "
        f"lr={best['best_lr']:g} tta={best['tta_val_acc']:.4f}"
    )


def plot_all(searches, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    agentic_plotter.plot_lr_search_grid(searches, output_dir / "lr_search_grid.png")
    agentic_plotter.plot_best_leaderboard(searches, output_dir / "best_leaderboard.png")
    agentic_plotter.plot_top_epoch_curves(
        searches, output_dir / "top_epoch_curves.png", top_n=top_n
    )
    agentic_plotter.plot_loss_accuracy_scatter(
        searches, output_dir / "train_loss_vs_tta.png"
    )
    plot_family_leaderboard(searches, output_dir / "family_leaderboard.png")
    agentic_plotter.search_plotter.write_csv(
        searches, output_dir / "lr_search_results.csv"
    )
    agentic_plotter.write_best_csv(searches, output_dir / "best_results.csv")
    agentic_plotter.write_summary(searches, output_dir / "summary.md")


def family_for(update: str) -> str:
    if update.startswith(("ns", "soft_", "contra_")):
        return "Muon/Aurora"
    if "soap" in update:
        return "SOAP"
    if "shampoo" in update or "spectral" in update:
        return "Shampoo"
    if update.startswith(("adam", "adafactor")):
        return "Adam/Adafactor"
    if "sign" in update or "lion" in update:
        return "Sign"
    if "psgd" in update or "pmuon" in update or "newton_cov" in update:
        return "Covariance"
    if "muloco" in update or "rre" in update:
        return "Extrapolation"
    return "Other"


def plot_family_leaderboard(searches, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = agentic_plotter.best_rows(searches)
    families = sorted({family_for(row["update"]) for row in rows})
    family_best = []
    for family in families:
        items = [row for row in rows if family_for(row["update"]) == family]
        best = max(items, key=lambda row: row["tta_val_acc"])
        family_best.append((family, best))
    family_best.sort(key=lambda item: item[1]["tta_val_acc"], reverse=True)

    labels = [family for family, _ in family_best]
    values = [row["tta_val_acc"] for _, row in family_best]
    methods = [row["label"] for _, row in family_best]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(labels, values, color=plt.get_cmap("tab10").colors[: len(labels)])
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    ax.set_ylabel("best TTA val accuracy")
    ax.set_title("Best Method by Family")
    ax.grid(axis="y", alpha=0.25)
    ymin = min(values)
    ymax = max(values)
    ax.set_ylim(max(0.0, ymin - 0.01), min(1.0, ymax + 0.01))
    ax.tick_params(axis="x", rotation=25)
    for i, method in enumerate(methods):
        ax.text(i, values[i] - 0.002, method, ha="center", va="top", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_cifar_baseline2_sgd_search_exp1 as search_plotter


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_agentic_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_agentic_exp1_plots")

UPDATE_LABELS = {
    "column_norm": "column norm",
    "column_norm_max": "column norm max",
    "sinkhorn_rows_first_norm": "sinkhorn rows first",
    "sinkhorn_columns_first_norm": "sinkhorn columns first",
    "row_centered_row_norm": "row centered rows",
    "column_centered_column_norm": "column centered columns",
    "double_centered_matrix_norm": "double centered matrix",
    "signed_sqrt_matrix_norm": "signed sqrt",
    "signed_cuberoot_matrix_norm": "signed cuberoot",
    "signed_square_matrix_norm": "signed square",
    "softsign_matrix_norm": "softsign",
    "tanh_matrix_norm": "tanh",
    "factorized_rms_norm": "factorized RMS",
    "inverse_factorized_rms_norm": "inverse factorized RMS",
    "row_norm_sqrt_weighted": "row sqrt weighted",
    "row_norm_inv_sqrt_weighted": "row inv-sqrt weighted",
    "qr_row_orthogonal": "QR row orthogonal",
    "zeropower_via_newtonschulz5_steps1": "Newton-Schulz steps=1",
    "zeropower_via_newtonschulz5_steps4": "Newton-Schulz steps=4",
    "zeropower_double_centered": "Newton-Schulz centered",
}


def label(update: str) -> str:
    return UPDATE_LABELS.get(update, update)


def color_map(searches):
    cmap = plt.get_cmap("tab20")
    ordered = [search.update for search in sorted(searches, key=lambda row: row.search)]
    return {update: cmap(i % cmap.N) for i, update in enumerate(ordered)}


def best_rows(searches):
    rows = []
    for search in searches:
        best = search.best_eval
        rows.append(
            {
                "search": search.search,
                "batch_size": search.batch_size,
                "update": search.update,
                "label": label(search.update),
                "initial_lr": search.initial_lr,
                "best_k": best.k,
                "best_lr": best.lr,
                "train25_loss": best.train25_loss,
                "val_acc": best.val_acc,
                "tta_val_acc": best.tta_val_acc,
                "time_seconds": best.time_seconds,
                "evaluated_lrs": len(search.evals),
            }
        )
    return sorted(rows, key=lambda row: row["tta_val_acc"], reverse=True)


def plot_lr_search_grid(searches, output_path: Path) -> None:
    colors = color_map(searches)
    ncols = 5
    nrows = math.ceil(len(searches) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.reshape(nrows, ncols)

    all_tta = [row.tta_val_acc for search in searches for row in search.evals]
    ymin = min(all_tta)
    ymax = max(all_tta)
    ypad = max(0.004, (ymax - ymin) * 0.10)

    for ax, search in zip(axes.flat, sorted(searches, key=lambda row: row.search)):
        points = sorted(search.evals, key=lambda row: row.k)
        xs = [row.k for row in points]
        ys = [row.tta_val_acc for row in points]
        best = search.best_eval
        color = colors[search.update]

        ax.plot(xs, ys, marker="o", linewidth=1.7, color=color)
        ax.scatter(
            [best.k],
            [best.tta_val_acc],
            marker="*",
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
        ax.axvline(0, color="0.65", linestyle="--", linewidth=0.9)
        ax.set_title(fill(label(search.update), width=24), fontsize=10)
        ax.set_xlabel("k")
        ax.grid(alpha=0.25)
        ax.set_ylim(max(0.0, ymin - ypad), min(1.0, ymax + ypad))
        ax.annotate(
            f"best lr={best.lr:g}\ntta={best.tta_val_acc:.4f}",
            xy=(0.03, 0.05),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.85),
        )

    for ax in axes.flat[len(searches) :]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("TTA val accuracy")

    fig.suptitle("Agentic SGD Update LR Search")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_leaderboard(searches, output_path: Path) -> None:
    rows = best_rows(searches)
    colors = color_map(searches)
    y = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    tta_values = [row["tta_val_acc"] for row in rows]
    lr_values = [row["best_lr"] for row in rows]
    bar_colors = [colors[row["update"]] for row in rows]
    labels = [row["label"] for row in rows]

    axes[0].barh(y, tta_values, color=bar_colors, alpha=0.9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("best TTA val accuracy")
    axes[0].set_title("Sorted by best TTA")
    axes[0].grid(axis="x", alpha=0.25)
    xmin = min(tta_values)
    xmax = max(tta_values)
    axes[0].set_xlim(max(0.0, xmin - 0.01), min(1.0, xmax + 0.01))
    for i, value in enumerate(tta_values):
        axes[0].text(value + 0.001, i, f"{value:.4f}", va="center", fontsize=8)

    axes[1].barh(y, lr_values, color=bar_colors, alpha=0.9)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("selected initial Muon LR")
    axes[1].set_title("Selected LR")
    axes[1].grid(axis="x", alpha=0.25)
    for i, value in enumerate(lr_values):
        axes[1].text(value, i, f" {value:g}", va="center", fontsize=8)

    fig.suptitle("Agentic SGD Update Leaderboard")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_epoch_curves(searches, output_path: Path, top_n: int = 8) -> None:
    rows = best_rows(searches)[:top_n]
    by_update = {search.update: search for search in searches}
    colors = color_map(searches)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for row in rows:
        search = by_update[row["update"]]
        best = search.best_eval
        epochs = [point["epoch"] for point in best.epoch_evals]
        vals = [point["val_acc"] for point in best.epoch_evals]
        ax.plot(
            epochs,
            vals,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=colors[search.update],
            label=f"{label(search.update)} lr={best.lr:g} tta={best.tta_val_acc:.4f}",
        )

    ax.set_title(f"Validation Accuracy by Epoch, Top {len(rows)} Selected LRs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_accuracy_scatter(searches, output_path: Path) -> None:
    rows = best_rows(searches)
    colors = color_map(searches)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for row in rows:
        ax.scatter(
            row["train25_loss"],
            row["tta_val_acc"],
            color=colors[row["update"]],
            s=70,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.annotate(
            str(row["search"]),
            (row["train25_loss"], row["tta_val_acc"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
        )

    ax.set_title("Best LR: 25-Batch Train Loss vs TTA Accuracy")
    ax.set_xlabel("25-batch train loss")
    ax.set_ylabel("TTA val accuracy")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_best_csv(searches, output_path: Path) -> None:
    rows = best_rows(searches)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(searches, output_path: Path) -> None:
    rows = best_rows(searches)
    lines = [
        "rank | search | update | initial_lr | best_k | best_lr | val_acc | tta_val_acc | train25_loss | evals",
        "---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank} | {row['search']} | {row['label']} | "
            f"{row['initial_lr']:g} | {row['best_k']:g} | {row['best_lr']:g} | "
            f"{row['val_acc']:.4f} | {row['tta_val_acc']:.4f} | "
            f"{row['train25_loss']:.4f} | {row['evaluated_lrs']}"
        )
    best = rows[0]
    lines.extend(
        [
            "",
            "Best overall:",
            (
                f"search={best['search']} update={best['label']} "
                f"lr={best['best_lr']:g} tta_val_acc={best['tta_val_acc']:.4f}"
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def plot_all(searches, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lr_search_grid(searches, output_dir / "lr_search_grid.png")
    plot_best_leaderboard(searches, output_dir / "best_leaderboard.png")
    plot_top_epoch_curves(searches, output_dir / "top_epoch_curves.png", top_n=top_n)
    plot_loss_accuracy_scatter(searches, output_dir / "train_loss_vs_tta.png")
    search_plotter.write_csv(searches, output_dir / "lr_search_results.csv")
    write_best_csv(searches, output_dir / "best_results.csv")
    write_summary(searches, output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_agentic_exp1 LR search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    searches = search_plotter.parse_log(args.log)
    if not searches:
        raise SystemExit(f"No LR searches parsed from {args.log}")

    plot_all(searches, args.output_dir, top_n=args.top_n)
    best = best_rows(searches)[0]
    print(f"Parsed {len(searches)} LR searches from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best overall: search={best['search']} update={best['label']} "
        f"lr={best['best_lr']:g} tta={best['tta_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()

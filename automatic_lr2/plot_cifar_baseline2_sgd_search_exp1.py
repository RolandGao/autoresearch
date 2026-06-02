from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).with_name("cifar_baseline2_sgd_search_exp1.log")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("cifar_baseline2_sgd_search_exp1_plots")

SEARCH_RE = re.compile(
    r"^cifar_baseline2_lr_search search=(?P<search>\d+) "
    r"batch_size=(?P<batch_size>\d+) initial_muon_lr=(?P<initial_lr>\S+) "
    r"update=(?P<update>\S+)"
)
EVAL_START_RE = re.compile(
    r"^lr_search_eval search(?P<search>\d+)_bs(?P<batch_size>\d+)_"
    r"update_(?P<update>.+)_k(?P<k>[-+.\deE]+)_lr(?P<lr>\S+) "
    r"initial_lr=(?P<initial_lr>\S+) rounded_lr=(?P<rounded_lr>\S+)"
)
CACHE_HIT_RE = re.compile(
    r"^lr_search_cache_hit search=(?P<search>\d+) batch_size=(?P<batch_size>\d+) "
    r"update=(?P<update>\S+) k=(?P<k>[-+.\deE]+) rounded_lr=(?P<rounded_lr>\S+)"
)
EPOCH_EVAL_RE = re.compile(
    r"^eval(?: run=\S+)? epoch=(?P<epoch>\d+) "
    r"val_acc=(?P<val_acc>\S+) time_seconds=(?P<time_seconds>\S+)"
)
FINAL_RE = re.compile(
    r"^eval epoch=final 25batch_train_loss=(?P<train25_loss>\S+) "
    r"val_acc=(?P<val_acc>\S+) tta_val_acc=(?P<tta_val_acc>\S+) "
    r"time_seconds=(?P<time_seconds>\S+)"
)
COMPLETE_RE = re.compile(
    r"^lr_search_complete search=(?P<search>\d+) batch_size=(?P<batch_size>\d+) "
    r"update=(?P<update>\S+) initial_lr=(?P<initial_lr>\S+) "
    r"best_k=(?P<best_k>[-+.\deE]+) best_lr=(?P<best_lr>\S+) "
    r"tta_val_acc=(?P<tta_val_acc>\S+) evaluated_lrs=(?P<evaluated_lrs>\d+)"
)

UPDATE_LABELS = {
    "row_norm": "row norm",
    "row_norm_max": "row norm max",
    "matrix_norm": "matrix norm",
    "matrix_norm_max": "matrix norm max",
    "zeropower_via_newtonschulz5": "Newton-Schulz",
    "zeropower_via_newtonschulz5_max": "Newton-Schulz max",
}
UPDATE_COLORS = {
    "row_norm": "tab:blue",
    "row_norm_max": "tab:cyan",
    "matrix_norm": "tab:green",
    "matrix_norm_max": "tab:olive",
    "zeropower_via_newtonschulz5": "tab:orange",
    "zeropower_via_newtonschulz5_max": "tab:red",
}


@dataclass
class LrEval:
    search: int
    batch_size: int
    update: str
    k: float
    lr: float
    initial_lr: float
    train25_loss: float = math.nan
    val_acc: float = math.nan
    tta_val_acc: float = math.nan
    time_seconds: float = math.nan
    epoch_evals: list[dict[str, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return UPDATE_LABELS.get(self.update, self.update)


@dataclass
class SearchRun:
    search: int
    batch_size: int
    update: str
    initial_lr: float
    evals: list[LrEval] = field(default_factory=list)
    cache_hits: list[dict[str, float | str]] = field(default_factory=list)
    best_k: float = math.nan
    best_lr: float = math.nan
    best_tta_val_acc: float = math.nan
    evaluated_lrs: int = 0

    @property
    def label(self) -> str:
        return UPDATE_LABELS.get(self.update, self.update)

    @property
    def best_eval(self) -> LrEval:
        return max(self.evals, key=lambda row: row.tta_val_acc)


def parse_float(value: str) -> float:
    return float(value)


def parse_log(path: Path) -> list[SearchRun]:
    searches: dict[int, SearchRun] = {}
    current_eval: LrEval | None = None

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if match := SEARCH_RE.match(line):
                search = int(match["search"])
                searches[search] = SearchRun(
                    search=search,
                    batch_size=int(match["batch_size"]),
                    update=match["update"],
                    initial_lr=parse_float(match["initial_lr"]),
                )
                current_eval = None
                continue

            if match := EVAL_START_RE.match(line):
                search = int(match["search"])
                current_eval = LrEval(
                    search=search,
                    batch_size=int(match["batch_size"]),
                    update=match["update"],
                    k=parse_float(match["k"]),
                    lr=parse_float(match["rounded_lr"]),
                    initial_lr=parse_float(match["initial_lr"]),
                )
                searches[search].evals.append(current_eval)
                continue

            if match := CACHE_HIT_RE.match(line):
                search = int(match["search"])
                searches[search].cache_hits.append(
                    {
                        "k": parse_float(match["k"]),
                        "rounded_lr": match["rounded_lr"],
                    }
                )
                current_eval = None
                continue

            if current_eval is not None and (match := EPOCH_EVAL_RE.match(line)):
                current_eval.epoch_evals.append(
                    {
                        "epoch": parse_float(match["epoch"]),
                        "val_acc": parse_float(match["val_acc"]),
                        "time_seconds": parse_float(match["time_seconds"]),
                    }
                )
                continue

            if current_eval is not None and (match := FINAL_RE.match(line)):
                current_eval.train25_loss = parse_float(match["train25_loss"])
                current_eval.val_acc = parse_float(match["val_acc"])
                current_eval.tta_val_acc = parse_float(match["tta_val_acc"])
                current_eval.time_seconds = parse_float(match["time_seconds"])
                continue

            if match := COMPLETE_RE.match(line):
                search = searches[int(match["search"])]
                search.best_k = parse_float(match["best_k"])
                search.best_lr = parse_float(match["best_lr"])
                search.best_tta_val_acc = parse_float(match["tta_val_acc"])
                search.evaluated_lrs = int(match["evaluated_lrs"])
                current_eval = None

    return sorted(searches.values(), key=lambda row: row.search)


def sorted_searches(searches: list[SearchRun]) -> list[SearchRun]:
    update_order = {name: i for i, name in enumerate(UPDATE_LABELS)}
    return sorted(
        searches,
        key=lambda row: (row.batch_size, update_order.get(row.update, 99), row.search),
    )


def batch_sizes(searches: list[SearchRun]) -> list[int]:
    return sorted({search.batch_size for search in searches})


def updates(searches: list[SearchRun]) -> list[str]:
    present = {search.update for search in searches}
    ordered = [name for name in UPDATE_LABELS if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_search_tta(searches: list[SearchRun], output_path: Path) -> None:
    rows = batch_sizes(searches)
    cols = updates(searches)
    lookup = {(search.batch_size, search.update): search for search in searches}
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(6.0 * len(cols), 3.6 * len(rows)),
        sharey=True,
        constrained_layout=True,
    )
    if len(rows) == 1:
        axes = [axes]

    for r, batch_size in enumerate(rows):
        for c, update in enumerate(cols):
            ax = axes[r][c] if len(cols) > 1 else axes[r]
            search = lookup.get((batch_size, update))
            if search is None:
                ax.axis("off")
                continue
            points = sorted(search.evals, key=lambda row: row.k)
            xs = [row.k for row in points]
            ys = [row.tta_val_acc for row in points]
            color = UPDATE_COLORS.get(update)
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=color)
            ax.axvline(0, color="0.65", linestyle="--", linewidth=1.0)
            best = search.best_eval
            ax.scatter(
                [best.k],
                [best.tta_val_acc],
                marker="*",
                s=150,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )
            for point in points:
                ax.annotate(
                    f"{point.lr:g}",
                    (point.k, point.tta_val_acc),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                )
            ax.set_title(f"bs={batch_size} {search.label}")
            ax.set_xlabel("k in initial_lr * 0.8^k")
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel("TTA val accuracy")

    fig.suptitle("LR Search TTA Curves")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_summary(searches: list[SearchRun], output_path: Path) -> None:
    batches = batch_sizes(searches)
    names = updates(searches)
    lookup = {(search.batch_size, search.update): search.best_eval for search in searches}
    x = list(range(len(batches)))
    width = 0.8 / max(1, len(names))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for i, update in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        positions = [pos + offset for pos in x]
        tta_values = [
            getattr(lookup.get((batch, update)), "tta_val_acc", math.nan)
            for batch in batches
        ]
        lr_values = [
            getattr(lookup.get((batch, update)), "lr", math.nan) for batch in batches
        ]
        bars = axes[0].bar(
            positions,
            tta_values,
            width=width,
            label=UPDATE_LABELS.get(update, update),
            color=UPDATE_COLORS.get(update),
            alpha=0.88,
        )
        axes[0].bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
        axes[1].bar(
            positions,
            lr_values,
            width=width,
            label=UPDATE_LABELS.get(update, update),
            color=UPDATE_COLORS.get(update),
            alpha=0.88,
        )

    all_tta_values = [search.best_eval.tta_val_acc for search in searches]
    tta_min = min(all_tta_values)
    tta_max = max(all_tta_values)
    tta_pad = max(0.005, (tta_max - tta_min) * 0.15)
    axes[0].set_title("Best TTA by search")
    axes[0].set_ylabel("TTA val accuracy")
    axes[0].set_ylim(max(0.0, tta_min - tta_pad), min(1.0, tta_max + tta_pad))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Selected rounded initial Muon LR")
    axes[1].set_ylabel("Muon LR")
    axes[1].set_xlabel("batch size")
    axes[1].set_xticks(x, [str(batch) for batch in batches])
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("cifar_baseline2_sgd_search_exp1")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_epoch_curves(searches: list[SearchRun], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    markers = {125: "o", 500: "s", 2000: "^"}
    linestyles = {
        "row_norm": "--",
        "row_norm_max": ":",
        "matrix_norm": "-.",
        "matrix_norm_max": ":",
        "zeropower_via_newtonschulz5": "-",
        "zeropower_via_newtonschulz5_max": ":",
    }

    for search in sorted_searches(searches):
        best = search.best_eval
        epochs = [row["epoch"] for row in best.epoch_evals]
        vals = [row["val_acc"] for row in best.epoch_evals]
        ax.plot(
            epochs,
            vals,
            marker=markers.get(search.batch_size, "o"),
            linestyle=linestyles.get(search.update, "-"),
            color=UPDATE_COLORS.get(search.update),
            linewidth=1.8,
            markersize=4.5,
            label=f"bs={search.batch_size} {search.label} lr={best.lr:g}",
        )

    ax.set_title("Validation accuracy by epoch for selected LRs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(searches: list[SearchRun], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "search",
                "batch_size",
                "update",
                "k",
                "muon_lr",
                "initial_muon_lr",
                "train25_loss",
                "val_acc",
                "tta_val_acc",
                "time_seconds",
                "selected",
            ]
        )
        for search in sorted_searches(searches):
            best = search.best_eval
            for row in sorted(search.evals, key=lambda item: item.k):
                writer.writerow(
                    [
                        row.search,
                        row.batch_size,
                        row.update,
                        row.k,
                        row.lr,
                        row.initial_lr,
                        row.train25_loss,
                        row.val_acc,
                        row.tta_val_acc,
                        row.time_seconds,
                        row is best,
                    ]
                )


def write_summary(searches: list[SearchRun], output_path: Path) -> None:
    lines = [
        "search | batch_size | update | initial_lr | best_k | best_lr | val_acc | tta_val_acc | evals",
        "---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for search in sorted_searches(searches):
        best = search.best_eval
        lines.append(
            f"{search.search} | {search.batch_size} | {search.label} | "
            f"{search.initial_lr:g} | {best.k:g} | {best.lr:g} | "
            f"{best.val_acc:.4f} | {best.tta_val_acc:.4f} | {len(search.evals)}"
        )
    best_overall = max((search.best_eval for search in searches), key=lambda row: row.tta_val_acc)
    lines.extend(
        [
            "",
            "Best overall:",
            (
                f"search={best_overall.search} batch_size={best_overall.batch_size} "
                f"update={best_overall.label} k={best_overall.k:g} "
                f"lr={best_overall.lr:g} tta_val_acc={best_overall.tta_val_acc:.4f}"
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def plot_all(searches: list[SearchRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_search_tta(searches, output_dir / "lr_search_tta_curves.png")
    plot_best_summary(searches, output_dir / "best_summary.png")
    plot_epoch_curves(searches, output_dir / "selected_epoch_curves.png")
    write_csv(searches, output_dir / "lr_search_results.csv")
    write_summary(searches, output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cifar_baseline2_sgd_search_exp1 LR search results."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    searches = parse_log(args.log)
    if not searches:
        raise SystemExit(f"No LR searches parsed from {args.log}")

    plot_all(searches, args.output_dir)
    best = max((search.best_eval for search in searches), key=lambda row: row.tta_val_acc)
    print(f"Parsed {len(searches)} LR searches from {args.log}")
    print(f"Wrote plots and summaries to {args.output_dir}")
    print(
        f"Best overall: search={best.search} bs={best.batch_size} "
        f"update={best.label} lr={best.lr:g} tta={best.tta_val_acc:.4f}"
    )


if __name__ == "__main__":
    main()

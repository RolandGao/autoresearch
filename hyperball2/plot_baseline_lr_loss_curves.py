#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_LINE_SEARCH_LOG = Path(
    "/workspace/neural_networks_optimization/autoresearch/hyperball2/"
    "baseline_line_search_exp1.log"
)
DEFAULT_FIXED_LOG = Path(
    "/workspace/neural_networks_optimization/autoresearch/hyperball2/"
    "hyperball_baseline4.log"
)

STEP_RE = re.compile(
    r"step\s+(?P<step>\d+).*?"
    r"\|\s+loss:\s+(?P<loss>[-+0-9.eE]+)\s+"
    r"\|\s+lrm:\s+(?P<lrm>[-+0-9.eE]+)"
    r"(?:\s+\|\s+lrs:\s+LR_MULT=(?P<lr_mult>[-+0-9.eE]+))?"
    r"(?:\s+\|\s+val_loss:\s+(?P<val_loss>[-+0-9.eE]+))?"
)


def parse_float(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def append_training_line(record, match):
    step = int(match.group("step"))
    loss = parse_float(match.group("loss"))
    lrm = parse_float(match.group("lrm"))
    lr_mult = parse_float(match.group("lr_mult"))
    val_loss = parse_float(match.group("val_loss"))

    record["step"].append(step)
    record["train_loss"].append(loss)
    record["lrm"].append(lrm)
    record["lr_mult"].append(lr_mult)
    record["effective_lr_mult"].append(
        None if lr_mult is None or lrm is None else lr_mult * lrm
    )
    record["step_val_loss"].append(val_loss)


def append_lr_search_line(record, line):
    _, payload = line.split("LR_SEARCH", 1)
    event = json.loads(payload.strip())
    best_lrs = event.get("best_lrs", {})
    best_depth2_lrs = event.get("best_depth2_lrs", {})

    record["lr_search_step"].append(event["step"])
    record["lr_search_loss"].append(event.get("best_val_loss"))
    record["lr_search_lr_mult"].append(best_lrs.get("LR_MULT"))
    record["lr_search_depth2_lr_mult"].append(best_depth2_lrs.get("LR_MULT"))


def parse_log(path):
    record = {
        "step": [],
        "train_loss": [],
        "lrm": [],
        "lr_mult": [],
        "effective_lr_mult": [],
        "step_val_loss": [],
        "lr_search_step": [],
        "lr_search_loss": [],
        "lr_search_lr_mult": [],
        "lr_search_depth2_lr_mult": [],
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            for line in raw_line.split("\r"):
                if "LR_SEARCH" in line:
                    append_lr_search_line(record, line)
                    continue
                match = STEP_RE.search(line)
                if match:
                    append_training_line(record, match)
    return record


def compact_xy(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return [], []
    return [x for x, _ in pairs], [y for _, y in pairs]


def plot_loss_curves(line_search, fixed, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        line_search["step"],
        line_search["train_loss"],
        label="baseline line search: train loss",
        linewidth=1.8,
    )
    ax.plot(
        fixed["step"],
        fixed["train_loss"],
        label="hyperball baseline4: train loss",
        linewidth=1.8,
    )

    x, y = compact_xy(
        line_search["lr_search_step"],
        line_search["lr_search_loss"],
    )
    if x:
        ax.plot(
            x,
            y,
            label="baseline line search: held-out search loss",
            linewidth=1.4,
            alpha=0.9,
        )

    ax.set_title("Loss Curves")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_curves(line_search, fixed, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        fixed["step"],
        fixed["lrm"],
        label="hyperball baseline4: scheduler lrm",
        linewidth=1.8,
    )
    ax.plot(
        line_search["step"],
        line_search["lrm"],
        label="baseline line search: scheduler lrm",
        linewidth=1.8,
    )

    x, y = compact_xy(line_search["step"], line_search["lr_mult"])
    if x:
        ax.plot(
            x,
            y,
            label="baseline line search: searched LR_MULT",
            linewidth=1.4,
        )

    x, y = compact_xy(line_search["step"], line_search["effective_lr_mult"])
    if x:
        ax.plot(
            x,
            y,
            label="baseline line search: LR_MULT * lrm",
            linewidth=1.4,
        )

    x, y = compact_xy(
        line_search["lr_search_step"],
        line_search["lr_search_depth2_lr_mult"],
    )
    if x:
        ax.plot(
            x,
            y,
            label="baseline line search: depth2 best LR_MULT",
            linewidth=1.0,
            alpha=0.65,
        )

    ax.set_title("LR Control Curves")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("logged LR multiplier")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and LR curves from baseline line-search logs."
    )
    parser.add_argument("--line-search-log", type=Path, default=DEFAULT_LINE_SEARCH_LOG)
    parser.add_argument("--fixed-log", type=Path, default=DEFAULT_FIXED_LOG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LINE_SEARCH_LOG.parent,
        help="Directory for generated PNG files.",
    )
    args = parser.parse_args()

    line_search = parse_log(args.line_search_log)
    fixed = parse_log(args.fixed_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = args.output_dir / "baseline_line_search_vs_hyperball_loss.png"
    lr_path = args.output_dir / "baseline_line_search_vs_hyperball_lr.png"
    plot_loss_curves(line_search, fixed, loss_path)
    plot_lr_curves(line_search, fixed, lr_path)

    print(f"wrote {loss_path}")
    print(f"wrote {lr_path}")
    print(
        "parsed "
        f"{len(line_search['step'])} line-search steps, "
        f"{len(line_search['lr_search_step'])} LR_SEARCH records, "
        f"{len(fixed['step'])} fixed-run steps"
    )


if __name__ == "__main__":
    main()

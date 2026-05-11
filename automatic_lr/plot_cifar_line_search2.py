import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


LOG_PREFIX = "LR_LANDSCAPE "
GROUPS = ("head", "muon")


def read_records(path):
    stream = sys.stdin if path == "-" else open(path)
    try:
        for line in stream:
            if LOG_PREFIX not in line:
                continue
            payload = line.split(LOG_PREFIX, 1)[1].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "lr_landscape":
                yield record
    finally:
        if path != "-":
            stream.close()


def grouped_landscape_records(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("phase") != "landscape":
            continue
        group = record.get("group")
        step = record.get("step")
        if group in GROUPS and step is not None:
            grouped[int(step)][group].append(record)
    return grouped


def positive_linthresh(records):
    positive_xs = [
        record["varied_lr"]
        for groups in records.values()
        for group_records in groups.values()
        for record in group_records
        if record.get("varied_lr", 0.0) > 0.0
    ]
    if not positive_xs:
        return 1e-8
    return max(min(positive_xs) * 0.5, 1e-12)


def plot_step(step, groups, out_dir, xscale):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    for ax, group in zip(axes, GROUPS):
        records = sorted(groups.get(group, []), key=lambda item: item["varied_lr"])
        if not records:
            ax.set_title(group)
            ax.text(0.5, 0.5, "no records", ha="center", va="center")
            continue

        xs = [record["varied_lr"] for record in records]
        train_losses = [record["train_loss"] for record in records]
        eval_losses = [record["eval_loss"] for record in records]
        pre_train_loss = records[0].get("pre_train_loss")
        pre_eval_loss = records[0].get("pre_eval_loss")

        ax.plot(xs, train_losses, marker="o", label="first batch")
        ax.plot(xs, eval_losses, marker="o", linestyle="--", label="second batch")

        if pre_train_loss is not None:
            ax.axhline(pre_train_loss, color="tab:blue", alpha=0.25)
        if pre_eval_loss is not None:
            ax.axhline(pre_eval_loss, color="tab:orange", alpha=0.25)

        for record in records:
            if record.get("point") in {"optimum", "boundary"}:
                ax.scatter(
                    [record["varied_lr"]],
                    [record["train_loss"]],
                    s=70,
                    marker="s" if record["point"] == "boundary" else "*",
                    color="black",
                    zorder=4,
                )

        ax.set_title(group)
        ax.set_xlabel("varied learning rate")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        if xscale == "symlog":
            ax.set_xscale("symlog", linthresh=positive_linthresh({step: groups}))
        else:
            ax.set_xscale(xscale)
        ax.legend()

    fig.suptitle(f"Line-search loss landscape, step {step}")
    output_path = os.path.join(out_dir, f"line_search_step_{step:03d}.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot LR_LANDSCAPE logs from cifar_line_search2.py."
    )
    parser.add_argument(
        "log_path",
        help="Training stdout log path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--out-dir",
        default="line_search_landscape_plots2",
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--xscale",
        choices=("linear", "log", "symlog"),
        default="symlog",
        help="X-axis scale. symlog keeps the lr=0 origin visible.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grouped = grouped_landscape_records(read_records(args.log_path))
    if not grouped:
        raise SystemExit("No LR_LANDSCAPE landscape records found.")

    for step in sorted(grouped):
        output_path = plot_step(step, grouped[step], args.out_dir, args.xscale)
        print(output_path)


if __name__ == "__main__":
    main()

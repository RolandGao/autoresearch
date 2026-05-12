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


def pre_step_records(records):
    result = []
    for record in records:
        if record.get("phase") == "pre_step" and record.get("step") is not None:
            result.append(record)
    return sorted(result, key=lambda record: int(record["step"]))


def optimal_lr_records(records):
    by_step = defaultdict(list)
    for record in records:
        if record.get("phase") == "coordinate_search" and record.get("step") is not None:
            by_step[int(record["step"])].append(record)

    result = []
    for step in sorted(by_step):
        step_records = by_step[step]
        best_train = min(step_records, key=lambda record: record["train_loss"])
        best_eval = min(step_records, key=lambda record: record["eval_loss"])
        result.append(
            dict(
                step=step,
                lrs=best_train["lrs"],
                train_loss=best_train["train_loss"],
                eval_loss=best_train["eval_loss"],
                second_batch_lrs=best_eval["lrs"],
                second_batch_train_loss=best_eval["train_loss"],
                second_batch_eval_loss=best_eval["eval_loss"],
            )
        )
    return result


def component_loss_decrease_records(records):
    grouped_pre_steps = {
        int(record["step"]): record
        for record in records
        if record.get("phase") == "pre_step" and record.get("step") is not None
    }
    grouped = defaultdict(dict)
    for record in records:
        if record.get("phase") != "landscape":
            continue
        if record.get("point") != "left_origin" or record.get("step") is None:
            continue
        step = int(record["step"])
        pre_step = grouped_pre_steps.get(step)
        if pre_step is None:
            continue

        landscape_group = record.get("group")
        if landscape_group == "muon":
            component_group = "head"
        elif landscape_group == "head":
            component_group = "muon"
        else:
            continue

        grouped[step][component_group] = dict(
            record,
            group=component_group,
            train_loss_decrease=pre_step["train_loss"] - record["train_loss"],
            eval_loss_decrease=pre_step["eval_loss"] - record["eval_loss"],
        )
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


def plot_loss_over_steps(records, out_dir):
    if not records:
        return None

    steps = [int(record["step"]) for record in records]
    train_losses = [record["train_loss"] for record in records]
    eval_losses = [record["eval_loss"] for record in records]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(steps, train_losses, marker="o", label="first batch")
    ax.plot(steps, eval_losses, marker="o", linestyle="--", label="second batch")
    ax.set_title("Loss over training steps")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = os.path.join(out_dir, "loss_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_optimal_lrs_over_steps(records, out_dir):
    if not records:
        return None

    steps = [int(record["step"]) for record in records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, group in zip(axes, GROUPS):
        ax.plot(
            steps,
            [record["lrs"][group] for record in records],
            marker="o",
            label="first batch",
        )
        ax.plot(
            steps,
            [record["second_batch_lrs"][group] for record in records],
            marker="o",
            linestyle="--",
            label="second batch",
        )
        ax.set_title(group)
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Optimal learning rates over steps")

    output_path = os.path.join(out_dir, "optimal_lrs_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_component_loss_decreases(grouped, out_dir):
    if not grouped:
        return None

    steps = sorted(grouped)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for group in GROUPS:
        group_records = [grouped[step].get(group) for step in steps]
        valid = [
            (step, record)
            for step, record in zip(steps, group_records)
            if record is not None
        ]
        if not valid:
            continue
        valid_steps = [step for step, _ in valid]
        ax.plot(
            valid_steps,
            [record["train_loss_decrease"] for _, record in valid],
            marker="o",
            label=f"first batch {group}",
        )
        ax.plot(
            valid_steps,
            [record["eval_loss_decrease"] for _, record in valid],
            marker="o",
            linestyle="--",
            label=f"second batch {group}",
        )

    ax.set_title("Isolated loss decrease from optimal group LR")
    ax.set_xlabel("step")
    ax.set_ylabel("pre-step loss minus isolated post-step loss")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.3)
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = os.path.join(out_dir, "component_loss_decrease_over_steps.png")
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
    records = list(read_records(args.log_path))
    grouped = grouped_landscape_records(records)
    if not grouped:
        raise SystemExit("No LR_LANDSCAPE landscape records found.")

    loss_output_path = plot_loss_over_steps(pre_step_records(records), args.out_dir)
    if loss_output_path is not None:
        print(loss_output_path)

    lr_output_path = plot_optimal_lrs_over_steps(
        optimal_lr_records(records), args.out_dir
    )
    if lr_output_path is not None:
        print(lr_output_path)

    decrease_output_path = plot_component_loss_decreases(
        component_loss_decrease_records(records), args.out_dir
    )
    if decrease_output_path is not None:
        print(decrease_output_path)

    for step in sorted(grouped):
        output_path = plot_step(step, grouped[step], args.out_dir, args.xscale)
        print(output_path)


if __name__ == "__main__":
    main()

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LOG_PREFIX = "LR_LANDSCAPE "
GROUPS = ("head", "muon")
VAL_LOSS_SCALE = 5.0


def read_records(path):
    records = []
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
                record["_ordinal"] = len(records)
                records.append(record)
    finally:
        if path != "-":
            stream.close()

    attach_run_keys(records)
    return records


def attach_run_keys(records):
    inferred_run = 1
    last_pre_step = None
    for record in records:
        explicit_run = record.get("run") or record.get("schedule")
        if explicit_run is not None:
            record["_run"] = str(explicit_run)
            continue

        step = record.get("step")
        if record.get("phase") == "pre_step" and step is not None:
            step = int(step)
            if last_pre_step is not None and step <= last_pre_step:
                inferred_run += 1
            last_pre_step = step
        record["_run"] = f"run_{inferred_run:02d}"


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def run_key(record):
    return record.get("_run", "run_01")


def grouped_by_run(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[run_key(record)].append(record)
    return dict(sorted(grouped.items()))


def step_number(record):
    return int(record["step"])


def step_label(step):
    return step + 1


def current_batch_index(record):
    return int(record.get("current_batch_index", 0))


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def val_loss(record, key="val_loss"):
    value = finite_float(record.get(key, float("nan")))
    return value / VAL_LOSS_SCALE


def batch_losses(record, pre=False):
    key = "pre_batch_train_losses" if pre else "batch_train_losses"
    values = record.get(key)
    if values is None and pre:
        values = record.get("batch_train_losses")
    if isinstance(values, list):
        return [finite_float(value) for value in values]

    # Backward-compatible fallback for older logs that only had first/second batch.
    losses = []
    if "train_loss" in record:
        losses.append(finite_float(record["train_loss"]))
    if "eval_loss" in record:
        losses.append(finite_float(record["eval_loss"]))
    return losses


def mean_loss(losses):
    finite_losses = [loss for loss in losses if math.isfinite(loss)]
    if not finite_losses:
        return float("nan")
    return sum(finite_losses) / len(finite_losses)


def loss_at(record, index, pre=False):
    losses = batch_losses(record, pre=pre)
    if 0 <= index < len(losses):
        return losses[index]
    return finite_float(record.get("train_loss", float("nan")))


def style_step_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", color="0.7", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)


def pre_step_records(records):
    return sorted(
        [
            record
            for record in records
            if record.get("phase") == "pre_step" and record.get("step") is not None
        ],
        key=step_number,
    )


def coordinate_records_by_step(records):
    grouped = defaultdict(list)
    for record in records:
        if (
            record.get("phase") == "coordinate_search"
            and record.get("step") is not None
        ):
            grouped[step_number(record)].append(record)
    return dict(sorted(grouped.items()))


def selected_lr_records(records):
    result = []
    for step, step_records in coordinate_records_by_step(records).items():
        best_decision = min(
            step_records,
            key=lambda record: record.get("decision_loss", record["train_loss"]),
        )
        best_mean = min(
            step_records, key=lambda record: mean_loss(batch_losses(record))
        )
        best_val = min(
            step_records,
            key=lambda record: val_loss(record),
        )
        result.append(
            dict(
                step=step,
                current_batch_index=current_batch_index(best_decision),
                decision_metric=best_decision.get("decision_metric", "current_batch"),
                lrs=best_decision["lrs"],
                train_loss=best_decision["train_loss"],
                val_loss=val_loss(best_decision),
                decision_loss=best_decision.get(
                    "decision_loss", best_decision["train_loss"]
                ),
                batch_train_losses=batch_losses(best_decision),
                mean_loss=mean_loss(batch_losses(best_decision)),
                mean_lrs=best_mean["lrs"],
                mean_selected_train_loss=best_mean["train_loss"],
                mean_selected_batch_train_losses=batch_losses(best_mean),
                mean_selected_mean_loss=mean_loss(batch_losses(best_mean)),
                mean_selected_val_loss=val_loss(best_mean),
                val_lrs=best_val["lrs"],
                val_selected_train_loss=best_val["train_loss"],
                val_selected_batch_train_losses=batch_losses(best_val),
                val_selected_mean_loss=mean_loss(batch_losses(best_val)),
                val_selected_val_loss=val_loss(best_val),
            )
        )
    return result


def grouped_landscape_records(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("phase") != "landscape":
            continue
        group = record.get("group")
        step = record.get("step")
        if group in GROUPS and step is not None:
            grouped[step_number(record)][group].append(record)
    return grouped


def component_loss_decrease_records(records):
    grouped_pre_steps = {
        step_number(record): record for record in pre_step_records(records)
    }
    grouped = defaultdict(dict)
    for record in records:
        if record.get("phase") != "landscape":
            continue
        if record.get("point") != "left_origin" or record.get("step") is None:
            continue
        step = step_number(record)
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

        index = current_batch_index(record)
        pre_losses = batch_losses(pre_step)
        post_losses = batch_losses(record)
        pre_val_loss = val_loss(pre_step)
        post_val_loss = val_loss(record)
        grouped[step][component_group] = dict(
            record,
            group=component_group,
            current_loss_decrease=loss_at(pre_step, index) - loss_at(record, index),
            mean_loss_decrease=mean_loss(pre_losses) - mean_loss(post_losses),
            val_loss_decrease=pre_val_loss - post_val_loss,
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


def truncate_over_original(xs, ys, original_loss):
    if not math.isfinite(finite_float(original_loss)):
        return xs, ys

    truncated_xs = []
    truncated_ys = []
    for x, y in zip(xs, ys):
        if not math.isfinite(y) or y > original_loss:
            break
        truncated_xs.append(x)
        truncated_ys.append(y)
    return truncated_xs, truncated_ys


def plot_loss_over_steps(pre_records, selected_records, out_dir):
    if not pre_records and not selected_records:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if pre_records:
        steps = [step_label(step_number(record)) for record in pre_records]
        ax.plot(
            steps,
            [loss_at(record, current_batch_index(record)) for record in pre_records],
            marker="o",
            color="tab:blue",
            alpha=0.45,
            label="current batch before step",
        )
        ax.plot(
            steps,
            [mean_loss(batch_losses(record)) for record in pre_records],
            marker="o",
            color="tab:green",
            alpha=0.45,
            label="mean of tracked batches before step",
        )
        ax.plot(
            steps,
            [val_loss(record) for record in pre_records],
            marker="o",
            color="tab:orange",
            alpha=0.45,
            label="validation before step",
        )
    if selected_records:
        steps = [step_label(record["step"]) for record in selected_records]
        ax.plot(
            steps,
            [record["train_loss"] for record in selected_records],
            marker="o",
            color="tab:blue",
            label="current batch selected",
        )
        ax.plot(
            steps,
            [record["mean_loss"] for record in selected_records],
            marker="o",
            linestyle="--",
            color="tab:green",
            label="mean of tracked batches at selected LR",
        )
        ax.plot(
            steps,
            [record["val_loss"] for record in selected_records],
            marker="o",
            linestyle=":",
            color="tab:orange",
            label="validation at selected LR",
        )

    ax.set_title("Loss Over Steps")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    style_step_axis(ax)
    ax.legend()

    output_path = os.path.join(out_dir, "loss_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_batch_loss_heatmap(selected_records, out_dir):
    if not selected_records:
        return None

    heatmaps = (
        ("decision-selected LR", "batch_train_losses"),
        ("best 25-batch mean LR", "mean_selected_batch_train_losses"),
        ("best validation LR", "val_selected_batch_train_losses"),
    )
    max_batches = max(
        len(record[key]) for _, key in heatmaps for record in selected_records
    )
    if max_batches == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    axes = axes.reshape(-1)
    steps = [step_label(record["step"]) for record in selected_records]

    for ax, (title, key) in zip(axes, heatmaps):
        matrix = []
        for record in selected_records:
            row = [float("nan")] * max_batches
            for index, loss in enumerate(record[key]):
                row[index] = loss
            matrix.append(row)

        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("tracked batch index")
        ax.set_xticks(range(max_batches))
        ax.set_xticklabels(range(max_batches))
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps)
        ax.scatter(
            [record["current_batch_index"] for record in selected_records],
            range(len(selected_records)),
            marker="s",
            facecolors="none",
            edgecolors="white",
            linewidths=1.4,
            label="decision batch",
        )
        ax.legend(loc="upper right")
        fig.colorbar(image, ax=ax, label="loss")
    axes[0].set_ylabel("step")
    fig.suptitle("Batch Losses Across Tracked Batches")

    output_path = os.path.join(out_dir, "selected_batch_loss_heatmap.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_lrs_over_steps(selected_records, out_dir):
    if not selected_records:
        return None

    steps = [step_label(record["step"]) for record in selected_records]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    for ax, group in zip(axes, GROUPS):
        ax.plot(
            steps,
            [record["lrs"][group] for record in selected_records],
            marker="o",
            label="selected by decision metric",
        )
        ax.plot(
            steps,
            [record["mean_lrs"][group] for record in selected_records],
            marker="o",
            linestyle="--",
            label="best mean tracked loss",
        )
        ax.plot(
            steps,
            [record["val_lrs"][group] for record in selected_records],
            marker="o",
            linestyle=":",
            label="best validation loss",
        )
        ax.set_title(group)
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        style_step_axis(ax)
        ax.legend()

    fig.suptitle("Selected Learning Rates")
    output_path = os.path.join(out_dir, "lrs_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_component_loss_decreases(grouped, out_dir):
    if not grouped:
        return None

    steps = sorted(grouped)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for group in GROUPS:
        valid = [
            (step, grouped[step][group]) for step in steps if group in grouped[step]
        ]
        if not valid:
            continue
        valid_steps = [step_label(step) for step, _ in valid]
        ax.plot(
            valid_steps,
            [record["current_loss_decrease"] for _, record in valid],
            marker="o",
            label=f"current batch {group}",
        )
        ax.plot(
            valid_steps,
            [record["mean_loss_decrease"] for _, record in valid],
            marker="o",
            linestyle="--",
            label=f"mean tracked batches {group}",
        )
        ax.plot(
            valid_steps,
            [record["val_loss_decrease"] for _, record in valid],
            marker="o",
            linestyle=":",
            label=f"validation {group}",
        )

    ax.set_title("Isolated Loss Decrease From Optimal Group LR")
    ax.set_xlabel("step")
    ax.set_ylabel("pre-step loss minus isolated post-step loss")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.3)
    style_step_axis(ax)
    ax.legend()

    output_path = os.path.join(out_dir, "component_loss_decrease_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


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
        batch_count = max(len(batch_losses(record)) for record in records)
        decision_batch = current_batch_index(records[0])
        pre_losses = batch_losses(records[0], pre=True)
        for batch_index in range(batch_count):
            ys = [
                losses[batch_index] if batch_index < len(losses) else float("nan")
                for losses in (batch_losses(record) for record in records)
            ]
            original_loss = (
                pre_losses[batch_index]
                if batch_index < len(pre_losses)
                else float("nan")
            )
            plot_xs, plot_ys = truncate_over_original(xs, ys, original_loss)
            if not plot_xs:
                continue
            if batch_index == decision_batch:
                ax.plot(
                    plot_xs,
                    plot_ys,
                    marker="o",
                    color="black",
                    linewidth=2.0,
                    label=f"decision batch {batch_index}",
                )
            else:
                ax.plot(plot_xs, plot_ys, color="0.65", linewidth=0.8, alpha=0.35)

        mean_losses = [mean_loss(batch_losses(record)) for record in records]
        pre_mean_loss = mean_loss(pre_losses)
        mean_xs, mean_ys = truncate_over_original(xs, mean_losses, pre_mean_loss)
        ax.plot(
            mean_xs,
            mean_ys,
            color="tab:green",
            linestyle="--",
            linewidth=1.6,
            label="mean tracked batches",
        )
        val_losses = [val_loss(record) for record in records]
        pre_val_loss = val_loss(records[0], "pre_val_loss")
        val_xs, val_ys = truncate_over_original(xs, val_losses, pre_val_loss)
        if any(math.isfinite(loss) for loss in val_losses):
            ax.plot(
                val_xs,
                val_ys,
                color="tab:orange",
                linestyle=":",
                linewidth=1.8,
                label="validation",
            )

        pre_current_loss = records[0].get("pre_train_loss")
        if pre_current_loss is not None:
            ax.axhline(pre_current_loss, color="black", alpha=0.2)
        if pre_losses:
            ax.axhline(pre_mean_loss, color="tab:green", alpha=0.2)
        if math.isfinite(pre_val_loss):
            ax.axhline(pre_val_loss, color="tab:orange", alpha=0.2)

        for record in records:
            if record.get("point") in {"optimum", "boundary"}:
                marker = "s" if record["point"] == "boundary" else "*"
                record_train_loss = finite_float(record["train_loss"])
                record_mean_loss = mean_loss(batch_losses(record))
                record_val_loss = val_loss(record)
                if record_train_loss <= finite_float(pre_current_loss):
                    ax.scatter(
                        [record["varied_lr"]],
                        [record_train_loss],
                        s=70,
                        marker=marker,
                        color="black",
                        zorder=4,
                    )
                if record_mean_loss <= pre_mean_loss:
                    ax.scatter(
                        [record["varied_lr"]],
                        [record_mean_loss],
                        s=70,
                        marker=marker,
                        color="tab:green",
                        zorder=4,
                    )
                if math.isfinite(record_val_loss) and record_val_loss <= pre_val_loss:
                    ax.scatter(
                        [record["varied_lr"]],
                        [record_val_loss],
                        s=70,
                        marker=marker,
                        color="tab:orange",
                        zorder=4,
                    )

        ax.set_title(f"{group}, decision batch {decision_batch}")
        ax.set_xlabel("varied learning rate")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        if xscale == "symlog":
            ax.set_xscale("symlog", linthresh=positive_linthresh({step: groups}))
        else:
            ax.set_xscale(xscale)
        ax.legend()

    fig.suptitle(f"Line-Search Loss Landscape, Step {step_label(step)}")
    output_path = os.path.join(out_dir, f"line_search_step_{step:03d}.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_run(run_name, records, base_out_dir, xscale):
    out_dir = os.path.join(base_out_dir, safe_name(run_name))
    os.makedirs(out_dir, exist_ok=True)

    outputs = []
    selected_records = selected_lr_records(records)
    for output_path in (
        plot_loss_over_steps(pre_step_records(records), selected_records, out_dir),
        plot_selected_batch_loss_heatmap(selected_records, out_dir),
        plot_lrs_over_steps(selected_records, out_dir),
        plot_component_loss_decreases(
            component_loss_decrease_records(records), out_dir
        ),
    ):
        if output_path is not None:
            outputs.append(output_path)

    grouped = grouped_landscape_records(records)
    for step in sorted(grouped):
        outputs.append(plot_step(step, grouped[step], out_dir, xscale))

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Plot LR_LANDSCAPE logs from cifar_line_search5.py."
    )
    parser.add_argument(
        "log_path",
        help="Training stdout log path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--out-dir",
        default="line_search_landscape5_plots2",
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
    records = read_records(args.log_path)
    if not records:
        raise SystemExit("No LR_LANDSCAPE records found.")

    outputs = []
    for run_name, run_records in grouped_by_run(records).items():
        outputs.extend(plot_run(run_name, run_records, args.out_dir, args.xscale))

    if not outputs:
        raise SystemExit("No plottable LR_LANDSCAPE records found.")
    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()

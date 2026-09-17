import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LOG_PREFIX = "MOMENTUM_LINE_SEARCH "
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
            if record.get("event") == "momentum_line_search":
                yield record
    finally:
        if path != "-":
            stream.close()


def run_key(record):
    return str(record.get("run", "run"))


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def group_records_by_run(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[run_key(record)].append(record)
    return dict(sorted(grouped.items()))


def step_number(record):
    return int(record["step"])


def step_label(step):
    return step + 1


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


def best_records(records):
    result = []
    for step, step_records in coordinate_records_by_step(records).items():
        best_train = min(step_records, key=lambda record: record["train_loss"])
        best_eval = min(step_records, key=lambda record: record["eval_loss"])
        result.append(
            dict(
                step=step,
                train_loss=best_train["train_loss"],
                eval_loss=best_train["eval_loss"],
                lrs=best_train["lrs"],
                momentums=best_train["momentums"],
                second_batch_train_loss=best_eval["train_loss"],
                second_batch_eval_loss=best_eval["eval_loss"],
                second_batch_lrs=best_eval["lrs"],
                second_batch_momentums=best_eval["momentums"],
            )
        )
    return result


def plot_loss_over_steps(pre_records, selected_records, out_dir):
    if not pre_records and not selected_records:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if pre_records:
        steps = [step_label(step_number(record)) for record in pre_records]
        ax.plot(
            steps,
            [record["train_loss"] for record in pre_records],
            marker="o",
            color="tab:blue",
            alpha=0.45,
            label="first batch before step",
        )
        ax.plot(
            steps,
            [record["eval_loss"] for record in pre_records],
            marker="o",
            color="tab:orange",
            alpha=0.45,
            label="second batch before step",
        )
    if selected_records:
        steps = [step_label(record["step"]) for record in selected_records]
        ax.plot(
            steps,
            [record["train_loss"] for record in selected_records],
            marker="o",
            color="tab:blue",
            label="first batch selected",
        )
        ax.plot(
            steps,
            [record["eval_loss"] for record in selected_records],
            marker="o",
            linestyle="--",
            color="tab:orange",
            label="second batch at selected",
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
            label="selected by first batch",
        )
        ax.plot(
            steps,
            [record["second_batch_lrs"][group] for record in selected_records],
            marker="o",
            linestyle="--",
            label="best second batch",
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


def max_momentum_count(selected_records):
    if not selected_records:
        return 0
    return max(
        len(record["momentums"].get(group, []))
        for record in selected_records
        for group in GROUPS
    )


def momentum_matrix(selected_records, group, key="momentums"):
    max_count = max_momentum_count(selected_records)
    matrix = []
    for record in selected_records:
        row = [float("nan")] * max_count
        for index, value in enumerate(record[key].get(group, [])):
            row[index] = value
        matrix.append(row)
    return matrix


def plot_momentum_heatmaps(selected_records, out_dir):
    if not selected_records or max_momentum_count(selected_records) == 0:
        return None

    steps = [step_label(record["step"]) for record in selected_records]
    max_count = max_momentum_count(selected_records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes = axes.reshape(-1)

    for ax, group in zip(axes, GROUPS):
        image = ax.imshow(
            momentum_matrix(selected_records, group),
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(group)
        ax.set_xlabel("momentum index")
        ax.set_ylabel("step")
        ax.set_xticks(range(max_count))
        ax.set_xticklabels([f"m{index + 1}" for index in range(max_count)])
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps)
        for row, record in enumerate(selected_records):
            for col, value in enumerate(record["momentums"].get(group, [])):
                ax.text(
                    col,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, label="momentum")

    fig.suptitle("Selected Momentum Schedule")
    output_path = os.path.join(out_dir, "momentum_schedule_heatmaps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_momentum_paths(selected_records, out_dir):
    if not selected_records or max_momentum_count(selected_records) == 0:
        return None

    max_count = max_momentum_count(selected_records)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)

    for ax, group in zip(axes, GROUPS):
        for index in range(max_count):
            points = [
                (step_label(record["step"]), record["momentums"][group][index])
                for record in selected_records
                if len(record["momentums"].get(group, [])) > index
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[index % len(colors)],
                label=f"m{index + 1}",
            )
        ax.set_title(group)
        ax.set_xlabel("step")
        ax.set_ylabel("momentum")
        ax.set_ylim(-0.05, 1.05)
        style_step_axis(ax)
        ax.legend(ncols=2, fontsize=8)

    fig.suptitle("Selected Momentum Constants Over Later Steps")
    output_path = os.path.join(out_dir, "momentum_paths.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_new_momentum_over_steps(selected_records, out_dir):
    records = [record for record in selected_records if record["momentums"].get("head")]
    if not records:
        return None

    steps = [step_label(record["step"]) for record in records]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for group in GROUPS:
        ax.plot(
            steps,
            [record["momentums"][group][-1] for record in records],
            marker="o",
            label=group,
        )
    ax.set_title("Newest Momentum Constant Per Step")
    ax.set_xlabel("step")
    ax.set_ylabel("new momentum")
    ax.set_ylim(-0.05, 1.05)
    style_step_axis(ax)
    ax.legend()

    output_path = os.path.join(out_dir, "new_momentum_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_search_candidate_counts(records, out_dir):
    grouped = coordinate_records_by_step(records)
    if not grouped:
        return None

    steps = [step_label(step) for step in grouped]
    counts = [len(step_records) for step_records in grouped.values()]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(steps, counts)
    ax.set_title("Line Search Candidate Evaluations")
    ax.set_xlabel("step")
    ax.set_ylabel("candidates evaluated")
    style_step_axis(ax)

    output_path = os.path.join(out_dir, "candidate_counts.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_candidate_losses(records, selected_records, out_dir):
    grouped = coordinate_records_by_step(records)
    if not grouped:
        return None

    fig, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(10, max(4, 2.0 * len(grouped))),
        sharex=False,
        constrained_layout=True,
    )
    if len(grouped) == 1:
        axes = [axes]
    selected_by_step = {record["step"]: record for record in selected_records}

    for ax, (step, step_records) in zip(axes, grouped.items()):
        ordered = sorted(step_records, key=lambda record: record["train_loss"])
        xs = list(range(1, len(ordered) + 1))
        ax.scatter(
            xs,
            [record["train_loss"] for record in ordered],
            s=18,
            label="first batch",
        )
        ax.scatter(
            xs,
            [record["eval_loss"] for record in ordered],
            s=18,
            marker="x",
            label="second batch",
        )
        selected = selected_by_step.get(step)
        if selected is not None:
            ax.axhline(selected["train_loss"], color="tab:blue", alpha=0.25)
            ax.axhline(selected["eval_loss"], color="tab:orange", alpha=0.25)
        ax.set_title(f"Step {step_label(step)}")
        ax.set_xlabel("candidate rank by first-batch loss")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Candidate Losses During Coordinate Search")
    output_path = os.path.join(out_dir, "candidate_losses.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_summary(pre_records, selected_records, records, out_dir):
    output_path = os.path.join(out_dir, "summary.txt")
    pre_by_step = {step_number(record): record for record in pre_records}
    candidate_counts = {
        step: len(step_records)
        for step, step_records in coordinate_records_by_step(records).items()
    }

    lines = [
        "Momentum line-search summary",
        "",
        "step  candidates  pre_train  selected_train  pre_second  selected_second  head_lr  muon_lr  head_m  muon_m",
    ]
    for record in selected_records:
        step = record["step"]
        pre = pre_by_step.get(step, {})
        lines.append(
            "%4d  %10d  %9.6g  %14.6g  %10.6g  %15.6g  %7.4g  %7.4g  %s  %s"
            % (
                step_label(step),
                candidate_counts.get(step, 0),
                pre.get("train_loss", float("nan")),
                record["train_loss"],
                pre.get("eval_loss", float("nan")),
                record["eval_loss"],
                record["lrs"]["head"],
                record["lrs"]["muon"],
                format_momentum_list(record["momentums"]["head"]),
                format_momentum_list(record["momentums"]["muon"]),
            )
        )

    with open(output_path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    return output_path


def plot_run_loss_comparison(run_summaries, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plotted = False
    for summary in run_summaries:
        records = summary["selected_records"]
        if not records:
            continue
        plotted = True
        steps = [step_label(record["step"]) for record in records]
        ax.plot(
            steps,
            [record["train_loss"] for record in records],
            marker="o",
            label=f"{summary['run']} first batch",
        )
        ax.plot(
            steps,
            [record["eval_loss"] for record in records],
            marker="o",
            linestyle="--",
            label=f"{summary['run']} second batch",
        )
    if not plotted:
        plt.close(fig)
        return None

    ax.set_title("Selected Loss Comparison")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    style_step_axis(ax)
    ax.legend(fontsize=8)

    output_path = os.path.join(out_dir, "run_loss_comparison.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_run_lr_comparison(run_summaries, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes = axes.reshape(-1)
    plotted = False
    for ax, group in zip(axes, GROUPS):
        for summary in run_summaries:
            records = summary["selected_records"]
            if not records:
                continue
            plotted = True
            steps = [step_label(record["step"]) for record in records]
            ax.plot(
                steps,
                [record["lrs"][group] for record in records],
                marker="o",
                label=summary["run"],
            )
        ax.set_title(group)
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        style_step_axis(ax)
        ax.legend(fontsize=8)
    if not plotted:
        plt.close(fig)
        return None

    fig.suptitle("Selected Learning Rate Comparison")
    output_path = os.path.join(out_dir, "run_lr_comparison.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def format_momentum_list(values):
    if not values:
        return "[]"
    return "[" + ", ".join(f"{value:.3g}" for value in values) + "]"


def plot_one_run(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pre_records = pre_step_records(records)
    selected_records = best_records(records)
    if not selected_records:
        return [], pre_records, selected_records

    outputs = [
        plot_loss_over_steps(pre_records, selected_records, out_dir),
        plot_lrs_over_steps(selected_records, out_dir),
        plot_momentum_heatmaps(selected_records, out_dir),
        plot_momentum_paths(selected_records, out_dir),
        plot_new_momentum_over_steps(selected_records, out_dir),
        plot_search_candidate_counts(records, out_dir),
        plot_candidate_losses(records, selected_records, out_dir),
        write_summary(pre_records, selected_records, records, out_dir),
    ]
    return (
        [output for output in outputs if output is not None],
        pre_records,
        selected_records,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot MOMENTUM_LINE_SEARCH logs from cifar_line_search4.py."
    )
    parser.add_argument(
        "log_path",
        help="Training stdout log path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--out-dir",
        default="line_search_landscape3_plots4",
        help="Directory for generated PNG and summary files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = list(read_records(args.log_path))
    if not records:
        raise SystemExit("No MOMENTUM_LINE_SEARCH records found.")

    run_records = group_records_by_run(records)
    run_summaries = []
    outputs = []
    for run, records_for_run in run_records.items():
        run_out_dir = (
            args.out_dir
            if len(run_records) == 1
            else os.path.join(args.out_dir, safe_name(run))
        )
        run_outputs, pre_records, selected_records = plot_one_run(
            records_for_run, run_out_dir
        )
        outputs.extend(run_outputs)
        run_summaries.append(
            dict(
                run=run,
                pre_records=pre_records,
                selected_records=selected_records,
            )
        )

    if not any(summary["selected_records"] for summary in run_summaries):
        raise SystemExit("No coordinate_search records found.")

    if len(run_summaries) > 1:
        outputs.extend(
            output
            for output in (
                plot_run_loss_comparison(run_summaries, args.out_dir),
                plot_run_lr_comparison(run_summaries, args.out_dir),
            )
            if output is not None
        )

    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()

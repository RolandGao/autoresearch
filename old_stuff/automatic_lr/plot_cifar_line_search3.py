import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


LOG_PREFIX = "LR_LANDSCAPE "
GROUPS = ("head", "muon")
SUMMARY_TRAIN_STEPS = (2, 4, 10, 20)


def config_key(record):
    head_momentum = record.get("head_momentum")
    muon_momentum = record.get("muon_momentum")
    if head_momentum is None or muon_momentum is None:
        return None
    return (float(head_momentum), float(muon_momentum))


def config_sort_key(config):
    if config is None:
        return (-1.0, -1.0)
    return config


def group_records_by_config(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[config_key(record)].append(record)
    return dict(sorted(grouped.items(), key=lambda item: config_sort_key(item[0])))


def momentum_label(value):
    return f"{value:g}"


def style_step_axis(ax):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="x", color="0.7", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)


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
    style_step_axis(ax)
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
        style_step_axis(ax)
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
    style_step_axis(ax)
    ax.legend()

    output_path = os.path.join(out_dir, "component_loss_decrease_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def momentum_values(config_records):
    head_momentums = sorted(
        {config[0] for config in config_records if config is not None}
    )
    muon_momentums = sorted(
        {config[1] for config in config_records if config is not None}
    )
    return head_momentums, muon_momentums


def component_total_records(records):
    grouped = component_loss_decrease_records(records)
    result = []
    for step in sorted(grouped):
        step_records = list(grouped[step].values())
        if not step_records:
            continue
        result.append(
            dict(
                step=step,
                train_loss_decrease=sum(
                    record["train_loss_decrease"] for record in step_records
                ),
                eval_loss_decrease=sum(
                    record["eval_loss_decrease"] for record in step_records
                ),
            )
        )
    return result


def loss_curves(records):
    step_records = pre_step_records(records)
    steps = [int(record["step"]) for record in step_records]
    return [
        dict(label="first batch", linestyle="-", steps=steps, values=[
            record["train_loss"] for record in step_records
        ]),
        dict(label="second batch", linestyle="--", steps=steps, values=[
            record["eval_loss"] for record in step_records
        ]),
    ]


def component_loss_decrease_curves(records):
    step_records = component_total_records(records)
    steps = [int(record["step"]) for record in step_records]
    return [
        dict(label="first batch", linestyle="-", steps=steps, values=[
            record["train_loss_decrease"] for record in step_records
        ]),
        dict(label="second batch", linestyle="--", steps=steps, values=[
            record["eval_loss_decrease"] for record in step_records
        ]),
    ]


def optimal_lr_curves(lr_group):
    def curves(records):
        step_records = optimal_lr_records(records)
        steps = [int(record["step"]) for record in step_records]
        return [
            dict(label="first batch", linestyle="-", steps=steps, values=[
                record["lrs"][lr_group] for record in step_records
            ]),
            dict(label="second batch", linestyle="--", steps=steps, values=[
                record["second_batch_lrs"][lr_group] for record in step_records
            ]),
        ]

    return curves


def draw_momentum_curve_panel(
    ax,
    config_records,
    fixed_axis,
    fixed_value,
    varied_values,
    colors,
    curves_fn,
    metric_label=None,
):
    for index, varied_value in enumerate(varied_values):
        config = (
            (fixed_value, varied_value)
            if fixed_axis == "head"
            else (varied_value, fixed_value)
        )
        records = config_records.get(config)
        if not records:
            continue
        for curve in curves_fn(records):
            ax.plot(
                curve["steps"],
                curve["values"],
                color=colors[index % len(colors)],
                linestyle=curve["linestyle"],
                linewidth=1.2,
                alpha=0.9,
            )

    fixed_label = "head" if fixed_axis == "head" else "muon"
    varied_label = "muon" if fixed_axis == "head" else "head"
    title = (
        f"{fixed_label} momentum = {momentum_label(fixed_value)}; vary {varied_label}"
    )
    if metric_label is not None:
        title = f"{metric_label}: {title}"
    ax.set_title(
        title,
        fontsize=9,
    )
    style_step_axis(ax)


def add_curve_grid_legends(fig, varied_values, colors):
    momentum_handles = [
        Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            lw=1.8,
            label=momentum_label(value),
        )
        for index, value in enumerate(varied_values)
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="first batch"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="second batch"),
    ]
    fig.legend(
        handles=momentum_handles,
        title="varied momentum",
        loc="outside upper center",
        ncols=len(varied_values),
    )
    fig.legend(handles=style_handles, loc="outside lower center", ncols=2)


def plot_14_momentum_curve_grid(
    config_records, curves_fn, title, ylabel, out_dir, filename
):
    head_momentums, muon_momentums = momentum_values(config_records)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        len(head_momentums),
        2,
        figsize=(15, max(10, 2.0 * len(head_momentums))),
        sharex=True,
        constrained_layout=True,
    )

    for row, head_momentum in enumerate(head_momentums):
        ax = axes[row][0]
        draw_momentum_curve_panel(
            ax,
            config_records,
            "head",
            head_momentum,
            muon_momentums,
            colors,
            curves_fn,
        )
        ax.set_ylabel(ylabel)

    for row, muon_momentum in enumerate(muon_momentums):
        ax = axes[row][1]
        draw_momentum_curve_panel(
            ax,
            config_records,
            "muon",
            muon_momentum,
            head_momentums,
            colors,
            curves_fn,
        )

    for ax in axes[-1]:
        ax.set_xlabel("step")

    add_curve_grid_legends(fig, muon_momentums, colors)
    fig.suptitle(title)
    output_path = os.path.join(out_dir, filename)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_loss_momentum_comparison(config_records, out_dir):
    return plot_14_momentum_curve_grid(
        config_records,
        loss_curves,
        "Loss over steps by momentum config",
        "loss",
        out_dir,
        "loss_over_steps.png",
    )


def plot_optimal_lr_momentum_comparison(config_records, out_dir):
    head_momentums, muon_momentums = momentum_values(config_records)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        len(head_momentums),
        4,
        figsize=(24, max(10, 2.0 * len(head_momentums))),
        sharex=True,
        constrained_layout=True,
    )

    for row, head_momentum in enumerate(head_momentums):
        draw_momentum_curve_panel(
            axes[row][0],
            config_records,
            "head",
            head_momentum,
            muon_momentums,
            colors,
            optimal_lr_curves("head"),
            metric_label="head LR",
        )
        axes[row][0].set_ylabel("head LR")
        draw_momentum_curve_panel(
            axes[row][2],
            config_records,
            "head",
            head_momentum,
            muon_momentums,
            colors,
            optimal_lr_curves("muon"),
            metric_label="muon LR",
        )
        axes[row][2].set_ylabel("muon LR")

    for row, muon_momentum in enumerate(muon_momentums):
        draw_momentum_curve_panel(
            axes[row][1],
            config_records,
            "muon",
            muon_momentum,
            head_momentums,
            colors,
            optimal_lr_curves("head"),
            metric_label="head LR",
        )
        draw_momentum_curve_panel(
            axes[row][3],
            config_records,
            "muon",
            muon_momentum,
            head_momentums,
            colors,
            optimal_lr_curves("muon"),
            metric_label="muon LR",
        )

    for ax in axes[-1]:
        ax.set_xlabel("step")

    add_curve_grid_legends(fig, muon_momentums, colors)
    fig.suptitle("Optimal learning rates over steps by momentum config")
    output_path = os.path.join(out_dir, "optimal_lrs_over_steps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_component_loss_decrease_momentum_comparison(config_records, out_dir):
    return plot_14_momentum_curve_grid(
        config_records,
        component_loss_decrease_curves,
        "Total isolated loss decrease over steps by momentum config",
        "loss decrease",
        out_dir,
        "component_loss_decrease_over_steps.png",
    )


def post_search_loss_after_steps(records, train_steps):
    target_step = train_steps - 1
    records_by_step = {
        int(record["step"]): record for record in optimal_lr_records(records)
    }
    record = records_by_step.get(target_step)
    if record is None:
        return None
    return dict(
        logged_step=target_step,
        train_loss=record["train_loss"],
        eval_loss=record["eval_loss"],
    )


def metric_grid(config_records, train_steps):
    head_momentums, muon_momentums = momentum_values(config_records)
    values = {}
    for config, records in config_records.items():
        if config is None:
            continue
        values[config] = post_search_loss_after_steps(records, train_steps)
    return head_momentums, muon_momentums, values


def metric_grid_matrix(config_records, train_steps, loss_key):
    head_momentums, muon_momentums, values = metric_grid(config_records, train_steps)
    matrix = []
    for head_momentum in head_momentums:
        row = []
        for muon_momentum in muon_momentums:
            result = values.get((head_momentum, muon_momentum))
            row.append(float("nan") if result is None else result[loss_key])
        matrix.append(row)
    return head_momentums, muon_momentums, matrix


def format_grid_value(result, loss_key):
    if result is None:
        return "NA"
    return f"{result[loss_key]:.6g}"


def format_metric_grid(config_records, train_steps, loss_key):
    head_momentums, muon_momentums, values = metric_grid(config_records, train_steps)
    header = ["head\\muon"] + [momentum_label(value) for value in muon_momentums]
    rows = [header]
    for head_momentum in head_momentums:
        row = [momentum_label(head_momentum)]
        for muon_momentum in muon_momentums:
            row.append(
                format_grid_value(values.get((head_momentum, muon_momentum)), loss_key)
            )
        rows.append(row)

    widths = [
        max(len(row[col]) for row in rows)
        for col in range(len(header))
    ]
    return "\n".join(
        "  ".join(value.rjust(widths[col]) for col, value in enumerate(row))
        for row in rows
    )


def ranking_rows(config_records, train_steps):
    rows = []
    for config, records in config_records.items():
        if config is None:
            continue
        result = post_search_loss_after_steps(records, train_steps)
        if result is None:
            continue
        head_momentum, muon_momentum = config
        rows.append(
            dict(
                head_momentum=head_momentum,
                muon_momentum=muon_momentum,
                train_loss=result["train_loss"],
                second_batch_loss=result["eval_loss"],
            )
        )
    return sorted(rows, key=lambda row: row["train_loss"])


def format_ranking(config_records, train_steps):
    rows = ranking_rows(config_records, train_steps)
    if not rows:
        return "No complete records found."

    lines = [
        "rank  head_momentum  muon_momentum  train_loss  second_batch_loss",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            "%4d  %13s  %13s  %10.6g  %17.6g"
            % (
                rank,
                momentum_label(row["head_momentum"]),
                momentum_label(row["muon_momentum"]),
                row["train_loss"],
                row["second_batch_loss"],
            )
        )
    return "\n".join(lines)


def draw_loss_heatmap(ax, config_records, train_steps, loss_key, title):
    head_momentums, muon_momentums, matrix = metric_grid_matrix(
        config_records, train_steps, loss_key
    )
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("muon momentum")
    ax.set_ylabel("head momentum")
    ax.set_xticks(range(len(muon_momentums)))
    ax.set_xticklabels([momentum_label(value) for value in muon_momentums])
    ax.set_yticks(range(len(head_momentums)))
    ax.set_yticklabels([momentum_label(value) for value in head_momentums])
    for row, row_values in enumerate(matrix):
        for col, value in enumerate(row_values):
            label = "NA" if value != value else f"{value:.3g}"
            ax.text(col, row, label, ha="center", va="center", fontsize=7)
    return image


def plot_momentum_summary_heatmaps(
    config_records, out_dir, train_steps_list=SUMMARY_TRAIN_STEPS
):
    fig, axes = plt.subplots(
        len(train_steps_list),
        2,
        figsize=(13, max(10, 3.0 * len(train_steps_list))),
        constrained_layout=True,
    )
    for row, train_steps in enumerate(train_steps_list):
        for col, (loss_key, loss_label) in enumerate(
            (("train_loss", "train loss"), ("eval_loss", "second-batch loss"))
        ):
            ax = axes[row][col]
            image = draw_loss_heatmap(
                ax,
                config_records,
                train_steps,
                loss_key,
                f"After {train_steps} steps: {loss_label}",
            )
            fig.colorbar(image, ax=ax, label=loss_label)

    fig.suptitle("Momentum config loss heatmaps")
    output_path = os.path.join(out_dir, "momentum_config_loss_heatmaps.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_momentum_summary(
    config_records, out_dir, train_steps_list=SUMMARY_TRAIN_STEPS
):
    output_path = os.path.join(out_dir, "momentum_config_rankings.txt")
    lines = [
        "Momentum config rankings",
        "",
        "Rows are head momentum; columns are muon momentum.",
        "Grid values are post-search losses; lower is better.",
        "Rankings are ordered by train loss.",
        "After N steps uses the LR-search result logged at step N - 1.",
    ]
    for train_steps in train_steps_list:
        lines.extend(
            [
                "",
                f"After {train_steps} steps: train-loss grid",
                format_metric_grid(config_records, train_steps, "train_loss"),
                "",
                f"After {train_steps} steps: second-batch-loss grid",
                format_metric_grid(config_records, train_steps, "eval_loss"),
                "",
                f"After {train_steps} steps: ranking best to worst",
                format_ranking(config_records, train_steps),
            ]
        )
    with open(output_path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot LR_LANDSCAPE logs from cifar_line_search3.py."
    )
    parser.add_argument(
        "log_path",
        help="Training stdout log path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--out-dir",
        default="line_search_landscape_plots3",
        help="Directory for generated PNG files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = list(read_records(args.log_path))
    grouped = grouped_landscape_records(records)
    if not grouped:
        raise SystemExit("No LR_LANDSCAPE landscape records found.")
    config_records = group_records_by_config(records)

    if len(config_records) == 1:
        only_records = next(iter(config_records.values()))
        loss_output_path = plot_loss_over_steps(
            pre_step_records(only_records), args.out_dir
        )
        if loss_output_path is not None:
            print(loss_output_path)

        lr_output_path = plot_optimal_lrs_over_steps(
            optimal_lr_records(only_records), args.out_dir
        )
        if lr_output_path is not None:
            print(lr_output_path)

        decrease_output_path = plot_component_loss_decreases(
            component_loss_decrease_records(only_records), args.out_dir
        )
        if decrease_output_path is not None:
            print(decrease_output_path)
    else:
        print(plot_loss_momentum_comparison(config_records, args.out_dir))
        print(plot_optimal_lr_momentum_comparison(config_records, args.out_dir))
        print(
            plot_component_loss_decrease_momentum_comparison(
                config_records, args.out_dir
            )
        )
        print(plot_momentum_summary_heatmaps(config_records, args.out_dir))
        print(write_momentum_summary(config_records, args.out_dir))


if __name__ == "__main__":
    main()

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def latest_log_path():
    log_paths = sorted(
        Path("logs").glob("*/log.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not log_paths:
        raise FileNotFoundError("No log.pt files found under ./logs")
    return log_paths[0]


def load_sweep_results(paths):
    results = []
    for path in paths:
        data = torch.load(path, map_location="cpu")
        for result in data["sweep_results"]:
            result = dict(result)
            result["source"] = str(path)
            results.append(result)
    return results


def lr_multiplier(result, x):
    shape = result["shape"]
    parameter = result["parameter"]
    if shape == "linear":
        return 1 - x
    if shape == "power":
        return (1 - x) ** parameter
    if shape == "exp":
        return math.exp(-parameter * x)
    raise ValueError("Unknown schedule shape: %s" % shape)


def piecewise_segment_lr(segment, step):
    span = max(segment["end_step"] - segment["start_step"], 1)
    x = min(max((step - segment["start_step"]) / span, 0.0), 1.0)
    start_lr = segment["start_lr"]
    end_lr = segment["end_lr"]
    if segment["shape"] == "linear":
        return start_lr + (end_lr - start_lr) * x
    if segment["shape"] == "power":
        return end_lr + (start_lr - end_lr) * ((1 - x) ** segment["power"])
    if segment["shape"] == "exp":
        safe_start = max(start_lr, 1e-5)
        safe_end = max(end_lr, 1e-5)
        return safe_start * ((safe_end / safe_start) ** x)
    raise ValueError("Unknown segment shape: %s" % segment["shape"])


def piecewise_lr(result, x):
    step = x * 200
    for segment in result["segments"]:
        if segment["start_step"] <= step < segment["end_step"]:
            return piecewise_segment_lr(segment, step)
    return piecewise_segment_lr(result["segments"][-1], step)


def schedule_lr(result, x):
    if "segments" in result:
        return piecewise_lr(result, x)
    return result["initial_lr"] * lr_multiplier(result, x)


def label_for(result):
    if "segments" in result:
        return "%s run=%s tta=%.4f" % (
            result.get("shape_pattern", "piecewise"),
            result.get("index", "?"),
            result["tta_val_acc"],
        )
    parameter = "" if result["parameter"] is None else " param=%.4g" % result["parameter"]
    return "%s lr=%.4g%s tta=%.4f" % (
        result["shape"],
        result["initial_lr"],
        parameter,
        result["tta_val_acc"],
    )


def plot_top_schedules(results, threshold, output_path):
    selected = [
        result for result in results
        if result.get("tta_val_acc", float("-inf")) >= threshold
    ]
    selected.sort(
        key=lambda result: (result["tta_val_acc"], result["val_acc"]),
        reverse=True,
    )
    if not selected:
        raise ValueError("No runs found with tta_val_acc >= %.4f" % threshold)

    xs = [i / 200 for i in range(201)]
    plt.figure(figsize=(11, 7))
    for result in selected:
        ys = [schedule_lr(result, x) for x in xs]
        plt.plot(xs, ys, linewidth=1.8, alpha=0.85, label=label_for(result))

    plt.title("Muon LR schedules with TTA val >= %.4f" % threshold)
    plt.xlabel("training progress x")
    plt.ylabel("Muon learning rate")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Plot all Muon LR schedules whose tta_val_acc meets a threshold."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="One or more sweep log.pt files. Defaults to the newest ./logs/*/log.pt.",
    )
    parser.add_argument("--threshold", type=float, default=0.94)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch/automatic_lr2/top_tta_schedules.png"),
    )
    args = parser.parse_args()

    log_paths = args.logs or [latest_log_path()]
    results = load_sweep_results(log_paths)
    selected = plot_top_schedules(results, args.threshold, args.output)
    print("Loaded %d runs from %d log file(s)" % (len(results), len(log_paths)))
    print("Plotted %d runs with tta_val_acc >= %.4f" % (len(selected), args.threshold))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

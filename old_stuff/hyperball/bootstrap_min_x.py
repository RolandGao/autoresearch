"""Bootstrap confidence intervals for the best-x value in hyperball logs."""

import argparse
import json
import math
import random
from pathlib import Path


DEFAULT_LOG = Path(__file__).with_name("hyperball_exp3.log")
SUMMARY_PREFIX = "RUN_SUMMARY "
ALL_SUMMARIES_PREFIX = "ALL_RUN_SUMMARIES "
DEFAULT_REPEATS = 10_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse RUN_SUMMARY records and bootstrap the x value selected by "
            "minimum error."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=DEFAULT_LOG,
        type=Path,
        help=f"log file to parse (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--x-field",
        action="append",
        default=None,
        help=(
            "numeric field to bootstrap, e.g. depth or "
            "hparams.matrix_initial_lrs.wte. May be passed multiple times. "
            "Defaults to every numeric field except metadata and error fields."
        ),
    )
    parser.add_argument(
        "--error-field",
        default=None,
        help="error field to minimize. Defaults to smooth_train_loss, then train_loss, then val_bpb.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.25,
        help="fraction of pairs to sample with replacement each bootstrap repeat",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="bootstrap repeats (default: 10000, i.e. 10^4)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def flatten_numeric(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_numeric(value, name))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        items[prefix] = float(obj)
    return items


def load_run_summaries(path):
    summaries = []
    all_summaries = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(SUMMARY_PREFIX):
                summaries.append(json.loads(line[len(SUMMARY_PREFIX) :]))
            elif line.startswith(ALL_SUMMARIES_PREFIX):
                all_summaries = json.loads(line[len(ALL_SUMMARIES_PREFIX) :])
    if summaries:
        return summaries
    if all_summaries is not None:
        return all_summaries
    raise ValueError(f"found no RUN_SUMMARY records in {path}")


def pick_error_field(flat_records, requested):
    if requested is not None:
        if requested not in flat_records[0]:
            raise ValueError(f"error field not found: {requested}")
        return requested
    for name in ("smooth_train_loss", "train_loss", "val_bpb"):
        if name in flat_records[0]:
            return name
    raise ValueError("could not infer error field; pass --error-field")


def percentile(sorted_values, pct):
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = pct / 100.0 * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_best_x(pairs, repeats, sample_frac, rng):
    n = len(pairs)
    sample_size = max(1, math.ceil(n * sample_frac))
    selected_x = []
    for _ in range(repeats):
        best_x = None
        best_error = None
        for _ in range(sample_size):
            x, error = pairs[rng.randrange(n)]
            if best_error is None or error < best_error:
                best_x = x
                best_error = error
        selected_x.append(best_x)
    selected_x.sort()
    return {
        "median": percentile(selected_x, 50),
        "ci_low": percentile(selected_x, 2.5),
        "ci_high": percentile(selected_x, 97.5),
    }


def fmt_num(value):
    return f"{value:.6g}"


def print_table(rows):
    headers = (
        "field",
        "observed_x",
        "best_error",
        "median_x",
        "ci95_low",
        "ci95_high",
    )
    widths = [len(header) for header in headers]
    formatted_rows = []
    for row in rows:
        formatted = (
            row["field"],
            fmt_num(row["observed_best_x"]),
            fmt_num(row["observed_best_error"]),
            fmt_num(row["median"]),
            fmt_num(row["ci_low"]),
            fmt_num(row["ci_high"]),
        )
        formatted_rows.append(formatted)
        widths = [max(width, len(value)) for width, value in zip(widths, formatted)]

    def render(values):
        return (
            values[0].ljust(widths[0])
            + "  "
            + "  ".join(
                value.rjust(width)
                for value, width in zip(values[1:], widths[1:])
            )
        )

    print(render(headers))
    print(
        "-" * widths[0]
        + "  "
        + "  ".join("-" * width for width in widths[1:])
    )
    for row in formatted_rows:
        print(render(row))


def candidate_x_fields(flat_records, error_field, requested):
    if requested:
        missing = [name for name in requested if name not in flat_records[0]]
        if missing:
            raise ValueError(f"x field(s) not found: {', '.join(missing)}")
        return requested
    excluded = {
        error_field,
        "train_loss",
        "smooth_train_loss",
        "val_bpb",
        "run_idx",
        "hparams.run_idx",
        "num_steps",
        "num_params_M",
        "total_seconds",
        "training_seconds",
        "peak_vram_mb",
        "mfu_percent",
        "total_tokens_M",
    }
    return sorted(name for name in flat_records[0] if name not in excluded)


def main():
    args = parse_args()
    summaries = load_run_summaries(args.log_path)
    flat_records = [flatten_numeric(summary) for summary in summaries]
    error_field = pick_error_field(flat_records, args.error_field)
    x_fields = candidate_x_fields(flat_records, error_field, args.x_field)
    rng = random.Random(args.seed)

    print(f"log_path: {args.log_path}")
    print(f"records: {len(flat_records)}")
    print(f"error_field: {error_field}")
    print(f"sample_fraction: {args.sample_frac}")
    print(f"sample_size: {max(1, math.ceil(len(flat_records) * args.sample_frac))}")
    print(f"repeats: {args.repeats}")
    print()

    rows = []
    for field in x_fields:
        pairs = [
            (record[field], record[error_field])
            for record in flat_records
            if field in record and error_field in record
        ]
        if not pairs:
            continue
        observed_best_x, observed_best_error = min(pairs, key=lambda pair: pair[1])
        result = bootstrap_best_x(pairs, args.repeats, args.sample_frac, rng)
        rows.append(
            {
                "field": field,
                "observed_best_x": observed_best_x,
                "observed_best_error": observed_best_error,
                **result,
            }
        )
    print_table(rows)


if __name__ == "__main__":
    main()
